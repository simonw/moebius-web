// Persistent model caching via the Cache Storage API — the same approach Transformers.js
// uses. We key by the STABLE Hugging Face URL (…/resolve/main/<file>), not the signed CDN
// URL it redirects to. HF's redirect target carries rotating Expires/Signature query params,
// so the browser HTTP cache never hits across loads; and a redirected fetch Response can't be
// cache.put() directly. So we download the bytes once, build a fresh Response, and store it
// under the stable URL. Subsequent loads (and future sessions) read straight from the cache.

const CACHE_NAME = "moebius-onnx-v1";

export interface DownloadProgress {
  (loaded: number, total: number, fromCache: boolean): void;
}

// Ask the browser to keep our storage (reduces eviction risk for ~1.2 GB of weights).
export async function requestPersistentStorage(): Promise<boolean> {
  try {
    if (navigator.storage?.persist) {
      if (await navigator.storage.persisted()) return true;
      return await navigator.storage.persist();
    }
  } catch {
    /* not supported */
  }
  return false;
}

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // e.g. private mode / cache disabled
  }
}

// Return the model bytes, from cache if present, else download (with progress) and cache.
export async function loadModelBytes(
  url: string,
  onProgress?: DownloadProgress,
): Promise<Uint8Array> {
  const cache = await openCache();

  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress?.(buf.byteLength, buf.byteLength, true);
      return new Uint8Array(buf);
    }
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → HTTP ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || 0;

  let bytes: Uint8Array;
  const reader = resp.body?.getReader();
  if (reader && total > 0) {
    // preallocate one buffer (memory-efficient for ~900 MB) and fill as chunks arrive
    bytes = new Uint8Array(total);
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, loaded);
      loaded += value.length;
      onProgress?.(loaded, total, false);
    }
  } else {
    const buf = await resp.arrayBuffer();
    bytes = new Uint8Array(buf);
    onProgress?.(bytes.length, bytes.length, false);
  }

  if (cache) {
    try {
      // build a fresh (non-redirected) Response so it can be stored under the stable URL
      await cache.put(
        url,
        new Response(bytes.buffer as ArrayBuffer, {
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-Length": String(bytes.length),
          },
        }),
      );
    } catch (e) {
      console.warn("[modelcache] cache.put failed (continuing uncached):", e);
    }
  }
  return bytes;
}

// For a "clear cache" affordance.
export async function clearModelCache(): Promise<void> {
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    /* ignore */
  }
}

export async function cachedBytes(): Promise<number> {
  const cache = await openCache();
  if (!cache) return 0;
  let total = 0;
  for (const req of await cache.keys()) {
    const r = await cache.match(req);
    const len = Number(r?.headers.get("content-length")) || 0;
    total += len;
  }
  return total;
}
