import { MoebiusPipeline } from "./pipeline.ts";
import { IMG, toSquareCanvas, type Fitted } from "./imaging.ts";
import { cachedBytes } from "./modelcache.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const imageCanvas = $<HTMLCanvasElement>("image");
const maskCanvas = $<HTMLCanvasElement>("mask");
const resultCanvas = $<HTMLCanvasElement>("result");
const resultPlaceholder = $("result-placeholder");
const statusEl = $("status");
const backendEl = $("backend");
const bar = $<HTMLDivElement>("bar");
const barLabel = $("bar-label");
const runBtn = $<HTMLButtonElement>("run");
const runHint = $("run-hint");
const downloadLink = $<HTMLAnchorElement>("download");

const ictx = imageCanvas.getContext("2d")!;
const mctx = maskCanvas.getContext("2d")!;

let hasImage = false;
let fitRect: Fitted["rect"] | null = null; // content rect inside the 512 letterbox
let loadPromise: Promise<MoebiusPipeline | null> | null = null;

// ---------- mask painting ----------
let painting = false;
let brush = 40;
$<HTMLInputElement>("brush").addEventListener("input", (e) => {
  brush = +(e.target as HTMLInputElement).value;
});

function canvasPos(e: PointerEvent): [number, number] {
  const r = maskCanvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * IMG, ((e.clientY - r.top) / r.height) * IMG];
}
function paintAt(x: number, y: number) {
  mctx.fillStyle = "rgba(110,168,254,0.6)";
  mctx.beginPath();
  mctx.arc(x, y, brush / 2, 0, Math.PI * 2);
  mctx.fill();
}
maskCanvas.addEventListener("pointerdown", (e) => {
  if (!hasImage) return;
  painting = true;
  maskCanvas.setPointerCapture(e.pointerId);
  const [x, y] = canvasPos(e);
  paintAt(x, y);
});
maskCanvas.addEventListener("pointermove", (e) => {
  if (!painting) return;
  const [x, y] = canvasPos(e);
  paintAt(x, y);
});
maskCanvas.addEventListener("pointerup", () => (painting = false));

$("clear-mask").addEventListener("click", () => mctx.clearRect(0, 0, IMG, IMG));

// ---------- image loading ----------
function setImage(img: HTMLImageElement) {
  const fitted = toSquareCanvas(img, img.naturalWidth, img.naturalHeight);
  fitRect = fitted.rect;
  ictx.clearRect(0, 0, IMG, IMG);
  ictx.drawImage(fitted.canvas, 0, 0);
  mctx.clearRect(0, 0, IMG, IMG);
  hasImage = true;
  maybeEnableRun();
}

$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => setImage(img);
  img.src = URL.createObjectURL(f);
});

$("sample").addEventListener("click", () => {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => setImage(img);
  img.onerror = () => (statusEl.textContent = "Could not load sample image.");
  img.src = `${import.meta.env.BASE_URL}sample.png`;
});

// ---------- model loading ----------
// Configurable so the production build can point at a Hugging Face model repo while
// local dev uses the on-disk symlink. Set VITE_MODEL_BASE / VITE_ORT_BASE at build time.
// Defaults derive from Vite's BASE_URL so the app works at root (dev, HF Space) or under
// a subpath (GitHub Pages project site, base "/<repo>/"). VITE_MODEL_BASE typically points
// at an absolute Hugging Face model-repo URL for deployment.
const MODEL_BASE = import.meta.env.VITE_MODEL_BASE ?? `${import.meta.env.BASE_URL}models`;
const ORT_BASE = import.meta.env.VITE_ORT_BASE ?? `${import.meta.env.BASE_URL}ort/`;

// Returns a shared promise that resolves only once the ONNX sessions are actually loaded.
// Concurrent callers (background prefetch + Run click) await the same promise, so Run never
// races ahead of session creation.
function ensureModels(): Promise<MoebiusPipeline | null> {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    MoebiusPipeline.configureRuntime(ORT_BASE);
    const p = new MoebiusPipeline();
    try {
      await p.load(MODEL_BASE, setProgress);
      backendEl.textContent = `Runtime: ONNX Runtime Web · ${p.backend.toUpperCase()}`;
      statusEl.textContent = "Models ready.";
      runHint.style.display = "none";
      return p;
    } catch (err) {
      statusEl.textContent = "Model load failed: " + (err as Error).message;
      loadPromise = null; // allow a retry on next click
      return null;
    }
  })();
  return loadPromise;
}

function maybeEnableRun() {
  runBtn.disabled = !hasImage;
}

// Shared progress renderer for both model loading (byte counts) and denoising (step counts).
function setProgress(stage: string, cur?: number, total?: number) {
  if (!total) {
    statusEl.textContent = stage + "…";
    barLabel.textContent = "";
    return;
  }
  const pct = (cur! / total) * 100;
  bar.style.width = `${pct}%`;
  statusEl.textContent = stage;
  // large totals ⇒ byte download; format as MB
  barLabel.textContent =
    total > 100000
      ? `${stage} — ${(cur! / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(0)} MB (${pct.toFixed(0)}%)`
      : `${stage} — ${cur} / ${total} (${pct.toFixed(0)}%)`;
}

// ---------- run ----------
$<HTMLInputElement>("steps").addEventListener("input", (e) => {
  $("steps-v").textContent = (e.target as HTMLInputElement).value;
});
$<HTMLInputElement>("cfg").addEventListener("input", (e) => {
  $("cfg-v").textContent = (e.target as HTMLInputElement).value;
});

runBtn.addEventListener("click", async () => {
  if (!hasImage) return;
  runBtn.disabled = true;
  const pipe = await ensureModels();
  if (!pipe) {
    runBtn.disabled = false;
    return;
  }
  const t0 = performance.now();
  try {
    const out = await pipe.run(imageCanvas, maskCanvas, {
      steps: +$<HTMLInputElement>("steps").value,
      guidance: +$<HTMLInputElement>("cfg").value,
      seed: +$<HTMLInputElement>("seed").value,
      paste: $<HTMLInputElement>("paste").checked,
      onProgress: setProgress,
      livePreview: () => $<HTMLInputElement>("live-preview").checked,
      onStep: (canvas) => {
        resultCanvas.getContext("2d")!.drawImage(canvas, 0, 0);
        resultPlaceholder.style.display = "none";
      },
    });
    resultCanvas.getContext("2d")!.drawImage(out, 0, 0);
    resultPlaceholder.style.display = "none";
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    statusEl.textContent = `Done in ${secs}s`;
    bar.style.width = "100%";
    // crop the 512 letterbox back to the original aspect ratio for download
    let dlCanvas: HTMLCanvasElement = resultCanvas;
    if (fitRect && (fitRect.w !== IMG || fitRect.h !== IMG)) {
      const crop = document.createElement("canvas");
      crop.width = fitRect.w;
      crop.height = fitRect.h;
      crop
        .getContext("2d")!
        .drawImage(resultCanvas, fitRect.x, fitRect.y, fitRect.w, fitRect.h, 0, 0, fitRect.w, fitRect.h);
      dlCanvas = crop;
    }
    downloadLink.href = dlCanvas.toDataURL("image/png");
    downloadLink.download = "moebius-inpaint.png";
    downloadLink.style.display = "inline-block";
  } catch (err) {
    statusEl.textContent = "Error: " + (err as Error).message;
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
});

// Don't download anything until the user clicks Run. But if the weights are already in the
// browser's Cache Storage from a previous visit, drop the "(downloads models)" hint.
window.addEventListener("load", async () => {
  if ((await cachedBytes()) > 1_200_000_000) runHint.style.display = "none";
});
