import * as ort from "onnxruntime-web/webgpu";
import { makeDDIM, ddimStep } from "./ddim.ts";
import { loadModelBytes, requestPersistentStorage } from "./modelcache.ts";
import {
  IMG,
  LAT,
  randn,
  canvasToCHW,
  maskCanvasToBinary,
  makeMaskedCHW,
  maskToLatent,
  chwToImageData,
  pasteBack,
} from "./imaging.ts";

const SCALING_FACTOR = 0.13025;
const NOISE_OFFSET = 0.0357;
const HALF_IDS = 10; // num_embeddings/2

export interface Progress {
  (stage: string, step?: number, total?: number): void;
}

export interface RunOptions {
  steps: number;
  guidance: number;
  seed: number;
  paste: boolean;
  onProgress?: Progress;
  onStep?: (canvas: HTMLCanvasElement) => void;
  livePreview?: () => boolean;
}

export class MoebiusPipeline {
  private enc!: ort.InferenceSession;
  private dec!: ort.InferenceSession;
  private unet!: ort.InferenceSession;
  public backend = "unknown";

  static configureRuntime(wasmPaths: string) {
    ort.env.wasm.wasmPaths = wasmPaths;
    // Multi-threaded WASM needs SharedArrayBuffer, which needs cross-origin isolation
    // (COOP/COEP). On hosts without it (e.g. a plain static HF Space) fall back to a
    // single thread — the WebGPU path doesn't need threads anyway.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(navigator.hardwareConcurrency || 4, 8)
      : 1;
  }

  async load(modelBase: string, onProgress?: Progress) {
    const ep: ("webgpu" | "wasm")[] =
      "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: ep,
      graphOptimizationLevel: "all",
    };
    void requestPersistentStorage();

    // Fetch each model's bytes through the persistent Cache Storage layer (keyed by the
    // stable URL), then build the session from the bytes. This avoids re-downloading on
    // every load despite Hugging Face's rotating signed CDN URLs.
    const get = (file: string, label: string, idx: number) =>
      loadModelBytes(`${modelBase}/${file}`, (loaded, total, fromCache) =>
        onProgress?.(
          fromCache ? `${label} (cached, ${idx}/3)` : `Downloading ${label} (${idx}/3)`,
          loaded,
          total,
        ),
      );

    this.enc = await ort.InferenceSession.create(await get("vae_encoder.onnx", "VAE encoder", 1), opts);
    this.dec = await ort.InferenceSession.create(await get("vae_decoder.onnx", "VAE decoder", 2), opts);
    this.unet = await ort.InferenceSession.create(await get("unet.onnx", "UNet", 3), opts);
    // report whichever EP actually got selected for the unet
    this.backend = ep[0];
  }

  private async encode(chw: Float32Array): Promise<Float32Array> {
    const t = new ort.Tensor("float32", chw, [1, 3, IMG, IMG]);
    const { moments } = await this.enc.run({ image: t });
    const m = moments.data as Float32Array;
    // take mean channels (first 4) * scaling_factor
    const out = new Float32Array(4 * LAT * LAT);
    for (let i = 0; i < out.length; i++) out[i] = m[i] * SCALING_FACTOR;
    return out;
  }

  private async decode(latent: Float32Array): Promise<ImageData> {
    const scaled = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] / SCALING_FACTOR;
    const t = new ort.Tensor("float32", scaled, [1, 4, LAT, LAT]);
    const { image } = await this.dec.run({ latent: t });
    return chwToImageData(image.data as Float32Array);
  }

  // Assemble the (2,9,64,64) CFG input from latents/mask/maskedLatent and run the UNet,
  // returning the classifier-free-guided noise prediction (4,64,64).
  private async unetCFG(
    latents: Float32Array,
    mask64: Float32Array,
    maskedLat: Float32Array,
    t: number,
    guidance: number,
  ): Promise<Float32Array> {
    const plane = LAT * LAT;
    const nine = new Float32Array(9 * plane);
    nine.set(latents.subarray(0, 4 * plane), 0); // ch 0-3 noisy latents
    nine.set(mask64, 4 * plane); // ch 4 mask
    nine.set(maskedLat.subarray(0, 4 * plane), 5 * plane); // ch 5-8 masked latents

    // batch of 2 (uncond, cond) — same 9ch, different input_ids
    const nine2 = new Float32Array(2 * 9 * plane);
    nine2.set(nine, 0);
    nine2.set(nine, 9 * plane);

    const ids = new BigInt64Array(2 * HALF_IDS);
    for (let i = 0; i < HALF_IDS; i++) {
      ids[i] = BigInt(HALF_IDS + i); // uncond [10..19]
      ids[HALF_IDS + i] = BigInt(i); // cond   [0..9]
    }
    const ts = new BigInt64Array([BigInt(t), BigInt(t)]);

    const out = await this.unet.run({
      latent: new ort.Tensor("float32", nine2, [2, 9, LAT, LAT]),
      timesteps: new ort.Tensor("int64", ts, [2]),
      input_ids: new ort.Tensor("int64", ids, [2, HALF_IDS]),
    });
    const noise = out.noise.data as Float32Array;
    const n = 4 * plane;
    const cfg = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u = noise[i];
      const c = noise[n + i];
      cfg[i] = u + guidance * (c - u);
    }
    return cfg;
  }

  async run(
    imageCanvas: HTMLCanvasElement,
    maskCanvas: HTMLCanvasElement,
    opts: RunOptions,
  ): Promise<HTMLCanvasElement> {
    const { steps, guidance, seed, paste, onProgress, onStep, livePreview } = opts;
    const ddim = makeDDIM(steps);

    onProgress?.("Encoding image");
    const imgCHW = canvasToCHW(imageCanvas);
    const maskBin = maskCanvasToBinary(maskCanvas);
    const maskedCHW = makeMaskedCHW(imgCHW, maskBin);
    const mask64 = maskToLatent(maskBin);

    const maskedLat = await this.encode(maskedCHW);
    // (image latents themselves aren't needed when strength≈1; we start from pure noise)

    // init latents = noise [+ noise_offset]
    const plane = LAT * LAT;
    let latents = randn(4 * plane, seed);
    const off = randn(4, seed ^ 0x9e3779b9);
    for (let c = 0; c < 4; c++) {
      for (let p = 0; p < plane; p++) latents[c * plane + p] += NOISE_OFFSET * off[c];
    }

    // Decode latents and apply paste-back if requested — shared by step previews and final result.
    const buildResult = async (lat: Float32Array): Promise<HTMLCanvasElement> => {
      const img = await this.decode(lat);
      if (paste) return pasteBack(img, imageCanvas, maskBin);
      const c = document.createElement("canvas");
      c.width = IMG;
      c.height = IMG;
      c.getContext("2d")!.putImageData(img, 0, 0);
      return c;
    };

    let lastStepCanvas: HTMLCanvasElement | null = null;

    const tl = ddim.timesteps;
    for (let i = 0; i < tl.length; i++) {
      const t = tl[i];
      const prevT = i + 1 < tl.length ? tl[i + 1] : -1;
      onProgress?.("Denoising", i + 1, tl.length);
      const eps = await this.unetCFG(latents, mask64, maskedLat, t, guidance);
      latents = ddimStep(eps, latents, t, prevT, ddim);
      if (onStep && livePreview?.()) {
        lastStepCanvas = await buildResult(latents);
        onStep(lastStepCanvas);
      } else {
        lastStepCanvas = null;
        // yield to UI so progress bar can repaint
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    onProgress?.("Decoding");
    return lastStepCanvas ?? await buildResult(latents);
  }
}
