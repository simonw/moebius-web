// Image <-> tensor helpers. Everything is fixed at 512x512 image / 64x64 latent
// because the model's cross-attention rel_pos_emb is tied to the trained resolution.

export const IMG = 512;
export const LAT = 64;

// Seedable PRNG (mulberry32) + Box–Muller for reproducible Gaussian noise.
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(n: number, seed: number): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // Box–Muller
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    out[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  return out;
}

// Draw a source image scaled into a 512x512 canvas, return that canvas.
export function toSquareCanvas(src: CanvasImageSource): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = IMG;
  c.height = IMG;
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, IMG, IMG);
  return c;
}

// RGB image canvas -> CHW Float32Array in [-1, 1]. (1,3,512,512)
export function canvasToCHW(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext("2d")!;
  const { data } = ctx.getImageData(0, 0, IMG, IMG); // RGBA, HWC
  const out = new Float32Array(3 * IMG * IMG);
  const plane = IMG * IMG;
  for (let p = 0; p < plane; p++) {
    out[p] = (data[p * 4] / 255) * 2 - 1; // R
    out[plane + p] = (data[p * 4 + 1] / 255) * 2 - 1; // G
    out[2 * plane + p] = (data[p * 4 + 2] / 255) * 2 - 1; // B
  }
  return out;
}

// Binary mask (1 = inpaint) from a mask canvas where painted (alpha>0 or white) = hole.
// Returns Float32Array length 512*512.
export function maskCanvasToBinary(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext("2d")!;
  const { data } = ctx.getImageData(0, 0, IMG, IMG);
  const out = new Float32Array(IMG * IMG);
  for (let p = 0; p < IMG * IMG; p++) {
    // use alpha channel of the overlay (painted strokes have alpha>0)
    out[p] = data[p * 4 + 3] >= 128 ? 1 : 0;
  }
  return out;
}

// masked image = image * (1 - mask), per channel. CHW in/out.
export function makeMaskedCHW(imgCHW: Float32Array, maskBin: Float32Array): Float32Array {
  const out = new Float32Array(imgCHW.length);
  const plane = IMG * IMG;
  for (let c = 0; c < 3; c++) {
    for (let p = 0; p < plane; p++) {
      // masked region (mask=1) -> value at -1? pipeline does image*(1-mask) on [-1,1] image,
      // so hole becomes 0.0 (mid-gray) in [-1,1] space. Match exactly.
      out[c * plane + p] = imgCHW[c * plane + p] * (1 - maskBin[p]);
    }
  }
  return out;
}

// Downsample 512x512 binary mask -> 64x64 using PyTorch-nearest (top-left of each 8x8 block).
export function maskToLatent(maskBin: Float32Array): Float32Array {
  const out = new Float32Array(LAT * LAT);
  const ratio = IMG / LAT; // 8
  for (let y = 0; y < LAT; y++) {
    for (let x = 0; x < LAT; x++) {
      out[y * LAT + x] = maskBin[y * ratio * IMG + x * ratio];
    }
  }
  return out;
}

// Decoder output (1,3,512,512) in [-1,1] -> ImageData (clipped 0..1 -> 0..255).
export function chwToImageData(chw: Float32Array): ImageData {
  const plane = IMG * IMG;
  const out = new ImageData(IMG, IMG);
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) {
      let v = (chw[c * plane + p] + 1) / 2;
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      out.data[p * 4 + c] = Math.round(v * 255);
    }
    out.data[p * 4 + 3] = 255;
  }
  return out;
}

// Paste-back: result*blur(mask) + (1-blur(mask))*original, with a 3px gaussian-blurred mask.
export function pasteBack(
  resultData: ImageData,
  originalCanvas: HTMLCanvasElement,
  maskBin: Float32Array,
): HTMLCanvasElement {
  // build a blurred mask canvas
  const mc = document.createElement("canvas");
  mc.width = IMG;
  mc.height = IMG;
  const mctx = mc.getContext("2d")!;
  const mdata = new ImageData(IMG, IMG);
  for (let p = 0; p < IMG * IMG; p++) {
    const v = maskBin[p] * 255;
    mdata.data[p * 4] = v;
    mdata.data[p * 4 + 1] = v;
    mdata.data[p * 4 + 2] = v;
    mdata.data[p * 4 + 3] = 255;
  }
  mctx.putImageData(mdata, 0, 0);
  // re-draw through blur filter
  const blur = document.createElement("canvas");
  blur.width = IMG;
  blur.height = IMG;
  const bctx = blur.getContext("2d")!;
  bctx.filter = "blur(3px)";
  bctx.drawImage(mc, 0, 0);
  const blurMask = bctx.getImageData(0, 0, IMG, IMG).data;

  const orig = originalCanvas.getContext("2d")!.getImageData(0, 0, IMG, IMG).data;

  const out = document.createElement("canvas");
  out.width = IMG;
  out.height = IMG;
  const octx = out.getContext("2d")!;
  const blended = new ImageData(IMG, IMG);
  for (let p = 0; p < IMG * IMG; p++) {
    const m = blurMask[p * 4] / 255;
    for (let c = 0; c < 3; c++) {
      blended.data[p * 4 + c] = Math.round(
        resultData.data[p * 4 + c] * m + orig[p * 4 + c] * (1 - m),
      );
    }
    blended.data[p * 4 + 3] = 255;
  }
  octx.putImageData(blended, 0, 0);
  return out;
}
