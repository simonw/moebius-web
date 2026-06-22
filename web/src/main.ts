import { MoebiusPipeline } from "./pipeline.ts";
import { IMG, toSquareCanvas } from "./imaging.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const imageCanvas = $<HTMLCanvasElement>("image");
const maskCanvas = $<HTMLCanvasElement>("mask");
const resultCanvas = $<HTMLCanvasElement>("result");
const resultPlaceholder = $("result-placeholder");
const statusEl = $("status");
const backendEl = $("backend");
const bar = $<HTMLDivElement>("bar");
const runBtn = $<HTMLButtonElement>("run");
const downloadLink = $<HTMLAnchorElement>("download");

const ictx = imageCanvas.getContext("2d")!;
const mctx = maskCanvas.getContext("2d")!;

let hasImage = false;
let pipeline: MoebiusPipeline | null = null;
let loadingModels = false;

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
function setImage(src: CanvasImageSource) {
  const sq = toSquareCanvas(src);
  ictx.drawImage(sq, 0, 0);
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
  img.src = "/sample.png";
});

// ---------- model loading ----------
async function ensureModels() {
  if (pipeline || loadingModels) return;
  loadingModels = true;
  MoebiusPipeline.configureRuntime("/ort/");
  pipeline = new MoebiusPipeline();
  try {
    await pipeline.load("/models", (stage) => (statusEl.textContent = stage + "…"));
    backendEl.textContent = `Runtime: ONNX Runtime Web · ${pipeline.backend.toUpperCase()}`;
    statusEl.textContent = "Models ready.";
  } catch (err) {
    statusEl.textContent = "Model load failed: " + (err as Error).message;
    pipeline = null;
  } finally {
    loadingModels = false;
  }
}

function maybeEnableRun() {
  runBtn.disabled = !hasImage;
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
  await ensureModels();
  if (!pipeline) {
    runBtn.disabled = false;
    return;
  }
  const t0 = performance.now();
  try {
    const out = await pipeline.run(imageCanvas, maskCanvas, {
      steps: +$<HTMLInputElement>("steps").value,
      guidance: +$<HTMLInputElement>("cfg").value,
      seed: +$<HTMLInputElement>("seed").value,
      paste: $<HTMLInputElement>("paste").checked,
      onProgress: (stage, step, total) => {
        statusEl.textContent = total ? `${stage} ${step}/${total}` : stage + "…";
        if (total) bar.style.width = `${(step! / total) * 100}%`;
      },
    });
    resultCanvas.getContext("2d")!.drawImage(out, 0, 0);
    resultPlaceholder.style.display = "none";
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    statusEl.textContent = `Done in ${secs}s`;
    bar.style.width = "100%";
    downloadLink.href = resultCanvas.toDataURL("image/png");
    downloadLink.download = "moebius-inpaint.png";
    downloadLink.style.display = "inline-block";
  } catch (err) {
    statusEl.textContent = "Error: " + (err as Error).message;
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
});

// Kick off model loading early (in the background) so first run is faster.
window.addEventListener("load", () => {
  setTimeout(ensureModels, 500);
});
