# Understanding `moebius-web`: a guided tour

A complete walk-through of [simonw/moebius-web](https://github.com/simonw/moebius-web) — an
in-browser image-inpainting demo that runs a 0.22-billion-parameter diffusion model **entirely
client-side** on your GPU, with no server doing the heavy lifting. This guide builds up every
concept you need: what the model is, what diffusion and inpainting mean, what ONNX is, how a
PyTorch model gets converted to ONNX, what WebGPU is, and how all the pieces fit together in the
actual code in this repo.

You don't need prior diffusion-model knowledge. You do need to be comfortable reading code and
basic linear-algebra-flavored ideas (tensors, channels, matrix multiply). Everything specific to
this repo is grounded in the files it ships.

---

## 0. The one-paragraph summary

The repo takes **Moebius**, a research image-inpainting model published as a PyTorch checkpoint,
and makes it run in a web browser. To do that it (1) exports the model's neural-network graphs to
**ONNX**, a portable format that runtime engines can execute; (2) ships those `.onnx` files plus a
small **TypeScript** program that re-implements the "orchestration" logic (the denoising loop, the
pre/post-processing) that normally lives in Python; and (3) runs the ONNX graphs in the browser via
**ONNX Runtime Web** on the **WebGPU** backend, which is what lets a phone or laptop GPU do the
matrix math fast enough to be usable. The clever part is that this particular model is unusually
browser-friendly, and the hard part is a long tail of practical problems (a Safari shader bug,
caching 1.27 GB of weights, numeric precision) that the repo's `notes.md` documents as it solves
them.

---

## 1. What "image inpainting" is, and what the demo does

**Inpainting** means filling in a missing or unwanted region of an image with plausible new
content. You paint a mask over part of a photo — say, a person you want removed — and the model
synthesizes pixels to replace that region so the result looks like a coherent, untouched photo.

The demo's user flow: upload an image, paint over a region, press go. The masked region gets
regenerated; everything outside the mask is preserved from the original. The whole computation
runs locally — the only network traffic is the one-time download of the model weights (~1.27 GB,
then browser-cached). A WebGPU-capable browser (recent Chrome or Safari) is required because the
math is too heavy for a CPU to do at interactive speed.

---

## 2. Diffusion models, the 10-minute version

To understand "the model" you need the shape of how modern image-generation diffusion models work.
Moebius is a *latent diffusion* inpainting model, which is the same family as Stable Diffusion.
There are four moving parts.

### 2.1 The VAE (working in "latent space" instead of pixels)

A 512×512 RGB image is 512×512×3 ≈ 786,000 numbers. Running a big neural net directly on that many
pixels every step would be enormously expensive. So latent diffusion uses a **VAE**
(variational auto-encoder), a pre-trained pair of networks:

- the **encoder** compresses an image down to a small **latent** tensor, and
- the **decoder** reconstructs an image back up from a latent.

In this model the VAE downsamples by a factor of 8 in each spatial dimension and uses 4 latent
channels, so a `(3, 512, 512)` image becomes a `(4, 64, 64)` latent — about 16,000 numbers instead
of 786,000, a ~48× reduction. All the expensive denoising happens in this compact latent space;
the VAE decoder is only run once at the very end to turn the final latent back into a picture.

One detail this repo hammers on: the VAE's **`scaling_factor` is `0.13025`**, *not* the usual
Stable-Diffusion value of `0.18215`. The scaling factor normalizes the latent to a sensible numeric
range. The encoder output is multiplied by it; the decoder input is divided by it. The repo's notes
warn that getting this wrong silently corrupts colors and contrast. (This VAE comes from a sibling
project, `hustvl/PixelHacker`.)

### 2.2 The UNet (the denoiser — this is "the model")

The heart of a diffusion model is a network — almost always shaped like a **UNet** — that is
trained to look at a *noisy* latent and predict the noise that was added to it. A UNet is an
encoder/decoder with skip connections: it progressively downsamples the spatial grid (capturing
coarse structure), then upsamples back (recovering detail), with "skip" links between matching
levels so fine information isn't lost.

In this repo the UNet is the file `unet.onnx`, and it's the big one (~907 MB at fp32). Everything
else exists to feed it and to consume its output.

### 2.3 The scheduler / the denoising loop (DDIM)

You don't denoise in one shot. You start from pure random noise and iteratively refine it over a
sequence of steps, each one a little less noisy, guided by the UNet's noise predictions. The math
that decides "given the current noisy latent and the predicted noise, what's the slightly-cleaner
latent for the next step" is the **scheduler**. This repo uses **DDIM** (Denoising Diffusion
Implicit Models), a deterministic, few-step sampler. Section 7 walks through the exact arithmetic,
because in this project the scheduler runs in *TypeScript*, not in the model.

### 2.4 Conditioning (telling the model *what* to generate)

A bare denoiser would hallucinate something random. **Conditioning** steers it. In Stable Diffusion
the conditioning is text: a CLIP text encoder turns "a photo of a cat" into vectors the UNet
attends to. **Moebius does this differently, and that difference is the whole reason this port is
tractable** — see the next section.

---

## 3. What makes Moebius specifically, and why it's browser-friendly

The repo's `research.md` is essentially an argument that Moebius is an unusually good candidate for
the browser. Three architectural facts drive that.

**(a) No text encoder — conditioning is a tiny lookup table.** Instead of a CLIP/text tower (tens
of millions of parameters, a tokenizer, BPE merge tables — all annoying to port), Moebius conditions
on a learned **embedding table**: a plain `nn.Embedding(20, 3072)`. Think of it as 20 rows of 3072
numbers each, learned during training. The "prompt" is just *which rows to look up*. There is
nothing to tokenize and no text model to ship. The rows are split into two fixed sets used for
classifier-free guidance (more on that below): IDs `0..9` are the **conditional** "prompt", and IDs
`10..19` are the **unconditional** one. The lookup stays *inside* the exported UNet graph as a cheap
gather operation; the browser code just passes the right integer IDs in.

**(b) Linear attention instead of quadratic attention.** Standard transformer attention builds an
N×N score matrix, which blows up memory and is awkward to tile on a constrained GPU. Moebius's
blocks (the repo calls the architecture
`UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8`) use **lambda-style linear
attention** expressed with `einsum` operations plus **depthwise-separable convolutions**. This
sidesteps the big-matrix problem and is exactly the kind of op a WebGPU runtime handles well.

**(c) A short, simple sampling loop and no mid-block.** The config has three down-blocks, **no
mid-block**, and three up-blocks, and runs DDIM at ~20 steps. That's a small, clean graph and a
short loop that's trivial arithmetic to re-implement in JavaScript.

The notes also flag a subtle architectural constraint that shapes the entire design: there are two
attention paths, and the **cross-attention path uses a learned relative-position embedding
(`rel_pos_emb`) whose size is tied to the trained spatial resolution.** Feed it a different spatial
size and the indexing goes out of bounds. **Consequence: the model is fixed at 512×512** (a 64×64
latent). The web app resizes any user input to a square 512×512, inpaints, then resizes the result
back and pastes it over the original. The self-attention path (`attn1`, a "MQSλ" block with a local
window `r=15`) is spatially flexible; it's the global cross-attention path (`attn2`, "MQCλ") that
pins the resolution.

---

## 4. The model, concretely: three graphs and their interfaces

The export is **three separate ONNX graphs**, not one monolithic model. Splitting them is
deliberate: the VAE encoder runs once at the start, the decoder once at the end, and the UNet runs
many times in the loop, so keeping them separate lets the TypeScript code call each at the right
moment.

| File | What it is | Input → Output | Size (fp32) |
|------|-----------|----------------|-------------|
| `vae_encoder.onnx` | image → latent moments | `image (B,3,512,512)` → `moments (B,8,64,64)` | ~137 MB |
| `unet.onnx` | the denoiser (embedding + lambda-UNet) | `latent (B,9,64,64)`, `timesteps (B,)`, `input_ids (B,10)` → `noise (B,4,64,64)` | ~907 MB |
| `vae_decoder.onnx` | latent → image | `latent (B,4,64,64)` → `image (B,3,512,512)` | ~198 MB |

Two interface details are worth internalizing because they recur everywhere:

**The encoder outputs 8 channels, but a latent is 4 channels.** A VAE encoder emits a *distribution*
per latent cell: 4 channels of **mean** and 4 channels of **log-variance** (together called
"moments"). For this pipeline you only need the mean, so the code takes the first 4 channels and
multiplies by the scaling factor. You can see this exact slice in `pipeline.ts`:

```ts
// take mean channels (first 4) * scaling_factor
const out = new Float32Array(4 * LAT * LAT);
for (let i = 0; i < out.length; i++) out[i] = m[i] * SCALING_FACTOR;
```

**The UNet input is 9 channels, not 4.** This is what makes it an *inpainting* UNet rather than a
plain generator. Each step, the 9-channel input is assembled as:

```
channels 0–3 : the current noisy latent       (what we're denoising)
channel  4   : the mask, downsampled to 64×64  (where to inpaint: 1 = fill, 0 = keep)
channels 5–8 : the masked-image latent         (the known context, hole zeroed out)
```

So the UNet always sees *what it's working on*, *where the hole is*, and *what the surrounding
known content looks like*. The assembly happens in `pipeline.ts`:

```ts
const nine = new Float32Array(9 * plane);
nine.set(latents.subarray(0, 4 * plane), 0);      // ch 0-3 noisy latents
nine.set(mask64, 4 * plane);                       // ch 4   mask
nine.set(maskedLat.subarray(0, 4 * plane), 5*plane);// ch 5-8 masked latents
```

---

## 5. ONNX: what it is and why this repo needs it

### 5.1 The format

**ONNX** (Open Neural Network Exchange) is a portable, framework-neutral file format for neural
networks. An `.onnx` file is essentially two things bundled together:

1. **A computation graph** — a directed graph of *nodes*, where each node is an **operator**
   (`Conv`, `MatMul`, `Add`, `Einsum`, `Softmax`, `Gather`, `Resize`, …) wired together by named
   tensors flowing between them. This is the "recipe" for the forward pass.
2. **The weights** — the learned parameter tensors (the convolution kernels, the embedding table,
   etc.), stored as initializers in that same graph.

Crucially, ONNX describes *what to compute*, abstractly, without saying *how* or *on what hardware*.
The operator set is versioned by an **opset** number (this repo uses **opset 18**), which pins down
exactly which operators exist and what their semantics are.

### 5.2 Why a portable format matters here

The model is trained and stored as a PyTorch checkpoint. PyTorch is a Python framework — you can't
run it in a browser. ONNX is the bridge: PyTorch can *export* its model to ONNX, and a completely
different engine written in C++/Rust/WASM can *load and execute* that ONNX graph with no Python and
no PyTorch involved. ONNX is the lingua franca that decouples "the model was built in PyTorch" from
"the model runs in a browser."

### 5.3 ONNX Runtime and "execution providers"

**ONNX Runtime (ORT)** is the engine that actually executes ONNX graphs. A key ORT concept is the
**execution provider (EP)** — a backend that supplies the actual kernels (implementations) for the
operators. The same `.onnx` file can run on the CPU EP, a CUDA EP, or — the one this repo cares
about — the **WebGPU EP** inside **ONNX Runtime Web** (the WebAssembly build of ORT that runs in a
browser). Section 8 covers WebGPU.

---

## 6. Converting the PyTorch model to ONNX (the export pipeline)

This is the part you asked most directly about. The relevant files are
`python/export_onnx.py` (the main export), `python/reexport_unet_no_conv3d.py` (a critical fix),
and `python/to_fp16.py` (precision experiments).

### 6.1 The core mechanism: `torch.onnx.export` and tracing

PyTorch converts a model to ONNX by **tracing**: you hand it the model and one set of *example
inputs*, it runs the forward pass once, records every tensor operation that happened, and emits
that recorded graph as ONNX. Here's the actual UNet export call from `export_onnx.py`:

```python
torch.onnx.export(
    unet, (latent9, timesteps, input_ids), unet_path, opset_version=18,
    input_names=["latent", "timesteps", "input_ids"], output_names=["noise"],
    dynamic_axes={"latent": {0: "B"}, "timesteps": {0: "B"},
                  "input_ids": {0: "B"}, "noise": {0: "B"}},
)
```

Three things to notice:

- **The example inputs define the shapes.** `latent9` is `(2, 9, 64, 64)`, `timesteps` is `(2,)`,
  `input_ids` is `(2, 10)`. Trace once with these and the graph "knows" those shapes.
- **`input_names` / `output_names`** give the graph's inputs and outputs stable string names. The
  TypeScript code later feeds tensors by exactly these names (`latent`, `timesteps`, `input_ids`)
  and reads the result by name (`noise`).
- **`dynamic_axes`** marks which dimensions are allowed to vary at runtime. Here only axis 0 (the
  batch dimension `B`) is dynamic — the spatial dimensions are *frozen at 64×64* because, as
  Section 3 explained, the model is resolution-locked. This is why the README says "static 512×512":
  the export deliberately bakes in the spatial size and only lets batch flex.

### 6.2 Why batch is dynamic: classifier-free guidance

Batch must flex because of **classifier-free guidance (CFG)**, the standard trick for making
conditioning stronger. Each denoising step runs the UNet *twice*: once with the conditional
"prompt" and once with the unconditional one. Instead of two separate calls, the code stacks them
into a batch of 2 and runs one forward pass. The final noise prediction is a weighted extrapolation
away from the unconditional toward the conditional:

```
guided = uncond + guidance_scale * (cond - uncond)
```

with `guidance_scale` defaulting to 2.0. You can see both halves in the export wrapper: the example
`input_ids` is `[[10..19], [0..9]]` — row 0 is the unconditional set, row 1 is the conditional set —
and the same idea appears in `pipeline.ts`'s `unetCFG`. So the model is *always* invoked with batch
2 in practice, which is why batch is a dynamic axis.

### 6.3 Wrapping the model for a clean graph

Before tracing, each model is wrapped in a tiny `nn.Module` whose `forward` returns plain tensors,
e.g.:

```python
class UNetExport(nn.Module):
    def __init__(self, removal_model):
        super().__init__(); self.m = removal_model
    def forward(self, latent_in9, timesteps, input_ids):
        return self.m(latent_in9, timesteps, input_ids).sample
```

The real model returns a structured object (`.sample` is the tensor inside it); ONNX wants tensors
in and tensors out, so the wrapper unwraps it. The VAE is split similarly into a `VaeEncExport`
(runs `encoder` then `quant_conv` to get the 8-channel moments) and a `VaeDecExport` (runs
`post_quant_conv` then `decoder`). This is also where the export *chooses* to expose moments rather
than a sampled latent, and to leave the scaling-factor multiply out of the graph (it's done in JS).

### 6.4 Verifying the export: parity checks

A traced graph can silently differ from the original (a wrong op, a precision quirk). So
immediately after each export, the script runs the ONNX graph through ONNX Runtime's CPU provider
on the *same* inputs and compares against PyTorch's output:

```python
sess = ort.InferenceSession(unet_path, providers=["CPUExecutionProvider"])
got = sess.run(None, {"latent": latent9.numpy(), ...})[0]
print(f"max|Δ| = {amax(ref, got):.3e}")
```

The reported parity is excellent: **UNet `max|Δ| ≈ 3.6e-6`, decoder `≈ 5.7e-5`** — i.e. the ONNX
graph reproduces PyTorch essentially exactly. There's also a *full-pipeline* parity test
(`python/onnx_pipeline.py`) that re-implements the entire DDIM + CFG + 9-channel + scaling loop in
**NumPy** on top of the ONNX sessions and compares the final decoded image against the PyTorch
reference using identical starting noise: **decoded-image `mean|Δ| ≈ 0.0022`**, i.e. visually
identical. This NumPy pipeline is essentially a rehearsal of the TypeScript port — get it right in
NumPy, then translate.

### 6.5 The Safari Conv3d problem (a perfect case study in op-coverage)

This is the single most instructive episode in the repo, because it shows what "export cleanly" and
"runs everywhere" actually require in practice.

The self-attention blocks contain a `pos_conv` that is a **3-D convolution**:
`Conv3d(1, dim_k, (1,15,15), padding=(0,7,7))` applied to a 5-D tensor `V` of shape
`(b, 1, v, hh, ww)`. It exported fine and ran fine in Chrome. But on **Safari's WebGPU (Metal)
backend it crashed at runtime**: ONNX Runtime Web generates Metal shader code, and for these
Conv3d ops it emitted `array<unsigned, 5>(a,b,c,d,e)`, a constructor Apple's Metal compiler rejects.
There are 15 such ops, so the whole thing failed on Safari.

The fix (`reexport_unet_no_conv3d.py`) is a lovely bit of reasoning: because the depth kernel over
the `v` axis has size **1** and the input channel count is **1**, that Conv3d is *mathematically
identical* to a 2-D convolution applied independently to each `v`-slice. So the script swaps each
`Conv3d` for an equivalent `Conv2d` wrapper that folds `(b, v)` into the batch dimension, runs a
plain `Conv2d(1, dim_k, 15, padding=7)` with the weights copied over via `squeeze(2)`, then reshapes
back:

```python
def forward(self, V):                 # V: (b, 1, v, hh, ww)
    b, u, v, hh, ww = V.shape          # u == 1
    x = V.reshape(b * v, 1, hh, ww)    # fold (b,v) into batch
    x = self.conv(x)                   # Conv2d(1, dim_k, 15, pad 7)
    return x.reshape(b, v, -1, hh, ww).permute(0, 2, 1, 3, 4)
```

It then re-exports, re-checks parity (torch self-parity `2.6e-6`; ONNX-vs-patched-torch `4.4e-6`),
and asserts the new graph contains **zero 3-D Conv nodes**. The lesson generalizes: exporting to
ONNX is necessary but not sufficient — you also have to land on operators your *target runtime's
specific backend* implements correctly. Here, "compiles on Apple's Metal shader compiler" was a
real constraint that reshaped the graph.

### 6.6 Precision: fp32 vs fp16

The shipped models are **fp32** (32-bit floats) for numeric parity with the reference.
`python/to_fp16.py` experiments with converting to **fp16** (16-bit), which would roughly halve the
download. The finding, recorded in the notes: fp16 is risky here. The lambda (linear-attention)
layers are precision-sensitive, and the VAE in particular is numerically unstable in fp16 (the
decoder has a Cast issue). The guidance is to keep the VAE in fp32 and validate carefully before
shipping any fp16 UNet. This is a recurring diffusion-model theme — these models accumulate small
values across many steps, and half precision can compound error visibly.

---

## 7. The denoising loop, ported to TypeScript (`ddim.ts`)

Normally the scheduler lives in a Python library (`diffusers.DDIMScheduler`). Here it's
hand-ported to TypeScript so it can run in the browser. `ddim.ts` is small and worth reading line
by line; here's the conceptual content.

**Building the schedule.** DDIM needs a precomputed table of `alphas_cumprod` — the cumulative
product of `(1 − beta)` across the 1000 training timesteps, where the betas follow a "scaled_linear"
schedule from `0.00085` to `0.012`:

```ts
// betas[i] = (linear-in-sqrt-space)^2 ; alphasCumprod = cumulative product of (1 - beta)
const s = a + (b - a) * (i / (NUM_TRAIN_TIMESTEPS - 1));  // a=√0.00085, b=√0.012
betas[i] = s * s;
```

The 20-step inference schedule picks evenly spaced timesteps `[950, 900, …, 50, 0]`, then
**`strength = 0.99` drops the first one**, leaving 19 actual steps `[900, …, 0]`. A subtle
correctness note the code calls out: the drop count uses `Math.floor(20 * 0.99) = 19`, matching
Python's integer truncation rather than rounding.

**One DDIM update step.** Given the predicted noise `eps` and the current `sample` at timestep `t`,
recover the implied clean latent, then re-noise it to the *previous* (less noisy) timestep:

```ts
const predX0 = (sample[i] - sqrtBetaT * eps[i]) / sqrtAcT;
out[i] = sqrtAcPrev * predX0 + sqrtOneMinusAcPrev * eps[i];
```

This is the `eta = 0`, `clip_sample = false` DDIM formula, and it matches the diffusers library to
about `5e-7`. The TypeScript port deliberately does **not** try to reproduce PyTorch's random-number
generator — diffusion is robust to *which* noise you draw, so the app uses a small seedable JS RNG
(`mulberry32` in `imaging.ts`) and still gets valid results, just not bit-identical to the Python
reference.

**The loop, in `pipeline.ts`.** Putting it together, `run()` does: encode the masked image to a
latent; build the 64×64 mask; start from pure noise (`+` a small `noise_offset = 0.0357`); then for
each timestep call `unetCFG` (assemble 9 channels, batch ×2, run the UNet, apply the guidance
formula) and feed the result through `ddimStep`. After the loop, decode the final latent and paste
it back over the original outside the mask. Notice it starts from pure noise because `strength ≈ 1`,
which is why it only needs the *masked-image* latent and never bothers encoding the full image.

---

## 8. WebGPU and ONNX Runtime Web (why it runs at all)

### 8.1 What WebGPU is

**WebGPU** is a modern browser API that gives web pages low-level access to the GPU for both
graphics and general-purpose compute. It superseded the older WebGL for compute work and is the
first browser API really suited to running neural-network kernels (large parallel matrix
operations) efficiently. It's available in recent Chrome and Safari, which is why those are the
demo's requirement.

### 8.2 ONNX Runtime Web on the WebGPU backend

`pipeline.ts` imports `onnxruntime-web/webgpu` and creates sessions that prefer the WebGPU EP and
fall back to WASM (CPU) if WebGPU is absent:

```ts
const ep = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
this.unet = await ort.InferenceSession.create(bytes, { executionProviders: ep, ... });
```

**Why WebGPU is non-negotiable here.** The notes measured CPU performance at ~8.9 seconds *per
step*; with 19 steps × CFG (×2) = 38 UNet passes, that's about 2 minutes 48 seconds on CPU — and
WASM/CPU is described bluntly as "unusable." WebGPU is "the whole game": it's what moves a single
denoising step from seconds to a fraction of a second.

### 8.3 Op coverage: the thing you must verify before promising speed

An execution provider only accelerates the operators it has kernels for. If the WebGPU EP lacks a
kernel for some op in your graph, ORT can silently fall back to running *that op* on CPU, which
tanks throughput. So the repo audited ONNX Runtime's source to confirm the heavy operators are
covered on WebGPU: `Einsum` ✓ (the linear-attention math), `Conv` ✓, `InstanceNormalization` ✓,
`MatMul`/`Gemm` ✓, `Softmax` ✓, the `Reduce*` family ✓, and `Transpose`/`Concat`/`Gather`/`Pad`/
`Resize`/`Where` ✓.

`GroupNorm` is an interesting case: there's no kernel registered under that name, but PyTorch
exports `nn.GroupNorm` as a decomposition (`Reshape → InstanceNormalization → Reshape → Mul → Add`),
and every piece of *that* is covered — which the excellent VAE-decoder parity number indirectly
confirms. The general principle: don't assume your runtime covers an op; check, because the failure
mode (a silent CPU fallback) is invisible until you profile.

### 8.4 One more reason the Conv3d fix mattered

Section 6.5 framed it as a Safari shader bug, but there's a runtime-architecture wrinkle the notes
record: EP fallback in ORT-Web is decided *per session-creation*, not *per op at runtime*. Safari
happily *created* the WebGPU session, then failed at the first *run* when it hit the bad shader — and
there's no automatic WASM rescue once you're running. So you can't lean on fallback; removing the
Conv3d outright was the real fix (and is faster anyway, since the Conv3d kernel was a naive
implementation).

---

## 9. The practical engineering tail (the stuff nobody warns you about)

A working model in the browser is maybe half the battle. `notes.md` documents the rest.

**Caching 1.27 GB of weights.** The biggest non-ML problem. The model files live on Hugging Face,
and a request to `huggingface.co/<repo>/resolve/main/<file>` 302-redirects to a CDN URL whose query
parameters (`Expires`, `Signature`, …) **rotate on every request**. The browser's normal HTTP cache
keys on the full URL, so it never gets a hit — meaning the full 1.27 GB would re-download on every
page load. Worse, a *redirected* fetch response can't be stored with the Cache API directly. The fix
in `modelcache.ts` (the same trick Transformers.js uses): download the bytes once, then store them
in the **Cache Storage API** keyed by the *stable* `…/resolve/main/<file>` URL, wrapped in a freshly
constructed `Response`. Subsequent loads — and future sessions — read straight from that cache. It
also calls `navigator.storage.persist()` to reduce the chance the browser evicts 1.2 GB.

**Streaming download with progress.** Rather than buffering the whole file, `loadModelBytes`
pre-allocates one `Uint8Array` of the known content length and fills it chunk-by-chunk from the
response stream, reporting progress (the UI shows "Downloading UNet 123/907 MB"). For a near-gigabyte
file, pre-allocating one buffer instead of concatenating chunks matters for memory.

**Cross-origin isolation (COOP/COEP).** Multi-threaded WASM needs `SharedArrayBuffer`, which browsers
only expose when the page is "cross-origin isolated" via COOP/COEP headers. The dev server sets them.
On hosts that *can't* set them, the code falls back to single-threaded WASM — fine, because the
WebGPU path doesn't need threads anyway.

**Serving the ORT runtime glue.** A Vite-specific gotcha: the ONNX Runtime `.mjs` glue must not sit
in Vite's `/public` (Vite tries to transform it as a module and breaks it); it's served as raw static
files via custom middleware instead. Small, but the kind of thing that costs an afternoon.

---

## 10. Putting the whole pipeline in one picture

End to end, ignoring caching, here is the data flow:

```
 user image (any size)            user-painted mask
        │                                │
        ▼                                ▼
 resize → 512×512 RGB            resize → 512×512, binarize
        │                                │
        ├──────────────► masked image = image × (1 − mask)
        │                                │
        ▼                                ▼
   (not needed,             VAE ENCODER (vae_encoder.onnx)
    strength≈1)            image(3,512,512) → moments(8,64,64)
                                  take mean[:4] × 0.13025
                                          │
                                 masked-image latent (4,64,64)
                                          │
 init latent = randn(4,64,64) + noise_offset·randn      mask → 64×64 (1 channel)
        │                                 │                       │
        └───────────────┬─────────────────┴───────────────────────┘
                        ▼
        ┌────────  DDIM LOOP  (19 steps)  ────────┐
        │  assemble 9 channels:                    │
        │    [ noisy latent(4) | mask(1) | masked latent(4) ]
        │  duplicate to batch 2  (uncond, cond)    │
        │                  │                        │
        │                  ▼                        │
        │       UNET  (unet.onnx)                   │
        │   (9,64,64)+timesteps+input_ids → noise(4,64,64)
        │                  │                        │
        │   CFG: guided = uncond + 2.0·(cond−uncond)│
        │                  │                        │
        │   DDIM step → slightly cleaner latent ────┘
        └──────────────────│───────────────────────┘
                           ▼
              final latent (4,64,64)
                           │  ÷ 0.13025
                           ▼
        VAE DECODER (vae_decoder.onnx)
        latent(4,64,64) → image(3,512,512), then (x+1)/2
                           │
                           ▼
   paste back: result inside mask, original outside
        (blurred-mask blend), resize to original size
                           │
                           ▼
                     final image
```

Read `pipeline.ts`'s `run()` method alongside this and every box maps to a few lines of code.

---

## 11. How to explore the repo yourself (a suggested reading order)

1. **`README.md`** — orientation and the "how it works" summary.
2. **`models/README.md`** (the Hugging Face model card) — the precise tensor shapes, the scaling
   factor, the DDIM constants. This is your reference sheet.
3. **`research.md`** — the up-front argument for *why* this model ports well. Read it before the
   code; it frames every later decision.
4. **`plan.md`** — the planned phases and the confirmed model facts.
5. **`python/reference.py`** — runs the original PyTorch pipeline to get ground truth.
6. **`python/export_onnx.py`** — the actual ONNX export with parity checks. The center of your
   "how do you convert a model to ONNX" question.
7. **`python/reexport_unet_no_conv3d.py`** — the Conv3d→Conv2d fix; the best single example of a
   real-world export constraint.
8. **`python/onnx_pipeline.py`** — the NumPy re-implementation of the full loop on ONNX sessions;
   the blueprint for the TypeScript port.
9. **`web/src/ddim.ts`** then **`web/src/pipeline.ts`** — the TypeScript port. Compare them directly
   against `onnx_pipeline.py`; they're deliberately parallel.
10. **`web/src/modelcache.ts`** — the caching solution.
11. **`notes.md`** — the running lab log. Read it last as a "how every problem got solved" capstone;
    it's where the Safari bug, the caching fix, and the op-coverage audit are recorded in detail.

---

## 12. Mini-glossary

- **Inpainting** — filling a masked region of an image with plausible generated content.
- **Latent / latent space** — the compact `(4,64,64)` representation the VAE compresses an image
  into; all the denoising happens here.
- **VAE** — variational auto-encoder; the encoder/decoder pair that converts between pixels and
  latents (8× spatial downsample, 4 channels here).
- **`scaling_factor`** — the constant (here **0.13025**, custom) that normalizes latents; multiply
  on encode, divide on decode.
- **UNet** — the denoiser network; predicts the noise in a noisy latent. Here it takes 9 input
  channels (noisy latent + mask + masked latent).
- **Diffusion / denoising loop** — iteratively refining pure noise into an image over several steps.
- **DDIM** — a deterministic few-step sampler; the exact step math is ported into `ddim.ts`.
- **Scheduler** — the math that turns "current latent + predicted noise" into the next step's latent.
- **Conditioning** — how you steer generation. Moebius uses a learned `nn.Embedding(20,3072)`, not
  a text encoder; IDs `0..9` = conditional, `10..19` = unconditional.
- **Classifier-free guidance (CFG)** — running the UNet conditionally and unconditionally and
  extrapolating: `uncond + scale·(cond − uncond)`. Why batch size is 2.
- **Embedding table** — a learned lookup matrix; a row per integer ID. Moebius's "prompt" is which
  rows to look up.
- **Linear / lambda attention** — attention that avoids the N×N score matrix (via `einsum` +
  depthwise convs); GPU- and browser-friendly.
- **ONNX** — portable neural-network format: a graph of operators plus weights, versioned by opset.
- **Operator / opset** — a node type in the graph (`Conv`, `Einsum`, …); the opset (here **18**)
  fixes which operators exist and what they mean.
- **Tracing** — how `torch.onnx.export` builds the graph: run the forward pass once on example
  inputs and record the operations.
- **`dynamic_axes`** — which tensor dimensions may vary at runtime (here only batch; spatial is
  frozen at 64×64 because the model is resolution-locked).
- **Parity check** — comparing ONNX output to PyTorch output on identical inputs to confirm a
  faithful export.
- **ONNX Runtime (ORT)** — the engine that executes ONNX graphs.
- **Execution provider (EP)** — an ORT backend (CPU, CUDA, **WebGPU**, WASM) that supplies the
  operator kernels.
- **ONNX Runtime Web** — the WebAssembly build of ORT that runs in a browser.
- **WebGPU** — the browser GPU-compute API that makes this fast enough to be usable; mandatory here.
- **Op coverage / fallback** — whether an EP has a kernel for an op; a missing kernel can silently
  drop to CPU and destroy performance.
- **fp32 / fp16** — 32- vs 16-bit floats; fp16 halves size but is numerically risky in these
  precision-sensitive layers (this repo ships fp32).
- **Cache Storage API** — the browser cache used to persist the ~1.27 GB of weights under a stable
  URL despite Hugging Face's rotating signed CDN links.
- **COOP/COEP** — headers that enable cross-origin isolation, required for multi-threaded WASM
  (`SharedArrayBuffer`).
