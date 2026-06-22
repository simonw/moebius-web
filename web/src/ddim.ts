// DDIM scheduler — direct port of the validated numpy implementation.
// scaled_linear beta schedule, eta=0, clip_sample=false. Matches diffusers to ~5e-7.

const NUM_TRAIN_TIMESTEPS = 1000;
const BETA_START = 0.00085;
const BETA_END = 0.012;

export interface DDIM {
  alphasCumprod: Float64Array; // length 1000
  timesteps: number[]; // descending, already strength-trimmed
}

export function makeDDIM(numSteps: number, strength = 0.99): DDIM {
  const betas = new Float64Array(NUM_TRAIN_TIMESTEPS);
  const a = Math.sqrt(BETA_START);
  const b = Math.sqrt(BETA_END);
  for (let i = 0; i < NUM_TRAIN_TIMESTEPS; i++) {
    const s = a + (b - a) * (i / (NUM_TRAIN_TIMESTEPS - 1));
    betas[i] = s * s;
  }
  const alphasCumprod = new Float64Array(NUM_TRAIN_TIMESTEPS);
  let acc = 1.0;
  for (let i = 0; i < NUM_TRAIN_TIMESTEPS; i++) {
    acc *= 1.0 - betas[i];
    alphasCumprod[i] = acc;
  }

  const stepRatio = Math.floor(NUM_TRAIN_TIMESTEPS / numSteps);
  const ts: number[] = [];
  for (let i = 0; i < numSteps; i++) ts.push(Math.round(i * stepRatio));
  ts.reverse(); // [950, 900, ..., 0]

  // strength<1 drops the first init_timestep: t_start = numSteps - min(round(numSteps*strength), numSteps)
  const initTimestep = Math.min(Math.round(numSteps * strength), numSteps);
  const tStart = Math.max(numSteps - initTimestep, 0);
  const timesteps = ts.slice(tStart);

  return { alphasCumprod, timesteps };
}

// One DDIM update step (in place on `sample`). eps and sample are Float32Array of equal length.
export function ddimStep(
  eps: Float32Array,
  sample: Float32Array,
  t: number,
  prevT: number,
  ddim: DDIM,
): Float32Array {
  const acT = ddim.alphasCumprod[t];
  const acPrev = prevT >= 0 ? ddim.alphasCumprod[prevT] : 1.0;
  const sqrtAcT = Math.sqrt(acT);
  const sqrtBetaT = Math.sqrt(1 - acT);
  const sqrtAcPrev = Math.sqrt(acPrev);
  const sqrtOneMinusAcPrev = Math.sqrt(1 - acPrev);

  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) {
    const predX0 = (sample[i] - sqrtBetaT * eps[i]) / sqrtAcT;
    out[i] = sqrtAcPrev * predX0 + sqrtOneMinusAcPrev * eps[i];
  }
  return out;
}
