"""Convert the exported ONNX models to fp16 and measure parity vs fp32 (CPU EP)."""
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

MODELS = "/tmp/Moebius/moebius-web/models"


def conv(name, feeds, keep_fp32_io=True):
    src = f"{MODELS}/{name}.onnx"
    dst = f"{MODELS}/{name}.fp16.onnx"
    m = onnx.load(src)
    m16 = float16.convert_float_to_float16(m, keep_io_types=keep_fp32_io)
    onnx.save(m16, dst)

    s32 = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    o32 = s32.run(None, feeds)[0]
    o16 = s16.run(None, feeds)[0]
    d = np.abs(o32.astype(np.float32) - o16.astype(np.float32))
    print(f"[fp16] {name:14s} max|Δ|={d.max():.4e} mean|Δ|={d.mean():.4e} "
          f"size={os.path.getsize(dst)/1e6:.1f}MB (was {os.path.getsize(src)/1e6:.1f})")


np.random.seed(0)
conv("vae_decoder", {"latent": np.random.randn(1, 4, 64, 64).astype(np.float32)})
conv("vae_encoder", {"image": np.random.randn(1, 3, 512, 512).astype(np.float32)})
conv("unet", {
    "latent": np.random.randn(2, 9, 64, 64).astype(np.float32),
    "timesteps": np.array([999, 999], dtype=np.int64),
    "input_ids": np.stack([np.arange(10, 20), np.arange(0, 10)]).astype(np.int64),
})
