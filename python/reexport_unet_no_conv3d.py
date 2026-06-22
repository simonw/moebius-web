"""Re-export unet.onnx with the self-attention pos_conv Conv3d replaced by an equivalent
Conv2d, so the graph compiles on Safari's WebGPU (Metal) backend.

The pos_conv is Conv3d(1, dim_k, (1, r, r), padding=(0, r//2, r//2)) applied to V of shape
(b, u=1, v, hh, ww). Depth kernel = 1 over the v axis ⇒ each v-slice is convolved
independently by a 2D (r×r) kernel on a single input channel. That is exactly a Conv2d.
"""
import os
import sys
import numpy as np
import torch
from torch import nn

MOEBIUS_REPO = "/tmp/Moebius/Moebius"
VAE_DIR = "/tmp/Moebius/moebius-web/weights/PixelHacker/vae"
WEIGHTS = "/tmp/Moebius/Moebius-weights/ft_places2/diffusion_pytorch_model.bin"
OUT = "/tmp/Moebius/moebius-web/models/unet.onnx"

sys.path.insert(0, MOEBIUS_REPO)
os.chdir(MOEBIUS_REPO)


class PosConv2d(nn.Module):
    """Drop-in for the Conv3d pos_conv. Input/Output keep the 5D (b,1|dim_k,v,hh,ww) layout."""
    def __init__(self, c3: nn.Conv3d):
        super().__init__()
        dim_k = c3.out_channels
        r = c3.kernel_size[1]
        self.conv = nn.Conv2d(1, dim_k, r, padding=r // 2)
        with torch.no_grad():
            self.conv.weight.copy_(c3.weight.squeeze(2))  # (dim_k,1,1,r,r) -> (dim_k,1,r,r)
            self.conv.bias.copy_(c3.bias)

    def forward(self, V):  # V: (b, 1, v, hh, ww)
        b, u, v, hh, ww = V.shape  # u == 1
        x = V.reshape(b * v, 1, hh, ww)         # fold (b,v) into batch; u=1 is the conv in-channel
        x = self.conv(x)                        # (b*v, dim_k, hh, ww)
        return x.reshape(b, v, -1, hh, ww).permute(0, 2, 1, 3, 4)  # (b, dim_k, v, hh, ww)


def replace_conv3d(model) -> int:
    n = 0
    for mod in model.modules():
        pc = getattr(mod, "pos_conv", None)
        if isinstance(pc, nn.Conv3d):
            mod.pos_conv = PosConv2d(pc)
            n += 1
    return n


class UNetExport(nn.Module):
    def __init__(self, removal_model):
        super().__init__()
        self.m = removal_model

    def forward(self, latent_in9, timesteps, input_ids):
        return self.m(latent_in9, timesteps, input_ids).sample


def main():
    from removal.v1_2.removal_model import build_removal_model, load_cfg, load_removal_model

    cfg = load_cfg(f"{MOEBIUS_REPO}/config/model_cfg/moebius.yaml")
    cfg["vae"]["model_dir"] = VAE_DIR
    m = build_removal_model(cfg, 20).eval()
    load_removal_model(m, WEIGHTS, "cpu")

    # reference inputs
    torch.manual_seed(0)
    latent9 = torch.randn(2, 9, 64, 64)
    ts = torch.tensor([811, 811], dtype=torch.int64)
    ids = torch.tensor([list(range(10, 20)), list(range(10))], dtype=torch.int64)

    wrap = UNetExport(m)
    with torch.no_grad():
        before = wrap(latent9, ts, ids).numpy()

    n = replace_conv3d(m)
    with torch.no_grad():
        after = wrap(latent9, ts, ids).numpy()
    d = np.abs(before - after).max()
    print(f"[fix] replaced {n} Conv3d pos_conv -> Conv2d. torch self-parity max|Δ| = {d:.3e}")
    assert d < 1e-4, "Conv2d replacement changed the math!"

    # export
    torch.onnx.export(
        wrap, (latent9, ts, ids), OUT, opset_version=18,
        input_names=["latent", "timesteps", "input_ids"], output_names=["noise"],
        dynamic_axes={"latent": {0: "B"}, "timesteps": {0: "B"},
                      "input_ids": {0: "B"}, "noise": {0: "B"}},
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"latent": latent9.numpy(), "timesteps": ts.numpy(),
                          "input_ids": ids.numpy()})[0]
    print(f"[fix] ONNX vs patched torch max|Δ| = {np.abs(after - got).max():.3e}")
    print(f"[fix] re-exported {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    # confirm no Conv3d remains in the graph
    import onnx
    g = onnx.load(OUT)
    n3 = sum(1 for node in g.graph.node if node.op_type == "Conv" and
             any(a.name == "kernel_shape" and len(a.ints) == 3 for a in node.attribute))
    print(f"[fix] remaining 3D Conv nodes in ONNX: {n3}")


if __name__ == "__main__":
    main()
