"""Run the real Moebius pipeline to produce a ground-truth inpaint + dump intermediates.

Usage:
  python python/reference.py --weight <ckpt.bin> --image <img.png> --mask <mask.png>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MOEBIUS_REPO = "/tmp/Moebius/Moebius"
VAE_DIR = "/tmp/Moebius/moebius-web/weights/PixelHacker/vae"
WEIGHTS_DIR = "/tmp/Moebius/Moebius-weights"

sys.path.insert(0, MOEBIUS_REPO)
os.chdir(MOEBIUS_REPO)  # pipeline uses relative imports / cwd-ish paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=f"{WEIGHTS_DIR}/ft_places2/diffusion_pytorch_model.bin")
    ap.add_argument("--config", default=f"{MOEBIUS_REPO}/config/model_cfg/moebius.yaml")
    ap.add_argument("--image", default=f"{MOEBIUS_REPO}/data/images/0.png")
    ap.add_argument("--mask", default=f"{MOEBIUS_REPO}/data/masks/000000.png")
    ap.add_argument("--out", default="/tmp/Moebius/moebius-web/reference_out")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cpu"
    dtype = torch.float32

    from diffusers import DDIMScheduler
    from removal.v1_2.removal_model import build_removal_model, load_cfg, load_removal_model
    from removal.v1_2.pipeline import RemovalSDXLPipeline_BatchMode
    from diffusers import AutoencoderKL
    build_vae = lambda c: AutoencoderKL.from_pretrained(c["vae"]["model_dir"])

    cfg = load_cfg(args.config)
    cfg["vae"]["model_dir"] = VAE_DIR

    print("[ref] building removal model ...")
    removal_model = build_removal_model(cfg, 20).to(device)
    msg = load_removal_model(removal_model, args.weight, device)
    print("[ref] load_state_dict:", msg)

    n_params = sum(p.numel() for p in removal_model.parameters())
    print(f"[ref] removal_model params: {n_params/1e6:.2f}M")

    vae = build_vae(cfg).to(device)
    print("[ref] vae scaling_factor:", vae.config.scaling_factor,
          "block_out_channels:", vae.config.block_out_channels)

    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, clip_sample=False)

    pipe = RemovalSDXLPipeline_BatchMode(
        removal_model=removal_model, vae=vae, scheduler=scheduler,
        device=device, dtype=dtype)

    # deterministic
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    print(f"[ref] image {image.size}  mask {mask.size}")

    out_list = pipe(
        [image], [mask],
        image_size=512,
        num_steps=args.steps,
        guidance_scale=args.cfg,
        paste=True, compensate=False,
        noise_offset=0.0357,
        mute=False,
    )
    out = out_list[0]
    out_path = os.path.join(args.out, "reference_result.png")
    out.save(out_path)
    print("[ref] saved", out_path, out.size)

    # also save the resized inputs the model actually saw (multiple-of-64)
    from removal.v1_2.pipeline import resize_image_to_multiple_of_64
    ri, rm = resize_image_to_multiple_of_64([image, mask.convert("RGB")], 512)
    ri.save(os.path.join(args.out, "input_resized.png"))
    rm.convert("L").save(os.path.join(args.out, "mask_resized.png"))
    print("[ref] model input size:", ri.size)


if __name__ == "__main__":
    main()
