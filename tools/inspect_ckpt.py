#!/usr/bin/env python3
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu")
print("type:", type(ckpt))
if isinstance(ckpt, dict):
    print("top keys:", list(ckpt.keys()))
    for k in ckpt:
        v = ckpt[k]
        if isinstance(v, dict):
            print(f"  {k}: dict len={len(v)}")
        elif hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("netG")
    if state is None:
        sample = list(ckpt.keys())[:5]
        if sample and all("." in str(s) for s in sample):
            state = ckpt
    if state:
        keys = list(state.keys())
        print(f"state_dict: {len(keys)} keys")
        for k in keys[:30]:
            t = state[k]
            print(f"  {k}: {tuple(t.shape)}")
        prefixes = sorted({k.split(".")[0] for k in keys})
        print("prefixes:", prefixes)
        for k in keys:
            if "gamma" in k and "evo" in k:
                print("evo gamma:", k, tuple(state[k].shape))
            if "gen_enc.inc" in k and "conv_x" in k and "weight" in k:
                print("gen_enc in:", k, tuple(state[k].shape))
            if "gen_enc.inc.double_conv.2.weight_orig" in k:
                print("gen_enc first conv:", k, tuple(state[k].shape))
            if "proj.conv_first" in k and "weight_bar" in k:
                print("proj conv_first:", k, tuple(state[k].shape))
            if "gen_dec.fc.weight" in k:
                print("gen_dec fc:", k, tuple(state[k].shape))
            if "gen_dec.conv_img.weight" in k:
                print("gen_dec conv_img:", k, tuple(state[k].shape))
            if "gen_dec.head_0.norm_0.mlp_shared.1.weight" in k:
                print("SPADE evo_ic (label_nc):", k, tuple(state[k].shape))
