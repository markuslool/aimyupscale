# run_inference.py — TinyFSR student ×2 upscale
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WEIGHTS_TXT = "TinyFSR student — 448 MACLR pixel.txt"
WEIGHTS_PTH = "tiny_fsr_student.pth"


# ---------- 1. Веса ----------
def _array(text, name):
    m = re.search(
        rf"{name}\s*\[[^\]]*\](\[[^\]]*\])*\s*=\s*\{{(.*?)\}}\s*;",
        text, re.S,
    )
    if not m:
        raise SystemExit(f"Не найден массив {name} в {WEIGHTS_TXT}")
    nums = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?f?", m.group(2))
    return np.array([float(n.rstrip("f")) for n in nums], dtype=np.float32)


def _shape(text, name):
    m = re.search(rf"{name}\s*((?:\[[^\]]*\])+)", text)
    return [int(d) for d in re.findall(r"\[(\d+)\]", m.group(1))]


def build_state_dict_from_txt(path):
    text = open(path, encoding="utf-8").read()
    mp = {
        "conv_in.weight":     "conv_in_weight",
        "conv_in.bias":       "conv_in_bias",
        "dw.weight":          "dw_weight",
        "dw.bias":            "dw_bias",
        "mix.weight":         "mix_weight",
        "mix.bias":           "mix_bias",
        "to_residual.weight": "to_residual_weight",
        "to_residual.bias":   "to_residual_bias",
    }
    sd = {}
    for k, arr_name in mp.items():
        data = _array(text, arr_name)
        shp = _shape(text, arr_name)
        assert data.size == int(np.prod(shp)), f"{arr_name}: {data.size} vs {shp}"
        sd[k] = torch.from_numpy(data.reshape(shp))
    return sd


def load_weights(dev):
    if os.path.exists(WEIGHTS_PTH):
        return torch.load(WEIGHTS_PTH, map_location=dev)
    if not os.path.exists(WEIGHTS_TXT):
        raise SystemExit(
            f"Нет ни {WEIGHTS_PTH}, ни {WEIGHTS_TXT}. Положи .txt рядом со скриптом."
        )
    print(f"{WEIGHTS_PTH} не найден — собираю из {WEIGHTS_TXT}")
    sd = build_state_dict_from_txt(WEIGHTS_TXT)
    torch.save(sd, WEIGHTS_PTH)
    print(f"Сохранён {WEIGHTS_PTH}")
    return sd


# ---------- 2. Модель ----------
class Net(nn.Module):
    """TinyFSR student: bicubic base + learned residual, ×2."""

    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.conv_in     = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
        self.dw          = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=True)
        self.mix         = nn.Conv2d(8, 8, kernel_size=1, bias=True)
        self.to_residual = nn.Conv2d(8, 3 * scale * scale, kernel_size=1, bias=True)
        self.relu        = nn.ReLU(inplace=True)
        self.ps          = nn.PixelShuffle(scale)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                             align_corners=False)
        y = self.relu(self.conv_in(x))
        y = self.relu(self.dw(y))
        y = self.relu(self.mix(y))
        y = self.to_residual(y)
        y = self.ps(y)
        return (base + y).clamp(0.0, 1.0)


# ---------- 3. Инференс ----------
def upscale(src, dst, net, dev):
    img = Image.open(src).convert("RGB")
    x = torch.from_numpy(np.array(img)).float().div(255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(dev)

    with torch.no_grad():
        y = net(x)

    out = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(dst)
    print(f"Готово: {src}  ->  {dst}  ({out.shape[1]}x{out.shape[0]})")


def main():
    if len(sys.argv) < 2:
        print("Использование: python run_inference.py <вход> [выход]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_x2.png"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Устройство:", dev)

    net = Net(scale=2).to(dev).eval()
    net.load_state_dict(load_weights(dev))
    upscale(src, dst, net, dev)


if __name__ == "__main__":
    main()
