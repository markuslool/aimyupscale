import os
import gc
import glob
import random
import multiprocessing
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T


# ============================================================
# CONFIG
# ============================================================
DIV2K_DIR       = "./DIV2K_train_HR"
VALID_DIR       = "./DIV2K_valid_HR"
CACHE_FILE      = "./div2k_cache_x2_64.pt"
TEACHER_CKPT    = "./checkpoints/best.pth"

SCALE_FACTOR    = 2
PATCH_SIZE      = 64
CACHE_LR_SIZE   = 32
PATCHES_PER_IMAGE_CACHE = 80

BATCH_SIZE      = 128
EPOCHS          = 80
NUM_WORKERS     = 2
PREFETCH        = 4
LEARNING_RATE   = 4e-4
USE_AMP         = True

VAL_EVERY       = 5

EMA_DECAY       = 0.999
PIXEL_WEIGHT    = 0.40
TEACHER_WEIGHT  = 0.30
GRAD_WEIGHT     = 0.15
HF_WEIGHT       = 0.15

# Disabled for the stable baseline. Hard mining can distort the training
# distribution and is intentionally NOT used in this version.
HARD_MINING     = False

CHKPT_DIR       = "./checkpoints_448"
os.makedirs(CHKPT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print("CUDA not found — CPU")


# ============================================================
# CACHE
# ============================================================
def build_cache_if_needed():
    if os.path.exists(CACHE_FILE):
        print(f"Cache found: {CACHE_FILE}")
        return

    print(f"Building cache from {DIV2K_DIR} ...")
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        img_paths += glob.glob(os.path.join(DIV2K_DIR, "**", ext), recursive=True)
        img_paths += glob.glob(os.path.join(DIV2K_DIR, "**", ext.upper()), recursive=True)
    img_paths = sorted(set(img_paths))

    if not img_paths:
        raise ValueError(f"No images found in {DIV2K_DIR}")

    hr_patches, lr_patches = [], []
    to_tensor = T.ToTensor()
    random.seed(42)

    for i, path in enumerate(img_paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"skip {path}: {e}")
            continue

        w, h = img.size
        if w < PATCH_SIZE or h < PATCH_SIZE:
            continue

        max_x = (w - PATCH_SIZE) // PATCH_SIZE + 1
        max_y = (h - PATCH_SIZE) // PATCH_SIZE + 1
        total = max_x * max_y
        k = min(total, PATCHES_PER_IMAGE_CACHE)

        for idx in random.sample(range(total), k):
            x = (idx % max_x) * PATCH_SIZE
            y = (idx // max_x) * PATCH_SIZE
            hr = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            lr = hr.resize((CACHE_LR_SIZE, CACHE_LR_SIZE), Image.BICUBIC)

            hr_patches.append((to_tensor(hr) * 255.0).round().byte())
            lr_patches.append((to_tensor(lr) * 255.0).round().byte())

    n = len(hr_patches)
    if n == 0:
        raise RuntimeError("No patches collected")

    lr_tensor = torch.empty(
        (n, 3, CACHE_LR_SIZE, CACHE_LR_SIZE), dtype=torch.uint8
    )
    hr_tensor = torch.empty(
        (n, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.uint8
    )

    for i, t in enumerate(lr_patches):
        lr_tensor[i] = t
    for i, t in enumerate(hr_patches):
        hr_tensor[i] = t

    del lr_patches, hr_patches
    gc.collect()

    torch.save({"lr": lr_tensor, "hr": hr_tensor}, CACHE_FILE)
    print(f"Cache saved: {tuple(lr_tensor.shape)}, {tuple(hr_tensor.shape)}")


class CachedPatchDataset(Dataset):
    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")
        self.lr = data["lr"]
        self.hr = data["hr"]
        self.n = self.lr.shape[0]

        print(
            f"Dataset: {self.n} pairs | "
            f"LR={tuple(self.lr.shape[1:])} HR={tuple(self.hr.shape[1:])}"
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        lr = self.lr[idx].float().div_(255.0)
        hr = self.hr[idx].float().div_(255.0)
        return lr, hr, idx


def augment_batch(lr, hr):
    B = lr.size(0)

    mask_h = torch.rand(B, device=lr.device) > 0.5
    if mask_h.any():
        lr[mask_h] = torch.flip(lr[mask_h], dims=[3])
        hr[mask_h] = torch.flip(hr[mask_h], dims=[3])

    mask_v = torch.rand(B, device=lr.device) > 0.5
    if mask_v.any():
        lr[mask_v] = torch.flip(lr[mask_v], dims=[2])
        hr[mask_v] = torch.flip(hr[mask_v], dims=[2])

    k = torch.randint(0, 4, (B,), device=lr.device)
    for kk in range(1, 4):
        mask_k = k == kk
        if mask_k.any():
            lr[mask_k] = torch.rot90(lr[mask_k], kk, dims=[2, 3])
            hr[mask_k] = torch.rot90(hr[mask_k], kk, dims=[2, 3])

    return lr, hr


# ============================================================
# TEACHER: original model, kept byte-for-byte compatible
# with the architecture in the supplied training script.
# ============================================================
class MiniESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 16, 5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 12, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.pixel_shuffle(self.conv3(x))


# ============================================================
# STUDENT
#
# 448 MAC/LR pixel:
#   3x3 3->8          = 216
#   depthwise 3x3 8   = 72
#   1x1 8->8          = 64
#   1x1 8->12         = 96
# total               = 448
#
# It predicts a residual over bicubic interpolation.
# ============================================================
class TinyFSRStudent(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.relu2 = nn.ReLU(inplace=True)

        self.mix = nn.Conv2d(8, 8, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.to_residual = nn.Conv2d(8, 12, 1)
        self.shuffle = nn.PixelShuffle(2)

        # Start close to "do nothing" residual, then learn detail.
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def forward(self, lr):
        x = self.relu1(self.conv_in(lr))
        x = self.relu2(self.dw(x))
        x = self.relu3(self.mix(x))
        residual = self.shuffle(self.to_residual(x))

        base = F.interpolate(
            lr, scale_factor=2, mode="bicubic", align_corners=False
        )

        return (base + residual).clamp(0.0, 1.0), residual


# ============================================================
# LOSSES
# ============================================================
def charbonnier(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))


def sobel_grad(x):
    # Fixed Sobel filters, applied independently to RGB.
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)

    c = x.shape[1]
    gx = F.conv2d(x, kx.expand(c, 1, 3, 3), padding=1, groups=c)
    gy = F.conv2d(x, ky.expand(c, 1, 3, 3), padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def high_frequency(x):
    # Residual after a small Gaussian-like blur.
    blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x - blur


def edge_weight(target):
    g = sobel_grad(target).mean(dim=1, keepdim=True)
    # Stable weighting: edges matter, but cannot dominate the loss.
    return 1.0 + 2.0 * torch.clamp(g / 0.35, 0.0, 1.0)


def weighted_charbonnier(pred, target, weight):
    d = torch.sqrt((pred - target) ** 2 + 1e-3 ** 2)
    return (d * weight).mean()


# ============================================================
# TRUE DIV2K VALIDATION
# ============================================================
class FullImageValidationDataset(Dataset):
    """Loads DIV2K validation HR images and makes exact x2 LR/HR pairs.
    No random crops and no augmentation: metrics are measured on held-out images.
    """
    def __init__(self, root):
        self.paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            self.paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
            self.paths += glob.glob(os.path.join(root, "**", ext.upper()), recursive=True)
        self.paths = sorted(set(self.paths))
        if not self.paths:
            raise FileNotFoundError(
                f"No validation images found in {root}. "
                "Expected DIV2K_valid_HR."
            )
        self.to_tensor = T.ToTensor()
        print(f"Validation images: {len(self.paths)} from {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        w, h = img.size
        # Exact x2 pair; crop odd dimensions from the HR side only.
        w -= w % 2
        h -= h % 2
        img = img.crop((0, 0, w, h))
        lr_img = img.resize((w // 2, h // 2), Image.BICUBIC)
        return self.to_tensor(lr_img), self.to_tensor(img), self.paths[idx]


# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate(student, teacher, loader):
    student.eval()
    teacher.eval()

    sum_psnr = 0.0
    sum_teacher_psnr = 0.0
    sum_bicubic_psnr = 0.0
    n = 0

    for lr, hr, _ in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Validation loader uses batch_size=1, so arbitrary DIV2K image sizes are safe.
        pred, _ = student(lr)
        t = teacher(lr).clamp(0, 1)
        bic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)

        for out, which in ((pred, "student"), (t, "teacher"), (bic, "bicubic")):
            mse = torch.mean((out - hr) ** 2, dim=(1, 2, 3))
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
            value = psnr.sum().item()
            if which == "student":
                sum_psnr += value
            elif which == "teacher":
                sum_teacher_psnr += value
            else:
                sum_bicubic_psnr += value
        n += lr.size(0)

    return (
        sum_psnr / max(n, 1),
        sum_teacher_psnr / max(n, 1),
        sum_bicubic_psnr / max(n, 1),
    )


# ============================================================
# HARD PATCH MINING
# ============================================================
# Intentionally disabled in this stable baseline. We keep the training
# distribution uniform so the loss curve and PSNR remain interpretable.

# ============================================================
# TRAIN
# ============================================================
def train():
    build_cache_if_needed()

    dataset = CachedPatchDataset(CACHE_FILE)

    # True held-out DIV2K validation images, not patches from the training cache.
    val_dataset = FullImageValidationDataset(VALID_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    teacher = MiniESPCN().to(device)

    if not os.path.exists(TEACHER_CKPT):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_CKPT}\n"
            "Train the original model first."
        )

    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    teacher_state = ckpt["model"] if "model" in ckpt else ckpt
    teacher.load_state_dict(teacher_state, strict=True)

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = TinyFSRStudent().to(device)
    ema = TinyFSRStudent().to(device)
    ema.load_state_dict(student.state_dict())

    for p in ema.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")
    print("Student compute budget: 448 MAC/LR-pixel")

    optimizer = optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-5
    )

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_psnr = -1e9
    global_step = 0

    print("\n--- Training student ---")
    print(
        "Loss = 0.40*pixel + 0.15*gradient + 0.15*high-frequency + 0.30*teacher residual (detail-weighted)"
    )
    print("Hard mining: DISABLED (uniform sampling for stable training)")

    for epoch in range(1, EPOCHS + 1):
        student.train()

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS if device.type == "cuda" else 0,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(device.type == "cuda" and NUM_WORKERS > 0),
            prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
            drop_last=True,
        )

        running = 0.0

        for lr, hr, _ in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            lr, hr = augment_batch(lr, hr)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, residual = student(lr)

                with torch.no_grad():
                    teacher_pred = teacher(lr).clamp(0, 1)

                # Teacher's residual relative to exactly the same bicubic base.
                base = F.interpolate(
                    lr, scale_factor=2,
                    mode="bicubic", align_corners=False
                )
                teacher_residual = teacher_pred - base

                w = edge_weight(hr)

                loss_pixel = weighted_charbonnier(pred, hr, w)

                loss_grad = charbonnier(
                    sobel_grad(pred),
                    sobel_grad(hr),
                )

                loss_hf = charbonnier(
                    high_frequency(pred),
                    high_frequency(hr),
                )

                # Distill the teacher's actual added detail over bicubic.
                # Smooth regions stay cheap; high-frequency teacher residuals
                # receive more weight so the 448-MAC student spends capacity
                # where it matters visually.
                teacher_hf = high_frequency(teacher_residual)
                hf_strength = teacher_hf.abs().mean(dim=1, keepdim=True)
                hf_weight = 1.0 + 3.0 * torch.clamp(hf_strength / 0.05, 0.0, 1.0)
                teacher_err = torch.sqrt(
                    (residual - teacher_residual) ** 2 + 1e-3 ** 2
                )
                loss_teacher = (teacher_err * hf_weight).mean()

                loss = (
                    PIXEL_WEIGHT * loss_pixel
                    + GRAD_WEIGHT * loss_grad
                    + HF_WEIGHT * loss_hf
                    + TEACHER_WEIGHT * loss_teacher
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            # EMA
            d = min(EMA_DECAY, (1 + global_step) / (10 + global_step))
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.mul_(d).add_(ps.detach(), alpha=1.0 - d)

                for be, bs in zip(ema.buffers(), student.buffers()):
                    be.copy_(bs)

            running += loss.item()

        scheduler.step()

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"loss={running / len(loader):.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            psnr, teacher_psnr, bic_psnr = validate(
                ema, teacher, val_loader
            )

            print(
                f"  PSNR | student EMA: {psnr:.3f} dB | "
                f"teacher: {teacher_psnr:.3f} dB | "
                f"bicubic: {bic_psnr:.3f} dB"
            )

            torch.save(
                {
                    "model": ema.state_dict(),
                    "epoch": epoch,
                    "psnr": psnr,
                    "teacher_psnr": teacher_psnr,
                    "bicubic_psnr": bic_psnr,
                    "mac_per_lr_pixel": 448,
                },
                os.path.join(CHKPT_DIR, f"student_ep{epoch:03d}.pth"),
            )

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(
                    {
                        "model": ema.state_dict(),
                        "psnr": psnr,
                        "mac_per_lr_pixel": 448,
                    },
                    os.path.join(CHKPT_DIR, "best_448.pth"),
                )
                print(f"  -> new best: {best_psnr:.3f} dB")

    return ema


# ============================================================
# HLSL EXPORT
# ============================================================
def export_to_hlsl(model, filename="weights_448.hlsl"):
    state = model.state_dict()

    with open(filename, "w", encoding="utf-8") as f:
        f.write("// TinyFSR student — 448 MAC/LR pixel\n")
        f.write("// bicubic base + learned residual\n\n")

        for key, tensor in state.items():
            arr = tensor.detach().cpu().numpy()
            name = key.replace(".", "_")

            if "weight" in key:
                shape = "".join(f"[{d}]" for d in arr.shape)
                f.write(f"static const float {name}{shape} = {{\n")
                flat = arr.flatten()

                for i, v in enumerate(flat):
                    f.write(f"{v:.8f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")

                f.write("\n};\n\n")

            elif "bias" in key:
                f.write(
                    f"static const float {name}[{arr.shape[0]}] = {{\n"
                )
                for v in arr:
                    f.write(f"{v:.8f}f, ")
                f.write("\n};\n\n")

    print(f"{filename} written.")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    model = train()

    torch.save(
        model.state_dict(),
        "tinyfsr_448.pth",
    )

    export_to_hlsl(
        model,
        "weights_448.hlsl",
    )

    print("\nDONE")
    print("Validation used held-out DIV2K_valid_HR full images.")
