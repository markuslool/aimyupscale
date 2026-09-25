import os
import gc
import glob
import random
import multiprocessing
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
DIV2K_DIR       = "./DIV2K_train_HR"
VALID_DIR       = "./DIV2K_valid_HR"
CACHE_FILE      = "./div2k_cache_x2_64.pt"
CACHE_LR_SIZE   = 32                            # LR-патч (HR = 64 при x2)

PATCHES_PER_IMAGE_CACHE = 80

SCALE_FACTOR    = 2
PATCH_SIZE      = 64                            # HR-патч
BATCH_SIZE      = 128
EPOCHS          = 60
NUM_WORKERS     = 2                             # Colab: 2 vCPU
PREFETCH        = 4
LEARNING_RATE   = 8e-4
USE_AMP         = True
VAL_EVERY       = 5
EMA_START_EPOCH = 10
CHKPT_DIR       = "./checkpoints_448"

# Дистилляция: старая MiniESPCN выступает teacher.
TEACHER_CKPT    = "./checkpoints/best.pth"

# 448 MAC/LR-pixel student:
# 3x3 3->8 = 216
# DW 3x3 8->8 = 72
# 1x1 8->8 = 64
# 1x1 8->12 = 96
# TOTAL = 448
STUDENT_CHANNELS = 8

# Loss weights. HR остаётся главным источником истины.
W_HR       = 0.70
W_TEACHER  = 0.20
W_GRAD     = 0.10

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(f"Запуск на GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA версия: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print("CUDA не найдена — работаем на CPU")

os.makedirs(CHKPT_DIR, exist_ok=True)

# ==========================================
# 2. ПРЕПРОЦЕССИНГ: кэш без OOM
# ==========================================
def build_cache_if_needed():
    """
    Один раз режет DIV2K на патчи 64x64 (HR) / 32x32 (LR).
    Из каждой картинки берёт случайные PATCHES_PER_IMAGE_CACHE патчей.
    Сохраняет как uint8-тензоры в один .pt-файл.
    Без torch.stack на всём массиве — предвыделяем через torch.empty.
    """
    if os.path.exists(CACHE_FILE):
        print(f"Кэш найден: {CACHE_FILE}")
        return

    print(f"Строю кэш из {DIV2K_DIR} ... (один раз, ~1-2 мин)")
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        img_paths += glob.glob(os.path.join(DIV2K_DIR, "**", ext), recursive=True)
        img_paths += glob.glob(os.path.join(DIV2K_DIR, "**", ext.upper()), recursive=True)
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise ValueError(f"Не найдены картинки в {DIV2K_DIR}!")
    print(f"Найдено картинок: {len(img_paths)}")

    hr_patches = []
    lr_patches = []
    to_tensor = T.ToTensor()
    random.seed(42)

    for i, path in enumerate(img_paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  пропуск {path}: {e}")
            continue

        w, h = img.size
        if w < PATCH_SIZE or h < PATCH_SIZE:
            continue

        max_x = (w - PATCH_SIZE) // PATCH_SIZE + 1
        max_y = (h - PATCH_SIZE) // PATCH_SIZE + 1
        total = max_x * max_y
        k = min(total, PATCHES_PER_IMAGE_CACHE)

        indices = random.sample(range(total), k)
        for idx in indices:
            x = (idx % max_x) * PATCH_SIZE
            y = (idx // max_x) * PATCH_SIZE
            hr = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            lr = hr.resize((CACHE_LR_SIZE, CACHE_LR_SIZE), Image.BICUBIC)
            hr_patches.append((to_tensor(hr) * 255.0).round().byte())
            lr_patches.append((to_tensor(lr) * 255.0).round().byte())

        if (i + 1) % 100 == 0:
            print(f"  обработано {i+1}/{len(img_paths)}, патчей: {len(hr_patches)}")

    n = len(hr_patches)
    if n == 0:
        raise RuntimeError("Ни одного патча не собрано!")
    print(f"Всего патчей: {n}. Сохраняю в {CACHE_FILE} ...")

    # предвыделяем тензоры и копируем поэлементно — меньше пик по RAM
    lr_tensor = torch.empty((n, 3, CACHE_LR_SIZE, CACHE_LR_SIZE), dtype=torch.uint8)
    hr_tensor = torch.empty((n, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.uint8)
    for i, t in enumerate(lr_patches):
        lr_tensor[i] = t
    for i, t in enumerate(hr_patches):
        hr_tensor[i] = t
    del lr_patches, hr_patches
    gc.collect()

    torch.save({'lr': lr_tensor, 'hr': hr_tensor}, CACHE_FILE)
    print(f"Кэш сохранён: LR {tuple(lr_tensor.shape)}, HR {tuple(hr_tensor.shape)}")


# ==========================================
# 3. ДАТАСЕТ ИЗ КЭША
# ==========================================
class CachedPatchDataset(Dataset):
    """
    Загружает весь кэш в RAM как uint8. Возвращает пару (lr, hr) в float32 [0,1].
    Аугментации — в train-цикле на GPU (augment_batch).
    """
    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location='cpu')
        self.lr = data['lr']    # [N, 3, 32, 32] uint8
        self.hr = data['hr']    # [N, 3, 64, 64] uint8
        self.n = self.lr.shape[0]
        print(f"CachedPatchDataset: {self.n} пар, "
              f"LR {tuple(self.lr.shape[1:])}, HR {tuple(self.hr.shape[1:])}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        lr = self.lr[idx].float().div_(255.0)
        hr = self.hr[idx].float().div_(255.0)
        return lr, hr


def augment_batch(lr, hr):
    """
    Аугментации на GPU-тензорах. flip/rot90 коммутируют с downscale,
    поэтому применяем ОДИНАКОВО к lr и hr.
    """
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
        mask_k = (k == kk)
        if mask_k.any():
            lr[mask_k] = torch.rot90(lr[mask_k], k=kk, dims=[2, 3])
            hr[mask_k] = torch.rot90(hr[mask_k], k=kk, dims=[2, 3])
    return lr, hr


# ==========================================
# 4. АРХИТЕКТУРА
# ==========================================
class MiniESPCN(nn.Module):
    """Original ~13K teacher. Kept byte-for-byte compatible with old checkpoints."""
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 12, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.pixel_shuffle(self.conv3(x))


class Student448(nn.Module):
    """
    Ultra-light 448-MAC/LR-pixel student.

    3->8 3x3             216 MAC
    depthwise 8x3x3       72 MAC
    8->8 1x1              64 MAC
    8->12 1x1             96 MAC
    --------------------------------
    total                 448 MAC/LR pixel

    Quality tricks:
      * residual reconstruction over bilinear upsample
      * depthwise spatial filtering preserves a 5x5 effective field
      * pointwise layers do channel mixing cheaply
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.mix = nn.Conv2d(8, 8, 1)
        self.out = nn.Conv2d(8, 12, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.act(self.conv1(x))
        y = self.act(self.dw(y))
        y = self.act(self.mix(y))
        residual = self.pixel_shuffle(self.out(y))
        # Residual is intentionally unrestricted; final clamp happens for validation/export use.
        return base + residual


def count_student_macs(model):
    # MACs per LR pixel for the convolutional part.
    return 3*8*9 + 8*9 + 8*8 + 8*12


# ==========================================
# 5. LOSS
# ==========================================
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def gradient_loss(pred, target):
    """
    L1 gradient loss. Helps preserve thin lines / texture while keeping
    pixel-domain Charbonnier as the main objective.
    """
    px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (F.l1_loss(px, tx) + F.l1_loss(py, ty))


# ==========================================
# 5. LOSS
# ==========================================
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


# ==========================================
# 6. ВАЛИДАЦИЯ
# ==========================================
class ValidationPatchDataset(Dataset):
    """Deterministic patches from DIV2K validation images, not training cache."""
    def __init__(self, root, max_patches=400):
        self.pairs = []
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
        paths = sorted(set(paths))
        if not paths:
            raise ValueError(f"Не найдены validation images в {root}")

        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            if w < PATCH_SIZE or h < PATCH_SIZE:
                continue

            # Deterministic grid; enough independent samples for model comparison.
            xs = list(range(0, w - PATCH_SIZE + 1, PATCH_SIZE))
            ys = list(range(0, h - PATCH_SIZE + 1, PATCH_SIZE))
            for y in ys:
                for x in xs:
                    self.pairs.append((path, x, y))
                    if len(self.pairs) >= max_patches:
                        break
                if len(self.pairs) >= max_patches:
                    break
            if len(self.pairs) >= max_patches:
                break

        self.to_tensor = T.ToTensor()
        print(f"ValidationPatchDataset: {len(self.pairs)} patches")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, x, y = self.pairs[idx]
        img = Image.open(path).convert("RGB")
        hr_img = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        lr_img = hr_img.resize((CACHE_LR_SIZE, CACHE_LR_SIZE), Image.BICUBIC)
        lr = self.to_tensor(lr_img)
        hr = self.to_tensor(hr_img)
        return lr, hr


@torch.no_grad()
def validate(model, val_loader):
    was_training = model.training
    model.eval()
    total_psnr, n = 0.0, 0
    for lr, hr in val_loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        out = model(lr).clamp(0, 1)
        mse = torch.mean((out - hr) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
        total_psnr += psnr.sum().item()
        n += lr.size(0)
    if was_training:
        model.train()
    return total_psnr / max(n, 1)


# ==========================================
# 7. ОБУЧЕНИЕ
# ==========================================
def load_teacher():
    teacher = MiniESPCN(scale_factor=2).to(device)
    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher loaded: {TEACHER_CKPT}")
    return teacher


def train():
    build_cache_if_needed()

    dataset = CachedPatchDataset(CACHE_FILE)
    val_dataset = ValidationPatchDataset(VALID_DIR, max_patches=400)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda" and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=0 if device.type != "cuda" else NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    teacher = load_teacher()
    model = Student448(scale_factor=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student params: {n_params:,}")
    print(f"Student MACs/LR-pixel: {count_student_macs(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-5
    )

    # EMA gives a cheap quality boost and costs almost nothing at inference.
    ema_model = Student448(scale_factor=2).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.999

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n--- Training student on {device} ---")
    print(f"AMP: {'on' if use_amp else 'off'}")
    print(f"Loss weights: HR={W_HR}, teacher={W_TEACHER}, gradient={W_GRAD}")
    print(f"LR: {LEARNING_RATE} -> 5e-5")
    print(f"Steps: {len(dataloader) * EPOCHS}")

    best_psnr = -1.0
    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0

        for lr, hr in dataloader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            lr, hr = augment_batch(lr, hr)

            with torch.no_grad():
                teacher_out = teacher(lr).clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                student_out = model(lr)

                # Keep the HR target as the primary objective.
                loss_hr = charbonnier_loss(student_out, hr)

                # Teacher term transfers texture/reconstruction behaviour.
                loss_teacher = charbonnier_loss(student_out, teacher_out)

                # Gradient term strongly discourages blurry edges.
                loss_grad = gradient_loss(student_out, hr)

                loss = (
                    W_HR * loss_hr
                    + W_TEACHER * loss_teacher
                    + W_GRAD * loss_grad
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            # EMA warm-up.
            d = min(ema_decay, (1 + global_step) / (10 + global_step))
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{EPOCHS}] - Loss {epoch_loss:.5f} | "
            f"LR {cur_lr:.2e} | step {global_step}"
        )

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            psnr_raw = validate(model, val_loader)
            psnr_ema = validate(ema_model, val_loader) if epoch >= EMA_START_EPOCH else None
            print(f"   -> Student PSNR: {psnr_raw:.3f} dB")
            if psnr_ema is not None:
                print(f"   -> Student EMA PSNR: {psnr_ema:.3f} dB")

            score = psnr_ema if psnr_ema is not None else psnr_raw
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": epoch_loss,
                "psnr_raw": psnr_raw,
                "psnr_ema": psnr_ema,
                "global_step": global_step,
                "macs_per_lr_pixel": count_student_macs(model),
            }
            torch.save(ckpt, os.path.join(CHKPT_DIR, f"ckpt_ep{epoch:03d}.pth"))

            if score > best_psnr:
                best_psnr = score
                torch.save(
                    {
                        "model": ema_model.state_dict() if psnr_ema is not None else model.state_dict(),
                        "psnr": score,
                        "macs_per_lr_pixel": count_student_macs(model),
                    },
                    os.path.join(CHKPT_DIR, "best.pth"),
                )
                print(f"   -> New best PSNR: {best_psnr:.3f} dB")

    print("Training complete!")
    return ema_model


# ==========================================
# 8. ЭКСПОРТ ВЕСОВ В HLSL
# ==========================================
def export_to_hlsl(model, filename="weights_448.hlsl"):
    print(f"Экспорт весов в {filename}...")
    state_dict = model.state_dict()

    with open(filename, "w") as f:
        f.write("// 448-MAC/LR-pixel Student Upscaler\n")
        f.write("// conv1: 3->8 3x3, depthwise: 8->8 3x3, mix/out: 1x1\n")
        f.write("// PixelShuffle x2 + bilinear residual base.\n\n")

        for key, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy()
            clean_name = key.replace(".", "_")

            if "weight" in key:
                shape_str = "".join([f"[{d}]" for d in arr.shape])
                f.write(f"static const float {clean_name}{shape_str} = {{\n")
                flat = arr.flatten()
                for i, val in enumerate(flat):
                    f.write(f"{val:.9g}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")
            elif "bias" in key:
                f.write(f"static const float {clean_name}[{arr.shape[0]}] = {{\n")
                for val in arr:
                    f.write(f"{val:.9g}f, ")
                f.write("\n};\n\n")

    print("weights_448.hlsl готов.")


# ==========================================
# 9. ГЕНЕРАЦИЯ HLSL COMPUTE SHADER
# ==========================================
def generate_hlsl_shader(filename="upscale_448_cs.hlsl"):
    print(f"Генерация {filename}...")

    hlsl = r"""// Ultra-light 2x Upscaler
// 448 MAC/LR-pixel neural residual + bilinear base.
//
// Dispatch: Dispatch(ceil(W_LR/8), ceil(H_LR/8), 1)
//
// Important:
// PyTorch training uses F.interpolate(..., mode='bilinear',
// align_corners=False) for the residual base.

#include "weights_448.hlsl"

Texture2D<float4>   InputTexture  : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

#define TG_X 8
#define TG_Y 8
#define HALO 1

#define IN_W (TG_X + 2 * HALO)
#define IN_H (TG_Y + 2 * HALO)

groupshared float3 s_input[IN_H][IN_W];

float3 LoadLR(int2 p, uint2 size)
{
    p = clamp(p, int2(0,0), int2(size) - 1);
    return InputTexture[p].rgb;
}

// Matches PyTorch F.interpolate(..., scale_factor=2, align_corners=False)
// for integer 2x upscaling.
float3 Bilinear2x(int2 op, uint2 size)
{
    float srcX = ((float)op.x + 0.5f) * 0.5f - 0.5f;
    float srcY = ((float)op.y + 0.5f) * 0.5f - 0.5f;

    int x0 = (int)floor(srcX);
    int y0 = (int)floor(srcY);
    float fx = srcX - x0;
    float fy = srcY - y0;

    float3 a = LoadLR(int2(x0,     y0),     size);
    float3 b = LoadLR(int2(x0 + 1, y0),     size);
    float3 c = LoadLR(int2(x0,     y0 + 1), size);
    float3 d = LoadLR(int2(x0 + 1, y0 + 1), size);

    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

[numthreads(TG_X, TG_Y, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint2 texSize;
    InputTexture.GetDimensions(texSize.x, texSize.y);

    int2 lrBase = int2(gid.xy * uint2(TG_X, TG_Y)) - int2(HALO, HALO);

    // Shared LR tile.
    for (int y = gtid.y; y < IN_H; y += TG_Y)
    {
        for (int x = gtid.x; x < IN_W; x += TG_X)
        {
            s_input[y][x] = LoadLR(lrBase + int2(x, y), texSize);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    int2 p = int2(gtid.xy) + int2(HALO, HALO);

    // Conv 3x3: 3 -> 8
    float f1[8];

    [unroll]
    for (int oc = 0; oc < 8; ++oc)
    {
        float sum = conv1_bias[oc];

        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                float3 v = s_input[p.y + dy][p.x + dx];
                sum += v.r * conv1_weight[oc][0][dy+1][dx+1];
                sum += v.g * conv1_weight[oc][1][dy+1][dx+1];
                sum += v.b * conv1_weight[oc][2][dy+1][dx+1];
            }
        }
        f1[oc] = max(0.0f, sum);
    }

    // Depthwise 3x3.
    float f2[8];

    [unroll]
    for (int c = 0; c < 8; ++c)
    {
        float sum = dw_bias[c];

        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                // For a true depthwise convolution we need neighboring f1
                // values, not just the center pixel. This tiny implementation
                // therefore recomputes conv1 neighbors below in a compact path.
            }
        }

        // Placeholder is replaced by generated shader below.
        f2[c] = sum;
    }
}
"""
    # The depthwise layer needs neighboring conv1 values. Generate a compact
    # correct shader using a local helper that evaluates conv1 at arbitrary
    # shared-memory positions. This avoids storing an 8-channel shared tile.
    hlsl = hlsl.replace(
"""        [unroll]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                // For a true depthwise convolution we need neighboring f1
                // values, not just the center pixel. This tiny implementation
                // therefore recomputes conv1 neighbors below in a compact path.
            }
        }

        // Placeholder is replaced by generated shader below.
        f2[c] = sum;""",
"""        // Placeholder.
        f2[c] = sum;"""
    )
    # Instead of emitting a potentially error-prone hand-written optimized
    # shader here, use a mathematically exact straightforward version below.
    hlsl = r"""// Ultra-light 2x Upscaler - 448 MAC/LR pixel
#include "weights_448.hlsl"

Texture2D<float4> InputTexture : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

#define TG_X 8
#define TG_Y 8
#define HALO 2
#define IN_W 12
#define IN_H 12

groupshared float3 s_input[IN_H][IN_W];

float3 LoadLR(int2 p, uint2 size)
{
    p = clamp(p, int2(0,0), int2(size)-1);
    return InputTexture[p].rgb;
}

float3 Bilinear2x(int2 op, uint2 size)
{
    float sx = ((float)op.x + 0.5f) * 0.5f - 0.5f;
    float sy = ((float)op.y + 0.5f) * 0.5f - 0.5f;
    int x0 = (int)floor(sx);
    int y0 = (int)floor(sy);
    float fx = sx - x0;
    float fy = sy - y0;

    float3 a = LoadLR(int2(x0, y0), size);
    float3 b = LoadLR(int2(x0+1, y0), size);
    float3 c = LoadLR(int2(x0, y0+1), size);
    float3 d = LoadLR(int2(x0+1, y0+1), size);
    return lerp(lerp(a,b,fx), lerp(c,d,fx),fy);
}

// Evaluate first 3x3 conv at a shared-memory LR coordinate.
float Conv1(int c, int sy, int sx)
{
    float sum = conv1_bias[c];
    [unroll]
    for (int dy=-1; dy<=1; ++dy)
    {
        [unroll]
        for (int dx=-1; dx<=1; ++dx)
        {
            float3 v = s_input[sy+dy][sx+dx];
            sum += v.r * conv1_weight[c][0][dy+1][dx+1];
            sum += v.g * conv1_weight[c][1][dy+1][dx+1];
            sum += v.b * conv1_weight[c][2][dy+1][dx+1];
        }
    }
    return max(0.0f, sum);
}

[numthreads(TG_X,TG_Y,1)]
void CSMain(uint3 gid:SV_GroupID, uint3 tid:SV_GroupThreadID)
{
    uint2 size;
    InputTexture.GetDimensions(size.x,size.y);

    int2 base = int2(gid.xy * uint2(TG_X,TG_Y)) - int2(HALO,HALO);

    for (int y=tid.y; y<IN_H; y+=TG_Y)
    {
        for (int x=tid.x; x<IN_W; x+=TG_X)
            s_input[y][x] = LoadLR(base+int2(x,y),size);
    }

    GroupMemoryBarrierWithGroupSync();

    int2 center = int2(tid.xy) + int2(HALO,HALO);

    // We need a 1-pixel halo around conv1 for depthwise 3x3.
    // Each thread evaluates its 3x3 conv1 neighborhood and then DW.
    float dwv[8];

    [unroll]
    for (int c=0;c<8;++c)
    {
        float sum=dw_bias[c];

        [unroll]
        for (int dy=-1;dy<=1;++dy)
        {
            [unroll]
            for (int dx=-1;dx<=1;++dx)
            {
                float v=Conv1(c,center.y+dy,center.x+dx);
                sum += v * dw_weight[c][0][dy+1][dx+1];
            }
        }
        dwv[c]=max(0.0f,sum);
    }

    // 1x1 channel mixing.
    float mixed[8];
    [unroll]
    for (int oc=0;oc<8;++oc)
    {
        float sum=mix_bias[oc];
        [unroll]
        for (int ic=0;ic<8;++ic)
            sum += dwv[ic]*mix_weight[oc][ic][0][0];
        mixed[oc]=max(0.0f,sum);
    }

    // 1x1 reconstruction to 12 subpixels.
    float r[12];
    [unroll]
    for (int oc=0;oc<12;++oc)
    {
        float sum=out_bias[oc];
        [unroll]
        for (int ic=0;ic<8;++ic)
            sum += mixed[ic]*out_weight[oc][ic][0][0];
        r[oc]=sum;
    }

    int2 lrPixel=int2(gid.xy*uint2(TG_X,TG_Y))+int2(tid.xy);
    if (lrPixel.x >= (int)size.x || lrPixel.y >= (int)size.y)
        return;

    int2 op=lrPixel*2;

    float3 b00=Bilinear2x(op+int2(0,0),size);
    float3 b10=Bilinear2x(op+int2(1,0),size);
    float3 b01=Bilinear2x(op+int2(0,1),size);
    float3 b11=Bilinear2x(op+int2(1,1),size);

    OutputTexture[op+int2(0,0)] = float4(b00+float3(r[0],r[4],r[8]),1);
    OutputTexture[op+int2(1,0)] = float4(b10+float3(r[1],r[5],r[9]),1);
    OutputTexture[op+int2(0,1)] = float4(b01+float3(r[2],r[6],r[10]),1);
    OutputTexture[op+int2(1,1)] = float4(b11+float3(r[3],r[7],r[11]),1);
}
"""
    with open(filename, "w") as f:
        f.write(hlsl)
    print(f"{filename} готов.")


# ==========================================
# 10. ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    trained_model = train()
    torch.save(trained_model.state_dict(), "student448.pth")
    print("student448.pth сохранён.")
    export_to_hlsl(trained_model, "weights_448.hlsl")
    generate_hlsl_shader("upscale_448_cs.hlsl")
    print("\n[OK] student448.pth, weights_448.hlsl, upscale_448_cs.hlsl")
