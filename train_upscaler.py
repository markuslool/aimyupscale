import os
import glob
import random
import multiprocessing
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
DIV2K_DIR       = "./DIV2K_train_HR"
VALID_DIR       = "./DIV2K_valid_HR"
CACHE_FILE      = "./div2k_cache_x2_64.pt"     # один файл со всем кэшем
CACHE_LR_SIZE   = 32                           # LR-патч (HR = 64 при x2)

SCALE_FACTOR    = 2
PATCH_SIZE      = 64                           # HR-патч (было 128)
BATCH_SIZE      = 128
EPOCHS          = 50
NUM_WORKERS     = 2                            # Colab: 2 vCPU
PREFETCH        = 4
LEARNING_RATE   = 1e-3
USE_AMP         = True
PATCHES_PER_IMG = 3                           # было 3 эпохи
VAL_EVERY       = 5
EMA_START_EPOCH = 15                           # валидировать EMA не раньше
CHKPT_DIR       = "./checkpoints"

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
# 2. ПРЕПРОЦЕССИНГ: один раз режем DIV2K на патчи и кэшируем в RAM
# ==========================================
def build_cache_if_needed():
    """
    Один раз проходит по DIV2K, режет каждую картинку на патчи 64x64 (HR) / 32x32 (LR),
    сохраняет в один .pt файл как uint8-тензоры (экономия RAM в 4 раза).
    Патчи сохраняются БЕЗ аугментаций — аугментации делаются на лету на GPU.
    """
    if os.path.exists(CACHE_FILE):
        print(f"Кэш найден: {CACHE_FILE}")
        return

    print(f"Строю кэш из {DIV2K_DIR} ... (один раз, может занять пару минут)")

    # Собираем все пути
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
    stride = PATCH_SIZE  # без перекрытия — самый быстрый вариант
    to_tensor = T.ToTensor()

    for i, path in enumerate(img_paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  пропуск {path}: {e}")
            continue

        w, h = img.size
        if w < PATCH_SIZE or h < PATCH_SIZE:
            continue

        # Режем на патчи с шагом stride
        for y in range(0, h - PATCH_SIZE + 1, stride):
            for x in range(0, w - PATCH_SIZE + 1, stride):
                hr = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                lr = hr.resize((CACHE_LR_SIZE, CACHE_LR_SIZE), Image.BICUBIC)

                hr_t = (to_tensor(hr) * 255.0).round().byte()  # uint8
                lr_t = (to_tensor(lr) * 255.0).round().byte()

                hr_patches.append(hr_t)
                lr_patches.append(lr_t)

        if (i + 1) % 100 == 0:
            print(f"  обработано {i+1}/{len(img_paths)}, патчей: {len(hr_patches)}")

    print(f"Всего патчей: {len(hr_patches)}. Сохраняю в {CACHE_FILE} ...")
    torch.save({
        'lr': torch.stack(lr_patches),   # [N, 3, 32, 32] uint8
        'hr': torch.stack(hr_patches),   # [N, 3, 64, 64] uint8
    }, CACHE_FILE)
    print(f"Кэш сохранён. Размер LR: {lr_patches[0].shape}, HR: {hr_patches[0].shape}")


# ==========================================
# 3. ДАТАСЕТ ИЗ КЭША (всё в RAM, аугментации на GPU)
# ==========================================
class CachedPatchDataset(Dataset):
    """
    Загружает весь кэш в RAM как uint8. На __getitem__ возвращает пару (lr, hr)
    в float32 [0,1] БЕЗ аугментаций — аугментации делаются в train-цикле на GPU.
    """
    def __init__(self, cache_file, augment=False):
        data = torch.load(cache_file, map_location='cpu')
        self.lr = data['lr']    # [N, 3, 32, 32] uint8
        self.hr = data['hr']    # [N, 3, 64, 64] uint8
        self.augment = augment
        self.n = self.lr.shape[0]
        print(f"CachedPatchDataset: {self.n} пар, LR {tuple(self.lr.shape[1:])}, HR {tuple(self.hr.shape[1:])}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # uint8 -> float32 [0,1]
        lr = self.lr[idx].float().div_(255.0)
        hr = self.hr[idx].float().div_(255.0)
        return lr, hr


def augment_batch(lr, hr):
    """
    Аугментации на GPU-тензорах. Применяются ОДИНАКОВО к lr и hr
    (flip/rot180 — они коммутируют с downscale).
    """
    B = lr.size(0)
    # horizontal flip
    mask_h = torch.rand(B, device=lr.device) > 0.5
    if mask_h.any():
        lr[mask_h] = torch.flip(lr[mask_h], dims=[3])
        hr[mask_h] = torch.flip(hr[mask_h], dims=[3])
    # vertical flip
    mask_v = torch.rand(B, device=lr.device) > 0.5
    if mask_v.any():
        lr[mask_v] = torch.flip(lr[mask_v], dims=[2])
        hr[mask_v] = torch.flip(hr[mask_v], dims=[2])
    # rot90 x k (только кратные 90, чтобы LR/HR оставались согласованы)
    k = torch.randint(0, 4, (B,), device=lr.device)
    for kk in range(1, 4):
        mask_k = (k == kk)
        if mask_k.any():
            lr[mask_k] = torch.rot90(lr[mask_k], k=kk, dims=[2, 3])
            hr[mask_k] = torch.rot90(hr[mask_k], k=kk, dims=[2, 3])
    return lr, hr


# ==========================================
# 4. АРХИТЕКТУРА (совпадает с HLSL)
# ==========================================
class MiniESPCN(nn.Module):
    """
    Conv3x3(3->32)+ReLU -> Conv5x5(32->16)+ReLU -> Conv3x3(16->12) -> PixelShuffle(2x)
    Receptive field всей сети: 9x9.
    Параметров: ~13K.
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        assert scale_factor == 2, "HLSL-шейдер написан под x2"
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x


# ==========================================
# 5. LOSS
# ==========================================
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


# ==========================================
# 6. ВАЛИДАЦИЯ
# ==========================================
@torch.no_grad()
def validate(model, val_loader):
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
    model.train()
    return total_psnr / max(n, 1)


# ==========================================
# 7. ОБУЧЕНИЕ
# ==========================================
def train():
    build_cache_if_needed()

    dataset = CachedPatchDataset(CACHE_FILE, augment=False)

    # Валидация — 100 случайных патчей из того же кэша (быстро и без I/O)
    n_val = min(100, len(dataset))
    val_idx = list(range(n_val))
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(device.type == 'cuda' and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=32, shuffle=False,
        num_workers=0 if device.type != 'cuda' else NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )

    model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели: {n_params:,}")

    criterion = charbonnier_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # eta_min=1e-4 вместо 1e-5 — LR не падает в ноль
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4
    )

    # EMA модель
    ema_model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.999

    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\n--- Обучение на {device} ---")
    print(f"AMP: {'вкл' if use_amp else 'выкл'} | Loss: Charbonnier | LR: {LEARNING_RATE} -> 1e-4 (Cosine)")
    print(f"Патчей в датасете: {len(dataset)} | Батчей на эпоху: {len(dataloader)}")
    print(f"Всего шагов за {EPOCHS} эпох: {len(dataloader) * EPOCHS}")
    model.train()

    best_psnr = -1.0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for lr, hr in dataloader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            # Аугментации на GPU
            lr, hr = augment_batch(lr, hr)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(lr)
                loss = criterion(outputs, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA с warm-up: на первых шагах быстро тянется к модели
            global_step += 1
            d = min(ema_decay, (1 + global_step) / (10 + global_step))
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.5f} | LR: {cur_lr:.2e} | step: {global_step}")

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            psnr_raw = validate(model, val_loader)
            print(f"   -> PSNR raw: {psnr_raw:.3f} dB")

            psnr_ema = None
            if epoch >= EMA_START_EPOCH:
                psnr_ema = validate(ema_model, val_loader)
                print(f"   -> PSNR EMA: {psnr_ema:.3f} dB")

            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': epoch_loss,
                'psnr_raw': psnr_raw,
                'psnr_ema': psnr_ema,
                'global_step': global_step,
            }
            torch.save(ckpt, os.path.join(CHKPT_DIR, f"ckpt_ep{epoch:03d}.pth"))

            # best по raw, если EMA ещё не валидируется
            score = psnr_ema if psnr_ema is not None else psnr_raw
            if score > best_psnr:
                best_psnr = score
                torch.save({'model': ema_model.state_dict() if psnr_ema is not None
                            else model.state_dict(),
                            'psnr': score},
                           os.path.join(CHKPT_DIR, "best.pth"))
                print(f"   -> Новый лучший PSNR: {best_psnr:.3f} dB")

    print("Обучение завершено!")
    return ema_model


# ==========================================
# 8. ЭКСПОРТ ВЕСОВ В HLSL
# ==========================================
def export_to_hlsl(model, filename="weights.hlsl"):
    print(f"Экспорт весов в {filename}...")
    state_dict = model.state_dict()

    with open(filename, "w") as f:
        f.write("// Автоматически сгенерированные веса (MiniESPCN, x2)\n")
        f.write("// Layout: convN_weight[out_c][in_c][ky][kx] — совпадает с PyTorch Conv2d.\n")
        f.write("// PixelShuffle: out[c, 2h+dh, 2w+dw] = in[c*r^2 + dh*r + dw, h, w], r=2, c_out=3.\n\n")

        for key, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy()
            clean_name = key.replace(".", "_")

            if "weight" in key:
                shape_str = "".join([f"[{d}]" for d in arr.shape])
                f.write(f"static const float {clean_name}{shape_str} = {{\n")
                flat = arr.flatten()
                for i, val in enumerate(flat):
                    f.write(f"{val:.6f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

            elif "bias" in key:
                f.write(f"static const float {clean_name}[{arr.shape[0]}] = {{\n")
                for val in arr:
                    f.write(f"{val:.6f}f, ")
                f.write("\n};\n\n")

    print("weights.hlsl готов.")


# ==========================================
# 9. ГЕНЕРАЦИЯ HLSL COMPUTE SHADER
# ==========================================
def generate_hlsl_shader(filename="upscale_cs.hlsl"):
    print(f"Генерация {filename}...")

    hlsl = r"""// Compute Shader 2x Upscaler (MiniESPCN)
// Архитектура: Conv3x3(3->32)+ReLU -> Conv5x5(32->16)+ReLU -> Conv3x3(16->12) -> PixelShuffle(2)
//
// Схема tiling:
//   TG = 8x8 LR-выходов -> 16x16 HR-выходов
//   halo = 4 (radius всей сети)
//   IN = 16x16 LR-входов
//
// Shared memory:
//   s_input: 16x16x3  = 3.0 KB
//   s_l1:    14x14x32 = 25.1 KB
//   s_l2:    10x10x16 = 6.4 KB
//   итого:   ~34.5 KB
//
// Dispatch: Dispatch(ceil(W_LR/8), ceil(H_LR/8), 1)

#include "weights.hlsl"

Texture2D<float4>   InputTexture  : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

#define TG_X 8
#define TG_Y 8
#define HALO 4

#define IN_W  (TG_X + 2 * HALO)   // 16
#define IN_H  (TG_Y + 2 * HALO)   // 16

#define L1_W  (IN_W - 2)          // 14
#define L1_H  (IN_H - 2)          // 14
#define L2_W  (L1_W - 4)          // 10
#define L2_H  (L1_H - 4)          // 10

groupshared float3 s_input[IN_H][IN_W];
groupshared float  s_l1[32][L1_H][L1_W];
groupshared float  s_l2[16][L2_H][L2_W];

[numthreads(TG_X, TG_Y, 1)]
void CSMain(uint3 gid  : SV_GroupID,
            uint3 gtid : SV_GroupThreadID)
{
    uint2 texSize;
    InputTexture.GetDimensions(texSize.x, texSize.y);

    int2 inBase = int2(gid.xy * uint2(TG_X, TG_Y)) - int2(HALO, HALO);

    // ---------- Загрузка входа с ZERO-padding ----------
    for (int y = gtid.y; y < IN_H; y += TG_Y) {
        for (int x = gtid.x; x < IN_W; x += TG_X) {
            int2 p = inBase + int2(x, y);
            if (p.x < 0 || p.y < 0 || p.x >= (int)texSize.x || p.y >= (int)texSize.y) {
                s_input[y][x] = float3(0.0f, 0.0f, 0.0f);
            } else {
                s_input[y][x] = InputTexture[p].rgb;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------- Conv1 3x3 (3->32) + ReLU ----------
    for (int y = gtid.y; y < L1_H; y += TG_Y) {
        for (int x = gtid.x; x < L1_W; x += TG_X) {
            int iy = y + 1, ix = x + 1;
            [unroll]
            for (int c = 0; c < 32; c++) {
                float sum = conv1_bias[c];
                [unroll]
                for (int dy = -1; dy <= 1; dy++) {
                    [unroll]
                    for (int dx = -1; dx <= 1; dx++) {
                        float3 v = s_input[iy + dy][ix + dx];
                        sum += v.r * conv1_weight[c][0][dy+1][dx+1]
                             + v.g * conv1_weight[c][1][dy+1][dx+1]
                             + v.b * conv1_weight[c][2][dy+1][dx+1];
                    }
                }
                s_l1[c][y][x] = max(0.0f, sum);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------- Conv2 5x5 (32->16) + ReLU ----------
    for (int y = gtid.y; y < L2_H; y += TG_Y) {
        for (int x = gtid.x; x < L2_W; x += TG_X) {
            int iy = y + 2, ix = x + 2;
            [unroll]
            for (int c = 0; c < 16; c++) {
                float sum = conv2_bias[c];
                [unroll]
                for (int in_c = 0; in_c < 32; in_c++) {
                    [unroll]
                    for (int dy = -2; dy <= 2; dy++) {
                        [unroll]
                        for (int dx = -2; dx <= 2; dx++) {
                            sum += s_l1[in_c][iy + dy][ix + dx]
                                 * conv2_weight[c][in_c][dy + 2][dx + 2];
                        }
                    }
                }
                s_l2[c][y][x] = max(0.0f, sum);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------- Conv3 3x3 (16->12), register accumulation ----------
    int iy = gtid.y + 1, ix = gtid.x + 1;

    float l3_0  = conv3_bias[0],  l3_1  = conv3_bias[1],  l3_2  = conv3_bias[2];
    float l3_3  = conv3_bias[3],  l3_4  = conv3_bias[4],  l3_5  = conv3_bias[5];
    float l3_6  = conv3_bias[6],  l3_7  = conv3_bias[7],  l3_8  = conv3_bias[8];
    float l3_9  = conv3_bias[9],  l3_10 = conv3_bias[10], l3_11 = conv3_bias[11];

    [unroll]
    for (int in_c = 0; in_c < 16; in_c++) {
        [unroll]
        for (int dy = -1; dy <= 1; dy++) {
            [unroll]
            for (int dx = -1; dx <= 1; dx++) {
                float v = s_l2[in_c][iy + dy][ix + dx];
                l3_0  += v * conv3_weight[0][in_c][dy+1][dx+1];
                l3_1  += v * conv3_weight[1][in_c][dy+1][dx+1];
                l3_2  += v * conv3_weight[2][in_c][dy+1][dx+1];
                l3_3  += v * conv3_weight[3][in_c][dy+1][dx+1];
                l3_4  += v * conv3_weight[4][in_c][dy+1][dx+1];
                l3_5  += v * conv3_weight[5][in_c][dy+1][dx+1];
                l3_6  += v * conv3_weight[6][in_c][dy+1][dx+1];
                l3_7  += v * conv3_weight[7][in_c][dy+1][dx+1];
                l3_8  += v * conv3_weight[8][in_c][dy+1][dx+1];
                l3_9  += v * conv3_weight[9][in_c][dy+1][dx+1];
                l3_10 += v * conv3_weight[10][in_c][dy+1][dx+1];
                l3_11 += v * conv3_weight[11][in_c][dy+1][dx+1];
            }
        }
    }

    // ---------- PixelShuffle + BOUNDS CHECK ----------
    int2 outPixel = int2(gid.xy * uint2(TG_X, TG_Y)) + int2(gtid.xy);

    if (outPixel.x >= (int)texSize.x || outPixel.y >= (int)texSize.y)
        return;

    int2 outBase = outPixel * 2;

    OutputTexture[outBase + int2(0, 0)] = float4(l3_0,  l3_4,  l3_8,  1.0f);
    OutputTexture[outBase + int2(1, 0)] = float4(l3_1,  l3_5,  l3_9,  1.0f);
    OutputTexture[outBase + int2(0, 1)] = float4(l3_2,  l3_6,  l3_10, 1.0f);
    OutputTexture[outBase + int2(1, 1)] = float4(l3_3,  l3_7,  l3_11, 1.0f);
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

    torch.save(trained_model.state_dict(), "mini_espcn.pth")
    print("mini_espcn.pth сохранён.")

    export_to_hlsl(trained_model, "weights.hlsl")
    generate_hlsl_shader("upscale_cs.hlsl")

    print("\n[УСПЕХ] Артефакты: mini_espcn.pth, weights.hlsl, upscale_cs.hlsl")
