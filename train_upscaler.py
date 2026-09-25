import os
import glob
import random
import multiprocessing
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
DIV2K_DIR       = "./DIV2K_train_HR"
VALID_DIR       = "./DIV2K_valid_HR"       # опционально; если нет — возьмём часть train
SCALE_FACTOR    = 2
PATCH_SIZE      = 128                      # HR-патч; LR = 64
BATCH_SIZE      = 64
EPOCHS          = 50
NUM_WORKERS     = 4
LEARNING_RATE   = 1e-3
USE_AMP         = True
PATCHES_PER_IMG = 3                        # виртуальная длина эпохи
VAL_EVERY       = 5
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
# 2. ДАТАСЕТ
# ==========================================
class DIV2KDataset(Dataset):
    def __init__(self, img_dir, patch_size=128, scale=2, patches_per_img=3, augment=True):
        self.patch_size = patch_size
        self.scale = scale
        self.patches_per_img = patches_per_img
        self.augment = augment

        self.img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            self.img_paths += glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
            self.img_paths += glob.glob(os.path.join(img_dir, "**", ext.upper()), recursive=True)
        self.img_paths = sorted(set(self.img_paths))

        if not self.img_paths:
            raise ValueError(f"Не найдены картинки в {img_dir}!")

        print(f"Найдено картинок: {len(self.img_paths)}")
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.img_paths) * self.patches_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_img

        for _ in range(5):
            img_path = self.img_paths[img_idx]
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except Exception:
                img_idx = random.randrange(len(self.img_paths))
        else:
            raise RuntimeError("Не удалось открыть картинку за 5 попыток")

        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            img = img.resize((max(w, self.patch_size), max(h, self.patch_size)))
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.augment:
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.randint(0, 3)
            if k:
                hr_patch = hr_patch.rotate(90 * k, expand=True)

        lr_size = self.patch_size // self.scale
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

# ==========================================
# 3. АРХИТЕКТУРА (совпадает с HLSL)
# ==========================================
class MiniESPCN(nn.Module):
    """
    Conv3x3(3->32)+ReLU -> Conv5x5(32->16)+ReLU -> Conv3x3(16->12) -> PixelShuffle(2x)
    Receptive field всей сети: 9x9 (1+2+1 с каждой стороны).
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
# 4. LOSS
# ==========================================
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

# ==========================================
# 5. ВАЛИДАЦИЯ
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
# 6. ОБУЧЕНИЕ
# ==========================================
def train():
    dataset = DIV2KDataset(DIV2K_DIR, patch_size=PATCH_SIZE, scale=SCALE_FACTOR,
                           patches_per_img=PATCHES_PER_IMG, augment=True)

    if os.path.isdir(VALID_DIR) and glob.glob(os.path.join(VALID_DIR, "**", "*.png"), recursive=True):
        val_set = DIV2KDataset(VALID_DIR, patch_size=PATCH_SIZE, scale=SCALE_FACTOR,
                               patches_per_img=1, augment=False)
    else:
        n_val = min(20, len(dataset.img_paths))
        val_set = Subset(
            DIV2KDataset(DIV2K_DIR, patch_size=PATCH_SIZE, scale=SCALE_FACTOR,
                         patches_per_img=1, augment=False),
            indices=list(range(n_val)),
        )

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0 and device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'),
    )

    model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели: {n_params:,}")

    criterion = charbonnier_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # EMA модель
    ema_model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.999

    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\n--- Обучение на {device} ---")
    print(f"AMP: {'вкл' if use_amp else 'выкл'} | Loss: Charbonnier | LR: {LEARNING_RATE} -> 1e-5 (Cosine)")
    model.train()

    best_psnr = -1.0

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for lr, hr in dataloader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(lr)
                loss = criterion(outputs, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.5f} | LR: {cur_lr:.2e}")

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            psnr_raw = validate(model, val_loader)
            psnr_ema = validate(ema_model, val_loader)
            print(f"   -> PSNR raw: {psnr_raw:.3f} dB | EMA: {psnr_ema:.3f} dB")

            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': epoch_loss,
                'psnr_raw': psnr_raw,
                'psnr_ema': psnr_ema,
            }
            torch.save(ckpt, os.path.join(CHKPT_DIR, f"ckpt_ep{epoch:03d}.pth"))

            if psnr_ema > best_psnr:
                best_psnr = psnr_ema
                torch.save({'model': ema_model.state_dict(), 'psnr': psnr_ema},
                           os.path.join(CHKPT_DIR, "best.pth"))
                print(f"   -> Новый лучший PSNR (EMA): {best_psnr:.3f} dB")

    print("Обучение завершено!")
    return ema_model

# ==========================================
# 7. ЭКСПОРТ ВЕСОВ В HLSL
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
# 8. ГЕНЕРАЦИЯ HLSL COMPUTE SHADER
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
// Dependency cone (с уменьшением размеров слоёв):
//   input  16x16
//   conv1  14x14  (radius 1)   -> храним в shared
//   conv2  10x10  (radius 2)   -> храним в shared
//   conv3   8x8   (radius 1)   -> в регистрах
//
// Shared memory:
//   s_input: 16x16x3  = 3.0 KB
//   s_l1:    14x14x32 = 25.1 KB
//   s_l2:    10x10x16 = 6.4 KB
//   итого:   ~34.5 KB  (влезает в 48 KB Kepler)
//
// Dispatch: Dispatch(ceil(W_LR/8), ceil(H_LR/8), 1)

// Compute Shader 2x Upscaler (MiniESPCN) — corrected
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

    // ---------- Загрузка входа с ZERO-padding (как PyTorch padding_mode='zeros') ----------
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
    int2 outTexSize = int2(texSize) * 2;

    // Проверка: не вышли ли за пределы выходной текстуры
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
# 9. ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    trained_model = train()

    torch.save(trained_model.state_dict(), "mini_espcn.pth")
    print("mini_espcn.pth сохранён.")

    export_to_hlsl(trained_model, "weights.hlsl")
    generate_hlsl_shader("upscale_cs.hlsl")

    print("\n[УСПЕХ] Артефакты: mini_espcn.pth, weights.hlsl, upscale_cs.hlsl")
