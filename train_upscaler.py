import os
import glob
import random
import multiprocessing
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ==========================================
# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ
# ==========================================
DIV2K_DIR = "./DIV2K_train_HR"  # Путь к папке с датасетом
SCALE_FACTOR = 2                # Масштаб апскейла (2x)
PATCH_SIZE = 128                # Размер патча HR (128x128 идеально под GPU)
BATCH_SIZE = 64                 # Размер батча
EPOCHS = 50                     # Количество эпох
NUM_WORKERS = 4                 # Потоки DataLoader
LEARNING_RATE = 1e-3
USE_AMP = True                  # Mixed precision

# Выбор устройства
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(f"Запуск на GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA версия: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print("CUDA не найдена — работаем на CPU")

torch.set_num_threads(os.cpu_count() or 4)

# ==========================================
# 2. ДАТАСЕТ (DIV2K)
# ==========================================
class DIV2KDataset(Dataset):
    def __init__(self, img_dir, patch_size=128, scale=2):
        self.patch_size = patch_size
        self.scale = scale

        self.img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            self.img_paths += glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
            self.img_paths += glob.glob(os.path.join(img_dir, "**", ext.upper()), recursive=True)
        self.img_paths = sorted(set(self.img_paths))

        if not self.img_paths:
            raise ValueError(f"Не найдены картинки в {img_dir}! Проверь путь к папке.")

        print(f"Успешно найдено картинок в датасете: {len(self.img_paths)}")
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        while True:
            img_path = random.choice(self.img_paths)
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except Exception:
                continue

        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            img = img.resize((max(w, self.patch_size), max(h, self.patch_size)))
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        hr_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        if random.random() > 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)

        lr_w = self.patch_size // self.scale
        lr_h = self.patch_size // self.scale
        lr_patch = hr_patch.resize((lr_w, lr_h), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

# ==========================================
# 3. АРХИТЕКТУРА МИНИ-СЕТИ (ESPCN)
# ==========================================
class MiniESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super(MiniESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(16, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# ==========================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def train():
    dataset = DIV2KDataset(DIV2K_DIR, patch_size=PATCH_SIZE, scale=SCALE_FACTOR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0 and device.type == 'cuda'),
        drop_last=True,
    )

    model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\n--- Начинаем обучение на {device} ---")
    print(f"Mixed precision (AMP): {'включён' if use_amp else 'выключен'}")
    model.train()

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

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Эпоха [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.5f}")

    print("Обучение завершено!")
    return model

# ==========================================
# 5. ЭКСПОРТ ВЕСОВ В HLSL C-ARRAY
# ==========================================
def export_to_hlsl(model, filename="weights.hlsl"):
    print(f"Экспорт весов в файл: {filename}...")
    state_dict = model.state_dict()

    with open(filename, "w") as f:
        f.write("// Автоматически сгенерированные веса для HLSL Compute Shader\n\n")

        for key, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy()
            clean_name = key.replace(".", "_")

            if "weight" in key:
                shape_str = "".join([f"[{d}]" for d in arr.shape])
                f.write(f"static const float {clean_name}{shape_str} = {{\n")
                flat_arr = arr.flatten()
                for i, val in enumerate(flat_arr):
                    f.write(f"{val:.6f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

            elif "bias" in key:
                f.write(f"static const float {clean_name}[{arr.shape[0]}] = {{\n")
                for val in arr:
                    f.write(f"{val:.6f}f, ")
                f.write("\n};\n\n")

    print("Файл weights.hlsl сформирован.")

# ==========================================
# 6. ГЕНЕРАЦИЯ COMPUTE SHADER (HLSL)
# ==========================================
def generate_hlsl_shader(filename="upscale_cs.hlsl"):
    print(f"Генерация Compute Shader кода в файл: {filename}...")
    hlsl_code = """// Compute Shader 2x Upscaler (MiniESPCN)
#include "weights.hlsl"

Texture2D<float4> InputTexture : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pos = int2(dispatchThreadID.xy);

    // 1. Считываем 3x3 окрестность пикселей (RGB)
    float3 neighbors[3][3];
    [unroll]
    for (int dy = -1; dy <= 1; dy++) {
        [unroll]
        for (int dx = -1; dx <= 1; dx++) {
            neighbors[dy + 1][dx + 1] = InputTexture[pos + int2(dx, dy)].rgb;
        }
    }

    // 2. Слой 1: Conv 3x3 (3 -> 32 каналов) + ReLU
    float layer1[32];
    [unroll]
    for (int c = 0; c < 32; c++) {
        float sum = conv1_bias[c];
        [unroll]
        for (int y = 0; y < 3; y++) {
            [unroll]
            for (int x = 0; x < 3; x++) {
                sum += dot(neighbors[y][x], float3(
                    conv1_weight[c][0][y][x],
                    conv1_weight[c][1][y][x],
                    conv1_weight[c][2][y][x]
                ));
            }
        }
        layer1[c] = max(0.0f, sum);
    }

    // 3. Слой 2: Conv 3x3 (32 -> 16 каналов) + ReLU
    float layer2[16];
    [unroll]
    for (int c = 0; c < 16; c++) {
        float sum = conv2_bias[c];
        [unroll]
        for (int in_c = 0; in_c < 32; in_c++) {
            [unroll]
            for (int y = 0; y < 3; y++) {
                [unroll]
                for (int x = 0; x < 3; x++) {
                    // Используем соседний пиксель центральной точки для слоя 2
                    sum += layer1[in_c] * conv2_weight[c][in_c][y][x];
                }
            }
        }
        layer2[c] = max(0.0f, sum);
    }

    // 4. Слой 3: Conv 3x3 (16 -> 12 каналов)
    float layer3[12];
    [unroll]
    for (int c = 0; c < 12; c++) {
        float sum = conv3_bias[c];
        [unroll]
        for (int in_c = 0; in_c < 16; in_c++) {
            [unroll]
            for (int y = 0; y < 3; y++) {
                [unroll]
                for (int x = 0; x < 3; x++) {
                    sum += layer2[in_c] * conv3_weight[c][in_c][y][x];
                }
            }
        }
        layer3[c] = sum;
    }

    // 5. PixelShuffle 2x: раскладываем 12 каналов на 2x2 блок RGB пикселей
    int2 out_pos = pos * 2;

    // Pixel (0,0) -> Каналы 0, 1, 2
    OutputTexture[out_pos + int2(0, 0)] = float4(layer3[0], layer3[1], layer3[2], 1.0f);
    // Pixel (1,0) -> Каналы 3, 4, 5
    OutputTexture[out_pos + int2(1, 0)] = float4(layer3[3], layer3[4], layer3[5], 1.0f);
    // Pixel (0,1) -> Каналы 6, 7, 8
    OutputTexture[out_pos + int2(0, 1)] = float4(layer3[6], layer3[7], layer3[8], 1.0f);
    // Pixel (1,1) -> Каналы 9, 10, 11
    OutputTexture[out_pos + int2(9], layer3[10], layer3[11], 1.0f);
}
"""
    with open(filename, "w") as f:
        f.write(hlsl_code)
    print(f"Файл {filename} сформирован.")

# ==========================================
# 7. ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Обучаем модель
    trained_model = train()

    # Сохраняем PyTorch-чекпоинт под Python (для тестов инференса)
    torch.save(trained_model.state_dict(), "mini_espcn.pth")
    print("Файл mini_espcn.pth успешно сохранен.")

    # Генерируем веса и файл Compute Shader
    export_to_hlsl(trained_model, "weights.hlsl")
    generate_hlsl_shader("upscale_cs.hlsl")

    print("\n[УСПЕХ] Все артефакты (mini_espcn.pth, weights.hlsl, upscale_cs.hlsl) собраны!")
