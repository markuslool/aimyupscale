import os
import glob
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ==========================================
# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ (CPU)
# ==========================================
DIV2K_DIR = "./DIV2K_train_HR"  # Путь к папочке с картинками DIV2K
SCALE_FACTOR = 2                # Масштаб апскейла (2x)
PATCH_SIZE = 64                 # Размер нарезанных кусочков HR
BATCH_SIZE = 32                 # Размер батча для CPU
EPOCHS = 15                     # Для быстрых результатов за ~20 мин на CPU
NUM_WORKERS = 4                 # Потоки загрузки данных
LEARNING_RATE = 1e-3

# Настройка потоков CPU в PyTorch
torch.set_num_threads(os.cpu_count() or 4)
device = torch.device('cpu')
print(f"Запуск на девайсе: {device} (доступно ядер: {os.cpu_count()})")

# ==========================================
# 2. ДАТАСЕТ: НАРЕЗКА DIV2K В РЕАЛЬНОМ ВРЕМЕНИ
# ==========================================
class DIV2KDataset(Dataset):
    def __init__(self, img_dir, patch_size=64, scale=2):
        self.patch_size = patch_size
        self.scale = scale

        # Рекурсивный поиск всех png, PNG, jpg, JPG во всех подпапках
        self.img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            self.img_paths += glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
            self.img_paths += glob.glob(os.path.join(img_dir, "**", ext.upper()), recursive=True)
        self.img_paths = sorted(set(self.img_paths))  # убрать дубликаты, если есть

        if not self.img_paths:
            raise ValueError(f"Не найдены картинки в {img_dir}! Проверь путь к папке.")

        print(f"Успешно найдено картинок: {len(self.img_paths)}")
        self.to_tensor = T.ToTensor()

    def __len__(self):
        # 2000 случайных патчей на одну эпоху для скорости
        return 2000

    def __getitem__(self, idx):
        # Берем случайное фото
        img_path = random.choice(self.img_paths)
        img = Image.open(img_path).convert("RGB")

        # Случайная кроп-зоны HR
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            img = img.resize((max(w, self.patch_size), max(h, self.patch_size)))
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        hr_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Аугментация: случайный поворот или отражение
        if random.random() > 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)

        # Создаем LR-патч сжатием
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
        # Conv1: 3 RGB -> 32 канала (3x3 kernel)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Conv2: 32 -> 16 каналов (3x3 kernel)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Conv3: 16 -> 12 каналов (3 RGB * 2x2 subpixels)
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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    criterion = nn.L1Loss() # L1 Loss для предотвращения "мыла"
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Начинаем обучение на CPU ---")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for batch_idx, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            outputs = model(lr)
            loss = criterion(outputs, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Эпоха [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.5f}")

    print("Обучение завершено!")
    return model

# ==========================================
# 5. КОНВЕРТАТОР ВЕСОВ В HLSL КАТАЛОГ
# ==========================================
def export_to_hlsl(model, filename="weights.hlsl"):
    print(f"Экспорт весов в файл: {filename}...")
    state_dict = model.state_dict()

    with open(filename, "w") as f:
        f.write("// Автоматически сгенерированные веса для HLSL Compute Shader\n\n")

        for key, tensor in state_dict.items():
            arr = tensor.cpu().numpy()
            clean_name = key.replace(".", "_")

            if "weight" in key:
                # Массив весов
                shape_str = "".join([f"[{d}]" for d in arr.shape])
                f.write(f"static const float {clean_name}{shape_str} = {{\n")
                flat_arr = arr.flatten()
                for i, val in enumerate(flat_arr):
                    f.write(f"{val:.6f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n\n")

            elif "bias" in key:
                # Массив сдвигов
                f.write(f"static const float {clean_name}[{arr.shape[0]}] = {{\n")
                for val in arr:
                    f.write(f"{val:.6f}f, ")
                f.write("\n};\n\n")

    print("Готово! Файл весов успешно сформирован.")

if __name__ == "__main__":
    trained_model = train()
    export_to_hlsl(trained_model, "weights.hlsl")
