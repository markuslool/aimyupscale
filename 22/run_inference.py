import torch
from PIL import Image
import torchvision.transforms as T
from train_upscaler import MiniESPCN  # Импортируем сеть

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Загружаем модель
model = MiniESPCN(scale_factor=2).to(device)
model.load_state_dict(torch.load("mini_espcn.pth", map_location=device))
model.eval()

# 2. Загружаем входной скриншот
img = Image.open("input_test.png").convert("RGB")
input_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

# 3. Инференс
with torch.no_grad():
    output_tensor = model(input_tensor).squeeze(0).cpu().clamp(0, 1)

# 4. Сохраняем результат
result_img = T.ToPILImage()(output_tensor)
result_img.save("output_upscaled_2x.png")
print("Готово! Результат сохранен в output_upscaled_2x.png")
