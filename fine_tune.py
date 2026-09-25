import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Импортируем готовые классы и функции из твоей архитектуры
from train_upscaler import (
    MiniESPCN, DIV2KDataset, charbonnier_loss, validate,
    export_to_hlsl, generate_hlsl_shader,
    DIV2K_DIR, VALID_DIR, SCALE_FACTOR, PATCH_SIZE, BATCH_SIZE,
    NUM_WORKERS, USE_AMP, CHKPT_DIR, device
)

# Настройки дообучения
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR     = 2e-4
BEST_CKPT_PATH   = os.path.join(CHKPT_DIR, "best.pth")

def fine_tune():
    # 1. Загрузка датасета и валидации
    dataset = DIV2KDataset(DIV2K_DIR, patch_size=PATCH_SIZE, scale=SCALE_FACTOR,
                           patches_per_img=3, augment=True)

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

    # 2. Инициализация модели и загрузка лучших весов
    model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)

    if os.path.exists(BEST_CKPT_PATH):
        checkpoint = torch.load(BEST_CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Загружен чекпоинт: {BEST_CKPT_PATH} (PSNR: {checkpoint.get('psnr', 0):.3f} dB)")
    elif os.path.exists("mini_espcn.pth"):
        model.load_state_dict(torch.load("mini_espcn.pth", map_location=device))
        print("Загружен базовый файл mini_espcn.pth")
    else:
        raise FileNotFoundError("Не найдены сохраненные веса для дообучения!")

    # 3. Настройка EMA, Optimizer, Scheduler
    ema_model = MiniESPCN(scale_factor=SCALE_FACTOR).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.999

    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS, eta_min=1e-6)

    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_psnr = validate(ema_model, val_loader)
    print(f"\n--- Стартовый PSNR дообучения: {best_psnr:.3f} dB ---")
    print(f"Дообучение на {FINE_TUNE_EPOCHS} эпох | LR: {FINE_TUNE_LR} -> 1e-6")

    model.train()

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        running_loss = 0.0
        for lr, hr in dataloader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(lr)
                loss = charbonnier_loss(outputs, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Обновление EMA
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        cur_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == FINE_TUNE_EPOCHS:
            psnr_ema = validate(ema_model, val_loader)
            print(f"Дообучение [{epoch}/{FINE_TUNE_EPOCHS}] - Loss: {epoch_loss:.5f} | LR: {cur_lr:.2e} | PSNR EMA: {psnr_ema:.3f} dB")

            if psnr_ema > best_psnr:
                best_psnr = psnr_ema
                torch.save({'model': ema_model.state_dict(), 'psnr': psnr_ema},
                           os.path.join(CHKPT_DIR, "best.pth"))
                print(f"   -> Обновлен лучший PSNR (EMA): {best_psnr:.3f} dB")

    print("\nДообучение завершено!")
    return ema_model

if __name__ == "__main__":
    fine_tuned_ema_model = fine_tune()

    # Сохраняем и экспортируем в HLSL
    torch.save(fine_tuned_ema_model.state_dict(), "mini_espcn.pth")
    export_to_hlsl(fine_tuned_ema_model, "weights.hlsl")
    generate_hlsl_shader("upscale_cs.hlsl")

    print("\n[УСПЕХ] Обновленный файл weights.hlsl и upscale_cs.hlsl готовы к тестам!")
