"""Тренер PixelTransformer с двумя фазами: train -> test.

Запуск вручную (из trainer/):
    python train.py --source synth          # быстрая офлайн-проверка пайплайна
    python train.py --source real --steps 2000
    python train.py --test-only             # только фаза тестов по сохранённым весам

После обучения печатается сводка фазы тестов (MSE на валидации + метрики
расстановки слоев) и сохраняются примеры картинок tests/sample_*.png.
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import PixelTransformer
from layers import arrange_layers, flat_to_grid


# ---------------------------------------------------------------------------
# Метрики качества "расстановки" (для ручной оценки без инференсера)
# ---------------------------------------------------------------------------

def centering_error(pixels: np.ndarray) -> float:
    """Расстояние от центра масс 'объектных' пикселей до центра канвы."""
    grid = flat_to_grid(pixels)
    bg = np.median(grid.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(grid - bg, axis=2)
    mask = dist > max(0.15, np.percentile(dist, 60))
    if not mask.any():
        return float("nan")
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    return float(np.hypot(cy - 7.5, cx - 7.5))


def color_consistency(pixels: np.ndarray) -> float:
    """Среднее расстояние пикселя до ближайшего соседа (малое = связный спрайт)."""
    grid = flat_to_grid(pixels)
    pad = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    nb = np.stack([pad[0:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, 0:-2], pad[1:-1, 2:]])
    return float(nb.sub(grid)[np.newaxis].min(axis=0).mean()) if False else \
        float(np.linalg.norm(nb - grid, axis=-1).min(axis=0).mean())


def save_png(pixels: np.ndarray, path: str, transparent_black=True):
    """Рендер [256,3] в PNG так же, как это делает Rust-инференсер."""
    from PIL import Image
    rgb = (np.clip(pixels, 0, 1) * 255).astype(np.uint8).reshape(16, 16, 3)
    rgba = np.dstack([rgb, np.full((16, 16), 255, np.uint8)])
    if transparent_black:
        black = (rgb.sum(axis=2) == 0)
        rgba[:, :, 3][black] = 0
    img = Image.fromarray(rgba).resize((128, 128), Image.Resampling.NEAREST)
    img.save(path)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

def get_loaders(source: str, batch_size: int):
    if source == "synth":
        from synth_dataset import get_synth_dataloader
        # train и val — разные seed'ы, чтобы валидация была честной
        return (get_synth_dataloader(batch_size, n_samples=4000, seed=0),
                get_synth_dataloader(batch_size, n_samples=512, seed=1))
    elif source == "real":
        from dataset import get_dataloader
        dl = get_dataloader(batch_size=batch_size)
        return dl, dl  # streaming-датасет: отдельной val-выборки нет
    raise ValueError(f"Неизвестный источник данных: {source}")


# ---------------------------------------------------------------------------
# Фазы
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=32):
    """Фаза тестов: MSE + метрики расстановки на свежих примерах."""
    model.eval()
    criterion = nn.MSELoss()
    losses, cent, cons = [], [], []
    samples = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        emb = batch["text_embedding"].to(device)
        target = batch["pixel_colors"].to(device)
        pred = model(emb)
        losses.append(criterion(pred, target).item())

        raw = pred[0].cpu().numpy()
        arranged = arrange_layers(raw)
        cent.append(centering_error(arranged))
        cons.append(color_consistency(arranged))
        if len(samples) < 6:
            cap = batch.get("caption", [f"sample_{i}"])
            samples.append((cap[0] if isinstance(cap, list) else str(cap), raw, arranged))

    model.train()
    metrics = {
        "val_mse": float(np.mean(losses)),
        "centering_err": float(np.nanmean(cent)),
        "color_consistency": float(np.mean(cons)),
    }
    return metrics, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synth", "real"], default="synth")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=0, help="макс. шагов за эпоху (0 = без лимита)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weights", default="pixel_model.pth")
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PixelTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_loader, val_loader = get_loaders(args.source, args.batch_size)
    total_steps = args.epochs * (args.steps or 250)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    os.makedirs("tests", exist_ok=True)

    if args.test_only:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        model.train()
        global_step = 0
        for epoch in range(args.epochs):
            running = 0.0
            for step, batch in enumerate(train_loader):
                if args.steps and step >= args.steps:
                    break
                emb = batch["text_embedding"].to(device)
                target = batch["pixel_colors"].to(device)

                optimizer.zero_grad()
                logits = model(emb)
                loss = criterion(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                running += loss.item()
                global_step += 1
                if global_step % 50 == 0:
                    print(f"[train] epoch {epoch} step {global_step} "
                          f"loss={running/50:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                    running = 0.0

            # мини-проверка между эпохами
            m, _ = evaluate(model, val_loader, device, max_batches=8)
            print(f"[valid] epoch {epoch}: val_mse={m['val_mse']:.4f} "
                  f"centering_err={m['centering_err']:.2f}")

        torch.save(model.state_dict(), args.weights)
        print(f"[save] веса: {args.weights}")

    # ============================ ФАЗА ТЕСТОВ ============================
    print("\n" + "=" * 60)
    print("FAZA TESTOV: оценка обученной модели на удержанных примерах")
    print("=" * 60)
    metrics, samples = evaluate(model, val_loader, device)
    print(f"  val MSE                 : {metrics['val_mse']:.4f}")
    print(f"  centering error (px)    : {metrics['centering_err']:.2f}  (меньше = лучше центровано)")
    print(f"  color consistency       : {metrics['color_consistency']:.4f}  (меньше = связнее спрайт)")

    for name, raw, arranged in samples:
        safe = name.replace(" ", "_")
        save_png(raw, f"tests/{safe}_raw.png")
        save_png(arranged, f"tests/{safe}_arranged.png")
        print(f"  тест '{name}': tests/{safe}_raw.png | tests/{safe}_arranged.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
