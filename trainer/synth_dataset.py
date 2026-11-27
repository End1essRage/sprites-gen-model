"""Синтетический датасет для тренировки расстановки слоев.

Каждый пример: пиксельный спрайт 16x16 (спайка/зелье/гриб...) на фоне + текст-описание.
Встроенный "текстовый энкодер" — детерминированный хэш сумматорий слов, чтобы
тренировка работала офлайн без скачивания SentenceTransformer.

Формат совпадает с PixelArtDataset из dataset.py:
    {"pixel_colors": [256,3], "text_embedding": [384]}
"""

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

EMBED_DIM = 384


def fake_text_encoder(text: str) -> np.ndarray:
    """Детерминированная замена MiniLM: мешок слов -> хэш -> псевдо-эмбеддинг.

    Одинаковые слова дают одинаковые вклады, поэтому модель реально может
    выучить связь 'red potion' -> красные пиксели в центре.
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for word in text.lower().split():
        h = hashlib.md5(word.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        vec += rng.standard_normal(EMBED_DIM).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


_PALETTES = {
    "red": (0.85, 0.15, 0.15), "blue": (0.15, 0.3, 0.85), "green": (0.2, 0.75, 0.25),
    "purple": (0.6, 0.2, 0.8), "yellow": (0.95, 0.8, 0.1), "orange": (0.95, 0.5, 0.1),
}
_SHAPES = ["potion", "sword", "mushroom", "gem"]


def _make_sprite(shape: str, color_name: str, rng: random.Random):
    grid = np.zeros((16, 16, 3), dtype=np.float32)
    c = np.array(_PALETTES[color_name], dtype=np.float32)
    dark = c * 0.6
    light = np.clip(c * 1.4, 0, 1)

    cx, cy = rng.randint(5, 10), rng.randint(5, 10)  # спрайт НЕ по центру — есть что центрировать
    if shape == "potion":
        for dy in range(-3, 3):
            w = 2 if abs(dy) < 2 else 1
            for dx in range(-w, w + 1):
                grid[cy + dy, cx + dx] = c if dy > -2 else dark
        grid[cy - 4, cx] = light  # пробка
    elif shape == "sword":
        for i in range(6):
            grid[cy - 3 + i, cx + i // 2 - i // 2] = light if i < 5 else dark
        grid[cy + 3, cx - 1] = dark
        grid[cy + 3, cx + 1] = dark
        grid[cy + 4, cx] = dark
    elif shape == "mushroom":
        for dx in range(-3, 4):
            grid[cy - 2, cx + dx] = c
            grid[cy - 1, cx + dx] = light if dx % 2 else c
        for dy in range(0, 3):
            grid[cy + dy, cx] = dark
            grid[cy + dy, cx + 1] = dark
    else:  # gem
        for dy in range(-2, 3):
            w = 3 - abs(dy)
            for dx in range(-w, w + 1):
                grid[cy + dy, cx + dx] = light if (dy + dx) % 3 == 0 else c

    # шум: случайные «мусорные» пиксели вокруг — их расстановщик должен вычищать
    for _ in range(rng.randint(4, 10)):
        y, x = rng.randrange(16), rng.randrange(16)
        grid[y, x] = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
    return grid


class SynthPixelDataset(IterableDataset):
    def __init__(self, n_samples=2000, seed=0):
        self.n_samples = n_samples
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.n_samples):
            shape = rng.choice(_SHAPES)
            color = rng.choice(list(_PALETTES))
            adj = rng.choice(["small", "big", "glowing", "old"])
            caption = f"{color} {adj} {shape}"
            grid = _make_sprite(shape, color, rng)
            yield {
                "pixel_colors": torch.tensor(grid.reshape(-1, 3), dtype=torch.float32),
                "text_embedding": torch.tensor(fake_text_encoder(caption), dtype=torch.float32),
                "caption": caption,
            }


def get_synth_dataloader(batch_size=16, n_samples=2000, seed=0):
    return DataLoader(SynthPixelDataset(n_samples, seed), batch_size=batch_size)
