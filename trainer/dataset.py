import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np

class PixelArtDataset(IterableDataset):
    def __init__(self, palette_size=16, max_samples=10000):
        # Загружаем потоком
        self.dataset = load_dataset("Angais/pixel-art-atlas-1.4m", split="train", streaming=True)
        self.palette_size = palette_size
        self.max_samples = max_samples
        
        # Собираем палитру (упрощенно: берем уникальные цвета из первых 1000 картинок)
        print("Собираем палитру из первых 1000 картинок...")
        self.palette = self._build_palette()
        print(f"Палитра собрана: {len(self.palette)} цветов")

    def _build_palette(self):
        colors = set()
        count = 0
        # ВАЖНО: Пересоздаем итератор, чтобы не тратить данные
        temp_dataset = load_dataset("Angais/pixel-art-atlas-1.4m", split="train", streaming=True)
        for example in temp_dataset:
            img = example["image"].convert("RGBA")
            # Берем уникальные цвета, игнорируя полностью прозрачные пиксели
            for pixel in img.getdata():
                if pixel[3] > 0: # Если не прозрачный
                    colors.add(pixel[:3]) # Только RGB
            count += 1
            if count >= 1000:
                break
        
        # Превращаем в список и обрезаем до нужного размера (или дополняем)
        palette_list = list(colors)[:self.palette_size]
        # Добавляем черный цвет как "прозрачность" на индекс 0 (или отдельный токен)
        # Для простоты: индекс 0 = прозрачность, индексы 1..N = цвета
        return [(0, 0, 0)] + palette_list 

    def _image_to_tokens(self, img: Image.Image):
        # Конвертируем картинку в токены на основе нашей палитры
        img = img.convert("RGBA")
        tokens = []
        for pixel in img.getdata():
            if pixel[3] == 0: # Прозрачный
                tokens.append(0)
            else:
                rgb = pixel[:3]
                # Ищем ближайший цвет в палитре (упрощенно - точное совпадение)
                try:
                    idx = self.palette.index(rgb)
                    tokens.append(idx)
                except ValueError:
                    # Если цвета нет в палитре, ставим прозрачный (или ближайший)
                    tokens.append(0) 
        return tokens

    def __iter__(self):
        count = 0
        for example in self.dataset:
            if count >= self.max_samples:
                break
                
            img = example["image"]
            caption = example["caption"]
            
            tokens = self._image_to_tokens(img)
            
            # Возвращаем тензоры
            # pixel_tokens: [256] (для 16x16)
            # text: [384] (заглушка, так как мы пока не подключили MiniLM)
            yield {
                "pixel_tokens": torch.tensor(tokens, dtype=torch.long),
                "text_embedding": torch.randn(384) # Заглушка
            }
            count += 1

def get_dataloader(batch_size=32):
    dataset = PixelArtDataset(max_samples=1000) # Для теста берем 1000
    return DataLoader(dataset, batch_size=batch_size)