import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

text_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class PixelArtDataset(IterableDataset):
    def __init__(self, max_samples=10000):
        self.dataset = load_dataset("Angais/pixel-art-atlas-1.4m", split="train", streaming=True)
        self.max_samples = max_samples

    def _process_image(self, img):
        img = img.convert("RGBA")
        pixels = []
        for pixel in img.getdata():
            if pixel[3] == 0: # Прозрачный -> черный
                pixels.append([0.0, 0.0, 0.0])
            else:
                pixels.append([pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0])
        return pixels

    def __iter__(self):
        count = 0
        for example in self.dataset:
            if count >= self.max_samples:
                break
                
            img = example["image"]
            caption = example["caption"]
            
            pixels = self._process_image(img)
            text_emb = text_encoder.encode(caption)
            
            yield {
                # [256, 3] — 256 пикселей, у каждого 3 канала (R, G, B)
                "pixel_colors": torch.tensor(pixels, dtype=torch.float32), 
                "text_embedding": torch.tensor(text_emb, dtype=torch.float32),
            }
            count += 1

def get_dataloader(batch_size=32):
    dataset = PixelArtDataset(max_samples=10000) 
    return DataLoader(dataset, batch_size=batch_size)