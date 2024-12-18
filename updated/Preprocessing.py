import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from collections import Counter
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption = self.preprocess_caption(caption)
        return image, caption

    def preprocess_caption(self, caption):
        words = caption.lower().split()
        caption_idx = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        return torch.tensor(caption_idx)

def build_vocab(captions, min_freq=1):
    counter = Counter()
    for caption in captions:
        words = caption.lower().split()
        counter.update(words)

    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def collate_fn(batch,vocab):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # 将文本序列进行填充，使得每个批次中的文本长度一致
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>'])

    return images, captions