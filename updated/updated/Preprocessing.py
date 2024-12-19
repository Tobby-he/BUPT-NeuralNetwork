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
        
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        # 确保 caption 被转换为整数序列
        if isinstance(caption, str):
            # 分词并转换为索引
            tokens = caption.split()
            caption = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # 确保所有的索引都是整数
        caption = [int(idx) for idx in caption]
        
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
    # 获取数据和标签
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    
    # 填充序列
    caption_lengths = [len(cap) for cap in captions]
    max_length = max(caption_lengths)
    padded_captions = torch.zeros(len(captions), max_length).long()  # 确保是长整型
    
    for i, cap in enumerate(captions):
        end = caption_lengths[i]
        # 确保 cap 是整数列表，而不是浮点数
        cap_indices = [int(idx) if isinstance(idx, (int, float)) else 0 for idx in cap[:end]]
        padded_captions[i, :end] = torch.tensor(cap_indices, dtype=torch.long)
    
    return images, padded_captions
def custom_collate_fn(batch, vocab):
    return collate_fn(batch, vocab)