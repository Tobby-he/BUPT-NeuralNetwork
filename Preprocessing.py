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

image_dir = '/home/u2022212035/jupyterlab/nn/image1'
captions_file = '/home/u2022212035/jupyterlab/nn/captions.json'

# 加载 captions.json 文件
with open(captions_file, 'r') as f:
    captions_data = json.load(f) 

# 前 1000 张
num_images_to_process = 1000

image_paths = []
captions = []

# 遍历图像文件夹，只选择存在描述的图像
for image_file in os.listdir(image_dir):
    if image_file.endswith('.jpg') and image_file in captions_data:
        image_paths.append(os.path.join(image_dir, image_file))
        captions.append(captions_data[image_file])
    
    # 如果处理的图像数量已达到限制，停止
    if len(image_paths) >= num_images_to_process:
        break

print(f"选择了 {len(image_paths)} 张图像及其描述。")

assert len(image_paths) == len(captions), "图像数量与描述数量不匹配！"

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None, save_dir=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 图像预处理（裁剪、缩放、归一化）
        if self.transform:
            image = self.transform(image)

        if self.save_dir:
            file_name = os.path.basename(image_path)
            save_path = os.path.join(self.save_dir, file_name)
            image_pil = transforms.ToPILImage()(image)  # 将Tensor转换为PIL.Image
            image_pil.save(save_path)  # 保存图像

        # 文本预处理（分词、编码）
        caption = self.preprocess_caption(caption)

        return image, caption

    def preprocess_caption(self, caption):
        # 按空格分词
        words = caption.lower().split()
        
        # 将词转换为词索引
        caption_idx = [self.vocab.get(word, self.vocab['<unk>']) for word in words]

        return torch.tensor(caption_idx)

# 构建词汇表
def build_vocab(captions, min_freq=1):
    counter = Counter()
    
    # 统计词频
    for caption in captions:
        words = caption.lower().split()
        counter.update(words)

    # 构建词汇表
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# 定义图像和文本的预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

save_image_dir = '/home/u2022212035/jupyterlab/nn/image2'
save_caption_file = '/home/u2022212035/jupyterlab/nn/captions.npy'

# 构建词汇表
vocab = build_vocab(captions)

# 创建数据集
dataset = CustomDataset(image_paths, captions, vocab, transform=transform, save_dir=save_image_dir)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch))

# collate_fn: 处理不等长的文本序列
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # 将文本序列进行填充，使得每个批次中的文本长度一致
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>'])

    return images, captions

# 保存处理后的文本数据（词索引）
def save_captions(captions, file_path):
    np.save(file_path, captions)

# 保存图像和文本数据
def save_data(dataloader, save_image_dir, save_caption_file):
    processed_images = []
    processed_captions = []

    # 处理每一个batch的数据
    for images, captions in dataloader:
        for image in images:
            processed_images.append(image.numpy())

        for caption in captions:
            processed_captions.append(caption.numpy()) 

    # 将图像数据保存为numpy文件
    np.save(os.path.join(save_image_dir, 'processed_images.npy'), np.array(processed_images))

    # 将文本数据保存为.npy文件
    save_captions(processed_captions, save_caption_file)

# 使用保存方法
save_data(dataloader, save_image_dir, save_caption_file)

print("数据保存完毕！")
