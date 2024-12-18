import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from collections import Counter
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# 图像文件夹路径和 captions.json 文件路径
image_dir = 'E:/images'
captions_file = 'E:/captions.json'

# 加载 captions.json 文件
with open(captions_file, 'r') as f:
    captions_data = json.load(f)  # 假设 captions 是一个字典，key 是图片文件名，value 是描述

# 设置随机种子保证结果可复现
random.seed(816)

# 获取所有图像文件名
all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and f in captions_data]

# 从图像文件中随机选取 5000 张
num_images_to_process = 5000
selected_image_files = random.sample(all_image_files, min(num_images_to_process, len(all_image_files)))

# 构造图像路径和描述列表
image_paths = [os.path.join(image_dir, image_file) for image_file in selected_image_files]
captions = [captions_data[image_file] for image_file in selected_image_files]

print(f"随机选择了 {len(image_paths)} 张图像及其描述。")

# 确保图像和描述数量一致
assert len(image_paths) == len(captions), "图像数量与描述数量不匹配！"

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None, save_dir=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        self.save_dir = save_dir

        # 如果指定了保存目录，创建该目录
        if self.save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像路径和对应的描述
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 图像预处理（裁剪、缩放、归一化）
        if self.transform:
            image = self.transform(image)

        # 如果指定了保存路径，则保存图像
        if self.save_dir:
            file_name = os.path.basename(image_path)
            save_path = os.path.join(self.save_dir, file_name)
            image_pil = transforms.ToPILImage()(image)  # 将Tensor转换为PIL.Image
            image_pil.save(save_path)  # 保存图像

        # 文本预处理（分词、编码）
        caption = self.preprocess_caption(caption)

        return image, caption

    def preprocess_caption(self, caption):
        # 分词（这里简单按空格分词）
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

# 创建保存目录
save_image_dir = './image2'  # 保存处理后的图像
save_caption_file = './captions.npy'  # 保存文本的词索引数据

# 构建词汇表
vocab = build_vocab(captions)

# 创建数据集
dataset = CustomDataset(image_paths, captions, vocab, transform=transform, save_dir=save_image_dir)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch))

# collate_fn: 处理不等长的文本序列
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # 将文本序列进行填充，使得每个批次中的文本长度一致
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>'])

    return images, captions

# 保存处理后的文本数据（词索引）
def save_captions(captions, file_path, max_len=None):
    if max_len is None:
        max_len = max(len(caption) for caption in captions)  # 计算最大长度

    # 填充序列到统一长度
    padded_captions = np.array([
        np.pad(caption, (0, max_len - len(caption)), constant_values=0) for caption in captions
    ])
    np.save(file_path, padded_captions)

# 保存图像和文本数据
# 保存图像和文本数据
def save_data(dataloader, save_image_dir, save_caption_file):
    processed_images = []
    processed_captions = []

    for images, captions in dataloader:
        for image in images:
            processed_images.append(image.cpu().numpy())  # 转到CPU再保存

        for caption in captions:
            processed_captions.append(caption.cpu().numpy())  # 转到CPU再保存

    # 保存图像数据为numpy文件
    np.save(os.path.join(save_image_dir, 'processed_images.npy'), np.array(processed_images))

    # 保存文本数据（词索引）并填充长度
    save_captions(processed_captions, save_caption_file)


# 使用保存方法
save_data(dataloader, save_image_dir, save_caption_file)

print("数据保存完毕！")
