import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from models import ViTTransformer, CNNGRU
from Preprocessing import CustomDataset, build_vocab, collate_fn
from metrics import rouge_l, cider_d
import time

image_dir = r"E:\Neuralwork\image2"
image2_dir=r"E:\Neuralwork\image2\image2"
caption_file = os.path.join(image_dir, "captions.npy")
save_model_dir_cnn_gru = r"E:\Neuralwork\image_captioning_project2\data\save_model_cnn_gru.pth"
save_model_dir_vit_transformer = r"E:\Neuralwork\image_captioning_project2\data\save_model_vit_transformer.pth"

# 加载图像路径和字幕数据
image_extensions = ['.jpg', '.png', '.jpeg']  # 定义有效图像扩展名
image_paths = [os.path.join(image2_dir, img) for img in os.listdir(image2_dir) if any(img.endswith(ext) for ext in image_extensions)]
captions = np.load(caption_file)
# 将 captions 中的 numpy.ndarray 元素转换为字符串
captions = [str(cap) if isinstance(cap, np.ndarray) else cap for cap in captions]

# 构建词汇表
vocab = build_vocab(captions)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.226, 0.224))
])
dataset = CustomDataset(image_paths, captions, vocab, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda batch: collate_fn(batch, vocab))

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ViT + Transformer 解码器模型
vit_transformer_model = ViTTransformer(len(vocab), embed_size=512, num_heads=8, num_layers=6, hidden_size=1024).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer_vit = optim.Adam(vit_transformer_model.parameters(), lr=0.0001)
# CNN + GRU 模型
cnn_gru_model = CNNGRU(len(vocab), embed_size=512, hidden_size=512, num_layers=3).to(device)
optimizer_cnn = optim.Adam(cnn_gru_model.parameters(), lr=0.0001)

# 训练模型函数
def train_model(model, optimizer, num_epochs=5):
    best_cider_d_score = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count=0
        for i, (images, captions)in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)
            caption_lengths = [len(cap) for cap in captions]
            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1], caption_lengths)
            targets = captions[:, 1:].contiguous().view(-1)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count+=1
            # 每 10 个批次显示一次进度
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
        
        # 每个 epoch 结束显示平均损失
        avg_loss = total_loss / batch_count
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")
        # 计算验证指标并保存最佳模型
        if (epoch + 1) % 5 == 0:  # 每5个epoch验证一次
            cider_d_score, rouge_l_score = evaluate_model(model)
            print(f"Validation Metrics - CIDEr-D: {cider_d_score:.4f}, ROUGE-L: {rouge_l_score:.4f}")
            if cider_d_score > best_cider_d_score:
                best_cider_d_score = cider_d_score
                if isinstance(model, ViTTransformer):
                    torch.save(model.state_dict(), save_model_dir_vit_transformer)
                elif isinstance(model, CNNGRU):
                    torch.save(model.state_dict(), save_model_dir_cnn_gru)
        end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}, CIDEr-D: {cider_d_score}, ROUGE-L: {rouge_l_score}, Time: {end_time - start_time}")

# 评估模型函数
def evaluate_model(model):
    model.eval()
    references = []
    candidates = []
    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            caption_lengths = [len(cap) for cap in captions]
            if isinstance(model, ViTTransformer):
                outputs = model(images, captions[:, :1], caption_lengths)
            elif isinstance(model, CNNGRU):
                outputs = model(images, captions[:, :1], caption_lengths)
            _, predicted = torch.max(outputs, dim=-1)
            for i in range(len(images)):
                reference = [vocab.get(word.item(), '<unk>') for word in captions[i]]
                candidate = [vocab.get(word.item(), '<unk>') for word in predicted[i]]
                references.append(' '.join([str(word) for word in reference]))
                candidates.append(' '.join([str(word) for word in candidate]))
    cider_d_score = cider_d(references, candidates)
    rouge_l_score = rouge_l(references[0], candidates[0])
    return cider_d_score, rouge_l_score

# 训练 ViT + Transformer 解码器模型
train_model(vit_transformer_model, optimizer_vit, num_epochs=10)
# 训练 CNN + GRU 模型
train_model(cnn_gru_model, optimizer_cnn, num_epochs=10)