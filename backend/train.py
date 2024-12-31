import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models import ViTTransformer, CNNGRU
from Preprocessing import build_vocab, collate_fn
from metrics import rouge_l, cider_d
import time
import nltk
import json

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading nltk punkt_tab...")
    nltk.download('punkt_tab')

# 设置参数
image_dir = r"E:\nn\BUPT-NeuralNetwork"  # 存放npy文件路径
image2_dir = r"E:\nn\BUPT-NeuralNetwork\image2"
caption_file = os.path.join(image_dir, "captions.npy")
processed_images_file = os.path.join(image2_dir, "processed_images.npy")
save_model_dir_cnn_gru = r"E:\nn\BUPT-NeuralNetwork\save_model_cnn_gru_demo.pth"#保存模型路径
save_model_dir_vit_transformer = r"E:\nn\BUPT-NeuralNetwork\save_model_vit_transformer_demo.pth"#保存模型路径

# 加载数据
captions = np.load(caption_file)
processed_images = np.load(processed_images_file)


print(f"加载的图像数据形状: {processed_images.shape}")
print(f"Caption数据的数量: {len(captions)}")

# 确保数据不为空
if len(processed_images) == 0:
    raise ValueError(f"在 {processed_images_file} 中没有找到任何图像数据")
if len(captions) == 0:
    raise ValueError(f"在 {caption_file} 中没有找到任何caption数据")

# 确保图像和描述数量匹配
assert len(processed_images) == len(captions), f"图像数量({len(processed_images)})与描述数量({len(captions)})不匹配！"

# 加载保存的词汇表
vocab_file = os.path.join(image_dir, "vocab.json")
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

print(f"\n词汇表大小: {len(vocab)}")
print("特殊标记:", {k: v for k, v in vocab.items() if k in ['<pad>', '<unk>', '<start>', '<end>']})

# 修改CustomDataset类以使用预处理后的numpy数组
class CustomDataset(Dataset):
    def __init__(self, images, captions, vocab):
        self.images = images
        self.captions = captions
        self.vocab = vocab

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        caption = self.captions[idx]
        return image, torch.tensor(caption)

# 创建数据集和dataloader
dataset = CustomDataset(processed_images, captions, vocab)
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, vocab)
)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ViT + Transformer 解码器模型
vit_transformer_model = ViTTransformer(len(vocab), embed_size=512, num_heads=8, num_layers=6, hidden_size=1024).to(
    device)
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
        batch_count = 0
        total_batches = len(dataloader)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        for i, (images, captions) in enumerate(dataloader):
            try:
                # 确保数据维度正确
                if images.dim() != 4:
                    print(f"Skip batch {i+1}: Invalid image dimension {images.shape}")
                    continue
                    
                images = images.to(device)
                captions = captions.to(device)
                
                # 获取caption长度
                caption_lengths = [len(cap) for cap in captions]
                if not caption_lengths:
                    print(f"Skip batch {i+1}: Empty captions")
                    continue
                
                optimizer.zero_grad()
                outputs = model(images, captions[:, :-1], caption_lengths)
                
                # 确保输出维度正确
                if outputs.size(-1) != len(vocab):
                    print(f"Skip batch {i+1}: Invalid output dimension {outputs.shape}")
                    continue
                
                targets = captions[:, 1:].contiguous().view(-1)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets)
                
                if torch.isnan(loss).any():
                    print(f"Skip batch {i+1}: NaN loss detected")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"\nError processing batch {i+1}: {str(e)}")
                continue
        
        # 防止除零错误
        if batch_count == 0:
            print("No valid batches in this epoch")
            continue
            
        avg_loss = total_loss / batch_count
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # 计算验证指标
        print("Calculating validation metrics...")
        cider_d_score, rouge_l_score = evaluate_model(model)
        print(f"Validation Metrics - CIDEr-D: {cider_d_score:.4f}, ROUGE-L: {rouge_l_score:.4f}")
        
        if cider_d_score > best_cider_d_score:
            best_cider_d_score = cider_d_score
            print("Saving best model...")
            if isinstance(model, ViTTransformer):
                torch.save(model.state_dict(), save_model_dir_vit_transformer)
            elif isinstance(model, CNNGRU):
                torch.save(model.state_dict(), save_model_dir_cnn_gru)

# 评估模型函数
def evaluate_model(model):
    model.eval()
    references = []
    candidates = []
    
    with torch.no_grad():
        for i, (images, captions) in enumerate(dataloader):
            try:
                images = images.to(device)
                batch_size = images.size(0)
                
                # 生成预测
                max_length = 20
                input_seq = torch.full((batch_size, 1), vocab['<start>'], dtype=torch.long).to(device)
                
                if isinstance(model, ViTTransformer):
                    # ViT模型的代码保持不变
                    outputs = model(images, input_seq, [1] * batch_size)
                    predicted_sequences = torch.argmax(outputs, dim=-1)
                    
                elif isinstance(model, CNNGRU):
                    # 修改CNN-GRU的预测部分
                    features = model.cnn(images)
                    features = features.reshape(batch_size, -1)
                    features = model.feature_fc(features)
                    
                    # 初始化隐藏状态
                    hidden = features.unsqueeze(0).repeat(model.num_layers, 1, 1)
                    
                    # 逐步生成序列
                    outputs = []
                    curr_input = input_seq
                    
                    for _ in range(max_length):
                        # 对当前输入进行嵌入
                        curr_embed = model.embed(curr_input)  # [batch_size, 1, embed_size]
                        
                        # 通过GRU生成下一个词
                        output, hidden = model.gru(curr_embed, hidden)  # output: [batch_size, 1, hidden_size]
                        logits = model.fc(output[:, -1])  # [batch_size, vocab_size]
                        next_word = torch.argmax(logits, dim=-1)  # [batch_size]
                        
                        outputs.append(next_word)
                        curr_input = next_word.unsqueeze(1)  # 准备下一步的输入
                        
                        # 检查是否所有序列都生成了结束标记
                        if (next_word == vocab['<end>']).all():
                            break
                    
                    predicted_sequences = torch.stack(outputs, dim=1)  # [batch_size, seq_len]
                
                # 转换为CPU和numpy
                predicted_sequences = predicted_sequences.cpu().numpy()
                captions = captions.cpu().numpy()
                
                # 处理预测结果
                for pred_seq, ref_seq in zip(predicted_sequences, captions):
                    # 处理预测序列
                    pred_words = []
                    for idx in pred_seq:
                        word = list(vocab.keys())[list(vocab.values()).index(idx)]
                        if word == '<end>':
                            break
                        if word not in ['<start>', '<pad>', '<unk>']:
                            pred_words.append(word)
                    
                    # 处理参考序列
                    ref_words = []
                    for idx in ref_seq:
                        word = list(vocab.keys())[list(vocab.values()).index(idx)]
                        if word == '<end>':
                            break
                        if word not in ['<start>', '<pad>', '<unk>']:
                            ref_words.append(word)
                    
                    candidates.append(' '.join(pred_words))
                    references.append(' '.join(ref_words))
                
            except Exception as e:
                print(f"Error in evaluation batch {i+1}: {str(e)}")
                continue
    
    # 计算评估指标
    if len(references) > 0 and len(candidates) > 0:
        try:
            cider_score = cider_d(references, candidates)
            rouge_score = sum(rouge_l(ref, cand) for ref, cand in zip(references, candidates)) / len(references)
            return cider_score, rouge_score
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return 0.0, 0.0
    return 0.0, 0.0

# 训练 ViT + Transformer 解码器模型
#train_model(vit_transformer_model, optimizer_vit, num_epochs=10)
# 训练 CNN + GRU 模型
train_model(cnn_gru_model, optimizer_cnn, num_epochs=10)