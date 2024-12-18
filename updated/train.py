import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models import ViTTransformerModel, CNNGRU
from Preprocessing import CustomDataset, build_vocab,collate_fn
from metrics import rouge_l, cider_d
import matplotlib.pyplot as plt
import numpy as np

# 设置参数
image_dir = r"E:\Neuralwork\image2"#存放图片和caption.npy文件路径
caption_file = os.path.join(image_dir, "captions.npy")
save_model_dir_cnn_gru = r"E:\Neuralwork\image_captioning_project2\data\save_model_cnn_gru.pth"#保存模型路径
save_model_dir_vit_transformer = r"E:\Neuralwork\image_captioning_project2\data\save_model_vit_transformer.pth"#保存模型路径

# 加载图片路径和描述
image_filenames = [os.path.join("E:\\Neuralwork\\image2\\image2", f) for f in os.listdir("E:\\Neuralwork\\image2\\image2") if f.endswith('.jpg') or f.endswith('.png')]
captions = np.load(caption_file, allow_pickle=True)#E:\\Neuralwork\\image2\\image2为存放图片路径
# 构建词汇表
captions = [str(cap) if isinstance(cap, np.ndarray) else cap for cap in captions]
vocab = build_vocab(captions)
# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
dataset = CustomDataset(image_filenames, captions, vocab, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=lambda batch: collate_fn(batch, vocab))

# 统计 captions 中最长文本长度
max_caption_length = max(len(cap) for cap in captions)
# 适当增加余量，比如增加 10
max_len_for_position_encoding = max_caption_length + 10
# 初始化模型
vit_transformer_model = ViTTransformerModel(vocab_size=len(vocab), image_size=224, patch_size=16, num_classes=1000,
                            hidden_size=768, num_layers=12, num_heads=12, mlp_dim=3072,
                            decoder_num_layers=6, decoder_num_heads=8, decoder_max_len=20,max_len=max_len_for_position_encoding)
cnn_gru_model = CNNGRU(vocab_size=len(vocab), embedding_dim=256, hidden_dim=512)

# 优化器
optimizer_vit_transformer = optim.Adam(vit_transformer_model.parameters(), lr=1e-4)
optimizer_cnn_gru = optim.Adam(cnn_gru_model.parameters(), lr=1e-4)

# 存储损失值用于可视化
losses_vit_transformer = []
losses_cnn_gru = []
window_size = 5  # 移动平均窗口大小

# 训练函数
def train_model(model, optimizer, dataloader, device, epochs=50):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for images, captions in dataloader:
            # 确保图像和描述数量匹配
            num_images = len(images)
            captions = captions[:num_images]
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:].contiguous().view(-1)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / batch_count}")
        if isinstance(model, ViTTransformerModel):
            losses_vit_transformer.append(total_loss / batch_count)
        else:
            losses_cnn_gru.append(total_loss / batch_count)
    return model

# 评估函数
def evaluate_model(model, dataloader, vocab, device):
    model.eval()
    references = []
    candidates = []
    with torch.no_grad():
        for images, captions in dataloader:
            # 确保图像和描述数量匹配
            num_images = len(images)
            captions = captions[:num_images]
            images = images.to(device)
            captions = captions.tolist()
            for caption in captions:
                reference = [vocab[idx] for idx in caption]
                references.append(reference)
            outputs = model.generate_caption(images, vocab, device=device)
            candidates.extend(outputs)
    rouge_l_scores = [rouge_l(ref, cand) for ref, cand in zip(references, candidates)]
    cider_d_scores = [cider_d([ref], [cand]) for ref, cand in zip(references, candidates)]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    avg_cider_d = sum(cider_d_scores) / len(cider_d_scores)
    return avg_rouge_l, avg_cider_d

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_transformer_model = vit_transformer_model.to(device)
    cnn_gru_model = cnn_gru_model.to(device)

    # 用于记录当前最佳的 CIDEr-D 分数以及对应的模型状态
    best_cider_d_score_vit_transformer = float('-inf')
    best_cider_d_score_cnn_gru = float('-inf')
    best_vit_transformer_model_state = None
    best_cnn_gru_model_state = None

    # 训练轮次
    epochs = 50
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # 训练 ViTTransformer 模型
        trained_vit_transformer_model = train_model(vit_transformer_model, optimizer_vit_transformer, dataloader, device, epochs=1)
        # 评估 ViTTransformer 模型
        rouge_l_score_vit_transformer, cider_d_score_vit_transformer = evaluate_model(trained_vit_transformer_model, dataloader, vocab, device)
        print(f"ViTTransformer ROUGE-L score: {rouge_l_score_vit_transformer}")
        print(f"ViTTransformer CIDEr-D score: {cider_d_score_vit_transformer}")

        # 判断是否为当前最佳的 ViTTransformer 模型（根据 CIDEr-D 分数）
        if cider_d_score_vit_transformer > best_cider_d_score_vit_transformer:
            best_cider_d_score_vit_transformer = cider_d_score_vit_transformer
            best_vit_transformer_model_state = trained_vit_transformer_model.state_dict()

        # 训练 CNNGRU 模型
        trained_cnn_gru_model = train_model(cnn_gru_model, optimizer_cnn_gru, dataloader, device, epochs=1)
        # 评估 CNNGRU 模型
        rouge_l_score_cnn_gru, cider_d_score_cnn_gru = evaluate_model(trained_cnn_gru_model, dataloader, vocab, device)
        print(f"CNNGRU ROUGE-L score: {rouge_l_score_cnn_gru}")
        print(f"CNNGRU CIDEr-D score: {cider_d_score_cnn_gru}")

        # 判断是否为当前最佳的 CNNGRU 模型（根据 CIDEr-D 分数）
        if cider_d_score_cnn_gru > best_cider_d_score_cnn_gru:
            best_cider_d_score_cnn_gru = cider_d_score_cnn_gru
            best_cnn_gru_model_state = trained_cnn_gru_model.state_dict()

    # 保存最佳的 ViTTransformer 模型及评估指标
    torch.save({
       'model_state_dict': best_vit_transformer_model_state,
        'rouge_l_score': rouge_l_score_vit_transformer,
        'cider_d_score': best_cider_d_score_vit_transformer
    }, save_model_dir_vit_transformer)
    print(f"Best ViTTransformer ROUGE-L score: {rouge_l_score_vit_transformer}")
    print(f"Best ViTTransformer CIDEr-D score: {best_cider_d_score_vit_transformer}")

    # 保存最佳的 CNNGRU 模型及评估指标
    torch.save({
       'model_state_dict': best_cnn_gru_model_state,
        'rouge_l_score': rouge_l_score_cnn_gru,
        'cider_d_score': best_cider_d_score_cnn_gru
    }, save_model_dir_cnn_gru)
    print(f"Best CNNGRU ROUGE-L score: {rouge_l_score_cnn_gru}")
    print(f"Best CNNGRU CIDEr-D score: {best_cider_d_score_cnn_gru}")

    # 可视化损失曲线
    plt.plot(range(1, len(losses_vit_transformer) + 1), losses_vit_transformer, label='ViTTransformer Loss', marker='o')
    plt.plot(range(1, len(losses_cnn_gru) + 1), losses_cnn_gru, label='CNNGRU Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()
