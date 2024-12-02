import torch
from torch import nn
from transformers import ViTModel, GPT2LMHeadModel
import torchvision.models as models

class CNNGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CNNGRU, self).__init__()
        # CNN 编码器
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_dim)
        # GRU 解码器
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, input_ids):
        # 提取图像特征
        image_features = self.cnn(images)
        # 嵌入文本输入
        embedded = self.embedding(input_ids)
        # GRU 处理
        output, _ = self.gru(embedded, None)
        # 预测输出
        output = self.fc(output)
        return output

class ViTTransformer(nn.Module):
    def __init__(self):
        super(ViTTransformer, self).__init__()
        # ViT 编码器
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # Transformer 解码器
        self.decoder = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, images, input_ids):
        # 提取图像特征
        image_features = self.vit(images).last_hidden_state
        # 生成描述
        output = self.decoder(input_ids=input_ids, encoder_hidden_states=image_features)
        return output.logits