import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision.models import resnet50
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 视觉Transformer (ViT) + Transformer解码器模型架构
class ViTTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_layers=6, hidden_size=1024, max_seq_length=196):
        super().__init__()
        # 使用 google/vit-base-patch16-224-in21k 预训练的 ViT 模型
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 添加一个线性层来调整 ViT 输出维度 (768 -> embed_size)
        self.feature_adapter = nn.Linear(768, embed_size)
        
        # 文本嵌入层
        self.text_embedding = nn.Embedding(vocab_size, embed_size)
        
        # 位置编码
        self.max_seq_length = max_seq_length
        self.position_embedding = nn.Embedding(max_seq_length, embed_size)
        
        # Transformer 解码器层
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, hidden_size), 
            num_layers
        )
        
        # 输出层
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions, caption_lengths):
        batch_size = images.size(0)
        
        # 获取 ViT 输出并调整维度
        vit_output = self.vit(images)
        image_features = vit_output.last_hidden_state  # [batch_size, 197, 768]
        image_features = self.feature_adapter(image_features)  # [batch_size, 197, embed_size]
        
        # 确保位置索引不超过最大长度
        seq_length = image_features.size(1)
        positions = torch.arange(0, min(seq_length, self.max_seq_length), 
                               device=images.device).unsqueeze(0).expand(batch_size, -1)
        
        # 截断并添加位置编码
        image_features = image_features[:, :self.max_seq_length, :]
        image_features = image_features + self.position_embedding(positions)
        
        # 调整维度顺序以适应 Transformer 解码器
        image_features = image_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
        
        # 处理文本输入：确保是长整型并添加文本嵌入
        captions = captions.long()  # 转换为长整型
        captions = self.text_embedding(captions)  # [batch_size, seq_len, embed_size]
        captions = captions.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
        
        # 生成注意力掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            captions.size(0), device=captions.device)
        
        # 使用 Transformer 解码器
        output = self.transformer_decoder(captions, image_features, tgt_mask=tgt_mask)
        
        # 预测下一个单词
        output = self.fc(output)
        return output
    def generate_caption(self, image, word_to_idx, idx_to_word, max_length=50):
        with torch.no_grad():
            # 使用ViT提取图像特征
            vit_output = self.vit(image)
            image_features = vit_output.last_hidden_state[:, 0, :]  # [batch_size, 768]
            image_features = self.feature_adapter(image_features)  # [batch_size, embed_size]

            # 初始化隐藏状态（对于Transformer解码器，不需要显式的隐藏状态初始化，这里为了统一结构，设为None）
            hidden = None

            # 初始输入为起始标记
            input_word = torch.tensor([word_to_idx['<start>']], dtype=torch.long).to(image.device)
            caption = []

            for _ in range(max_length):
                # 嵌入输入单词
                embedded_word = self.text_embedding(input_word).unsqueeze(1)  # [1, 1, embed_size]
                # 通过Transformer解码器计算输出（这里与GRU不同，不需要传入隐藏状态）
                output = self.transformer_decoder(embedded_word, image_features.unsqueeze(0), tgt_mask=None)
                # 预测下一个单词
                output = self.fc(output.squeeze(0))
                _, predicted_word_idx = output.max(1)
                predicted_word = idx_to_word[predicted_word_idx.item()]
                if predicted_word == '<end>':
                    break
                caption.append(predicted_word)
                input_word = predicted_word_idx

            return''.join(caption)
# CNN + GRU 模型架构
class CNNGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(CNNGRU, self).__init__()
        # 使用预训练的 ResNet50 作为 CNN 提取图像特征
        self.cnn = resnet50(pretrained=True)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 调整 CNN 输出特征维度
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)
        # 文本嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        # GRU 层
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        # 线性层将 GRU 输出映射到词汇表大小
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions, caption_lengths):
        # 获取图像特征
        image_features = self.cnn(images)
        image_features = image_features.unsqueeze(1)
        # 嵌入文本
        embedded_captions = self.embed(captions)
        # 打包文本序列以处理不同长度
        packed_captions = pack_padded_sequence(embedded_captions, caption_lengths, batch_first=True, enforce_sorted=False)
        # 通过 GRU 层
        output, _ = self.gru(packed_captions, image_features)
        # 解包文本序列
        output, _ = pad_packed_sequence(output, batch_first=True)
        # 预测下一个单词
        output = self.fc(output)
        return output
    def generate_caption(self, image, word_to_idx, idx_to_word, max_length=50):
        with torch.no_grad():
            # 使用CNN提取图像特征
            image_features = self.cnn(image)
            image_features = image_features.view(image_features.size(0), -1)
            image_features = self.cnn.fc(image_features)

            # 初始化隐藏状态
            hidden = torch.zeros(self.gru.num_layers, 1, self.gru.hidden_size).to(image.device)

            # 初始输入为起始标记
            input_word = torch.tensor([word_to_idx['<start>']], dtype=torch.long).to(image.device)
            caption = []

            for _ in range(max_length):
                # 嵌入输入单词
                embedded_word = self.embed(input_word).unsqueeze(1)
                # 通过GRU计算隐藏状态
                output, hidden = self.gru(embedded_word, hidden)
                # 预测下一个单词
                output = self.fc(output.squeeze(1))
                _, predicted_word_idx = output.max(1)
                predicted_word = idx_to_word[predicted_word_idx.item()]
                if predicted_word == '<end>':
                    break
                caption.append(predicted_word)
                input_word = predicted_word_idx

            return''.join(caption)
