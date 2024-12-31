import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision.models import resnet50
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

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

        # 生成��掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            captions.size(0), device=captions.device)

        # 使用 Transformer 解码器
        output = self.transformer_decoder(captions, image_features, tgt_mask=tgt_mask)

        # 预测下一个单词
        output = self.fc(output)
        return output
    def generate_caption(self, image, word_to_idx, idx_to_word, max_length=50, temperature=1.0):
      with torch.no_grad():
        # 提取ViT特征
        vit_output = self.vit(image)
        image_features = vit_output.last_hidden_state  # [batch_size, 197, 768]
        image_features = self.feature_adapter(image_features)  # [batch_size, 197, embed_size]
        image_features = image_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]

        # 初始化生成序列
        input_word_idx = word_to_idx['<start>']
        generated_sequence = [input_word_idx]
        caption = []

        for _ in range(max_length):
            # 嵌入输入单词
            input_tensor = torch.tensor([generated_sequence], dtype=torch.long).to(image.device)  # [1, seq_len]
            embedded_sequence = self.text_embedding(input_tensor).permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded_sequence.size(0)).to(image.device)

            # 解码
            output = self.transformer_decoder(embedded_sequence, image_features, tgt_mask=tgt_mask)
            output = self.fc(output[-1])  # 取最后一个时间步
            output = F.softmax(output / temperature, dim=-1)

            # 采样下一个单词
            predicted_word_idx = torch.multinomial(output, num_samples=1).item()
            predicted_word = idx_to_word[predicted_word_idx]

            if predicted_word == '<end>':
                break

            caption.append(predicted_word)
            generated_sequence.append(predicted_word_idx)

        return ' '.join(caption)



# CNN + GRU 模型架构
class CNNGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(CNNGRU, self).__init__()
        self.num_layers = num_layers  # 保存层数
        # 使用预训练的 ResNet50 作为 CNN 提取图像特征
        resnet = resnet50(pretrained=True)
        # 移除最后的全连接层
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        
        # 添加一个新的全连接层来调整特征维度
        self.feature_fc = nn.Linear(2048, hidden_size)  # ResNet50的输出是2048维
        
        # 文本嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # GRU 层
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, 
                         dropout=dropout if num_layers > 1 else 0, 
                         batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, images, captions, caption_lengths):
        batch_size = images.size(0)
        
        try:
            # 获取图像特征，并打印维度信息
            features = self.cnn(images)  # [batch_size, 2048, 1, 1]
            
            # 移除最后两个维度
            features = features.reshape(batch_size, -1)  # [batch_size, 2048]
            
            # 通过特征适配层
            features = self.feature_fc(features)  # [batch_size, hidden_size]
            
            # 准备GRU的隐藏状态
            hidden = features.unsqueeze(0)  # [1, batch_size, hidden_size]
            hidden = hidden.repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_size]
            
            # 处理文本输入
            embeddings = self.dropout(self.embed(captions))  # [batch_size, seq_len, embed_size]
            
            # 修正caption_lengths：确保不超过实际序列长度
            max_seq_len = captions.size(1)
            caption_lengths = [min(length, max_seq_len) for length in caption_lengths]
            
            # 通过GRU
            packed = pack_padded_sequence(embeddings, caption_lengths, 
                                        batch_first=True, enforce_sorted=False)
            outputs, _ = self.gru(packed, hidden)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            
            # 通过输出层
            outputs = self.fc(outputs)  # [batch_size, seq_len, vocab_size]
            
            return outputs
            
        except Exception as e:
            print(f"\nError in CNNGRU forward pass:")
            print(f"Input images shape: {images.shape}")
            print(f"Input captions shape: {captions.shape}")
            print(f"Original caption lengths: {caption_lengths}")
            print(f"Max sequence length: {captions.size(1)}")
            raise e
    def generate_caption(self, image, word_to_idx, idx_to_word, max_length=50):
        with torch.no_grad():
            # 使用CNN提取图像特征
            features = self.cnn(image)  # [batch_size, 2048, 1, 1]
            features = features.reshape(features.size(0), -1)  # [batch_size, 2048]
            features = self.feature_fc(features)  # [batch_size, hidden_size]
            
            # 初始化隐藏状态
            hidden = features.unsqueeze(0)  # [1, batch_size, hidden_size]
            hidden = hidden.repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_size]
            
            # 初始输入为起始标记
            input_word = torch.tensor([word_to_idx['<start>']], dtype=torch.long).to(image.device)
            caption = []
            
            for _ in range(max_length):
                # 嵌入输入单词
                embedded_word = self.embed(input_word).unsqueeze(1)  # [1, 1, embed_size]
                # 通过GRU计算输出
                output, hidden = self.gru(embedded_word, hidden)  # output: [1, 1, hidden_size]
                # 预测下一个单词
                output = self.fc(output.squeeze(1))  # [1, vocab_size]
                _, predicted_word_idx = output.max(1)
                predicted_word = idx_to_word[predicted_word_idx.item()]
                
                if predicted_word == '<end>':
                    break
                caption.append(predicted_word)
                input_word = predicted_word_idx
            
            return ' '.join(caption)  # 添加空格分隔单词