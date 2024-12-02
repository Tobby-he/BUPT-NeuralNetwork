import torch
from torch import nn

# 自定义位置编码类
class CustomPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(CustomPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 自定义多头注意力机制类
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CustomMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output

# 自定义基于注意力机制的 Transformer 解码器层类
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, num_heads, dropout)
        self.multihead_attn = CustomMultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(nn.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# 自定义基于注意力机制的 Transformer 解码器类
class CustomTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = CustomPositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, encoder_hidden_states, mask=None):
        embedded = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        pos_encoded = self.positional_encoding(embedded)
        output = pos_encoded
        for layer in self.layers:
            output = layer(output, encoder_hidden_states, tgt_mask=mask)
        output = self.fc(output)
        return output

# 自定义 ViT 编码器类
class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = CustomPositionalEncoding(dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # 将图像分割为 patches 并嵌入
        patches = self.patch_embedding(x).flatten(2).transpose(1, 2)
        # 添加位置编码
        pos_encoded_patches = self.positional_encoding(patches)
        # 通过 Transformer 编码器层
        for block in self.blocks:
            pos_encoded_patches = block(pos_encoded_patches)
        # 归一化
        output = self.norm(pos_encoded_patches)
        # 平均池化（可选，根据任务需求）
        output = torch.mean(output, dim=1)
        # 全连接层分类（可根据任务调整，这里假设用于分类任务）
        output = self.fc(output)
        return output

# ViT + 自定义 Transformer 解码器模型架构类
class ViTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, image_size, patch_size, num_classes, mlp_dim):
        super(ViTTransformer, self).__init__()
        # 初始化 ViT 编码器
        self.vit = ViTEncoder(image_size, patch_size, num_classes, d_model, num_layers, num_heads, mlp_dim)
        # 初始化自定义 Transformer 解码器
        self.decoder = CustomTransformerDecoder(vocab_size, d_model, num_heads, num_layers)

    def forward(self, images, input_ids):
        # 提取图像特征
        image_features = self.vit(images)
        # 生成文本描述
        output = self.decoder(input_ids=input_ids, encoder_hidden_states=image_features)
        return output.logits

# 模型架构验证示例
if __name__ == "__main__":
    vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    image_size = 224
    patch_size = 16
    num_classes = 1000
    mlp_dim = 2048
    model = ViTTransformer(vocab_size, d_model, num_heads, num_layers, image_size, patch_size, num_classes, mlp_dim)
    # 生成随机图像数据和文本输入标识（示例数据，实际需根据数据集调整）
    random_images = torch.randn(2, 3, 224, 224)
    random_input_ids = torch.randint(0, vocab_size, (2, 50))
    output_logits = model(random_images, random_input_ids)
    print(output_logits.shape) 