import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义线性层用于计算Q、K、V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(attn_output)
        output = self.W_o(output)
        return output
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_size, num_layers, num_heads, mlp_dim):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size
        
        # 修改 patch embedding 以确保输出维度正确
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, hidden_size, H', W']
        
        # 打印维度，帮助调试
        print(f"After patch embedding: {x.shape}")
        
        x = x.flatten(2)  # [B, hidden_size, N]
        print(f"After flatten: {x.shape}")
        
        x = x.transpose(1, 2)  # [B, N, hidden_size]
        print(f"After transpose: {x.shape}")
        
        # 添加 cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"After adding cls token: {x.shape}")
        
        # 添加位置编码
        x = x + self.position_embedding
        
        # Layer Norm
        x = self.norm(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        print(f"Final output: {x.shape}")
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(d_model, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True  # 添加这个参数
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt_embedding = self.token_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)
        
        # 使用 transformer_decoder 而不是直接使用层
        output = self.transformer_decoder(tgt_embedding, memory)
        output = self.fc(output)
        return output

# 修改模型参数
class ViTTransformerModel(nn.Module):
    def __init__(self, vocab_size, image_size, patch_size, num_classes, hidden_size, 
                 num_layers, num_heads, mlp_dim, decoder_num_layers, decoder_num_heads, decoder_max_len):
        super(ViTTransformerModel, self).__init__()
        
        # 确保编码器和解码器使用相同的 hidden_size
        self.vit_encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            hidden_size=hidden_size,  # 这个值应该与解码器的 d_model 相同
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim
        )
        
        self.transformer_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=hidden_size,  # 使用相同的 hidden_size
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            max_len=decoder_max_len
        )

    def forward(self, images, captions):
        image_features = self.vit_encoder(images)
        output = self.transformer_decoder(captions, image_features)
        return output
# 定义模型参数
batch_size = 2
image_size = 224
patch_size = 16
num_classes = 1000
hidden_size = 768
num_layers = 12
num_heads = 12
mlp_dim = 3072
vocab_size = 5000
decoder_num_layers = 6
decoder_num_heads = 8
decoder_max_len = 20
num_patches = (image_size // patch_size) ** 2 
# 创建模型实例
model = ViTTransformerModel(vocab_size, image_size, patch_size, num_classes, hidden_size, num_layers, num_heads, mlp_dim, decoder_num_layers, decoder_num_heads, decoder_max_len)

# 创建随机输入数据
images = torch.randn(batch_size, 3, image_size, image_size)
captions = torch.randint(0, vocab_size, (batch_size, decoder_max_len))

# 前向传播
output = model(images, captions)
print(output.shape)
    print(output_logits.shape) 
