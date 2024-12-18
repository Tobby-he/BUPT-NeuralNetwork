import torch
from torch import nn

# 自定义 CNN 编码器类
class CustomCNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomCNNEncoder, self).__init__()
        # 卷积层和池化层用于提取图像特征
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()# ReLU 激活函数层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #采用最大池化，池化核大小为 2x2，步长为 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层将卷积层输出映射到指定维度
        self.fc = nn.Linear(128 * 28 * 28, out_channels)
        #根据后续训练以及对接再进行调整
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)#进行扁平化处理，将多维的特征图转换为一维向量
        x = self.fc(x)
        return x

# 自定义 GRU 解码器类
class CustomGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(CustomGRUDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)#将输入的词汇索引（由 input_ids 提供）转换为对应的低维向量表示
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)#批次维度会放在最前面
        self.fc = nn.Linear(hidden_size, vocab_size)#将 GRU 输出的隐藏状态向量转换为与词汇表大小相同维度的向量，以便后续根据这个向量来预测下一个可能的词汇。
        # 添加一个线性层来调整 encoder_output 的维度
        self.hidden_transform = nn.Linear(input_size, hidden_size)  

    def forward(self, encoder_output, input_ids):
        embedded = self.embedding(input_ids)
        # 使用线性层转换 encoder_output 的维度
        hidden = self.hidden_transform(encoder_output).unsqueeze(0)  #在第 0 维（批次维度）上增加一个维度，因为 GRU 期望的隐藏状态输入形状是（1、批次大小、隐藏状态维度）
        output, _ = self.gru(embedded, hidden)#output 是 GRU 在每个时间步的输出，不关心 GRU 内部的隐藏状态更新过程中的中间状态
        output = self.fc(output)
        return output

# CNN + GRU 模型架构类
class CNNGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CNNGRU, self).__init__()
        # 初始化 CNN 编码器
        self.cnn = CustomCNNEncoder(3, embedding_dim)
        # 初始化 GRU 解码器
        self.decoder = CustomGRUDecoder(embedding_dim, hidden_dim, vocab_size)

    def forward(self, images, input_ids):
        # 提取图像特征
        image_features = self.cnn(images)
        # 生成文本描述
        output = self.decoder(image_features, input_ids)
        return output
    
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__() #super调用父类（nn.module）的init函数进行初始化
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" #断言处理d_model，能够被num_heads整除，后续要将d_model均匀分配到多个头（heads）上。
        self.d_model = d_model #模型维度
        self.num_heads = num_heads #头的数量
        self.d_k = d_model // num_heads #每个头的维度，舍去余数部分

        # 定义线性层用于计算Q、K、V以及最终输出
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    # 实现缩放点积注意力机制
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5) # 计算注意力分数，矩阵乘法将K,Q相乘，除以sqrt(d_k)进行缩放，这是为了防止注意力分数过大导致梯度消失或爆炸等问题
        if mask is not None: #若有mask，则将mask中为 0 的位置对应的attn_scores填充为一个很小的值（-1e9），这样在后续进行softmax操作时，这些位置的概率就会趋近于 0，从而实现对某些位置的注意力屏蔽。
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) #通过 masked_fill 方法对 attn_scores 进行处理。将 mask 中值为 0 的位置对应的 attn_scores 中的值填充为一个很小的值（-1e9）。
        attn_probs = torch.softmax(attn_scores, dim=-1)#计算得到注意力概率
        output = torch.matmul(attn_probs, V)
        return output
    # 进行维度变换
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size() #获取维度信息
        # 使用 view 方法对 x 进行维度重塑，将d_model 拆分成 self.num_heads 个 self.d_k 维度，将第 2 维和第 3 维交换。
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 更便于后续在每个头上分别进行注意力计算等操作。
    # 将多个头结果进行合并，使用 contiguous 方法确保张量在内存中的存储是连续的
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    #前向传播
    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(attn_output)
        output = self.W_o(output)
        return output
import math
#位置编码
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)#创建形状为 (max_len, d_model) 的全零张量 pe，用于存储计算得到的位置编码信息。max_len 为序列的最大长度，d_model 是模型的维度，也是位置编码要匹配的输入数据的维度。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #unsqueeze(1) 在第二个维度（维度索引为 1）上增加一个维度，将其形状变为 (max_len, 1)，为了方便后续与其他张量进行乘法运算。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#选取d_model 维度中的偶数索引位置的元素组成的张量。对这个张量的每个元素进行指数运算，在不同维度上生成不同频率的位置编码。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #让位置编码张量的维度顺序与后续输入数据的维度顺序相匹配，便于后续的加法操作。
        self.register_buffer('pe', pe) #将计算得到的位置编码张量 pe 注册为模型的缓冲区。
        #在模型保存和加载时，张量会自动被处理，它的值在训练过程中不会被更新，只是作为一个固定的位置编码信息添加到输入数据中。
    #self.pe[:x.size(0), :] 选取与x 长度匹配的位置编码部分（因为 x 的长度可能小于 max_len）
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
import torch.nn.functional as F
#Vit编码器
class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_size, num_layers, num_heads, mlp_dim):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size #输入图像的尺寸
        self.patch_size = patch_size #划分图像成 patches 的尺寸
        self.num_patches = (image_size // patch_size) ** 2 #将图像在水平和垂直方向分别划分
        self.hidden_size = hidden_size #隐藏层大小
        
        # 修改二维卷积层 patch embedding 以确保输出维度正确
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size, #通道数设置为 hidden_size
            kernel_size=patch_size,#卷积核大小和步长都设置为 patch_size
            stride=patch_size
        )
        #一个特殊的 patch，整个图像的一个整体表示，形状为 (1, 1, hidden_size) 的张量，通过 nn.Parameter 将其标记为模型的可学习参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))#给cls_token添加位置信息
        self.norm = nn.LayerNorm(hidden_size)#对输入进行归一化处理
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,#前馈过程中中间层的维度设置
            batch_first=True#批次维度会放在最前面
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Patch embedding
        x = self.patch_embedding(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        #print("After patch embedding and flattening:", x.shape)

     # 添加 cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #print("After adding cls token:", x.shape)

        # 添加位置编码
        x = x + self.position_embedding
        # Layer Norm
        x = self.norm(x)       

    # Transformer编码
        x = self.transformer_encoder(x)
        #print("Final output:", x.shape)
        return x
#transformers解码器架构
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)#将输入的词汇索引转换为对应的向量表示。
        self.position_encoding = PositionEncoding(d_model, max_len)#经过词嵌入后的目标序列添加位置信息
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,#前馈神经网络维度设置为模型维度4倍
            batch_first=True  # 添加这个参数
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        #将经过 TransformerDecoder 处理后的输出向量转换为与词汇表大小对应的维度
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
                 num_layers, num_heads, mlp_dim, decoder_num_layers, decoder_num_heads, decoder_max_len,max_len):
        super(ViTTransformerModel, self).__init__()
        
        # 确保编码器和解码器使用相同的 hidden_size
        self.vit_encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            hidden_size=hidden_size,  # 这个值应该与解码器的 d_model 相同
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
        )
        
        self.transformer_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=hidden_size,  # 使用相同的 hidden_size
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            max_len=max_len
        )

    def forward(self, images, captions):
        image_features = self.vit_encoder(images)
        output = self.transformer_decoder(captions, image_features)
        return output
