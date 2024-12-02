import torch
from torch import nn

# 自定义 CNN 编码器类
class CustomCNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomCNNEncoder, self).__init__()
        # 卷积层和池化层用于提取图像特征
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层将卷积层输出映射到指定维度
        self.fc = nn.Linear(128 * 28 * 28, out_channels)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 自定义 GRU 解码器类
class CustomGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(CustomGRUDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_output, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.gru(embedded, encoder_output.unsqueeze(0))
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

# 模型架构验证示例
if __name__ == "__main__":
    vocab_size = 5000
    embedding_dim = 256
    hidden_dim = 512
    model = CNNGRU(vocab_size, embedding_dim, hidden_dim)
    # 生成随机图像数据和文本输入标识（示例数据，实际需根据数据集调整）
    random_images = torch.randn(2, 3, 224, 224)
    random_input_ids = torch.randint(0, vocab_size, (2, 50))
    output_prediction = model(random_images, random_input_ids)
    print(output_prediction.shape) 