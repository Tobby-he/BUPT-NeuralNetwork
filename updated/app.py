from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import os
import numpy as np
import torchvision.transforms as transforms
from models import ViTTransformer, CNNGRU
from collections import Counter
def build_vocab(captions, min_freq=1):
    counter = Counter()
    for caption in captions:
        words = caption.lower().split()
        counter.update(words)

    vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# 定义模型保存路径（与训练代码中的路径一致）
save_model_dir_cnn_gru = r"E:\Neuralwork\image_captioning_project2\data\save_model_cnn_gru.pth"#cnngru模型储存地址
save_model_dir_vit_transformer = r"E:\Neuralwork\image_captioning_project2\data\save_model_vit_transformer.pth"#vit模型储存地址

app = Flask(__name__)

# 加载词汇表（根据train.py中的逻辑构建词汇表，这里图片目录和caption文件路径与train.py中一致）
image_dir = r"E:\Neuralwork\image2"
caption_file = os.path.join(image_dir, "captions.npy")
captions = np.load(caption_file)
captions = [str(cap) if isinstance(cap, np.ndarray) else cap for cap in captions]
vocab = build_vocab(captions)
word_to_idx = vocab
idx_to_word = {v: k for k, v in vocab.items()}

# 加载模型
cnn_gru_model = CNNGRU(len(word_to_idx), embed_size=512, hidden_size=512, num_layers=3)
cnn_gru_model.load_state_dict(torch.load(save_model_dir_cnn_gru))
cnn_gru_model.eval()

vit_transformer_model = ViTTransformer(len(word_to_idx), embed_size=512, num_heads=8, num_layers=6, hidden_size=1024)
vit_transformer_model.load_state_dict(torch.load(save_model_dir_vit_transformer))
vit_transformer_model.eval()

# 图像预处理步骤（与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.226, 0.224))
])

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img_tensor = transform(img).unsqueeze(0)

    # 根据请求参数选择使用哪个模型生成描述（这里假设前端可以传递一个'model'参数，值为'cnn_gru'或'vit_transformer'）
    model_type = request.form.get('model', 'cnn_gru')
    if model_type == 'cnn_gru':
        with torch.no_grad():
            caption = cnn_gru_model.generate_caption(img_tensor, word_to_idx, idx_to_word)
    elif model_type == 'vit_transformer':
        with torch.no_grad():
            caption = vit_transformer_model.generate_caption(img_tensor, word_to_idx, idx_to_word)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    return jsonify({'description': caption})

if __name__ == '__main__':
    app.run(debug=True)