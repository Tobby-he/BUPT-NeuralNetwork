from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import os
import json
import numpy as np
import torchvision.transforms as transforms
from models import ViTTransformer, CNNGRU

# 定义模型保存路径
save_model_dir_cnn_gru = r"E:\nn\BUPT-NeuralNetwork\save_model_cnn_gru.pth"
save_model_dir_vit_transformer = r"E:\nn\BUPT-NeuralNetwork\save_model_vit_transformer_demo.pth"

app = Flask(__name__)
CORS(app)
# 直接加载保存的词汇表
vocab_file = r"E:\nn\BUPT-NeuralNetwork\vocab.json"
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
word_to_idx = vocab
idx_to_word = {v: k for k, v in vocab.items()}

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_gru_model = CNNGRU(len(word_to_idx), embed_size=512, hidden_size=512, num_layers=3).to(device)
cnn_gru_model.load_state_dict(torch.load(save_model_dir_cnn_gru))
cnn_gru_model.eval()

vit_transformer_model = ViTTransformer(len(word_to_idx), embed_size=512, num_heads=8, num_layers=6, hidden_size=1024).to(device)
vit_transformer_model.load_state_dict(torch.load(save_model_dir_vit_transformer))
vit_transformer_model.eval()

# 图像预处理步骤
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
    img_tensor = transform(img).unsqueeze(0).to(device)

    model_type = request.form.get('model', 'cnn_gru')
    if model_type == 'cnn_gru':
        with torch.no_grad():
            caption = cnn_gru_model.generate_caption(img_tensor, word_to_idx, idx_to_word)
    elif model_type == 'vit_transformer':
        with torch.no_grad():
            caption = cnn_gru_model.generate_caption(img_tensor, word_to_idx, idx_to_word)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    return jsonify({'description': caption})

if __name__ == '__main__':
    app.run(debug=True)