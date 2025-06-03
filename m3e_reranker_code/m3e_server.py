import os
import torch
from torch.cuda import device
from flask import Flask, request, render_template, jsonify, Blueprint
import time
import json
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    print("GPU is available")
    # 读取显卡ID环境变量，如果不存在，默认为0
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    # 设置要使用的显卡ID
    device(int(cuda_visible_devices))
else:
    print("No GPU detected")
    # 确保PyTorch使用CPU
    # 对于PyTorch，清空此环境变量可避免GPU使用
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  
    # 绕过PyTorch的GPU检查
    torch.cuda.is_available = lambda: False 

if os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  
    torch.cuda.is_available = lambda: False

model_path = os.getenv('MODEL_WEIGHTS_PATH', 'models/finetuned-model/m3e-base')

m3e_server_bp = Blueprint("m3e_server", __name__, template_folder='templates')

model = SentenceTransformer(model_path)

@m3e_server_bp.route('/m3e_sft_batch_query_embedding', methods=['POST'])
def process_file():
    sentences = request.get_json()
    process_start = time.time()
    embeddings = model.encode(sentences)

    process_duration = time.time() - process_start
    print(f"推理时常 {process_duration:.2f} s")
    data = []
    for sentence, embedding in zip(sentences, embeddings):
        embedding_list = embedding.tolist()
        item = {
            'sentence': sentence,
            'embedding': embedding_list
        }
        data.append(item)
    json_data = json.dumps(data)
    response = jsonify(data)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


app = Flask(__name__)
app.register_blueprint(m3e_server_bp)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8104, debug=False)
