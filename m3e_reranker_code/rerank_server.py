from flask import Flask, request, jsonify
import time
import logging
from BCEmbedding import RerankerModel
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model once when the application starts
start_time = time.time()
model = RerankerModel(model_name_or_path="/app/models/bce-reranker-base_v1")
end_time = time.time()
logging.info(f"Model loaded in {end_time - start_time:.2f} seconds")

@app.route('/rerank', methods=['POST'])
def rerank():
    start_time = time.time()
    
    # Get data from the POST request
    data = request.get_json()
    query = data.get('query', '')
    slices = data.get('slices', [])
    
    if not query or not slices:
        return jsonify({"error": "Both 'query' and 'slices' are required."}), 400
    
    try:
        # Calculate scores or rerank slices
        rerank_results = model.rerank(query, slices)
        
        end_time = time.time()
        logging.info(f"Reranking completed in {end_time - start_time:.2f} seconds")
        
        return jsonify(rerank_results), 200
    except Exception as e:
        logging.error(f"Error during reranking: {e}")
        return jsonify({"error": "An error occurred during reranking."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9001, debug=False)

