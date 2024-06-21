# playing with pytorch and REST api
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/add', methods=['POST'])
def add_tensors():
    data = request.get_json()
    tensor_a = torch.tensor(data['tensor_a'])
    tensor_b = torch.tensor(data['tensor_b'])
    result = torch.add(tensor_a, tensor_b)
    return jsonify(result.tolist())

@app.route('/multiply', methods=['POST'])
def multiply_tensors():
    data = request.get_json()
    tensor_a = torch.tensor(data['tensor_a'])
    tensor_b = torch.tensor(data['tensor_b'])
    result = torch.mul(tensor_a, tensor_b)
    return jsonify(result.tolist())

@app.route('/relu', methods=['POST'])
def relu_activation():
    data = request.get_json()
    tensor = torch.tensor(data['tensor'])
    result = torch.nn.functional.relu(tensor)
    return jsonify(result.tolist())

@app.route('/matrix_multiply', methods=['POST'])
def matrix_multiply():
    data = request.get_json()
    matrix_a = torch.tensor(data['matrix_a'])
    matrix_b = torch.tensor(data['matrix_b'])
    result = torch.mm(matrix_a, matrix_b)
    return jsonify(result.tolist())

if __name__ == '__main__':
    app.run(debug=True)