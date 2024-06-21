from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)
sentiment_model = pipeline('sentiment-analysis')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text')
    if text:
        result = sentiment_model(text)
        return jsonify(result)
    return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)