from flask import Flask, render_template, request, jsonify
from testLLM import generate_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'})
    
    response = generate_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True) 