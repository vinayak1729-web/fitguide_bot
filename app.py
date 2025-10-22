from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-domain requests

# Initialize RAG engine at startup
print("🚀 Initializing RAG Engine...")
rag_engine = RAGEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get answer from RAG engine
        answer = rag_engine.get_answer(user_message)
        
        return jsonify({
            'success': True,
            'message': answer
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Serve the embed script
@app.route('/embed.js')
def embed_script():
    with open('static/embed.js', 'r') as f:
        js_content = f.read()
    return js_content, 200, {'Content-Type': 'application/javascript'}

# Serve widget CSS
@app.route('/widget.css')
def widget_css():
    with open('static/css/widget.css', 'r') as f:
        css_content = f.read()
    return css_content, 200, {'Content-Type': 'text/css'}

# Serve widget JS
@app.route('/widget.js')
def widget_js():
    with open('static/js/widget.js', 'r') as f:
        js_content = f.read()
    return js_content, 200, {'Content-Type': 'application/javascript'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
