from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from rag_engine import RAGEngine
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app, supports_credentials=True)

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
        
        if 'stage' not in session:
            session['stage'] = 'greet'
            session['user_data'] = {}
        
        answer, stage_info = rag_engine.get_answer(
            user_message, 
            session.get('stage', 'greet'),
            session.get('user_data', {})
        )
        
        # Update session
        session['stage'] = stage_info['current_stage']
        session['user_data'] = stage_info['user_data']
        
        # Handle table data display
        if stage_info['current_stage'] == 'show_results' and 'table_data' in stage_info:
            make = stage_info['user_data'].get('make', '')
            model = stage_info['user_data'].get('model', '')
            component = stage_info.get('component', '')
            table_data = stage_info.get('table_data', [])
            
            # Format table as HTML
            table_html = f"""
            <div class="result-container">
                <h3>🎵 Speaker Information</h3>
                <p><strong>{make} {model}</strong> - {component}</p>
                <table class="speaker-table">
                    <thead>
                        <tr>
                            <th>Size</th>
                            <th>Location</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            if table_data:
                for row in table_data:
                    table_html += f"""
                        <tr>
                            <td>{row.get('Size', 'N/A')}</td>
                            <td>{row.get('Location', 'N/A')}</td>
                            <td>{row.get('Notes', 'N/A')}</td>
                        </tr>
                    """
            else:
                table_html += """
                        <tr>
                            <td colspan="3">No speaker data available</td>
                        </tr>
                """
            
            table_html += """
                    </tbody>
                </table>
            </div>
            """
            
            answer = table_html
        
        return jsonify({
            'success': True,
            'message': answer,
            'buttons': stage_info.get('buttons', [])
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
