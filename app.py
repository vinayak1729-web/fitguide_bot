

# from flask import Flask, render_template, request, jsonify, session
# from flask_cors import CORS
# from rag_engine import RAGEngine
# import secrets
# import pandas as pd
# import os


# app = Flask(__name__)
# app.secret_key = secrets.token_hex(16)
# CORS(app, supports_credentials=True)


# print("🚀 Initializing RAG Engine...")
# rag_engine = RAGEngine()


# # Load CSV once at startup for suggestions
# CSV_PATH = os.path.join('data', 'tbl_Speakers_Complete.csv')
# try:
#     df = pd.read_csv(CSV_PATH)
#     print(f"✅ CSV loaded successfully with {len(df)} records")
    
#     # PRE-BUILD SEARCH INDEX for fast autocomplete
#     print("🔍 Building search index...")
#     search_index = {
#         'makes': set(),
#         'models': set(),
#         'years': set(),
#         'full_entries': []
#     }
    
#     for _, row in df.iterrows():
#         make = str(row.get('Make', '')).strip()
#         model = str(row.get('Model', '')).strip()
#         year = str(row.get('Year', '')).strip()
        
#         if make and make.lower() != 'nan':
#             search_index['makes'].add(make)
#         if model and model.lower() != 'nan':
#             search_index['models'].add(model)
#         if year and year.lower() != 'nan':
#             search_index['years'].add(year)
        
#         # Store full entries for quick lookup
#         if make and model and make.lower() != 'nan' and model.lower() != 'nan':
#             entry = f"{make} {model}"
#             if year and year.lower() != 'nan':
#                 entry += f" {year}"
#             search_index['full_entries'].append(entry)
    
#     # Convert sets to sorted lists for faster searching
#     search_index['makes'] = sorted(list(search_index['makes']))
#     search_index['models'] = sorted(list(search_index['models']))
#     search_index['years'] = sorted(list(search_index['years']))
    
#     print(f"✅ Search index built: {len(search_index['makes'])} makes, {len(search_index['models'])} models, {len(search_index['full_entries'])} entries")
    
# except Exception as e:
#     print(f"⚠️ Warning: Could not load CSV for suggestions: {e}")
#     df = pd.DataFrame()
#     search_index = {'makes': [], 'models': [], 'years': [], 'full_entries': []}


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         data = request.get_json()
#         user_message = data.get('message', '').strip()
        
#         print(f"📨 Received message: '{user_message}'")
        
#         if not user_message:
#             return jsonify({'error': 'Empty message'}), 400
        
#         # Initialize session if not present
#         if 'stage' not in session:
#             session['stage'] = 'greet'
#             session['user_data'] = {}
#             print(f"🔧 Initialized session with stage: {session['stage']}")
        
#         print(f"📊 Current stage: {session.get('stage')}, User data: {session.get('user_data')}")
        
#         # Get answer from RAG engine
#         answer, stage_info = rag_engine.get_answer(
#             user_message, 
#             session.get('stage', 'greet'),
#             session.get('user_data', {})
#         )
        
#         print(f"✅ RAG response - Stage: {stage_info['current_stage']}, Buttons: {len(stage_info.get('buttons', []))}")
        
#         # Update session
#         session['stage'] = stage_info['current_stage']
#         session['user_data'] = stage_info['user_data']
#         session.modified = True  # Force session save
        
#         # Handle table data display - IMPROVED
#         if stage_info['current_stage'] == 'show_results' and 'table_data' in stage_info:
#             make = stage_info['user_data'].get('make', '')
#             model = stage_info['user_data'].get('model', '')
#             component = stage_info.get('component', '')
#             table_data = stage_info.get('table_data', [])
            
#             print(f"📋 Showing results for {make} {model} - {component}")
            
#             # Format as Markdown table instead of HTML
#             year_display = f" ({stage_info['user_data'].get('year', '')})" if stage_info['user_data'].get('year') else ""
            
#             answer = f"### 🎵 Speaker Information\n\n"
#             answer += f"**{make} {model}{year_display}** - {component}\n\n"
            
#             if table_data:
#                 # Create markdown table
#                 answer += "| Size | Location | Notes |\n"
#                 answer += "|------|----------|-------|\n"
#                 for row in table_data:
#                     size = row.get('Size', 'N/A')
#                     location = row.get('Location', 'N/A')
#                     notes = row.get('Notes', 'N/A')
#                     answer += f"| {size} | {location} | {notes} |\n"
#             else:
#                 answer += "*No speaker data available for this configuration.*\n"
        
#         response_data = {
#             'success': True,
#             'message': answer,
#             'buttons': stage_info.get('buttons', [])
#         }
        
#         print(f"📤 Sending response with {len(response_data['buttons'])} buttons")
#         return jsonify(response_data)
    
#     except Exception as e:
#         print(f"❌ Error in /chat: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500


# @app.route('/suggest', methods=['POST'])
# def suggest():
#     """OPTIMIZED autocomplete endpoint using pre-built index"""
#     try:
#         query = request.json.get('query', '').strip()
        
#         if not query or len(query) < 2:
#             return jsonify({'suggestions': []})
        
#         query_lower = query.lower()
#         results = []
        
#         # Fast search using pre-built index
#         # 1. Check makes
#         for make in search_index['makes']:
#             if query_lower in make.lower():
#                 results.append(make)
#                 if len(results) >= 8:
#                     break
        
#         # 2. Check models if we don't have enough results
#         if len(results) < 8:
#             for model in search_index['models']:
#                 if query_lower in model.lower() and model not in results:
#                     results.append(model)
#                     if len(results) >= 8:
#                         break
        
#         # 3. Check full entries for comprehensive matches
#         if len(results) < 8:
#             for entry in search_index['full_entries']:
#                 if query_lower in entry.lower() and entry not in results:
#                     results.append(entry)
#                     if len(results) >= 8:
#                         break
        
#         return jsonify({'suggestions': results[:8]})
        
#     except Exception as e:
#         print(f"Suggestion error: {e}")
#         return jsonify({'suggestions': []})


# @app.route('/reset', methods=['POST'])
# def reset():
#     """Reset session"""
#     session.clear()
#     return jsonify({'success': True, 'message': 'Session reset successfully'})


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


from flask import Flask, render_template, request, jsonify, session, send_from_directory, Response
from flask_cors import CORS
from rag_engine import RAGEngine
import secrets
import pandas as pd
import os


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# CORS configuration for embeddable widget
CORS(app, supports_credentials=True, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=['Content-Type'],
     methods=['GET', 'POST', 'OPTIONS'])


print("🚀 Initializing RAG Engine...")
rag_engine = RAGEngine()


# Load CSV once at startup for suggestions
CSV_PATH = os.path.join('data', 'tbl_Speakers_Complete.csv')
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV loaded successfully with {len(df)} records")
    
    # PRE-BUILD SEARCH INDEX for fast autocomplete
    print("🔍 Building search index...")
    search_index = {
        'makes': set(),
        'models': set(),
        'years': set(),
        'full_entries': []
    }
    
    for _, row in df.iterrows():
        make = str(row.get('Make', '')).strip()
        model = str(row.get('Model', '')).strip()
        year = str(row.get('Year', '')).strip()
        
        if make and make.lower() != 'nan':
            search_index['makes'].add(make)
        if model and model.lower() != 'nan':
            search_index['models'].add(model)
        if year and year.lower() != 'nan':
            search_index['years'].add(year)
        
        if make and model and make.lower() != 'nan' and model.lower() != 'nan':
            entry = f"{make} {model}"
            if year and year.lower() != 'nan':
                entry += f" {year}"
            search_index['full_entries'].append(entry)
    
    search_index['makes'] = sorted(list(search_index['makes']))
    search_index['models'] = sorted(list(search_index['models']))
    search_index['years'] = sorted(list(search_index['years']))
    
    print(f"✅ Search index built: {len(search_index['makes'])} makes, {len(search_index['models'])} models, {len(search_index['full_entries'])} entries")
    
except Exception as e:
    print(f"⚠️ Warning: Could not load CSV for suggestions: {e}")
    df = pd.DataFrame()
    search_index = {'makes': [], 'models': [], 'years': [], 'full_entries': []}


@app.route('/')
def index():
    return render_template('index.html')


# FIXED: Route to serve embed.js
@app.route('/embed.js')
def serve_embed_js():
    """Serve the embeddable widget JavaScript file"""
    embed_js_path = os.path.join(os.path.dirname(__file__), 'static', 'embed.js')
    
    print(f"📁 Looking for embed.js at: {embed_js_path}")
    
    # Check if file exists
    if not os.path.exists(embed_js_path):
        print(f"❌ File not found: {embed_js_path}")
        print(f"💡 Please create the file at: static/embed.js")
        return jsonify({
            'error': 'embed.js not found',
            'expected_path': 'static/embed.js',
            'current_dir': os.getcwd()
        }), 404
    
    try:
        with open(embed_js_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ Successfully loaded embed.js ({len(content)} bytes)")
        return Response(content, mimetype='application/javascript')
    
    except Exception as e:
        print(f"❌ Error reading embed.js: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        print(f"📨 Received message: '{user_message}'")
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        if 'stage' not in session:
            session['stage'] = 'greet'
            session['user_data'] = {}
            print(f"🔧 Initialized session with stage: {session['stage']}")
        
        print(f"📊 Current stage: {session.get('stage')}, User data: {session.get('user_data')}")
        
        answer, stage_info = rag_engine.get_answer(
            user_message, 
            session.get('stage', 'greet'),
            session.get('user_data', {})
        )
        
        print(f"✅ RAG response - Stage: {stage_info['current_stage']}, Buttons: {len(stage_info.get('buttons', []))}")
        
        session['stage'] = stage_info['current_stage']
        session['user_data'] = stage_info['user_data']
        session.modified = True
        
        # Format table as Markdown
        if stage_info['current_stage'] == 'show_results' and 'table_data' in stage_info:
            make = stage_info['user_data'].get('make', '')
            model = stage_info['user_data'].get('model', '')
            component = stage_info.get('component', '')
            table_data = stage_info.get('table_data', [])
            
            print(f"📋 Showing results for {make} {model} - {component}")
            
            year_display = f" ({stage_info['user_data'].get('year', '')})" if stage_info['user_data'].get('year') else ""
            
            answer = f"### 🎵 Speaker Information\n\n"
            answer += f"**{make} {model}{year_display}** - {component}\n\n"
            
            if table_data:
                answer += "| Size | Location | Notes |\n"
                answer += "|------|----------|-------|\n"
                for row in table_data:
                    size = row.get('Size', 'N/A')
                    location = row.get('Location', 'N/A')
                    notes = row.get('Notes', 'N/A')
                    answer += f"| {size} | {location} | {notes} |\n"
            else:
                answer += "*No speaker data available for this configuration.*\n"
        
        response_data = {
            'success': True,
            'message': answer,
            'buttons': stage_info.get('buttons', [])
        }
        
        print(f"📤 Sending response with {len(response_data['buttons'])} buttons")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ Error in /chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/suggest', methods=['POST'])
def suggest():
    """OPTIMIZED autocomplete endpoint using pre-built index"""
    try:
        query = request.json.get('query', '').strip()
        
        if not query or len(query) < 2:
            return jsonify({'suggestions': []})
        
        query_lower = query.lower()
        results = []
        
        for make in search_index['makes']:
            if query_lower in make.lower():
                results.append(make)
                if len(results) >= 8:
                    break
        
        if len(results) < 8:
            for model in search_index['models']:
                if query_lower in model.lower() and model not in results:
                    results.append(model)
                    if len(results) >= 8:
                        break
        
        if len(results) < 8:
            for entry in search_index['full_entries']:
                if query_lower in entry.lower() and entry not in results:
                    results.append(entry)
                    if len(results) >= 8:
                        break
        
        return jsonify({'suggestions': results[:8]})
        
    except Exception as e:
        print(f"Suggestion error: {e}")
        return jsonify({'suggestions': []})


@app.route('/reset', methods=['POST'])
def reset():
    """Reset session"""
    session.clear()
    return jsonify({'success': True, 'message': 'Session reset successfully'})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Car Audio Bot Server Starting...")
    print("="*50)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Static folder: {os.path.join(os.getcwd(), 'static')}")
    print(f"📄 embed.js should be at: {os.path.join(os.getcwd(), 'static', 'embed.js')}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
