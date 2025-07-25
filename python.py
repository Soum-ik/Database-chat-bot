import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_deepseek import ChatDeepSeek

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Database Connection ---
# Direct database URL
db_uri = os.getenv("DB_URI")

try:
    # Connect to the database
    db = SQLDatabase.from_uri(db_uri)
    print(f"Successfully connected to database: {db_uri}")
except Exception as e:
    print(f"Error connecting to database: {e}")
    print("Please check your database configuration and ensure the database is running.")
    exit(1)

# --- Language Model ---
# Initialize the language model you want to use
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# --- SQL Agent ---
# Create the SQL agent with the database and language model
agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)

# --- Flask API Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Database Chatbot API is running',
        'timestamp': time.time()
    })

@app.route('/query', methods=['POST'])
def query_database():
    """Main endpoint for database queries"""
    try:
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question in request body',
                'example': {'question': 'What tables are available?'}
            }), 400
        
        question = data['question']
        if not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Start timing
        start_time = time.time()
        
        # Process the query
        response = agent_executor.invoke({"input": question})
        
        # End timing and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': response["output"],
            'duration_seconds': round(duration, 2),
            'timestamp': time.time()
        })
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time if 'start_time' in locals() else 0
        
        return jsonify({
            'success': False,
            'error': str(e),
            'duration_seconds': round(duration, 2),
            'timestamp': time.time()
        }), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Get information about the database and API"""
    try:
        return jsonify({
            'database_uri': db_uri.replace('FleetBloxDev', '***'),  # Hide password
            'model': 'deepseek-chat',
            'agent_type': 'tool-calling',
            'endpoints': {
                'GET /health': 'Health check',
                'POST /query': 'Query database with natural language',
                'GET /info': 'Get API information'
            },
            'example_request': {
                'url': '/query',
                'method': 'POST',
                'body': {'question': 'Show me all tables in the database'}
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run the Flask App ---
if __name__ == "__main__":
    print(f"\n🚀 Starting Database Chatbot API...")
    print(f"📊 Connected to database: {db_uri}")
    print(f"🤖 Using model: deepseek-chat")
    print(f"\n📡 API Endpoints:")
    print(f"   GET  /health - Health check")
    print(f"   POST /query  - Query database")
    print(f"   GET  /info   - API information")
    print(f"\n🌐 Starting server on http://localhost:5000")
    print(f"\n" + "="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)