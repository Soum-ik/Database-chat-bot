import os
import time
import ast
import re
import logging
from typing_extensions import TypedDict, Annotated
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Configure logging
logging.basicConfig(
    level=[logging.INFO, logging.DEBUG, logging.ERROR],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.debug("Loading environment variables...")
load_dotenv()
logger.debug("Environment variables loaded successfully")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Database Connection ---
logger.ERROR("Starting database connection process...")
# Direct database URL
db_uri = os.getenv("DB_URI")
logger.info(f"Database URI loaded: {db_uri[:20]}..." if db_uri else "No DB_URI found in environment")

try:
    # Connect to the database
    logger.info("Attempting to connect to database...")
    db = SQLDatabase.from_uri(db_uri)
    logger.info(f"Successfully connected to database: {db_uri}")
    logger.info(f"Database dialect: {db.dialect}")
    logger.info(f"Available tables: {db.get_usable_table_names()}")
except Exception as e:
    logger.error(f"Error connecting to database: {e}")
    logger.error("Please check your database configuration and ensure the database is running.")
    exit(1)

# --- Language Model ---
logger.info("Initializing language model...")
# Initialize the language model you want to use
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
logger.info(f"DeepSeek API key loaded: {'Yes' if deepseek_api_key else 'No'}")

try:
    llm = ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=deepseek_api_key,
    )
    logger.info("Language model initialized successfully: deepseek-reasoner")
except Exception as e:
    logger.error(f"Error initializing language model: {e}")
    exit(1)

# --- State Definition for LangGraph Chain ---
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    approval_required: bool
    approved: bool

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# --- Prompt Templates ---
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

# --- Helper Functions ---
def query_as_list(db, query):
    """Parse database query result into a list of unique elements."""
    logger.debug(f"Executing query_as_list with query: {query}")
    try:
        res = db.run(query)
        logger.debug(f"Raw query result: {res}")
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        result = list(set(res))
        logger.debug(f"Processed result: {len(result)} unique items")
        return result
    except Exception as e:
        logger.error(f"Error in query_as_list: {e}")
        return []

# --- Chain Functions ---
def write_query(state: State):
    """Generate SQL query to fetch information."""
    logger.info(f"Starting write_query for question: {state['question']}")
    
    try:
        prompt = query_prompt_template.invoke({
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        })
        logger.debug("Prompt created successfully")
        
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        logger.info(f"Generated SQL query: {result['query']}")
        
        # Check if query contains potentially dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        approval_required = any(keyword in result["query"].upper() for keyword in dangerous_keywords)
        logger.info(f"Query safety check - Approval required: {approval_required}")
        
        return {
            "query": result["query"],
            "approval_required": approval_required,
            "approved": not approval_required  # Auto-approve safe queries
        }
    except Exception as e:
        logger.error(f"Error in write_query: {e}")
        return {
            "query": "SELECT 1 as error",
            "approval_required": False,
            "approved": False
        }

def execute_query(state: State):
    """Execute SQL query if approved."""
    logger.info(f"Starting execute_query for query: {state.get('query', 'No query')}")
    
    if not state.get("approved", False):
        logger.warning("Query execution not approved")
        return {"result": "Query execution not approved"}
    
    try:
        logger.info("Executing SQL query...")
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        logger.info(f"Query executed successfully. Result length: {len(str(result))}")
        logger.debug(f"Query result: {result}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"result": f"Error executing query: {str(e)}"}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    logger.info("Starting generate_answer...")
    
    try:
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}"
        )
        logger.debug(f"Generated prompt for LLM")
        
        response = llm.invoke(prompt)
        logger.info(f"Answer generated successfully. Length: {len(response.content)}")
        logger.debug(f"Generated answer: {response.content[:200]}...")
        
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        return {"answer": f"Error generating answer: {str(e)}"}

# --- Initialize Enhanced Tools ---
logger.info("Initializing enhanced tools...")
try:
    # Create embeddings for proper noun retrieval
    logger.info("Creating embeddings for proper noun retrieval...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    logger.info("Embeddings created successfully")
    
    # Get proper nouns from database for retrieval tool
    logger.info("Extracting proper nouns from database...")
    try:
        logger.debug("Querying for artists...")
        artists = query_as_list(db, "SELECT DISTINCT Name FROM Artist LIMIT 100")
        logger.info(f"Found {len(artists)} artists")
        
        # Try to get albums if table exists
        try:
            logger.debug("Querying for albums...")
            albums = query_as_list(db, "SELECT DISTINCT Title FROM Album LIMIT 100")
            logger.info(f"Found {len(albums)} albums")
        except Exception as album_error:
            logger.warning(f"Could not retrieve albums: {album_error}")
            albums = []
        
        proper_nouns = artists + albums
        logger.info(f"Total proper nouns collected: {len(proper_nouns)}")
        
        if proper_nouns:
            logger.info("Creating vector store for proper nouns...")
            vector_store = InMemoryVectorStore(embeddings)
            vector_store.add_texts(proper_nouns)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            retriever_tool = create_retriever_tool(
                retriever,
                name="search_proper_nouns",
                description="Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search."
            )
            logger.info("Proper noun retriever tool created successfully")
        else:
            logger.warning("No proper nouns found, retriever tool disabled")
            retriever_tool = None
    except Exception as e:
        logger.error(f"Could not create proper noun retriever: {e}")
        retriever_tool = None
        
except Exception as e:
    logger.error(f"Could not initialize embeddings: {e}")
    retriever_tool = None

# --- SQL Agent with Enhanced Tools ---
logger.info("Creating SQL agent with enhanced tools...")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
logger.info(f"Base SQL tools loaded: {len(tools)} tools")

if retriever_tool:
    tools.append(retriever_tool)
    logger.info("Added proper noun retriever tool to toolkit")
else:
    logger.info("Proper noun retriever tool not available")

# Enhanced system message for agent
agent_system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.

If you need to filter on a proper noun like a Name, you must ALWAYS first look up
the filter value using the 'search_proper_nouns' tool! Do not try to
guess at the proper name - use this function to find similar ones.
""".format(dialect=db.dialect, top_k=5)

# Create the enhanced agent
logger.info("Creating enhanced ReAct agent...")
agent_executor = create_react_agent(llm, tools, prompt=agent_system_message)
logger.info("Enhanced agent created successfully")

# --- LangGraph Chain (Alternative to Agent) ---
logger.info("Building LangGraph chain...")
# Build the chain graph
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
logger.info("Chain graph structure created")

# Add memory for human-in-the-loop
memory = MemorySaver()
chain_graph = graph_builder.compile(checkpointer=memory)
logger.info("Chain graph compiled with memory support")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return jsonify({
        'status': 'healthy',
        'message': 'Database Chatbot API is running',
        'timestamp': time.time()
    })

@app.route('/query', methods=['POST'])
def query_database():
    """Main endpoint for database queries using enhanced agent"""
    logger.info("Query endpoint accessed")
    try:
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            logger.warning("Request missing question parameter")
            return jsonify({
                'error': 'Missing question in request body',
                'example': {'question': 'What tables are available?'}
            }), 400
        
        question = data['question']
        if not question.strip():
            logger.warning("Empty question received")
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Get mode preference (agent or chain)
        mode = data.get('mode', 'agent')  # Default to agent
        logger.info(f"Processing question: '{question}' in {mode} mode")
        
        # Start timing
        start_time = time.time()
        
        if mode == 'chain':
            # Use the LangGraph chain approach
            config = {"configurable": {"thread_id": str(int(time.time()))}}
            response_data = {}
            
            for step in chain_graph.stream(
                {"question": question}, 
                config, 
                stream_mode="updates"
            ):
                response_data.update(step)
            
            answer = response_data.get('answer', 'No answer generated')
            query_used = response_data.get('query', 'No query generated')
            
            # End timing and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            return jsonify({
                'success': True,
                'mode': 'chain',
                'question': question,
                'answer': answer,
                'query_used': query_used,
                'duration_seconds': round(duration, 2),
                'timestamp': time.time()
            })
        
        else:
            # Use the enhanced agent approach (default)
            response = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
            
            # End timing and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract the final answer from agent response
            final_message = response["messages"][-1]
            answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            return jsonify({
                'success': True,
                'mode': 'agent',
                'question': question,
                'answer': answer,
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

@app.route('/query/validate', methods=['POST'])
def validate_query():
    """Endpoint to validate SQL queries before execution"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query in request body',
                'example': {'query': 'SELECT * FROM table_name LIMIT 5'}
            }), 400
        
        query = data['query']
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        is_dangerous = any(keyword in query.upper() for keyword in dangerous_keywords)
        
        # Basic syntax validation (you could enhance this)
        validation_result = {
            'query': query,
            'is_dangerous': is_dangerous,
            'is_select_only': query.strip().upper().startswith('SELECT'),
            'estimated_safety': 'SAFE' if not is_dangerous and query.strip().upper().startswith('SELECT') else 'DANGEROUS'
        }
        
        return jsonify({
            'success': True,
            'validation': validation_result,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/search/entities', methods=['POST'])
def search_entities():
    """Endpoint to search for proper nouns/entities in the database"""
    try:
        if not retriever_tool:
            return jsonify({
                'error': 'Entity search not available - proper noun retrieval not initialized'
            }), 503
        
        data = request.get_json()
        if not data or 'search_term' not in data:
            return jsonify({
                'error': 'Missing search_term in request body',
                'example': {'search_term': 'alice chains'}
            }), 400
        
        search_term = data['search_term']
        if not search_term.strip():
            return jsonify({'error': 'Search term cannot be empty'}), 400
        
        # Use the retriever tool to find similar entities
        results = retriever_tool.invoke(search_term)
        
        return jsonify({
            'success': True,
            'search_term': search_term,
            'results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Get information about the database and API"""
    try:
        # Get basic database info
        table_names = db.get_usable_table_names()
        
        # Check if proper noun search is available
        entity_search_available = retriever_tool is not None
        
        return jsonify({
            'success': True,
            'database': {
                'dialect': str(db.dialect),
                'tables': table_names,
                'total_tables': len(table_names)
            },
            'api': {
                'version': '2.0',
                'endpoints': ['/health', '/query', '/query/validate', '/search/entities', '/info'],
                'description': 'Enhanced Database Chatbot API powered by LangChain with SQL Q&A best practices',
                'features': {
                    'structured_output': True,
                    'query_validation': True,
                    'entity_search': entity_search_available,
                    'dual_mode': 'agent and chain modes available'
                }
            },
            'usage': {
                'query_modes': ['agent', 'chain'],
                'query_example': {'question': 'How many customers are there?', 'mode': 'agent'},
                'validation_example': {'query': 'SELECT COUNT(*) FROM Customer'},
                'entity_search_example': {'search_term': 'alice chains'}
            },
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

# --- Run the Flask App ---
if __name__ == "__main__":
    print(f"\n🚀 Starting Enhanced Database Chatbot API...")
    print(f"📊 Connected to database: {db_uri}")
    print(f"🤖 Using model: gemini-2.5-flash")
    print(f"🔧 Features: Structured output, Query validation, Entity search")
    print(f"\n📡 API Endpoints:")
    print(f"   GET  /health          - Health check")
    print(f"   POST /query           - Query database (agent/chain modes)")
    print(f"   POST /query/validate  - Validate SQL queries")
    print(f"   POST /search/entities - Search for proper nouns")
    print(f"   GET  /info            - API information")
    print(f"\n🌐 Starting server on http://localhost:5000")
    print(f"\n" + "="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)