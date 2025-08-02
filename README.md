# 🤖 Database Chatbot API

A powerful Flask-based REST API that allows you to query your PostgreSQL database using natural language. Built with LangChain and DeepSeek AI, this chatbot can understand complex database questions and provide intelligent responses.

## ✨ Features

- 🗣️ **Natural Language Queries**: Ask questions about your database in plain English
- ⚡ **RESTful API**: Clean HTTP endpoints for easy integration
- 🌐 **Web Interface**: Beautiful HTML interface for interactive querying
- ⏱️ **Performance Monitoring**: Built-in timing for all database operations
- 🔒 **Environment Variables**: Secure configuration management
- 📊 **Real-time Status**: Live API health monitoring
- 🎨 **Modern UI**: Responsive design with gradient themes

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL database
- DeepSeek API key

### Installation

1. **Clone or download the project files**
   ```bash
   cd c:\Users\Acer\OneDrive\Desktop\Chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Edit the `.env` file with your credentials:
   ```env
   # Database Configuration
   DB_URI="postgresql://username:password@host:port/database"
   
   # DeepSeek API Configuration
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

4. **Start the API server**
   ```bash
   python python.py
   ```

5. **Open the web interface**
   
   Double-click `index.html` or open it in your browser:
   ```
   file:///c:/Users/Acer/OneDrive/Desktop/Chatbot/index.html
   ```

## 📡 API Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "message": "Database Chatbot API is running",
  "timestamp": 1643123456.789
}
```

### Query Database
```http
POST /query
Content-Type: application/json

{
  "question": "What tables are available in the database?"
}
```

**Response:**
```json
{
  "success": true,
  "question": "What tables are available in the database?",
  "answer": "The database contains the following tables: users, orders, products...",
  "duration_seconds": 2.34,
  "timestamp": 1643123456.789
}
```

### API Information
```http
GET /info
```
**Response:**
```json
{
  "database_uri": "postgresql://username:***@host:port/database",
  "model": "deepseek-chat",
  "agent_type": "tool-calling",
  "endpoints": {
    "GET /health": "Health check",
    "POST /query": "Query database with natural language",
    "GET /info": "Get API information"
  }
}
```

## 💻 Web Interface

The included HTML interface provides:

- **🎯 Interactive Query Input**: Type questions or use example buttons
- **📊 Real-time Status**: See if your API is online/offline
- **⏱️ Performance Metrics**: View response times for each query
- **📱 Responsive Design**: Works on desktop and mobile
- **🔄 Auto-refresh**: Automatic API status monitoring

### Example Questions

- "What tables are available in the database?"
- "How many records are in each table?"
- "Show me the structure of the users table"
- "What are the column names in the orders table?"
- "Find all inactive cars in the system"
- "Show me users from a specific country"

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_URI` | PostgreSQL connection string | `postgresql://user:pass@host:port/db` |
| `DEEPSEEK_API_KEY` | Your DeepSeek API key | `sk-xxxxxxxxxxxxxxxx` |

### Model Configuration

The chatbot uses the DeepSeek Chat model with the following settings:
- **Model**: `deepseek-chat`
- **Temperature**: `0` (deterministic responses)
- **Agent Type**: `tool-calling`
- **Max Retries**: `2`

## 📁 Project Structure

```
Chatbot/
├── python.py          # Main Flask API server
├── index.html         # Web interface
├── requirements.txt   # Python dependencies
├── .env              # Environment variables
└── README.md         # This file
```

## 🔧 Dependencies

- **Flask 3.0.0**: Web framework
- **Flask-CORS 4.0.0**: Cross-origin resource sharing
- **LangChain Community**: Database utilities and SQL agent
- **LangChain DeepSeek**: AI model integration
- **psycopg2-binary**: PostgreSQL adapter
- **python-dotenv**: Environment variable management

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure the Flask server is running on port 5000
   - Check if the database connection is successful

2. **Database Connection Error**
   - Verify your `DB_URI` in the `.env` file
   - Ensure the PostgreSQL server is accessible

3. **DeepSeek API Error**
   - Check your `DEEPSEEK_API_KEY` is valid
   - Verify you have API credits remaining

4. **CORS Issues**
   - The API includes CORS headers for all origins
   - If issues persist, check your browser's developer console

### Debug Mode

The Flask app runs in debug mode by default. Check the console output for detailed error messages.

## 🔒 Security Notes

- API keys are stored in environment variables (not hardcoded)
- Database passwords are masked in API responses
- CORS is enabled for development (consider restricting in production)

## 🎯 Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:5000/health

# Query database
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users are in the database?"}'

# Get API info
curl http://localhost:5000/info
```

### Using the Web Interface

1. Open `index.html` in your browser
2. Wait for the green status indicator
3. Type your question or click an example
4. View the response with timing information

## 📊 Performance

- **Response Times**: Typically 1-5 seconds depending on query complexity
- **Concurrent Requests**: Flask development server (single-threaded)
- **Database Queries**: Optimized through LangChain SQL agent

## 🤝 Contributing

Feel free to enhance this chatbot by:
- Adding new API endpoints
- Improving the web interface
- Adding authentication
- Implementing caching
- Adding more AI models

## 📝 License

This project is open source and available under the MIT License.

---

**Made by Soumik ❤️ using Flask, LangChain, and DeepSeek AI**
