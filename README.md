# ChatWhiz

ChatWhiz is an AI assistant implementation that provides personalized responses about my professional experience and projects using Google's Gemini AI model. Built with FastAPI and Python, it features real-time data synchronization, conversation persistence, and robust user session management.

## Features

### Core Functionality
- AI-powered conversational interface using Google Gemini-3.1-flash-lite model
- Dynamic data synchronization from portfolio JSON sources
- In-session conversation history per user
- User session management with UUID-based tracking
- Rate limiting system (50 requests/24 hours per user)

### Technical Features
- Real-time data updates every 2 hours
- 5-message conversation context window
- CORS middleware for secure cross-origin requests
- Comprehensive error handling and logging
- Health monitoring endpoints

## API Endpoints

### Primary Endpoints
```
POST /query
- Process user queries and generate AI responses
- Request body: { "query_text": "string" }
- Response: { "response": "string" }

GET /history
- Retrieve conversation history for current user
- Response: { "history": [...] }

DELETE /history
- Clear conversation history for current user
- Response: { "message": "string" }
```

### Monitoring Endpoints
```
GET /health
- Check system health and data update status
- Response: {
    "status": "healthy",
    "timestamp": "ISO timestamp",
    "last_data_update": "ISO timestamp",
    "data_available": boolean
}

GET /api/remaining-requests
- Check remaining API requests for current user
- Response: {
    "remaining_requests": int,
    "rate_limit": int,
    "window_hours": float
}
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/YourUsername/ChatWhiz.git
cd ChatWhiz
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create .env file with the following variables
GEMINI_API_KEY=your_gemini_api_key
PORTFOLIO_DATA_BASE_URL=https://www.nafisui.com
```

## Requirements
```
fastapi
uvicorn
python-dotenv
google-generativeai
requests
pypdf
pydantic
python-multipart
```

## Usage

1. Start the server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

2. The API will be available at `http://localhost:8000`

## Data Management

### Data Sources
The system synchronizes data from the deployed portfolio site (default: `https://www.nafisui.com`):
- Projects: `/projects.json`
- Work Experience: `/work.json`
- Research: `/research.json`
- Resume: `/Resume.pdf` (text extracted at sync time)

Set `PORTFOLIO_DATA_BASE_URL` to override the site used for data sync.

### Data Update Mechanism
- Automatic updates every 2 hours
- Updates only occur if data has changed
- Failed updates retry on next cycle

## Conversation Management

### Storage
- Conversations are kept in memory for the current session
- Each conversation includes:
  - Timestamp
  - User ID
  - Query
  - Response

### Context Window
- Maintains last 5 conversations per user
- Used for context-aware responses
- Automatically cleared after session ends

## Rate Limiting

### Limits
- 50 requests per user per 24 hours
- Based on IP address with UUID mapping
- Automatic cleanup of expired request counts

### Monitoring
- Real-time request counting
- Remaining request checking endpoint
- Automatic rate limit enforcement

## Security

### CORS Configuration
- Restricted to specified domains
- Secure credential handling
- Protected endpoints

### Error Handling
- Comprehensive error catching
- Detailed error responses
- Failed request logging

## Development

### Running Tests
```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=.
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact me at wasiflh@connect.hku.hk

## Acknowledgments

- Google Generative AI team for the Gemini model
- FastAPI team for the excellent framework
