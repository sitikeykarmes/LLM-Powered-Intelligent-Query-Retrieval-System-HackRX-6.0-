# HackRX Document Query System

An intelligent document query and retrieval system powered by LLM (Large Language Models) that can process PDF and DOCX documents, extract information, and answer questions based on the document content.

## 🚀 Features

- **Document Processing**: Supports PDF and DOCX file formats
- **Intelligent Chunking**: Breaks documents into meaningful chunks with overlap
- **Vector Search**: Uses sentence transformers and FAISS for semantic search
- **LLM Integration**: Powered by OpenRouter API for generating contextual answers
- **RESTful API**: FastAPI-based REST endpoints
- **Authentication**: Bearer token authentication
- **CORS Support**: Cross-origin resource sharing enabled

## 📋 Dependencies

This project requires the following Python packages:

- `fastapi` - Modern web framework for building APIs
- `uvicorn[standard]` - ASGI server implementation
- `pydantic` - Data validation using Python type annotations
- `requests` - HTTP library for downloading documents
- `PyPDF2` - PDF processing library
- `python-docx` - Microsoft Word document processing
- `sentence-transformers` - Semantic text embeddings
- `faiss-cpu` - Efficient similarity search and clustering
- `scikit-learn` - Machine learning utilities
- `openai` - OpenAI/OpenRouter API client
- `python-multipart` - Multipart form data parsing

## 🛠 Installation

**⚠️ Important: Do NOT install dependencies using a requirements.txt file. Install each package individually to avoid version conflicts.**

Install each dependency one by one using pip:

```bash
pip install fastapi
pip install uvicorn[standard]
pip install pydantic
pip install requests
pip install PyPDF2
pip install python-docx
pip install sentence-transformers
pip install faiss-cpu
pip install scikit-learn
pip install openai
pip install python-multipart
```

## 🔑 Environment Setup

1. **Get OpenRouter API Key**:

   - Visit [OpenRouter](https://openrouter.ai/)
   - Create an account and get your API key
   - Some models are free, others require credits

2. **Set Environment Variable**:

   **Windows (PowerShell):**

   ```powershell
   $env:OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```

   **Windows (Command Prompt):**

   ```cmd
   set OPENROUTER_API_KEY=your-openrouter-api-key-here
   ```

   **Linux/macOS:**

   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```

   **Alternative - Create .env file in app folder:**

   ```
   OPENROUTER_API_KEY=your-openrouter-api-key-here
   ```

## 🚀 Running the Project

Contact sitikeykarmes(Instagram) for help.

### 1. Start the API Server

Navigate to your project directory(app) and run:

```bash
python main.py
```

Or alternatively:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

### 2. Verify the Setup

Check if the server is running:

```bash
curl http://localhost:8000/health
```

You should see:

```json
{
  "status": "healthy",
  "timestamp": "2025-08-03T..."
}
```

### 3. Test the API

Run the test script to validate your setup:

```bash
python test_openrouter_api.py
```

## 📖 API Usage

### Authentication

All requests require a Bearer token in the Authorization header:

```
Authorization: Bearer 6316e01746d83a3078c19510945475dd0aa9c7f218659c845184a49e455bf8e0
```

### Main Endpoint

**POST** `/hackrx/run`

**Request Body:**

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response:**

```json
{
  "answers": [
    "The grace period for premium payment is 30 days...",
    "The waiting period for pre-existing diseases is 2 years..."
  ]
}
```

### Alternative Endpoint

**POST** `/api/v1/hackrx/run`

Same functionality as the main endpoint, provided for compatibility.

## 🔧 Configuration

You can modify the following settings in the `Config` class:

- `MAX_CHUNK_SIZE`: Maximum size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity for chunk retrieval (default: 0.3)

## 🎯 Supported Document Formats

- **PDF Files**: `.pdf`
- **Microsoft Word**: `.docx`, `.doc`

Documents are automatically processed, cleaned, and chunked for optimal retrieval.

## 🧪 Testing

The project includes a comprehensive test suite (`test_openrouter_api.py`) that validates:

- ✅ Environment setup
- ✅ OpenRouter API connection
- ✅ Health endpoints
- ✅ Authentication
- ✅ Main functionality
- ✅ Response format validation
- ✅ Performance testing

## 🚨 Troubleshooting

### Common Issues

1. **"Invalid API key" error**:

   - Verify your OpenRouter API key is set correctly
   - Check that the environment variable is properly configured

2. **Document processing fails**:

   - Ensure the document URL is accessible
   - Check that the document format is supported (PDF or DOCX)

3. **Slow response times**:

   - OpenRouter free models may have rate limits
   - Consider using premium models for faster responses

4. **Import errors**:
   - Install dependencies individually as instructed
   - Avoid using requirements.txt to prevent version conflicts

### Debug Mode

To run with debug logging:

```bash
python main.py --log-level debug
```

## 📝 API Documentation

Once the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🏗 Architecture

```
Document URL → Download → Extract Text → Clean & Chunk → Generate Embeddings → FAISS Index
                                                                                      ↓
Question → Generate Embedding → Vector Search → Retrieve Chunks → LLM Processing → Answer
```

## 🤝 Contributing

1. Fork the repository
2. Install dependencies individually (not via requirements.txt)
3. Make your changes
4. Run the test suite
5. Submit a pull request

## 📄 License

This project belongs to team BitLords and is part of the HackRX competition.

---

**🎉 Ready to submit? Your webhook URL should be: `https://your-domain.com/hackrx/run`**
