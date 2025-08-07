import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
from dotenv import load_dotenv
load_dotenv()

# FastAPI and HTTP dependencies
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import uvicorn

# Document processing
import requests
import PyPDF2
import docx
from io import BytesIO
import re

# Vector search and embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# LLM integration
from openai import OpenAI


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX Document Query System",
    description="LLM-Powered Intelligent Query-Retrieval System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Configuration
class Config:
    BEARER_TOKEN = "6316e01746d83a3078c19510945475dd0aa9c7f218659c845184a49e455bf8e0"
    OPENAI_API_KEY = os.getenv('OPENROUTER_API_KEY')  # Changed to OPENROUTER_API_KEY
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 7
    SIMILARITY_THRESHOLD = 0.2 # Lowered threshold for better retrieval

config = Config()

# Pydantic models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document blob")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

class DocumentChunk(BaseModel):
    content: str
    chunk_id: str
    source: str
    metadata: Dict[str, Any] = {}

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Document processing utilities
class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc']
    
    async def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process DOCX: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)]', ' ', text)
        # Remove extra spaces
        text = text.strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or config.MAX_CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def process_document(self, url: str) -> List[DocumentChunk]:
        """Process document and return chunks"""
        content = await self.download_document(url)
        
        # Determine file type from URL
        url_lower = url.lower()
        if '.pdf' in url_lower:
            text = self.extract_text_from_pdf(content)
        elif '.docx' in url_lower or '.doc' in url_lower:
            text = self.extract_text_from_docx(content)
        else:
            # Try PDF first, then DOCX
            try:
                text = self.extract_text_from_pdf(content)
            except:
                try:
                    text = self.extract_text_from_docx(content)
                except:
                    raise HTTPException(status_code=400, detail="Unsupported document format")
        
        # Clean and chunk text
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{url}_{i}_{chunk[:50]}".encode()).hexdigest()
            document_chunks.append(DocumentChunk(
                content=chunk,
                chunk_id=chunk_id,
                source=url,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            ))
        
        return document_chunks

# Vector search system
class VectorSearchEngine:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        return self.encoder.encode(texts)
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index from document chunks"""
        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        self.embeddings = self.encode_texts(texts)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Built FAISS index with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = None) -> List[tuple]:
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        # Encode query
        query_embedding = self.encode_texts([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Log search results for debugging
        logger.info(f"Search query: {query}")
        logger.info(f"Top scores: {scores[0]}")
        
        # Filter by similarity threshold
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= config.SIMILARITY_THRESHOLD:
                results.append((self.chunks[idx], float(score)))
                logger.info(f"Added chunk with score {score:.3f}: {self.chunks[idx].content[:100]}...")
        
        logger.info(f"Found {len(results)} chunks above threshold {config.SIMILARITY_THRESHOLD}")
        return results

# LLM integration
# LLM integration with OpenRouter configuration
class LLMEngine:
    def __init__(self):
        # Configure OpenAI client to use OpenRouter
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1"  # This is the key addition!
        )
    
    async def generate_answer(self, question: str, context_chunks: List[tuple]) -> str:
        """Generate answer using LLM with retrieved context"""
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer this question."
        
        # Prepare context with more detailed formatting
        context_parts = []
        for i, (chunk, score) in enumerate(context_chunks):
            context_parts.append(f"[Context {i+1} - Relevance: {score:.2f}]\n{chunk.content}\n")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt for better extraction
        prompt = f"""You are an expert document analyst. Based on the following document excerpts, please answer the question accurately and in detail.

Document Context:
{context}

Question: {question}

Instructions:
1. Carefully read through all the context provided above
2. Answer based ONLY on the information found in the context
3. If the exact information is not available, say "The document does not contain specific information about [topic]"
4. Be specific and include relevant details like numbers, dates, conditions, and clauses
5. If there are multiple relevant pieces of information, combine them in your answer
6. Use direct quotes from the document when possible
7. Structure your answer clearly and concisely

Answer:"""

        try:
            # Use free models available on OpenRouter
            models_to_try = [
                "moonshotai/kimi-k2:free",  # Free model
                "openai/gpt-3.5-turbo",  # Free tier available
                "microsoft/wizardlm-2-8x22b",  # Free alternative
                "meta-llama/llama-3-8b-instruct:free"  # Free Llama model
            ]
            
            for model in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on document content. Provide accurate, concise answers based only on the provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.1
                    )
                    logger.info(f"Successfully used model: {model}")
                    return response.choices[0].message.content.strip()
                    
                except Exception as model_error:
                    logger.warning(f"Model {model} failed: {model_error}")
                    if model == models_to_try[-1]:  # Last model in list
                        # If all models fail, return a fallback response
                        logger.error("All models failed, providing fallback response")
                        return self._generate_fallback_answer(question, context_chunks)
                    continue
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._generate_fallback_answer(question, context_chunks)
    
    def _generate_fallback_answer(self, question: str, context_chunks: List[tuple]) -> str:
        """Generate a simple fallback answer when LLM fails"""
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer this question."
        
        # Simple keyword-based extraction as fallback
        question_lower = question.lower()
        relevant_text = ""
        
        for chunk, score in context_chunks[:2]:  # Use top 2 chunks
            relevant_text += chunk.content + " "
        
        # Truncate if too long
        if len(relevant_text) > 300:
            relevant_text = relevant_text[:300] + "..."
        
        return f"Based on the document content: {relevant_text}"

# Main processing pipeline
class QueryProcessor:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_engine = VectorSearchEngine()
        self.llm_engine = LLMEngine()
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main processing pipeline"""
        try:
            # Step 1: Process document
            logger.info(f"Processing document: {request.documents}")
            chunks = await self.doc_processor.process_document(request.documents)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Step 2: Build vector index
            self.vector_engine.build_index(chunks)
            
            # Step 3: Process each question
            answers = []
            for i, question in enumerate(request.questions):
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
                
                # Retrieve relevant chunks
                relevant_chunks = self.vector_engine.search(question)
                logger.info(f"Found {len(relevant_chunks)} relevant chunks")
                
                # Generate answer
                answer = await self.llm_engine.generate_answer(question, relevant_chunks)
                answers.append(answer)
            
            return QueryResponse(answers=answers)
        
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Initialize processor
query_processor = QueryProcessor()

# API Routes
@app.get("/")
async def root():
    return {
        "message": "HackRX Document Query System",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Main endpoint for processing document queries"""
    logger.info(f"Received query request with {len(request.questions)} questions")
    
    try:
        result = await query_processor.process_query(request)
        logger.info("Query processing completed successfully")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Additional utility endpoints
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query_alt(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Alternative endpoint path for compatibility"""
    return await run_query(request, token)

if __name__ == "__main__":
    # For local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )