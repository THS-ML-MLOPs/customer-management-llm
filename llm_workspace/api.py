"""
================================================================================
    CUSTOMER MANAGEMENT SYSTEM - FASTAPI BACKEND
    Complete LLM API with Ollama, DuckDB, CLIP, and Plotly
================================================================================
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime
import uvicorn
import os
import json
import sqlite3
import uuid
import base64
import io
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PATH DETECTION - Works on Local Windows AND Google Colab
# ============================================================================

def detect_environment():
    """Detect if running on Colab or local"""
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'

def get_project_root():
    """Get project root path based on environment"""
    env = detect_environment()

    if env == 'colab':
        return Path("/content/drive/MyDrive/ML_Projects/Customer_Management")
    else:
        # Local: go up from llm_workspace to project root
        return Path(__file__).parent.parent

# Set paths
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONVERSATIONS_DIR = PROJECT_ROOT / "llm_workspace" / "conversations"
FEEDBACK_DB = PROJECT_ROOT / "llm_workspace" / "feedback.db"

# Create directories
CONVERSATIONS_DIR.mkdir(exist_ok=True, parents=True)

logger.info(f"Environment: {detect_environment()}")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Data Directory: {DATA_DIR}")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Customer Management LLM API",
    description="Complete API with Ollama, DuckDB, CLIP, and Plotly",
    version="1.0.0"
)

# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict] = {}

class QueryResponse(BaseModel):
    success: bool
    data: Optional[List[Dict]] = None
    sql_used: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

class ChartRequest(BaseModel):
    data: Dict
    chart_type: str
    title: str
    config: Optional[Dict] = {}

class ChartResponse(BaseModel):
    success: bool
    chart_html: Optional[str] = None
    chart_json: Optional[Dict] = None
    error: Optional[str] = None

class ImageSearchRequest(BaseModel):
    image: str  # base64 encoded
    top_k: int = 5

class ImageSearchResponse(BaseModel):
    success: bool
    products: Optional[List[Dict]] = None
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    message_id: str
    feedback_type: str  # 'like' or 'dislike'
    comment: Optional[str] = None

# ============================================================================
# OLLAMA INTEGRATION
# ============================================================================

class OllamaClient:
    """Client for Ollama LLM"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.1"

    def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """
        Send chat message to Ollama
        """
        try:
            import requests

            # Build messages
            messages = conversation_history or []
            messages.append({
                "role": "user",
                "content": message
            })

            # Call Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return f"Error: Ollama returned status {response.status_code}"

        except Exception as e:
            logger.error(f"Ollama exception: {str(e)}")
            return f"Error communicating with Ollama: {str(e)}"

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

# Initialize Ollama client
ollama_client = OllamaClient()

# ============================================================================
# DUCKDB INTEGRATION
# ============================================================================

class DuckDBClient:
    """Client for DuckDB queries on parquet files"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._db = None

    def get_connection(self):
        """Get or create DuckDB connection"""
        try:
            import duckdb

            if self._db is None:
                self._db = duckdb.connect(":memory:")
                logger.info("DuckDB connection created")

                # Register key parquet files as views
                self._register_views()

            return self._db
        except Exception as e:
            logger.error(f"DuckDB connection error: {str(e)}")
            raise

    def _register_views(self):
        """Register parquet files as DuckDB views"""
        try:
            conn = self._db

            # Core tables
            tables_to_register = {
                'customers': 'customers_clean.parquet',
                'products': 'fashion_catalog_clean.parquet',
                'discounts': 'discounts_clean.parquet',
                'transactions': 'transaction_items.parquet',
                'churn_risk': 'features/customer_churn_risk.parquet',
                'customer_segments': 'features/customer_segments.parquet',
                'customer_ltv': 'features/customer_lifetime_value.parquet',
            }

            for table_name, filename in tables_to_register.items():
                file_path = self.data_dir / filename

                if file_path.exists():
                    conn.execute(f"""
                        CREATE OR REPLACE VIEW {table_name} AS
                        SELECT * FROM read_parquet('{file_path}')
                    """)
                    logger.info(f"Registered view: {table_name}")
                else:
                    logger.warning(f"File not found: {file_path}")

        except Exception as e:
            logger.error(f"Error registering views: {str(e)}")

    def query(self, sql: str) -> List[Dict]:
        """Execute SQL query and return results as list of dicts"""
        try:
            conn = self.get_connection()

            # Execute query
            result = conn.execute(sql).fetchdf()

            # Convert to list of dicts
            return result.to_dict('records')

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            raise

    def natural_language_to_sql(self, nl_query: str) -> str:
        """
        Convert natural language to SQL using Ollama
        """
        # Get schema info
        schema_info = self._get_schema_info()

        # Create prompt for Ollama
        prompt = f"""You are a SQL expert. Convert this natural language query to SQL.

Available tables and columns:
{schema_info}

Natural language query: {nl_query}

Return ONLY the SQL query, no explanations. Use DuckDB syntax.
Limit results to 1000 rows unless specified otherwise.

SQL:"""

        # Get SQL from Ollama
        sql = ollama_client.chat(prompt)

        # Clean up SQL (remove markdown code blocks if present)
        sql = sql.replace('```sql', '').replace('```', '').strip()

        return sql

    def _get_schema_info(self) -> str:
        """Get schema information for prompt"""
        try:
            conn = self.get_connection()

            # Get list of views
            views = conn.execute("SHOW TABLES").fetchall()

            schema_lines = []
            for (view_name,) in views:
                # Get columns for this view
                columns = conn.execute(f"DESCRIBE {view_name}").fetchdf()
                cols_str = ", ".join([f"{row['column_name']} ({row['column_type']})"
                                     for _, row in columns.iterrows()])
                schema_lines.append(f"- {view_name}: {cols_str}")

            return "\n".join(schema_lines)

        except:
            return "customers, products, discounts, transactions, churn_risk, customer_segments"

# Initialize DuckDB client
duckdb_client = DuckDBClient(DATA_DIR)

# ============================================================================
# CLIP IMAGE SEARCH
# ============================================================================

class CLIPSearcher:
    """CLIP-based image search for products"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.embeddings = None
        self.product_mapping = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of CLIP model"""
        if self._initialized:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import numpy as np
            import torch

            logger.info("Initializing CLIP model...")

            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Try to load pre-computed embeddings
            embeddings_file = PROJECT_ROOT / "product_embeddings.npy"
            mapping_file = PROJECT_ROOT / "embedding_mapping.parquet"

            if embeddings_file.exists() and mapping_file.exists():
                logger.info("Loading pre-computed embeddings...")
                self.embeddings = np.load(str(embeddings_file))

                import pandas as pd
                self.product_mapping = pd.read_parquet(str(mapping_file))

                logger.info(f"Loaded {len(self.embeddings)} product embeddings")
            else:
                logger.warning("Pre-computed embeddings not found. Image search may be limited.")

            self._initialized = True

        except Exception as e:
            logger.error(f"CLIP initialization error: {str(e)}")
            raise

    def search(self, image_bytes: bytes, top_k: int = 5) -> List[Dict]:
        """Search for similar products by image"""
        try:
            self.initialize()

            if self.embeddings is None:
                raise ValueError("Product embeddings not available")

            from PIL import Image
            import numpy as np
            import torch

            # Load and process image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")

            # Get image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_embedding = image_features.cpu().numpy()[0]

            # Normalize
            image_embedding = image_embedding / np.linalg.norm(image_embedding)

            # Compute similarities
            similarities = np.dot(self.embeddings, image_embedding)

            # Get top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Build results
            results = []
            for idx in top_indices:
                product_info = self.product_mapping.iloc[idx].to_dict()
                product_info['similarity'] = float(similarities[idx])
                results.append(product_info)

            return results

        except Exception as e:
            logger.error(f"Image search error: {str(e)}")
            raise

# Initialize CLIP searcher
clip_searcher = CLIPSearcher()

# ============================================================================
# PLOTLY CHART GENERATION
# ============================================================================

def generate_plotly_chart(data: Dict, chart_type: str, title: str, config: Dict = None) -> str:
    """Generate Plotly chart as HTML"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Generate chart based on type
        if chart_type == 'bar':
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)

        elif chart_type == 'line':
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)

        elif chart_type == 'scatter':
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=title)

        elif chart_type == 'pie':
            fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)

        else:
            # Auto-detect best chart type
            if len(df.columns) >= 2:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

        # Apply custom config if provided
        if config:
            fig.update_layout(**config)

        # Return as HTML
        return fig.to_html(include_plotlyjs='cdn', div_id='chart')

    except Exception as e:
        logger.error(f"Chart generation error: {str(e)}")
        raise

# ============================================================================
# CONVERSATION STORAGE
# ============================================================================

class ConversationStorage:
    """Store and retrieve conversation history"""

    @staticmethod
    def save(conversation_id: str, messages: List[Dict]):
        """Save conversation to file"""
        try:
            file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"

            with open(file_path, 'w') as f:
                json.dump({
                    'conversation_id': conversation_id,
                    'messages': messages,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")

    @staticmethod
    def load(conversation_id: str) -> List[Dict]:
        """Load conversation from file"""
        try:
            file_path = CONVERSATIONS_DIR / f"{conversation_id}.json"

            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return data.get('messages', [])

            return []

        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}")
            return []

# ============================================================================
# FEEDBACK STORAGE
# ============================================================================

class FeedbackManager:
    """Manage user feedback in SQLite"""

    @staticmethod
    def init_db():
        """Initialize feedback database"""
        try:
            conn = sqlite3.connect(str(FEEDBACK_DB))
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    comment TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error initializing feedback DB: {str(e)}")

    @staticmethod
    def save_feedback(message_id: str, feedback_type: str, comment: str = None):
        """Save feedback"""
        try:
            conn = sqlite3.connect(str(FEEDBACK_DB))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO feedback (message_id, feedback_type, comment, timestamp)
                VALUES (?, ?, ?, ?)
            """, (message_id, feedback_type, comment, datetime.now().isoformat()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")

# Initialize feedback DB
FeedbackManager.init_db()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Customer Management LLM API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/chat",
            "query": "/query",
            "chart": "/chart",
            "search_image": "/search_image",
            "conversation": "/conversation/{id}",
            "feedback": "/feedback"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""

    # Check Ollama
    ollama_status = ollama_client.is_available()

    # Check DuckDB
    duckdb_status = False
    try:
        duckdb_client.get_connection()
        duckdb_status = True
    except:
        pass

    return {
        "status": "healthy" if (ollama_status and duckdb_status) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama": "online" if ollama_status else "offline",
            "duckdb": "online" if duckdb_status else "offline",
            "clip": "ready" if clip_searcher._initialized else "not_loaded"
        },
        "environment": detect_environment(),
        "project_root": str(PROJECT_ROOT)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with Ollama LLM
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Load conversation history
        history = ConversationStorage.load(conversation_id)

        # Get response from Ollama
        response_text = ollama_client.chat(request.message, history)

        # Generate message ID
        message_id = str(uuid.uuid4())

        # Update history
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": response_text, "message_id": message_id})

        # Save conversation
        ConversationStorage.save(conversation_id, history)

        return ChatResponse(
            success=True,
            response=response_text,
            message_id=message_id,
            conversation_id=conversation_id
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return ChatResponse(
            success=False,
            error=str(e)
        )

@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """
    Execute natural language query on data
    """
    try:
        import time
        start_time = time.time()

        # Convert natural language to SQL
        sql = duckdb_client.natural_language_to_sql(request.query)

        logger.info(f"Generated SQL: {sql}")

        # Execute query
        results = duckdb_client.query(sql)

        execution_time = time.time() - start_time

        return QueryResponse(
            success=True,
            data=results,
            sql_used=sql,
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return QueryResponse(
            success=False,
            error=str(e)
        )

@app.post("/chart", response_model=ChartResponse)
async def generate_chart(request: ChartRequest):
    """
    Generate Plotly chart
    """
    try:
        chart_html = generate_plotly_chart(
            request.data,
            request.chart_type,
            request.title,
            request.config
        )

        return ChartResponse(
            success=True,
            chart_html=chart_html
        )

    except Exception as e:
        logger.error(f"Chart error: {str(e)}")
        return ChartResponse(
            success=False,
            error=str(e)
        )

@app.post("/search_image", response_model=ImageSearchResponse)
async def search_image(request: ImageSearchRequest):
    """
    Search similar products by image using CLIP
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)

        # Search
        results = clip_searcher.search(image_bytes, request.top_k)

        return ImageSearchResponse(
            success=True,
            products=results
        )

    except Exception as e:
        logger.error(f"Image search error: {str(e)}")
        return ImageSearchResponse(
            success=False,
            error=str(e)
        )

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history
    """
    try:
        messages = ConversationStorage.load(conversation_id)

        return {
            "success": True,
            "conversation_id": conversation_id,
            "messages": messages
        }

    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a message
    """
    try:
        FeedbackManager.save_feedback(
            request.message_id,
            request.feedback_type,
            request.comment
        )

        return {
            "success": True,
            "message": "Feedback submitted"
        }

    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    logger.info("="*80)
    logger.info("  CUSTOMER MANAGEMENT LLM API - STARTING")
    logger.info("="*80)
    logger.info(f"Environment: {detect_environment()}")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Ollama Available: {ollama_client.is_available()}")
    logger.info("="*80)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown"""
    logger.info("API shutting down...")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
