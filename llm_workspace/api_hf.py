"""
================================================================================
    CUSTOMER MANAGEMENT SYSTEM - HUGGING FACE CLIENT (STREAMLIT VERSION)
    Core functionality for HuggingFace, DuckDB, and CLIP
    No FastAPI needed - optimized for Streamlit Cloud
================================================================================
"""

import os
import json
import sqlite3
import uuid
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Hugging Face
from huggingface_hub import InferenceClient

# Data Loader (loads from HF Dataset)
from .data_loader import get_data_loader, get_parquet_path, get_embeddings_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PATH DETECTION - Works on Local Windows AND HF Spaces
# ============================================================================

def detect_environment():
    """Detect if running on HF Spaces, Colab, or local"""
    if os.environ.get("SPACE_ID"):
        return 'huggingface'
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'

def get_project_root():
    """Get project root path based on environment"""
    env = detect_environment()

    if env == 'huggingface':
        # HF Spaces: everything is in /app or current directory
        return Path.cwd()
    elif env == 'colab':
        return Path("/content/drive/MyDrive/ML_Projects/Customer_Management")
    else:
        # Local: go up from llm_workspace to project root
        return Path(__file__).parent.parent

# Set paths
PROJECT_ROOT = get_project_root()
CONVERSATIONS_DIR = PROJECT_ROOT / "llm_workspace" / "conversations"
FEEDBACK_DB = PROJECT_ROOT / "llm_workspace" / "feedback.db"

# Create directories
CONVERSATIONS_DIR.mkdir(exist_ok=True, parents=True)

# Initialize data loader (downloads from HF Dataset if needed)
data_loader = get_data_loader()

logger.info(f"Environment: {detect_environment()}")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Data Loader initialized - will load from HF Dataset")

# ============================================================================
# HUGGING FACE INTEGRATION
# ============================================================================

class HuggingFaceClient:
    """Client for Hugging Face Inference API"""

    def __init__(self):
        # Get HF token from environment or Streamlit Secrets
        self.hf_token = os.environ.get("HF_TOKEN")

        # Fallback to Streamlit Secrets if not in environment
        if not self.hf_token:
            try:
                import streamlit as st
                self.hf_token = st.secrets.get("HF_TOKEN")
                if self.hf_token:
                    logger.info("HF_TOKEN loaded from Streamlit Secrets")
            except:
                pass

        if not self.hf_token:
            logger.warning("HF_TOKEN not found in environment or Streamlit Secrets. API calls may fail.")

        # Initialize client with Llama model
        self.client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=self.hf_token
        )

        logger.info("HuggingFace client initialized with Llama-3.1-8B-Instruct")

    def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """
        Send chat message to Hugging Face
        """
        try:
            # Build prompt from conversation history
            if conversation_history:
                prompt = self._build_prompt_from_history(conversation_history, message)
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            # Call HF Inference API
            response = self.client.text_generation(
                prompt=prompt,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            return f"Error: {str(e)}"

    def _build_prompt_from_history(self, history: List[Dict], new_message: str) -> str:
        """Build Llama 3.1 format prompt from conversation history"""
        prompt = "<|begin_of_text|>"

        for msg in history:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
            elif role == 'assistant':
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

        # Add new user message
        prompt += f"<|start_header_id|>user<|end_header_id|>\n{new_message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return prompt

    def is_available(self) -> bool:
        """Check if HF API is available"""
        return self.hf_token is not None

# Initialize HF client
hf_client = HuggingFaceClient()

# ============================================================================
# DUCKDB INTEGRATION
# ============================================================================

class DuckDBClient:
    """Client for DuckDB queries on parquet files (loaded from HF Dataset)"""

    def __init__(self):
        self._db = None
        self._views_registered = False

    def get_connection(self):
        """Get or create DuckDB connection"""
        try:
            import duckdb

            if self._db is None:
                self._db = duckdb.connect(":memory:")
                logger.info("DuckDB connection created (views will be registered on first query)")

            return self._db
        except Exception as e:
            logger.error(f"DuckDB connection error: {str(e)}")
            raise

    def ensure_views_registered(self):
        """Ensure views are registered (lazy loading)"""
        if not self._views_registered:
            logger.info("Registering DuckDB views (downloading from HF Dataset)...")
            self._register_views()
            self._views_registered = True

    def _register_views(self):
        """Register parquet files as DuckDB views (downloads from HF if needed)"""
        try:
            conn = self._db

            # Tables to register (keys match data_loader.py)
            tables_to_register = [
                'customers',
                'products',
                'discounts',
                'transactions',
                'churn_risk',
                'customer_segments',
                'customer_ltv',
            ]

            for table_name in tables_to_register:
                # Get file path from HF Dataset (downloads if needed)
                file_path = get_parquet_path(table_name)

                if file_path:
                    conn.execute(f"""
                        CREATE OR REPLACE VIEW {table_name} AS
                        SELECT * FROM read_parquet('{file_path}')
                    """)
                    logger.info(f"✅ Registered view: {table_name} from {file_path}")
                else:
                    logger.warning(f"⚠️  Skipping {table_name}: file not available")

        except Exception as e:
            logger.error(f"Error registering views: {str(e)}")

    def query(self, sql: str) -> List[Dict]:
        """Execute SQL query and return results as list of dicts"""
        try:
            conn = self.get_connection()

            # Ensure views are registered (lazy loading)
            self.ensure_views_registered()

            # Execute query
            result = conn.execute(sql).fetchdf()

            # Convert to list of dicts
            return result.to_dict('records')

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            raise

    def natural_language_to_sql(self, nl_query: str) -> str:
        """
        Convert natural language to SQL using HuggingFace
        """
        # Get schema info
        schema_info = self._get_schema_info()

        # Create prompt for HF model
        prompt = f"""You are a SQL expert. Convert this natural language query to SQL.

Available tables and columns:
{schema_info}

Natural language query: {nl_query}

Important rules:
- Return ONLY the SQL query, no explanations
- Use DuckDB syntax
- Limit results to 1000 rows unless specified
- Use proper SQL formatting

SQL:"""

        # Get SQL from HuggingFace
        sql = hf_client.chat(prompt)

        # Clean up SQL (remove markdown code blocks if present)
        sql = sql.replace('```sql', '').replace('```', '').strip()

        # Remove any text after the SQL query
        if ';' in sql:
            sql = sql.split(';')[0] + ';'

        return sql

    def _get_schema_info(self) -> str:
        """Get schema information for prompt"""
        try:
            conn = self.get_connection()

            # Ensure views are registered
            self.ensure_views_registered()

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

# Initialize DuckDB client (will load data from HF Dataset)
duckdb_client = DuckDBClient()

# ============================================================================
# CLIP IMAGE SEARCH (Optional - can be disabled for faster startup)
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

            # Load pre-computed embeddings from HF Dataset
            embeddings_paths = get_embeddings_paths()

            if embeddings_paths:
                embeddings_file, mapping_file = embeddings_paths
                logger.info("Loading pre-computed embeddings from HF Dataset...")

                self.embeddings = np.load(str(embeddings_file))

                import pandas as pd
                self.product_mapping = pd.read_parquet(str(mapping_file))

                logger.info(f"✅ Loaded {len(self.embeddings)} product embeddings")
            else:
                logger.warning("⚠️  Pre-computed embeddings not found. Image search disabled.")

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

# Initialize CLIP searcher (lazy)
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
# MODULE INITIALIZATION
# ============================================================================

logger.info("="*80)
logger.info("  CUSTOMER MANAGEMENT LLM - STREAMLIT VERSION")
logger.info("="*80)
logger.info(f"Environment: {detect_environment()}")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Data Source: HF Dataset ({data_loader.repo_id})")
logger.info(f"HuggingFace Available: {hf_client.is_available()}")
logger.info("="*80)
