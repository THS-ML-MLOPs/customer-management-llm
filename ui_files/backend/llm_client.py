"""
================================================================================
    LLM CLIENT (STREAMLIT VERSION)
    Direct integration with HF client - no HTTP requests needed
================================================================================
"""

from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import logging

# Import HF client directly (no HTTP needed)
from llm_workspace.api_hf import (
    hf_client,
    duckdb_client,
    ConversationStorage,
    FeedbackManager,
    generate_plotly_chart,
    clip_searcher
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for LLM operations - calls hf_client directly
    No FastAPI server needed!
    """

    def __init__(self):
        """Initialize LLM client"""
        self.hf_client = hf_client
        self.duckdb_client = duckdb_client
        logger.info("LLMClient initialized (direct mode)")

    def health_check(self) -> bool:
        """
        Check if HF client is available

        Returns:
            bool: True if HF API is available
        """
        return self.hf_client.is_available()

    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat message to LLM (direct call to hf_client)

        Args:
            message: User message
            conversation_id: Optional conversation ID for context
            stream: Not used (kept for compatibility)

        Returns:
            Dict with response
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Load conversation history
            history = ConversationStorage.load(conversation_id)

            # Get response from HuggingFace (direct call)
            response_text = self.hf_client.chat(message, conversation_history=history)

            # Generate message ID
            message_id = str(uuid.uuid4())

            # Update history
            history.append({"role": "user", "content": message})
            history.append({
                "role": "assistant",
                "content": response_text,
                "message_id": message_id
            })

            # Save conversation
            ConversationStorage.save(conversation_id, history)

            return {
                "success": True,
                "response": response_text,
                "message_id": message_id,
                "conversation_id": conversation_id
            }

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def query_data(
        self,
        query: str,
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute DuckDB query (direct call to duckdb_client)

        Args:
            query: Natural language or SQL query
            parameters: Optional query parameters

        Returns:
            Dict with query results
        """
        try:
            import time
            start_time = time.time()

            # Convert natural language to SQL
            sql = self.duckdb_client.natural_language_to_sql(query)

            logger.info(f"Generated SQL: {sql}")

            # Execute query
            results = self.duckdb_client.query(sql)

            execution_time = time.time() - start_time

            return {
                "success": True,
                "data": results,
                "sql_used": sql,
                "execution_time": execution_time
            }

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def generate_chart(
        self,
        data: Dict[str, Any],
        chart_type: str,
        title: str,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate Plotly chart (direct call to generate_plotly_chart)

        Args:
            data: Chart data
            chart_type: Type of chart (bar, line, scatter, etc)
            title: Chart title
            config: Optional chart configuration

        Returns:
            Dict with chart HTML and metadata
        """
        try:
            chart_html = generate_plotly_chart(
                data,
                chart_type,
                title,
                config or {}
            )

            return {
                "success": True,
                "chart_html": chart_html
            }

        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def search_image(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search similar products by image using CLIP (direct call to clip_searcher)

        Args:
            image_bytes: Image as bytes
            top_k: Number of results to return

        Returns:
            Dict with similar products
        """
        try:
            results = clip_searcher.search(image_bytes, top_k)

            return {
                "success": True,
                "products": results
            }

        except Exception as e:
            logger.error(f"Image search error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieve conversation history

        Args:
            conversation_id: Conversation ID

        Returns:
            Dict with conversation data
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

    def submit_feedback(
        self,
        message_id: str,
        feedback_type: str,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit feedback (like/dislike)

        Args:
            message_id: Message ID to give feedback on
            feedback_type: 'like' or 'dislike'
            comment: Optional comment

        Returns:
            Dict with success status
        """
        try:
            FeedbackManager.save_feedback(
                message_id,
                feedback_type,
                comment
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


# Singleton instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """
    Get or create LLM client singleton

    Returns:
        LLMClient instance
    """
    global _llm_client

    if _llm_client is None:
        _llm_client = LLMClient()

    return _llm_client
