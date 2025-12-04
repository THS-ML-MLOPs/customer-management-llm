"""
================================================================================
    LLM CLIENT
    HTTP client for FastAPI backend
    Handles: Chat, Queries, Charts, Image Search
================================================================================
"""

import requests
import json
from typing import Dict, List, Optional, Any, Generator
from pathlib import Path
import base64
import time

class LLMClient:
    """
    Client for FastAPI LLM backend
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize LLM client
        
        Args:
            base_url: FastAPI server URL
        """
        self.base_url = base_url
        self.timeout = 120  # 2 minutes for complex queries
    
    def health_check(self) -> bool:
        """
        Check if API is alive
        
        Returns:
            bool: True if API is responsive
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def chat(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat message to LLM
        
        Args:
            message: User message
            conversation_id: Optional conversation ID for context
            stream: Whether to stream response
        
        Returns:
            Dict with response
        """
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out. The query may be too complex.",
                "success": False
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API error: {str(e)}",
                "success": False
            }
    
    def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream chat response
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
        
        Yields:
            Chunks of response text
        """
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "stream": True
        }
        
        try:
            with requests.post(
                f"{self.base_url}/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'chunk' in data:
                                yield data['chunk']
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            yield f"\n\n❌ Error: {str(e)}"
    
    def query_data(
        self,
        query: str,
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute DuckDB query
        
        Args:
            query: Natural language or SQL query
            parameters: Optional query parameters
        
        Returns:
            Dict with query results
        """
        payload = {
            "query": query,
            "parameters": parameters or {}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            return {
                "error": f"Query error: {str(e)}",
                "success": False
            }
    
    def generate_chart(
        self,
        data: Dict[str, Any],
        chart_type: str,
        title: str,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate Plotly chart
        
        Args:
            data: Chart data
            chart_type: Type of chart (bar, line, scatter, etc)
            title: Chart title
            config: Optional chart configuration
        
        Returns:
            Dict with chart HTML and metadata
        """
        payload = {
            "data": data,
            "chart_type": chart_type,
            "title": title,
            "config": config or {}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chart",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            return {
                "error": f"Chart generation error: {str(e)}",
                "success": False
            }
    
    def search_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search similar products by image using CLIP
        
        Args:
            image_path: Path to image file
            image_bytes: Image as bytes
            top_k: Number of results to return
        
        Returns:
            Dict with similar products
        """
        # Read image
        if image_path:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        elif image_bytes:
            image_data = base64.b64encode(image_bytes).decode('utf-8')
        else:
            return {"error": "No image provided", "success": False}
        
        payload = {
            "image": image_data,
            "top_k": top_k
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/search_image",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            return {
                "error": f"Image search error: {str(e)}",
                "success": False
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
            response = requests.get(
                f"{self.base_url}/conversation/{conversation_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            return {
                "error": f"Error retrieving conversation: {str(e)}",
                "success": False
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
        payload = {
            "message_id": message_id,
            "feedback_type": feedback_type,
            "comment": comment
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/feedback",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            return {
                "error": f"Feedback error: {str(e)}",
                "success": False
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
