"""
================================================================================
    CUSTOMER MANAGEMENT SYSTEM - HUGGING FACE SPACES APP
    All-in-one Streamlit app with integrated backend
    No Ollama needed - uses HuggingFace Inference API
================================================================================
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configurations
from ui_files.config.styles import load_styles
from ui_files.utils.session_state import init_session_state, get_current_view

# Import views
from ui_files.views import home, churn_dashboard

# Import HF backend client
from llm_workspace.api_hf import (
    hf_client,
    duckdb_client,
    ConversationStorage,
    FeedbackManager
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Management System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZATION
# ============================================================================

# Load custom CSS
load_styles()

# Initialize session state
init_session_state()

# Check HF token
if not os.environ.get("HF_TOKEN"):
    st.error("""
    ⚠️ **HuggingFace Token não configurado!**

    Para usar este app, você precisa:
    1. Criar uma conta no HuggingFace (gratuito)
    2. Gerar um token em: https://huggingface.co/settings/tokens
    3. Configurar o token como Secret no HF Spaces

    **Para rodar localmente:**
    ```bash
    export HF_TOKEN="your_token_here"
    ```
    """)
    st.stop()

# ============================================================================
# FOOTER (minimal status)
# ============================================================================

# Minimal status in footer (not sidebar)
with st.sidebar:
    st.markdown("---")
    st.caption("🤖 Powered by AI")

# ============================================================================
# ROUTER
# ============================================================================

def main():
    """Main router function"""

    current_view = get_current_view()

    if current_view == 'home':
        home.render()

    elif current_view == 'churn':
        churn_dashboard.render()

    elif current_view == 'target1':
        st.info("🚧 Sales Prediction - Coming soon!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()

    elif current_view == 'target2':
        st.info("🚧 Return Prediction - Coming soon!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()

    elif current_view == 'target3':
        st.info("🚧 Understanding Sales Behavior - Coming soon!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()

    elif current_view == 'target4':
        st.info("🚧 Understanding Sales Patterns - Coming soon!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()

    else:
        st.session_state.current_view = 'home'
        st.rerun()

if __name__ == "__main__":
    main()
