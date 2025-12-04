"""
================================================================================
    CUSTOMER MANAGEMENT SYSTEM - MAIN APPLICATION
    Entry point and router for all views
================================================================================
"""

import streamlit as st
import sys
from pathlib import Path

# Add ui_files to Python path (works on local and Colab)
UI_PATH = Path(__file__).parent
sys.path.insert(0, str(UI_PATH))

# Import configurations
from ui_files.config.styles import load_styles
from ui_files.config.paths import PATHS

# Import utilities
from ui_files.utils.session_state import init_session_state, get_current_view

# Import views
from ui_files.views import home, churn_dashboard

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
        st.info("🚧 Sales Prediction - Coming in Phase 4!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()
    
    elif current_view == 'target2':
        st.info("🚧 Return Prediction - Coming in Phase 4!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()
    
    elif current_view == 'target3':
        st.info("🚧 Understanding Sales Behavior - Coming in Phase 4!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()
    
    elif current_view == 'target4':
        st.info("🚧 Understanding Sales Patterns - Coming in Phase 4!")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()
    
    else:
        st.session_state.current_view = 'home'
        st.rerun()

if __name__ == "__main__":
    main()
