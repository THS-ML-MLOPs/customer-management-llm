"""
================================================================================
    SESSION STATE UTILITIES
    Manage Streamlit session state for navigation and data persistence
================================================================================
"""

import streamlit as st

def init_session_state():
    """
    Initialize session state variables
    """
    
    # Navigation state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'home'
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # User input
    if 'current_input' not in st.session_state:
        st.session_state.current_input = ""
    
    # LLM processing state
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    # Dashboard loading state
    if 'loading_dashboard' not in st.session_state:
        st.session_state.loading_dashboard = False

def reset_session():
    """
    Reset session state
    """
    st.session_state.chat_history = []
    st.session_state.current_input = ""
    st.session_state.current_view = 'home'
    st.session_state.loading_dashboard = False

def get_current_view():
    """
    Get current view name
    """
    return st.session_state.get('current_view', 'home')

def set_current_view(view_name):
    """
    Set current view and trigger rerun
    """
    st.session_state.current_view = view_name
    st.rerun()

def add_chat_message(role, content):
    """
    Add message to chat history
    """
    st.session_state.chat_history.append({
        'role': role,
        'content': content
    })
