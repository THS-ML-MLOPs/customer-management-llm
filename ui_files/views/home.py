"""
================================================================================
    HOME VIEW
    Homepage with chat interface integrated
================================================================================
"""

import streamlit as st
from components.sidebar import render_sidebar
from components.action_cards import render_action_cards
from components.chat_interface import render_chat_interface

def render():
    """
    Render homepage with integrated chat
    """
    
    # Render sidebar
    render_sidebar()
    
    # Check if we should show welcome or chat
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
    
    if not st.session_state.show_chat:
        # Show welcome screen
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💬</div>
            <h1 class="welcome-title">Welcome to Customer Management System</h1>
            <p class="welcome-subtitle">Ask questions about your customer data, generate insights, and make data-driven decisions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action cards
        render_action_cards()
        
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        
        # Chat input (triggers chat mode)
        col_input, col_button = st.columns([9, 1])
        
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Type your question about customer data...",
                label_visibility="collapsed",
                key="welcome_chat_input"
            )
        
        with col_button:
            if st.button("➤", key="welcome_send_btn", type="primary", use_container_width=True):
                if user_input:
                    # Switch to chat mode
                    st.session_state.show_chat = True
                    st.session_state.first_message = user_input
                    st.rerun()
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #555555; font-size: 12px; margin-top: 20px;">
            LLM can make mistakes. Verify important information.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Show full chat interface
        render_chat_interface()
        
        # Process first message if exists
        if 'first_message' in st.session_state:
            # This will be processed by chat_interface
            st.session_state.chat_input_field = st.session_state.first_message
            del st.session_state.first_message
