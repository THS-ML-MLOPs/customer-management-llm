"""
================================================================================
    SIDEBAR COMPONENT
    Reusable sidebar with navigation, templates, recent chats, and user info
================================================================================
"""

import streamlit as st

def render_sidebar():
    """
    Render sidebar with all navigation elements
    This sidebar is consistent across all views
    """
    
    with st.sidebar:
        # Header
        st.markdown(
            '<div class="sidebar-header">Customer Management<br>System</div>', 
            unsafe_allow_html=True
        )
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary", key="new_chat"):
            st.session_state.current_view = 'home'
            st.rerun()
        
        # Search
        st.text_input(
            "🔍 Search Chats", 
            placeholder="Search...", 
            label_visibility="collapsed", 
            key="search"
        )
        
        # Templates UI Section
        st.markdown('<div class="sidebar-section">TEMPLATES UI</div>', unsafe_allow_html=True)
        
        templates = [
            {"icon": "📊", "color": "#4A90E2", "title": "Sales Prediction", "view": "target1"},
            {"icon": "🔄", "color": "#9B59B6", "title": "Return Prediction", "view": "target2"},
            {"icon": "💡", "color": "#2ECC71", "title": "Understanding Sales Behavior", "view": "target3"},
            {"icon": "📈", "color": "#E67E22", "title": "Understanding Sales Patterns", "view": "target4"}
        ]
        
        for template in templates:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(
                    f'<div style="background-color: {template["color"]}; width: 32px; height: 32px; '
                    f'border-radius: 6px; display: flex; align-items: center; justify-content: center; '
                    f'font-size: 16px;">{template["icon"]}</div>',
                    unsafe_allow_html=True
                )
            with col2:
                if st.button(
                    template["title"], 
                    key=f"template_{template['title']}", 
                    use_container_width=True
                ):
                    st.session_state.current_view = template["view"]
                    st.rerun()
        
        # Recent Chats Section
        st.markdown('<div class="sidebar-section">RECENT CHATS</div>', unsafe_allow_html=True)
        
        all_chats = [
            "Customer segmentation an...",
            "Q4 sales forecast review",
            "Return rate by product cat...",
            "Customer lifetime value tren...",
            "Churn prediction model",
            "Regional performance metri...",
            "Customer feedback senti...",
            "Product recommendation in...",
            "Sales analysis Q3",
            "Market trends 2024"
        ]
        
        # Show only first 6 chats
        recent_chats = all_chats[:6]
        
        for i, chat in enumerate(recent_chats):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(
                    '<div style="width: 24px; height: 24px; display: flex; '
                    'align-items: center; justify-content: center; font-size: 14px;">💬</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.button(chat, key=f"chat_{i}_{chat}", use_container_width=True)
        
        # User info at bottom of sidebar
        st.markdown("""
        <div class="user-info-container">
            <div class="user-avatar">JA</div>
            <div class="user-details">
                <p class="user-name">John Anderson</p>
                <p class="user-status">Online</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
