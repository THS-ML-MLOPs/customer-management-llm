"""
================================================================================
    LLM HOMEPAGE - CHURN WITH DIRECT LINK
================================================================================
"""

import streamlit as st

st.set_page_config(
    page_title="Customer Management System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; padding-top: 0rem !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }
    .sidebar-header { color: #FFFFFF; font-size: 20px; font-weight: 600; padding: 5px 0 10px 0; margin-top: 0px !important; margin-bottom: 20px; line-height: 1.3; }
    .sidebar-section { color: #888888; font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 30px; margin-bottom: 15px; }
    [data-testid="stSidebar"] button[kind="secondary"] { background-color: #FFFFFF !important; color: #000000 !important; border: none; border-radius: 6px; }
    [data-testid="stSidebar"] button[kind="secondary"]:hover { background-color: #CCCCCC !important; color: #FFFFFF !important; }
    [data-testid="stSidebar"] button[kind="secondary"] p { color: #000000 !important; }
    [data-testid="stSidebar"] button[kind="secondary"]:hover p { color: #FFFFFF !important; }
    [data-testid="stSidebar"] button[kind="primary"] { background-color: #FF4B4B !important; color: #FFFFFF !important; }
    [data-testid="stSidebar"] button[kind="primary"]:hover { background-color: #FF6B6B !important; color: #FFFFFF !important; }
    .welcome-container { text-align: center; padding: 40px 20px; margin-bottom: 40px; }
    .welcome-icon { width: 80px; height: 80px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 40px; margin-bottom: 20px; }
    .welcome-title { color: #FFFFFF; font-size: 32px; font-weight: 600; margin-bottom: 10px; }
    .welcome-subtitle { color: #888888; font-size: 16px; }
    .action-card { background-color: #1E1E1E; border-radius: 12px; padding: 24px; cursor: pointer; transition: all 0.3s; border: 1px solid transparent; margin-bottom: 20px; text-decoration: none; display: block; }
    .action-card:hover { background-color: #252525; border-color: #3E3E3E; transform: translateY(-2px); }
    .card-icon { width: 48px; height: 48px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; font-size: 24px; margin-bottom: 15px; }
    .card-title { color: #FFFFFF; font-size: 18px; font-weight: 600; margin-bottom: 8px; }
    .card-description { color: #888888; font-size: 14px; line-height: 1.5; }
    .user-info-container { background-color: #262626; border-radius: 8px; padding: 12px; margin-top: 40px; display: flex; align-items: center; gap: 12px; }
    .user-avatar { width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 16px; flex-shrink: 0; }
    .user-details { flex: 1; }
    .user-name { color: #FFFFFF; font-size: 14px; font-weight: 500; margin: 0; padding: 0; line-height: 1.4; }
    .user-status { color: #888888; font-size: 12px; margin: 0; padding: 0; line-height: 1.4; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Customer Management<br>System</div>', unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.rerun()
    st.text_input("🔍 Search Chats", placeholder="Search...", label_visibility="collapsed")
    st.markdown('<div class="sidebar-section">TEMPLATES UI</div>', unsafe_allow_html=True)
    
    templates = [
        {"icon": "📊", "color": "#4A90E2", "title": "Sales Prediction"},
        {"icon": "🔄", "color": "#9B59B6", "title": "Return Prediction"},
        {"icon": "💡", "color": "#2ECC71", "title": "Understanding Sales Behavior"},
        {"icon": "📈", "color": "#E67E22", "title": "Understanding Sales Patterns"}
    ]
    
    for template in templates:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f'<div style="background-color: {template["color"]}; width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 16px;">{template["icon"]}</div>', unsafe_allow_html=True)
        with col2:
            st.button(template["title"], key=template["title"], use_container_width=True)
    
    st.markdown('<div class="sidebar-section">RECENT CHATS</div>', unsafe_allow_html=True)
    recent_chats = ["Customer segmentation an...", "Q4 sales forecast review", "Return rate by product cat...", "Customer lifetime value tren...", "Churn prediction model", "Regional performance metri..."]
    
    for i, chat in enumerate(recent_chats):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<div style="width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 14px;">💬</div>', unsafe_allow_html=True)
        with col2:
            st.button(chat, key=f"chat_{i}", use_container_width=True)
    
    st.markdown('<div class="user-info-container"><div class="user-avatar">JA</div><div class="user-details"><p class="user-name">John Anderson</p><p class="user-status">Online</p></div></div>', unsafe_allow_html=True)

st.markdown('<div class="welcome-container"><div class="welcome-icon">💬</div><h1 class="welcome-title">Welcome to Customer Management System</h1><p class="welcome-subtitle">Ask questions about your customer data, generate insights, and make data-driven decisions</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="action-card"><div class="card-icon" style="background-color: rgba(74, 144, 226, 0.2);"><span style="color: #4A90E2;">📊</span></div><div class="card-title">Analyze Sales Trends</div><div class="card-description">Get insights on sales performance across different time periods and regions</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="action-card"><div class="card-icon" style="background-color: rgba(46, 204, 113, 0.2);"><span style="color: #2ECC71;">📈</span></div><div class="card-title">Predict Future Revenue</div><div class="card-description">Forecast revenue based on historical data and market trends</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="action-card"><div class="card-icon" style="background-color: rgba(155, 89, 182, 0.2);"><span style="color: #9B59B6;">👥</span></div><div class="card-title">Customer Segmentation</div><div class="card-description">Identify customer groups based on behavior, demographics, and purchase patterns</div></div>', unsafe_allow_html=True)
    
    # CHURN CARD - Using <a> tag for direct link
    st.markdown("""
    <a href="file:///C:/Users/thiag/Desktop/Project/01_churn_risk/dashboard_complete_churn_risk.html" target="_blank" class="action-card" style="text-decoration: none;">
        <div class="card-icon" style="background-color: rgba(231, 76, 60, 0.2);">
            <span style="color: #E74C3C;">🔴</span>
        </div>
        <div class="card-title">Identify Churn Risk</div>
        <div class="card-description">Detect customers at risk of churning and get retention strategies</div>
    </a>
    """, unsafe_allow_html=True)

st.markdown("<br>" * 2, unsafe_allow_html=True)

col_input, col_button = st.columns([9, 1])
with col_input:
    st.text_input("Message", placeholder="Type your question about customer data...", label_visibility="collapsed")
with col_button:
    st.button("➤", type="primary", use_container_width=True)

st.markdown('<div style="text-align: center; color: #555555; font-size: 12px; margin-top: 20px;">LLM can make mistakes. Verify important information.</div>', unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
