"""
================================================================================
    STYLES CONFIGURATION
    Custom CSS styles for Customer Management System
    Dark theme - 100% approved
================================================================================
"""

import streamlit as st

def load_styles():
    """
    Load custom CSS styles for the application
    This includes dark theme, sidebar styling, cards, buttons, etc.
    """
    
    st.markdown("""
    <style>
        /* Main background */
        .main {
            background-color: #0E1117;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            padding-top: 0rem !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem !important;
        }
        
        /* Sidebar header */
        .sidebar-header {
            color: #FFFFFF;
            font-size: 20px;
            font-weight: 600;
            padding: 5px 0 10px 0;
            margin-top: 0px !important;
            margin-bottom: 20px;
            line-height: 1.3;
        }
        
        /* Section titles */
        .sidebar-section {
            color: #888888;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        /* Buttons - white background */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: none;
            border-radius: 6px;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #CCCCCC !important;
            color: #FFFFFF !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"] p {
            color: #000000 !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover p {
            color: #FFFFFF !important;
        }
        
        /* New Chat button - red */
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #FF4B4B !important;
            color: #FFFFFF !important;
        }
        
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #FF6B6B !important;
            color: #FFFFFF !important;
        }
        
        /* Welcome section */
        .welcome-container {
            text-align: center;
            padding: 40px 20px;
            margin-bottom: 40px;
        }
        
        .welcome-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 20px;
        }
        
        .welcome-title {
            color: #FFFFFF;
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .welcome-subtitle {
            color: #888888;
            font-size: 16px;
        }
        
        /* Action cards */
        .action-card {
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 24px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid transparent;
            margin-bottom: 20px;
        }
        
        .action-card:hover {
            background-color: #252525;
            border-color: #3E3E3E;
            transform: translateY(-2px);
        }
        
        .card-icon {
            width: 48px;
            height: 48px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 15px;
        }
        
        .card-title {
            color: #FFFFFF;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .card-description {
            color: #888888;
            font-size: 14px;
            line-height: 1.5;
        }
        
        /* User info inside sidebar */
        .user-info-container {
            background-color: #262626;
            border-radius: 8px;
            padding: 12px;
            margin-top: 40px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .user-details {
            flex: 1;
        }
        
        .user-name {
            color: #FFFFFF;
            font-size: 14px;
            font-weight: 500;
            margin: 0;
            padding: 0;
            line-height: 1.4;
        }
        
        .user-status {
            color: #888888;
            font-size: 12px;
            margin: 0;
            padding: 0;
            line-height: 1.4;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
