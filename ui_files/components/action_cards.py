"""
================================================================================
    ACTION CARDS COMPONENT
================================================================================
"""

import streamlit as st

def render_action_cards():
    """
    Render 4 styled buttons
    """
    
    st.markdown("""
    <style>
        /* FORCE light blue background on ALL buttons */
        button[kind="secondary"] {
            background-color: #E3F2FD !important;
            color: #1565C0 !important;
            border: 2px solid #BBDEFB !important;
        }
        
        button[kind="secondary"]:hover {
            background-color: #BBDEFB !important;
            border-color: #90CAF9 !important;
        }
        
        /* Main content area buttons */
        .main button {
            background-color: #E3F2FD !important;
            color: #1565C0 !important;
            min-height: 180px !important;
            text-align: left !important;
            padding: 24px !important;
            border-radius: 12px !important;
            border: 2px solid #BBDEFB !important;
            white-space: pre-line !important;
        }
        
        .main button:hover {
            background-color: #BBDEFB !important;
            border-color: #90CAF9 !important;
            transform: translateY(-2px) !important;
        }
        
        .main button:disabled {
            opacity: 0.6 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.button(
            "📊  Analyze Sales Trends\n\nGet insights on sales performance",
            key="sales_card",
            disabled=True,
            use_container_width=True,
            type="secondary"
        )
        
        st.button(
            "📈  Predict Future Revenue\n\nForecast revenue based on data",
            key="revenue_card",
            disabled=True,
            use_container_width=True,
            type="secondary"
        )
    
    with col2:
        st.button(
            "👥  Customer Segmentation\n\nIdentify customer groups",
            key="segment_card",
            disabled=True,
            use_container_width=True,
            type="secondary"
        )
        
        if st.button(
            "🔴  Identify Churn Risk\n\nDetect at-risk customers",
            key="churn_card",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.current_view = 'churn'
            st.rerun()
