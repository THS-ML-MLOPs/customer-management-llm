"""
================================================================================
    CHURN DASHBOARD VIEW
    Opens dashboard from local path using webbrowser
================================================================================
"""

import streamlit as st
import webbrowser
import os
from pathlib import Path

# ============================================================================
# AUTO-DETECT DASHBOARD PATH
# ============================================================================

def get_dashboard_path():
    """Auto-detect dashboard path based on environment"""
    try:
        import google.colab
        # Colab
        return Path("/content/drive/MyDrive/ML_Projects/Customer_Management/dashboards/01_churn_risk/dashboard_complete_churn_risk.html")
    except:
        # Local: navigate from ui_files/views to project root
        project_root = Path(__file__).parent.parent.parent
        return project_root / "dashboards" / "01_churn_risk" / "dashboard_complete_churn_risk.html"

DASHBOARD_PATH = get_dashboard_path()

# ============================================================================

def detect_environment():
    """
    Detect if running on Colab or local machine
    """
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'

def open_dashboard_local(path):
    """
    Open dashboard in default browser
    Returns True if successful, False otherwise
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            return False, f"File not found: {abs_path}"
        
        # Open in browser
        webbrowser.open(f'file:///{abs_path}')
        
        return True, "Dashboard opened in browser!"
        
    except Exception as e:
        return False, f"Error opening dashboard: {str(e)}"

def render():
    """
    Render churn dashboard view
    """
    
    # Back button
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button("← Back", key="back_to_home", type="secondary"):
            st.session_state.current_view = 'home'
            st.rerun()
    
    with col2:
        st.markdown("### 🔴 Churn Risk Analysis Dashboard")
    
    st.markdown("---")
    
    # Detect environment
    env = detect_environment()
    
    if env == 'local':
        # LOCAL MACHINE - Use webbrowser.open()
        st.markdown("""
        ### 🚀 Opening Dashboard Locally

        The dashboard will open in your default browser.
        """)

        # Show path being used
        with st.expander("📁 Dashboard location"):
            st.code(str(DASHBOARD_PATH))

        # Open button
        col_a, col_b, col_c = st.columns([1, 2, 1])

        with col_b:
            if st.button("🌐 Open Dashboard in Browser", type="primary", use_container_width=True):
                success, message = open_dashboard_local(str(DASHBOARD_PATH))
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("💡 Check your browser - a new tab should have opened!")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
                    st.warning("⚠️ Make sure the path is correct and the file exists.")
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        ### 📋 How it works:
        
        1. Click the button above
        2. Dashboard opens in your default browser
        3. All 8 charts load instantly! ⚡
        4. Return to this page anytime using the Back button
        
        **Note:** The dashboard opens in a new browser tab outside of Streamlit.
        """)
        
        # Troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **Dashboard not opening?**

            1. **Check the path:** Make sure this file exists:
```
               {DASHBOARD_PATH}
```
            
            2. **Update the path:** If your dashboard is in a different location,
               edit the path at the top of `churn_dashboard.py`
            
            3. **Check browser:** Make sure you have a default browser set
            
            4. **Manual open:** You can also open the file directly:
               - Navigate to the folder in File Explorer
               - Double-click `dashboard_complete_churn_risk.html`
            """)
    
    else:
        # COLAB - Fallback to inline view
        st.warning("⚠️ Running on Colab - local path not available")
        st.info("💡 Using inline view instead (may be slow)")
        
        dashboard_path = Path(COLAB_DASHBOARD_PATH)
        
        if dashboard_path.exists():
            with st.spinner("Loading dashboard..."):
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    dashboard_html = f.read()
                
                st.components.v1.html(dashboard_html, height=3500, scrolling=True)
        else:
            st.error("❌ Dashboard not found in Colab path")
