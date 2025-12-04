"""
================================================================================
    TARGET 4: HISTORICAL ANALYSIS UI
    Streamlit interface for natural language data queries
================================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import json
import io
from typing import Optional, Dict, List

# Page config
st.set_page_config(
    page_title="Historical Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "https://precious-canonistic-ayden.ngrok-free.dev"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-card {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .sql-code {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9rem;
        border: 1px solid #dee2e6;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.5rem;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_chart' not in st.session_state:
    st.session_state.current_chart = None

# ============================================================================
# Helper Functions
# ============================================================================

def execute_query(natural_language_query: str, limit: int = 100) -> Dict:
    """Execute natural language query via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={
                "natural_language_query": natural_language_query,
                "limit": limit
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "error": f"API returned status {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def generate_chart_from_query(description: str, nl_query: str) -> Dict:
    """Generate chart from query via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chart",
            json={
                "chart_description": description,
                "natural_language_query": nl_query,
                "chart_type": None  # Auto-detect
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "error": f"API returned status {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def auto_create_chart(df: pd.DataFrame, query: str) -> Optional[go.Figure]:
    """Automatically create appropriate chart based on data structure"""
    if df.empty:
        return None
    
    try:
        # Detect numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID columns
        numeric_cols = [c for c in numeric_cols if not any(x in c.lower() for x in ['id', '_id'])]
        categorical_cols = [c for c in categorical_cols if not any(x in c.lower() for x in ['id', '_id'])]
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            # Bar chart: categorical vs numeric
            fig = px.bar(
                df.head(20),  # Limit to top 20 for readability
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0].title()} by {categorical_cols[0].title()}",
                template='plotly_white'
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
            return fig
            
        elif len(numeric_cols) >= 2:
            # Scatter plot: numeric vs numeric
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[1].title()} vs {numeric_cols[0].title()}",
                template='plotly_white'
            )
            fig.update_layout(height=500)
            return fig
            
        elif len(categorical_cols) >= 2:
            # Count plot
            counts = df[categorical_cols[0]].value_counts().head(15)
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                title=f"Distribution of {categorical_cols[0].title()}",
                labels={'x': categorical_cols[0].title(), 'y': 'Count'},
                template='plotly_white'
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
            return fig
            
    except Exception as e:
        st.warning(f"Could not auto-generate chart: {str(e)}")
    
    return None

def export_to_excel(df: pd.DataFrame, filename: str = "data.xlsx") -> bytes:
    """Export dataframe to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    return output.getvalue()

# ============================================================================
# Main UI
# ============================================================================

# Header
st.markdown('<div class="main-header">📊 Historical Analysis System</div>', unsafe_allow_html=True)

# Sidebar - Query Examples and History
with st.sidebar:
    st.header("💡 Query Examples")
    
    example_queries = [
        "Show total revenue by category",
        "Top 10 products by profit",
        "Which countries have the most customers?",
        "Average discount by product category",
        "Monthly revenue trends for 2024",
        "Stores with highest sales",
        "Products with most returns",
        "Customer distribution by age group"
    ]
    
    st.info("Click an example to try it:")
    
    for example in example_queries:
        if st.button(f"📌 {example}", key=f"example_{example}", use_container_width=True):
            st.session_state.selected_example = example
    
    st.divider()
    
    # Query history
    if st.session_state.query_history:
        st.header("📜 Query History")
        
        for i, hist in enumerate(reversed(st.session_state.query_history[-10:])):
            if st.button(
                f"🕐 {hist['query'][:40]}...",
                key=f"history_{i}",
                use_container_width=True
            ):
                st.session_state.selected_example = hist['query']

# Main content
col_query, col_settings = st.columns([3, 1])

with col_query:
    st.subheader("🔍 Natural Language Query")
    
    # Check if example was selected
    default_query = st.session_state.get('selected_example', '')
    if default_query:
        del st.session_state.selected_example
    
    query_text = st.text_area(
        "Enter your question in plain English:",
        value=default_query,
        height=100,
        placeholder="E.g., Show me total revenue by category"
    )

with col_settings:
    st.subheader("⚙️ Settings")
    
    result_limit = st.number_input(
        "Max Results",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    auto_chart = st.checkbox(
        "Auto-generate chart",
        value=True
    )

# Execute button
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

with col_btn1:
    execute_button = st.button("🚀 Execute Query", type="primary", use_container_width=True)

with col_btn2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.current_data = None
        st.session_state.current_chart = None
        st.rerun()

with col_btn3:
    if st.button("📜 History", use_container_width=True):
        pass  # Placeholder

st.divider()

# Execute query
if execute_button and query_text:
    with st.spinner("Executing query..."):
        result = execute_query(query_text, result_limit)
        
        if result['status'] == 'success':
            # Store in session state
            df = pd.DataFrame(result['data'])
            st.session_state.current_data = df
            
            # Add to history
            st.session_state.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query_text,
                'num_rows': len(df)
            })
            
            # Success message
            st.success(f"✅ Query executed successfully! Retrieved {len(df)} rows.")
            
            # Display metadata
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            
            with col_meta1:
                st.markdown(f'<div class="metric-badge">📊 {len(df)} rows</div>', unsafe_allow_html=True)
            
            with col_meta2:
                st.markdown(f'<div class="metric-badge">📋 {len(df.columns)} columns</div>', unsafe_allow_html=True)
            
            with col_meta3:
                exec_time = result.get('execution_time', 0)
                st.markdown(f'<div class="metric-badge">⏱️ {exec_time:.2f}s</div>', unsafe_allow_html=True)
            
            # Show SQL used
            if 'sql_used' in result:
                with st.expander("📝 View SQL Query"):
                    st.code(result['sql_used'], language='sql')
            
            st.divider()
            
            # Data table
            st.subheader("📋 Query Results")
            
            # Display options
            col_display1, col_display2 = st.columns([1, 3])
            
            with col_display1:
                display_rows = st.selectbox(
                    "Rows to display:",
                    [10, 25, 50, 100, "All"],
                    index=1
                )
            
            # Display dataframe
            display_df = df if display_rows == "All" else df.head(display_rows)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Auto-generate chart if enabled
            if auto_chart:
                st.divider()
                st.subheader("📊 Visualization")
                
                with st.spinner("Generating chart..."):
                    fig = auto_create_chart(df, query_text)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.current_chart = fig
                    else:
                        st.info("💡 Tip: For better visualizations, try queries that return numeric data grouped by categories.")
            
            st.divider()
            
            # Export section
            st.subheader("💾 Export Data")
            
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_export2:
                excel_data = export_to_excel(df)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_export3:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
        else:
            st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")

elif execute_button:
    st.warning("⚠️ Please enter a query")

# Show previous results if available
elif st.session_state.current_data is not None:
    df = st.session_state.current_data
    
    st.info(f"📊 Showing previous query results ({len(df)} rows)")
    
    st.dataframe(
        df.head(25),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    if st.session_state.current_chart:
        st.plotly_chart(st.session_state.current_chart, use_container_width=True)
    
    # Export buttons
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        excel_data = export_to_excel(df)
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_export3:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    # Welcome screen
    st.info("""
    👋 **Welcome to the Historical Analysis System!**
    
    This tool lets you explore historical sales data using natural language queries.
    
    **How to use:**
    1. Type your question in plain English (or click an example)
    2. Click "Execute Query" to see results
    3. View data in the table and auto-generated charts
    4. Export results to CSV, Excel, or JSON
    
    **Example queries:**
    - "Show me total revenue by category"
    - "Which products have the highest profit margins?"
    - "How many customers are there in each country?"
    - "What are the monthly sales trends?"
    
    **Features:**
    - 🔍 Natural language to SQL conversion
    - 📊 Automatic chart generation
    - 💾 Export to multiple formats
    - 📜 Query history tracking
    """)

# Footer
st.divider()
st.caption("🤖 Powered by Customer Management LLM System | Built with Streamlit")
