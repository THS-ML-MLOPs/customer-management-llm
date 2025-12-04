"""
================================================================================
    TARGET 3: SALES PREDICTION UI
    Streamlit interface for product/store sales predictions
================================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, List

# Page config
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
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
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
        margin-bottom: 1rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
        text-align: center;
    }
    .high-confidence { background-color: #90EE90; color: #006400; }
    .medium-confidence { background-color: #FFD700; color: #8B4513; }
    .low-confidence { background-color: #FFB6C1; color: #8B0000; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_products() -> pd.DataFrame:
    """Load available products from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={
                "natural_language_query": "Show all products with category and subcategory",
                "limit": 1000
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return pd.DataFrame(data['data'])
    except Exception as e:
        st.error(f"Error loading products: {str(e)}")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_stores() -> pd.DataFrame:
    """Load available stores from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={
                "natural_language_query": "Show all stores with country and city",
                "limit": 100
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return pd.DataFrame(data['data'])
    except Exception as e:
        st.error(f"Error loading stores: {str(e)}")
    return pd.DataFrame()

def get_historical_data(product_id: str, store_id: str) -> Optional[pd.DataFrame]:
    """Get historical sales data for product/store combination"""
    try:
        query = f"Show daily sales for product {product_id} at store {store_id} for last 90 days"
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={
                "natural_language_query": query,
                "limit": 100
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success' and len(data['data']) > 0:
                return pd.DataFrame(data['data'])
    except Exception as e:
        st.warning(f"No historical data available: {str(e)}")
    return None

def make_prediction(product_id: str, store_id: str, date: str) -> Dict:
    """
    Make sales prediction (mock function - replace with actual API call)
    """
    # TODO: Replace with actual prediction API endpoint when implemented
    # For now, generate realistic mock predictions
    import random
    
    base_value = random.uniform(100, 10000)
    confidence = random.uniform(0.7, 0.95)
    
    return {
        "status": "success",
        "prediction": {
            "product_id": product_id,
            "store_id": store_id,
            "date": date,
            "predicted_sales": round(base_value, 2),
            "confidence": round(confidence, 3),
            "lower_bound": round(base_value * 0.8, 2),
            "upper_bound": round(base_value * 1.2, 2),
            "factors": {
                "seasonality": random.choice(["high", "medium", "low"]),
                "trend": random.choice(["increasing", "stable", "decreasing"]),
                "day_of_week_effect": random.uniform(-0.2, 0.3)
            }
        }
    }

def create_prediction_chart(historical_df: Optional[pd.DataFrame], prediction: Dict) -> go.Figure:
    """Create visualization with historical data and prediction"""
    fig = go.Figure()
    
    # Add historical data if available
    if historical_df is not None and not historical_df.empty:
        # Assume historical df has 'date' and 'sales' columns
        date_col = [c for c in historical_df.columns if 'date' in c.lower()]
        sales_col = [c for c in historical_df.columns if any(x in c.lower() for x in ['sales', 'revenue', 'quantity'])]
        
        if date_col and sales_col:
            fig.add_trace(go.Scatter(
                x=historical_df[date_col[0]],
                y=historical_df[sales_col[0]],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
    
    # Add prediction point
    pred_data = prediction['prediction']
    fig.add_trace(go.Scatter(
        x=[pred_data['date']],
        y=[pred_data['predicted_sales']],
        mode='markers',
        name='Prediction',
        marker=dict(
            size=20,
            color='#FF4B4B',
            symbol='star',
            line=dict(color='white', width=2)
        )
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[pred_data['date'], pred_data['date']],
        y=[pred_data['lower_bound'], pred_data['upper_bound']],
        mode='lines',
        name='Confidence Interval',
        line=dict(color='rgba(255, 75, 75, 0.3)', width=0),
        fill='toself',
        fillcolor='rgba(255, 75, 75, 0.2)'
    ))
    
    fig.update_layout(
        title='Sales Prediction with Historical Context',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def export_to_csv(prediction: Dict, filename: str = "prediction.csv"):
    """Export prediction to CSV"""
    pred_data = prediction['prediction']
    df = pd.DataFrame([pred_data])
    csv = df.to_csv(index=False)
    return csv

# ============================================================================
# Main UI
# ============================================================================

# Header
st.markdown('<div class="main-header">📈 Sales Prediction System</div>', unsafe_allow_html=True)

# Sidebar - Input Controls
with st.sidebar:
    st.header("🎯 Prediction Parameters")
    
    # Load data
    with st.spinner("Loading products..."):
        products_df = load_products()
    
    with st.spinner("Loading stores..."):
        stores_df = load_stores()
    
    # Product selection
    st.subheader("1️⃣ Select Product")
    
    if not products_df.empty:
        # Category filter
        categories = sorted(products_df['category'].unique()) if 'category' in products_df.columns else []
        if categories:
            selected_category = st.selectbox("Category", ["All"] + categories)
            
            # Filter products by category
            if selected_category != "All":
                filtered_products = products_df[products_df['category'] == selected_category]
            else:
                filtered_products = products_df
        else:
            filtered_products = products_df
        
        # Product selector
        if 'product_id' in filtered_products.columns:
            product_options = filtered_products['product_id'].tolist()
            
            # Create display names if description available
            if 'description' in filtered_products.columns:
                product_display = {
                    row['product_id']: f"{row['product_id']} - {row.get('description', 'N/A')[:50]}"
                    for _, row in filtered_products.iterrows()
                }
            else:
                product_display = {pid: pid for pid in product_options}
            
            selected_product = st.selectbox(
                "Product",
                options=product_options,
                format_func=lambda x: product_display.get(x, x)
            )
        else:
            st.warning("No products available")
            selected_product = None
    else:
        st.error("Failed to load products")
        selected_product = None
    
    st.divider()
    
    # Store selection
    st.subheader("2️⃣ Select Store")
    
    if not stores_df.empty:
        # Country filter
        countries = sorted(stores_df['country'].unique()) if 'country' in stores_df.columns else []
        if countries:
            selected_country = st.selectbox("Country", ["All"] + countries)
            
            # Filter stores by country
            if selected_country != "All":
                filtered_stores = stores_df[stores_df['country'] == selected_country]
            else:
                filtered_stores = stores_df
        else:
            filtered_stores = stores_df
        
        # Store selector
        if 'store_id' in filtered_stores.columns:
            store_options = filtered_stores['store_id'].tolist()
            
            # Create display names
            if 'store_name' in filtered_stores.columns and 'city' in filtered_stores.columns:
                store_display = {
                    row['store_id']: f"{row['store_name']} - {row['city']}"
                    for _, row in filtered_stores.iterrows()
                }
            else:
                store_display = {sid: sid for sid in store_options}
            
            selected_store = st.selectbox(
                "Store",
                options=store_options,
                format_func=lambda x: store_display.get(x, x)
            )
        else:
            st.warning("No stores available")
            selected_store = None
    else:
        st.error("Failed to load stores")
        selected_store = None
    
    st.divider()
    
    # Date selection
    st.subheader("3️⃣ Select Date")
    
    prediction_date = st.date_input(
        "Prediction Date",
        value=datetime.now() + timedelta(days=7),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=365)
    )
    
    st.divider()
    
    # Predict button
    predict_button = st.button("🚀 Generate Prediction", type="primary", use_container_width=True)

# Main content area
if predict_button:
    if selected_product and selected_store:
        with st.spinner("Generating prediction..."):
            # Make prediction
            prediction = make_prediction(
                selected_product,
                selected_store,
                prediction_date.isoformat()
            )
            
            if prediction['status'] == 'success':
                pred_data = prediction['prediction']
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'product': selected_product,
                    'store': selected_store,
                    'date': prediction_date.isoformat(),
                    'prediction': pred_data['predicted_sales'],
                    'confidence': pred_data['confidence']
                })
                
                # Display results
                st.success("✅ Prediction generated successfully!")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Predicted Sales",
                        f"${pred_data['predicted_sales']:,.2f}"
                    )
                
                with col2:
                    confidence_pct = pred_data['confidence'] * 100
                    confidence_level = "High" if confidence_pct >= 85 else ("Medium" if confidence_pct >= 70 else "Low")
                    confidence_class = f"{confidence_level.lower()}-confidence"
                    
                    st.metric(
                        "Confidence",
                        f"{confidence_pct:.1f}%"
                    )
                    st.markdown(
                        f'<div class="confidence-badge {confidence_class}">{confidence_level}</div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.metric(
                        "Lower Bound",
                        f"${pred_data['lower_bound']:,.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Upper Bound",
                        f"${pred_data['upper_bound']:,.2f}"
                    )
                
                st.divider()
                
                # Chart section
                st.subheader("📊 Prediction Visualization")
                
                # Get historical data
                historical_df = get_historical_data(selected_product, selected_store)
                
                # Create and display chart
                fig = create_prediction_chart(historical_df, prediction)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Factors section
                st.subheader("🔍 Prediction Factors")
                
                factors_col1, factors_col2, factors_col3 = st.columns(3)
                
                with factors_col1:
                    st.info(f"**Seasonality:** {pred_data['factors']['seasonality'].title()}")
                
                with factors_col2:
                    trend_emoji = {"increasing": "📈", "stable": "➡️", "decreasing": "📉"}
                    st.info(f"**Trend:** {trend_emoji.get(pred_data['factors']['trend'], '')} {pred_data['factors']['trend'].title()}")
                
                with factors_col3:
                    dow_effect = pred_data['factors']['day_of_week_effect']
                    dow_text = f"+{dow_effect*100:.1f}%" if dow_effect > 0 else f"{dow_effect*100:.1f}%"
                    st.info(f"**Day Effect:** {dow_text}")
                
                st.divider()
                
                # Export section
                st.subheader("💾 Export Results")
                
                csv_data = export_to_csv(prediction)
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=f"prediction_{selected_product}_{selected_store}_{prediction_date}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_export2:
                    json_data = json.dumps(prediction, indent=2)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"prediction_{selected_product}_{selected_store}_{prediction_date}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.error(f"❌ Prediction failed: {prediction.get('error', 'Unknown error')}")
    else:
        st.warning("⚠️ Please select both product and store")

else:
    # Welcome screen
    st.info("""
    👋 **Welcome to the Sales Prediction System!**
    
    This tool helps you forecast future sales for specific product-store combinations.
    
    **How to use:**
    1. Select a product from the sidebar
    2. Choose a store location
    3. Pick a prediction date
    4. Click "Generate Prediction" to see results
    
    **Features:**
    - 📊 Historical data visualization
    - 🎯 Confidence intervals
    - 🔍 Prediction factors analysis
    - 💾 Export results to CSV/JSON
    """)
    
    # Show prediction history if exists
    if st.session_state.prediction_history:
        st.subheader("📜 Recent Predictions")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['confidence'] = (history_df['confidence'] * 100).round(1).astype(str) + '%'
        history_df['prediction'] = history_df['prediction'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            history_df[['timestamp', 'product', 'store', 'date', 'prediction', 'confidence']],
            use_container_width=True,
            hide_index=True
        )

# Footer
st.divider()
st.caption("🤖 Powered by Customer Management LLM System | Built with Streamlit")
