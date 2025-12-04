"""
================================================================================
    CHAT INTERFACE COMPONENT
    Full-featured chat UI with streaming, image upload, charts
================================================================================
"""

import streamlit as st
from ui_files.backend.llm_client import get_llm_client
import uuid
from datetime import datetime
from PIL import Image
import io

def render_chat_interface():
    """
    Render full chat interface with all features
    """
    
    # Initialize LLM client
    llm_client = get_llm_client()
    
    # Check API health
    if 'api_health_checked' not in st.session_state:
        with st.spinner("🔍 Checking API connection..."):
            api_healthy = llm_client.health_check()
            st.session_state.api_healthy = api_healthy
            st.session_state.api_health_checked = True
    
    # Show API status
    if not st.session_state.get('api_healthy', False):
        st.error("❌ API not responding. Make sure FastAPI server is running.")
        st.code("python -m uvicorn api:app --host 0.0.0.0 --port 8000")
        return
    
    # Initialize conversation
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat header
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("### 💬 Chat with AI Assistant")
    
    with col2:
        if st.button("🔄 New Chat", type="secondary"):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    
    with col3:
        if st.button("📥 Upload Image", type="secondary"):
            st.session_state.show_image_upload = not st.session_state.get('show_image_upload', False)
            st.rerun()
    
    st.markdown("---")
    
    # Image upload area
    if st.session_state.get('show_image_upload', False):
        st.markdown("#### 📸 Upload Product Image")
        
        uploaded_file = st.file_uploader(
            "Upload an image to search for similar products",
            type=['png', 'jpg', 'jpeg'],
            key="image_uploader"
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Search button
            if st.button("🔍 Find Similar Products", type="primary", use_container_width=True):
                with st.spinner("🔍 Searching similar products..."):
                    # Convert image to bytes
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    
                    # Search
                    result = llm_client.search_image(
                        image_bytes=img_bytes,
                        top_k=5
                    )
                    
                    if result.get('success'):
                        # Add to chat
                        st.session_state.messages.append({
                            'role': 'user',
                            'content': '[Image uploaded for similarity search]',
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Format results
                        products = result.get('products', [])
                        response_text = f"🎯 Found {len(products)} similar products:\n\n"
                        
                        for i, prod in enumerate(products, 1):
                            response_text += f"{i}. **Product ID**: {prod['product_id']} | "
                            response_text += f"**Similarity**: {prod['similarity']:.2%}\n"
                        
                        st.session_state.messages.append({
                            'role': 'assistant',
                            'content': response_text,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        st.session_state.show_image_upload = False
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")
        
        st.markdown("---")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
                    
                    # Show feedback buttons
                    if 'message_id' in msg:
                        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 10])
                        
                        with col_fb1:
                            if st.button("👍", key=f"like_{msg['message_id']}"):
                                llm_client.submit_feedback(
                                    msg['message_id'],
                                    'like'
                                )
                                st.toast("✅ Feedback submitted!")
                        
                        with col_fb2:
                            if st.button("👎", key=f"dislike_{msg['message_id']}"):
                                llm_client.submit_feedback(
                                    msg['message_id'],
                                    'dislike'
                                )
                                st.toast("✅ Feedback submitted!")
    
    # Chat input
    st.markdown("---")
    
    col_input, col_send = st.columns([9, 1])
    
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about sales, products, trends, or request charts...",
            label_visibility="collapsed",
            key="chat_input_field"
        )
    
    with col_send:
        send_button = st.button("➤", type="primary", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            
            with st.spinner("🤔 Thinking..."):
                # Call LLM
                result = llm_client.chat(
                    message=user_input,
                    conversation_id=st.session_state.conversation_id,
                    stream=False
                )
                
                if result.get('success'):
                    response_text = result.get('response', '')
                    message_id = result.get('message_id')
                    
                    response_placeholder.markdown(response_text)
                    
                    # Add to messages
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': response_text,
                        'message_id': message_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # If chart was generated, display it
                    if 'chart_html' in result:
                        st.components.v1.html(result['chart_html'], height=500)
                else:
                    error_msg = result.get('error', 'Unknown error')
                    response_placeholder.error(f"❌ {error_msg}")
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("💡 Tip: Ask me to analyze sales trends, generate charts, or search for products by image!")
