
# Streamlit UI Component for Feedback

import streamlit as st

def render_feedback_buttons(
    conversation_id: str,
    message_id: str,
    assistant_message: str,
    user_message: str = None
):
    """
    Render like/dislike buttons below AI response
    
    Usage in Streamlit:
        render_feedback_buttons(conv_id, msg_id, ai_response)
    """
    
    col1, col2, col3 = st.columns([1, 1, 10])
    
    with col1:
        if st.button("👍", key=f"like_{message_id}"):
            feedback_manager.save_feedback(
                conversation_id=conversation_id,
                message_id=message_id,
                assistant_message=assistant_message,
                rating=FeedbackRating.LIKE,
                user_message=user_message
            )
            st.success("Thanks for your feedback!")
    
    with col2:
        if st.button("👎", key=f"dislike_{message_id}"):
            feedback_manager.save_feedback(
                conversation_id=conversation_id,
                message_id=message_id,
                assistant_message=assistant_message,
                rating=FeedbackRating.DISLIKE,
                user_message=user_message
            )
            
            # Optional: Ask for comment
            st.text_input(
                "What went wrong? (optional)",
                key=f"comment_{message_id}"
            )

# Example usage in chat interface:
st.chat_message("assistant").write(ai_response)
render_feedback_buttons(conv_id, msg_id, ai_response, user_msg)
