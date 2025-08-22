"""
Styling utilities for Product Review Intelligence System
Contains CSS, page configuration, and UI helper functions
"""

import streamlit as st
from typing import Dict, Any

def get_page_config() -> Dict[str, Any]:
    """Get Streamlit page configuration"""
    return {
        "page_title": "Product Review Intelligence",
        "page_icon": "🛍️",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }

def load_custom_css():
    """Load custom CSS for professional styling"""
    st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub header styling */
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Search container styling */
    .search-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Product card styling */
    .product-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border-left: 4px solid #28a745;
        transition: transform 0.2s ease-in-out;
        min-height: 200px;
    }
    
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .product-card h3 {
        margin-bottom: 1rem !important;
        line-height: 1.4 !important;
    }
    
    .product-card p {
        margin-bottom: 0.8rem !important;
        line-height: 1.5 !important;
    }
    
    /* Sidebar content styling */
    .sidebar-content {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Message styling */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    /* Info card with gradient */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .info-card h3 {
        margin-top: 0;
        color: white;
    }
    
    /* Rating badge styling */
    .rating-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .rating-badge.excellent {
        background-color: #28a745;
    }
    
    .rating-badge.good {
        background-color: #ffc107;
        color: #000;
    }
    
    .rating-badge.poor {
        background-color: #dc3545;
    }
    
    /* Sample review styling */
    .sample-review {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-style: italic;
        margin-top: 0.5rem;
        border-left: 3px solid #007bff;
        position: relative;
    }
    
    .sample-review::before {
        content: '"';
        font-size: 2rem;
        color: #007bff;
        position: absolute;
        left: 0.5rem;
        top: -0.5rem;
    }
    
    /* Analysis results styling */
    .analysis-result {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-top: 4px solid #007bff;
    }
    
    /* Stats container */
    .stats-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Loading spinner customization */
    .stSpinner > div > div {
        border-top-color: #667eea !important;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .search-container {
            padding: 1rem;
        }
        
        .product-card {
            padding: 1rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .product-card, .search-container, .analysis-result {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        .sample-review {
            background-color: #3b3b3b;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = "", help_text: str = ""):
    """Create a styled metric card"""
    delta_color = "normal"
    if delta.startswith("+"):
        delta_color = "normal"
    elif delta.startswith("-"):
        delta_color = "inverse"
    
    st.markdown(f"""
    <div class="metric-container">
        <h4 style="margin: 0; color: #1f77b4;">{title}</h4>
        <h2 style="margin: 0.5rem 0; color: #333;">{value}</h2>
        <small style="color: #666;">{delta}</small>
    </div>
    """, unsafe_allow_html=True)

def create_info_card(title: str, content: str, icon: str = "ℹ️"):
    """Create an information card with gradient background"""
    st.markdown(f"""
    <div class="info-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_success_message(message: str):
    """Create a success message"""
    st.markdown(f"""
    <div class="success-message">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)

def create_warning_message(message: str):
    """Create a warning message"""
    st.markdown(f"""
    <div class="warning-message">
        ⚠️ {message}
    </div>
    """, unsafe_allow_html=True)

def create_error_message(message: str):
    """Create an error message"""
    st.markdown(f"""
    <div class="error-message">
        ❌ {message}
    </div>
    """, unsafe_allow_html=True)

def create_rating_badge(rating: float) -> str:
    """Create a rating badge with appropriate color"""
    if rating >= 4.0:
        css_class = "excellent"
        emoji = "⭐"
    elif rating >= 3.0:
        css_class = "good"
        emoji = "👍"
    else:
        css_class = "poor"
        emoji = "👎"
    
    return f"""
    <span class="rating-badge {css_class}">
        {emoji} {rating:.1f}/5
    </span>
    """

def display_loading_message(message: str):
    """Display a loading message with spinner"""
    with st.spinner(f"🔄 {message}"):
        return True

def format_large_number(number: int) -> str:
    """Format large numbers with commas"""
    return f"{number:,}"

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    sentiment_emojis = {
        "POSITIVE": "😊",
        "NEGATIVE": "😞", 
        "NEUTRAL": "😐",
        "positive": "😊",
        "negative": "😞",
        "neutral": "😐"
    }
    return sentiment_emojis.get(sentiment, "😐")

def get_risk_indicator(risk_level: str, probability: float) -> str:
    """Get risk indicator with color and emoji"""
    if probability > 0.7:
        return "🔴 HIGH RISK"
    elif probability > 0.3:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"