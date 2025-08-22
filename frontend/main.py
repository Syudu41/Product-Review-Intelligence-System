"""
Product Review Intelligence System - Single Page Application
Clean navigation with everything in the main content area
"""

import streamlit as st
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add frontend to path for local imports
frontend_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(frontend_root)

# Import page modules
from pages import home, analytics, review_intel, recommendations, insights
from components.api_client import APIClient
from utils.styling import load_custom_css, get_page_config

# Configure Streamlit page
st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar
)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Initialize API client
@st.cache_resource
def get_api_client():
    """Initialize and cache API client"""
    return APIClient()

def main():
    """Main application entry point"""
    # Load custom styling
    load_custom_css()
    
    # Initialize API client
    api_client = get_api_client()
    
    # Hide sidebar completely
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {display: none !important;}
        .main > div {padding-top: 2rem;}
        div[data-testid="stSidebarNav"] {display: none !important;}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("# 🛍️ Product Review Intelligence")
    st.markdown("#### *AI-Powered Product Analysis & Recommendation System*")
    st.markdown("")  # Add space
    
    # Navigation tabs
    create_navigation_tabs()
    
    # System status indicator  
    display_system_status_header(api_client)
    
    # Route to selected page
    route_to_page(api_client)

def create_navigation_tabs():
    """Create navigation tabs in main content"""
    # Navigation tabs with clean spacing
    tab_col1, tab_col2, tab_col3, tab_col4, tab_col5 = st.columns(5)
    
    with tab_col1:
        if st.button("🏠 Home & Search", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "Home" else "secondary",
                    key="nav_home"):
            st.session_state.current_page = "Home"
            st.rerun()
    
    with tab_col2:
        if st.button("📊 Product Analytics", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Analytics" else "secondary",
                    key="nav_analytics"):
            st.session_state.current_page = "Analytics"
            st.rerun()
    
    with tab_col3:
        if st.button("🔍 Review Intelligence", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Reviews" else "secondary",
                    key="nav_reviews"):
            st.session_state.current_page = "Reviews"
            st.rerun()
    
    with tab_col4:
        if st.button("🎯 Recommendations", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Recommendations" else "secondary",
                    key="nav_recommendations"):
            st.session_state.current_page = "Recommendations"
            st.rerun()
    
    with tab_col5:
        if st.button("📈 Insights Dashboard", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Insights" else "secondary",
                    key="nav_insights"):
            st.session_state.current_page = "Insights"
            st.rerun()
    
    st.markdown("---")

def display_system_status_header(api_client):
    """Display system status in header"""
    st.markdown("#### 🔄 System Status")
    
    status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
    
    try:
        health_status = api_client.check_health()
        
        with status_col1:
            if health_status.get("healthy"):
                st.success("✅ API Online", icon="🔗")
            else:
                st.error("❌ API Offline", icon="🔗")
        
        with status_col2:
            st.success("✅ Database Ready", icon="🗄️")
        
        with status_col3:
            models = health_status.get("models", {})
            if models.get("sentiment_analyzer"):
                st.success("✅ Sentiment AI", icon="🎯")
            else:
                st.warning("⚠️ Sentiment AI", icon="🎯")
        
        with status_col4:
            if models.get("fake_detector"):
                st.success("✅ Fake Detection", icon="🕵️")
            else:
                st.warning("⚠️ Fake Detection", icon="🕵️")
        
        with status_col5:
            if models.get("recommendation_engine"):
                st.success("✅ Recommendations", icon="🎨")
            else:
                st.warning("⚠️ Recommendations", icon="🎨")
                
    except Exception as e:
        with status_col1:
            st.error("❌ System Error", icon="⚠️")
        with status_col2:
            st.caption(f"Error: {str(e)[:50]}...")
    
    st.markdown("---")

def route_to_page(api_client):
    """Route to the selected page"""
    try:
        if st.session_state.current_page == "Home":
            home.show_page(api_client)
        elif st.session_state.current_page == "Analytics":
            analytics.show_page(api_client)
        elif st.session_state.current_page == "Reviews":
            review_intel.show_page(api_client)
        elif st.session_state.current_page == "Recommendations":
            recommendations.show_page(api_client)
        elif st.session_state.current_page == "Insights":
            insights.show_page(api_client)
        else:
            home.show_page(api_client)
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.info("Please refresh the page or try a different page.")
        
        # Fallback to home
        if st.button("🏠 Return to Home"):
            st.session_state.current_page = "Home"
            st.rerun()

if __name__ == "__main__":
    main()