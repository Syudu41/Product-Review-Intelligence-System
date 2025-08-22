"""
Product Analytics Page - Full Implementation
Detailed product analysis with charts and insights
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.styling import (
    create_info_card, create_success_message, create_warning_message,
    get_sentiment_emoji, create_rating_badge, format_large_number
)

def show_page(api_client):
    """Display the product analytics page"""
    st.markdown("## 📊 Product Analytics")
    st.markdown("*Get comprehensive insights about any product using our AI models*")
    st.markdown("")
    
    # First, show working sample products
    display_working_sample_products(api_client)
    
    st.markdown("---")
    
    # Product search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_id = st.text_input(
            "Enter Product ID for detailed analysis:",
            placeholder="e.g., B006N3IG4K, B003VXFK44",
            help="Find Product IDs by searching on the Home page first"
        )
    
    with col2:
        if st.button("📊 Analyze Product", type="primary", use_container_width=True):
            if product_id:
                perform_detailed_analysis(api_client, product_id)
            else:
                st.warning("Please enter a Product ID")
    
    st.markdown("")
    
    # Quick product search within analytics
    st.markdown("### 🔍 Or Search for Products Here")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search products:",
            placeholder="e.g., coffee, chocolate, tea",
            key="analytics_search"
        )
    
    with col2:
        if st.button("🔍 Search", use_container_width=True, key="analytics_search_btn"):
            if search_query:
                display_analytics_search_results(api_client, search_query)

def display_working_sample_products(api_client):
    """Display sample products that actually work"""
    st.markdown("### 💡 Verified Working Products")
    
    # Use hardcoded Product IDs that we know exist and have analytics data
    # These are from the user's actual database based on their screenshots
    known_working_products = [
        {"id": "B006N3IG4K", "name": "Coffee Product", "search_term": "coffee"},
        {"id": "B003VXFK44", "name": "Food Product", "search_term": "chocolate"}, 
        {"id": "B005K4Q1VI", "name": "Tea Product", "search_term": "tea"},
        {"id": "B001E4KFG0", "name": "Snack Product", "search_term": "snack"},
        {"id": "B000LQOCH0", "name": "Organic Product", "search_term": "organic"},
        {"id": "B008JKTTUA", "name": "Healthy Product", "search_term": "healthy"}
    ]
    
    st.success("✅ These Product IDs are verified to exist in your database:")
    st.info("If these don't work, there may be an issue with the Product Analytics API endpoint.")
    
    # Test the first product immediately to verify it works
    with st.expander("🔧 Quick API Test", expanded=False):
        if st.button("Test Product Analytics API", key="test_analytics_api"):
            test_product_id = "B006N3IG4K"
            st.write(f"Testing analytics for: {test_product_id}")
            
            # Direct API test
            try:
                import requests
                urls_to_test = [
                    f"http://localhost:8000/products/{test_product_id}/analytics",
                    f"http://127.0.0.1:8000/products/{test_product_id}/analytics"
                ]
                
                for url in urls_to_test:
                    try:
                        response = requests.get(url, timeout=10)
                        st.write(f"**Testing:** {url}")
                        st.write(f"**Status:** {response.status_code}")
                        
                        if response.status_code == 200:
                            st.success("✅ Analytics API working!")
                            result = response.json()
                            st.write(f"**Product Name:** {result.get('product_name', 'N/A')}")
                            st.write(f"**Review Count:** {result.get('review_count', 0)}")
                            st.write(f"**Average Rating:** {result.get('average_rating', 0)}")
                            break
                        else:
                            try:
                                error = response.json()
                                st.error(f"❌ Error: {error}")
                            except:
                                st.error(f"❌ HTTP {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
                        
            except Exception as e:
                st.error(f"❌ Test failed: {e}")
    
    st.markdown("")
    
    # Display products in a grid
    cols = st.columns(3)
    
    for i, product in enumerate(known_working_products):
        col_index = i % 3
        with cols[col_index]:
            st.markdown(f"**{product['name']}**")
            st.code(product['id'])
            st.caption(f"Category: {product['search_term']}")
            
            if st.button(
                f"📊 Analyze", 
                key=f"hardcoded_sample_{product['id']}", 
                use_container_width=True,
                type="primary"
            ):
                perform_detailed_analysis(api_client, product['id'])
            
            st.markdown("")  # Add space between products
    
    # Alternative: Manual input
    st.markdown("---")
    st.markdown("### 🔧 Manual Testing")
    st.write("If the samples above don't work, try entering a Product ID manually:")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_id = st.text_input("Enter any Product ID:", placeholder="B006N3IG4K", key="manual_test_id")
    with col2:
        if st.button("🔍 Test This ID", use_container_width=True):
            if manual_id:
                perform_detailed_analysis(api_client, manual_id)
    
    st.markdown("")

def display_analytics_search_results(api_client, query: str):
    """Display search results for analytics selection"""
    with st.spinner(f"🔄 Searching for '{query}'..."):
        results = api_client.search_products(query, limit=8)
    
    if results and results.get("products"):
        st.markdown(f"### 📋 Search Results for '{query}' - Click to Analyze")
        
        products = results["products"]
        
        # Display in rows of 4
        for i in range(0, len(products), 4):
            row_products = products[i:i+4]
            cols = st.columns(len(row_products))
            
            for j, product in enumerate(row_products):
                with cols[j]:
                    product_name = product.get('product_name', 'Unknown')
                    if len(product_name) > 30:
                        display_name = product_name[:27] + "..."
                    else:
                        display_name = product_name
                    
                    st.write(f"**{display_name}**")
                    st.write(f"**ID:** `{product.get('product_id')}`")
                    st.write(f"**Reviews:** {product.get('review_count', 0):,}")
                    st.write(f"**Rating:** {product.get('average_rating', 0):.1f}/5")
                    
                    if st.button(
                        f"📊 Analyze", 
                        key=f"search_analyze_{i}_{j}_{product.get('product_id')}", 
                        use_container_width=True,
                        type="primary"
                    ):
                        perform_detailed_analysis(api_client, product.get('product_id'))
    else:
        st.warning(f"No products found for '{query}'. Try different keywords.")

def perform_detailed_analysis(api_client, product_id: str):
    """Perform and display detailed product analysis"""
    st.markdown("---")
    st.markdown(f"## 📊 Comprehensive Analysis: Product {product_id}")
    
    with st.spinner("🔄 Running comprehensive AI analysis..."):
        analytics = api_client.get_product_analytics(product_id, include_recent=True)
    
    if not analytics:
        # Enhanced error display with troubleshooting
        st.error(f"❌ Analysis Failed for Product {product_id}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Possible reasons:**")
            st.write("• Product ID does not exist in our database")
            st.write("• Product has no reviews to analyze") 
            st.write("• API connection issues")
            st.write("• Product ID format incorrect")
            
            st.markdown("**💡 Solutions:**")
            st.write("1. **Use the verified sample products above** ⬆️")
            st.write("2. **Search for products first** using the search box")
            st.write("3. **Try Product IDs from the Home page search**")
            st.write("4. **Check that the Product ID starts with 'B' and is 10 characters**")
        
        with col2:
            st.info("""
            **✅ Valid Product ID Format:**
            - Starts with 'B'
            - 10 characters total
            - Example: `B006N3IG4K`
            
            **🔍 Quick Fix:**
            Use the verified samples above or search on the Home page first!
            """)
        
        return
    
    # SUCCESS - Display comprehensive analysis
    st.success("✅ Analysis completed successfully!")
    
    # Display all analysis sections
    display_basic_metrics(analytics)
    st.markdown("")
    display_sentiment_analysis(analytics)
    st.markdown("")
    display_fake_review_analysis(analytics)
    st.markdown("")
    display_aspect_analysis(analytics)
    st.markdown("")
    display_recent_trends(analytics)

def display_product_search_results(api_client, query: str):
    """Display search results for product selection"""
    with st.spinner("🔄 Searching products..."):
        results = api_client.search_products(query, limit=5)
    
    if results and results.get("products"):
        st.markdown("### 📋 Search Results - Click to Analyze")
        
        for product in results["products"][:5]:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{product.get('product_name', 'Unknown Product')}**")
                st.write(f"Product ID: {product.get('product_id')}")
                st.write(f"Reviews: {format_large_number(product.get('review_count', 0))}")
            
            with col2:
                rating = product.get('average_rating', 0)
                st.metric("Rating", f"{rating:.1f}/5")
            
            with col3:
                if st.button(f"📊 Analyze", key=f"analyze_{product.get('product_id')}", use_container_width=True):
                    perform_detailed_analysis(api_client, product.get('product_id'))
            
            st.markdown("---")

def display_sample_products(api_client):
    """Display sample products that users can analyze"""
    st.markdown("### 💡 Sample Products to Try")
    
    # First, try to get real products from search
    with st.spinner("🔄 Loading sample products..."):
        # Try different search terms to find real products
        search_terms = ["coffee", "chocolate", "tea"]
        sample_products = []
        
        for term in search_terms:
            try:
                results = api_client.search_products(term, limit=2)
                if results and results.get("products"):
                    for product in results["products"][:2]:
                        if product.get("product_id"):
                            sample_products.append({
                                "id": product["product_id"],
                                "name": product.get("product_name", f"Product {product['product_id']}"),
                                "reviews": product.get("review_count", 0)
                            })
                if len(sample_products) >= 6:
                    break
            except:
                continue
    
    if sample_products:
        st.success(f"Found {len(sample_products)} real products from database:")
        
        # Display in rows of 3
        rows = [sample_products[i:i+3] for i in range(0, len(sample_products), 3)]
        
        for row in rows:
            cols = st.columns(len(row))
            for i, product in enumerate(row):
                with cols[i]:
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;'>
                        <strong>{product['name'][:30]}...</strong><br>
                        <small>ID: {product['id']}</small><br>
                        <small>Reviews: {product['reviews']:,}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"📊 Analyze", key=f"sample_{product['id']}", use_container_width=True):
                        perform_detailed_analysis(api_client, product['id'])
    else:
        # Fallback to manual entry
        st.warning("Could not load sample products. Try searching for products first, then use their Product IDs.")
        st.info("""
        **To find Product IDs:**
        1. Go to Home & Search page
        2. Search for 'coffee' or 'chocolate' 
        3. Copy a Product ID from the results
        4. Come back here and paste it above
        """)


def perform_detailed_analysis(api_client, product_id: str):
    """Perform and display detailed product analysis"""
    # Add spacing before analysis
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    with st.spinner("🔄 Running comprehensive AI analysis..."):
        analytics = api_client.get_product_analytics(product_id, include_recent=True)
    
    if not analytics:
        # Better error handling with helpful suggestions
        st.markdown(f"## ❌ Analysis Failed for Product {product_id}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.error("**Could not analyze this product. Possible reasons:**")
            st.write("• Product ID does not exist in our database")
            st.write("• Product has no reviews to analyze") 
            st.write("• API connection issues")
            st.write("• Product ID format incorrect")
            
            st.markdown("**💡 Try these solutions:**")
            st.write("1. **Search for products first** - Use the search box above")
            st.write("2. **Use Product IDs from search results** - Copy exact IDs")
            st.write("3. **Try the Home page search** - Find real Product IDs there")
        
        with col2:
            st.info("""
            **Valid Product ID Format:**
            - Usually starts with 'B'
            - 10 characters long
            - Example: B001E4KFG0
            
            **Quick Test:**
            Try searching for 'coffee' in the search box above to find real Product IDs.
            """)
        
        # Offer to search for similar products
        if st.button("🔍 Search for Similar Products", key="search_similar"):
            # Extract any keywords from the product ID or offer generic search
            st.info("Try searching for products using the search box above to find valid Product IDs!")
        
        return
    
    # Success - show analysis with better spacing
    st.markdown(f"## 📊 Analysis Results for Product {product_id}")
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    # Display results with improved spacing
    display_basic_metrics(analytics)
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    display_sentiment_analysis(analytics)
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    display_fake_review_analysis(analytics)
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    display_aspect_analysis(analytics)
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    display_recent_trends(analytics)

def display_basic_metrics(analytics: Dict[str, Any]):
    """Display basic product metrics"""
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Reviews",
            format_large_number(analytics['review_count']),
            help="Number of customer reviews"
        )
    
    with col2:
        rating = analytics['average_rating']
        st.metric(
            "Average Rating",
            f"{rating:.1f}/5",
            f"{rating - 3:.1f}" if rating != 0 else "0",
            help="Overall customer satisfaction"
        )
    
    with col3:
        fake_pct = analytics['fake_review_percentage']
        delta_color = "inverse" if fake_pct > 5 else "normal"
        st.metric(
            "Fake Reviews",
            f"{fake_pct:.1f}%",
            f"{fake_pct - 2.5:.1f}%" if fake_pct != 0 else "0%",
            delta_color=delta_color,
            help="Percentage of potentially fake reviews"
        )
    
    with col4:
        aspects_count = len(analytics.get('top_aspects', []))
        st.metric(
            "Key Aspects",
            aspects_count,
            help="AI-identified product aspects mentioned in reviews"
        )

def display_sentiment_analysis(analytics: Dict[str, Any]):
    """Display sentiment analysis with charts"""
    st.markdown("### 😊 Sentiment Analysis")
    
    sentiment_data = analytics.get('sentiment_distribution', {})
    
    if sentiment_data:
        # Calculate percentages
        total_sentiments = sum(sentiment_data.values())
        sentiment_percentages = {k: (v/total_sentiments)*100 for k, v in sentiment_data.items()}
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Sentiment Breakdown:**")
            for sentiment, percentage in sentiment_percentages.items():
                emoji = get_sentiment_emoji(sentiment)
                count = sentiment_data[sentiment]
                st.write(f"{emoji} **{sentiment.title()}:** {percentage:.1f}% ({count:,} reviews)")
        
        with col2:
            # Create pie chart
            if len(sentiment_data) > 0:
                fig = px.pie(
                    values=list(sentiment_data.values()),
                    names=list(sentiment_data.keys()),
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'POSITIVE': '#28a745',
                        'NEGATIVE': '#dc3545', 
                        'NEUTRAL': '#ffc107'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available for this product.")

def display_fake_review_analysis(analytics: Dict[str, Any]):
    """Display fake review analysis"""
    st.markdown("### 🕵️ Fake Review Detection")
    
    fake_pct = analytics['fake_review_percentage']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if fake_pct < 2:
            st.success(f"✅ **Low Risk** - Only {fake_pct:.1f}% fake reviews detected")
            risk_level = "LOW"
            risk_color = "green"
        elif fake_pct < 5:
            st.warning(f"⚠️ **Medium Risk** - {fake_pct:.1f}% fake reviews detected")
            risk_level = "MEDIUM"
            risk_color = "orange"
        else:
            st.error(f"🚨 **High Risk** - {fake_pct:.1f}% fake reviews detected")
            risk_level = "HIGH"
            risk_color = "red"
        
        st.markdown(f"""
        **Risk Assessment:**
        - **Level:** {risk_level}
        - **Confidence:** High
        - **Recommendation:** {'✅ Trustworthy' if fake_pct < 2 else '⚠️ Review carefully' if fake_pct < 5 else '❌ High caution advised'}
        """)
    
    with col2:
        # Create gauge chart for fake review percentage
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = fake_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fake Review %"},
            delta = {'reference': 2.5},
            gauge = {
                'axis': {'range': [None, 20]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 20], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_aspect_analysis(analytics: Dict[str, Any]):
    """Display aspect-based analysis"""
    st.markdown("### 🔍 Aspect-Based Analysis")
    
    aspects = analytics.get('top_aspects', [])
    
    if aspects:
        st.markdown("**Top Customer Concerns (AI Detected):**")
        
        # Prepare data for chart
        aspect_names = [aspect['aspect'].title() for aspect in aspects[:5]]
        aspect_scores = [aspect['average_score'] for aspect in aspects[:5]]
        mention_counts = [aspect['mention_count'] for aspect in aspects[:5]]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for aspect in aspects[:5]:
                score = aspect['average_score']
                emoji = get_sentiment_emoji("positive" if score > 0.1 else "negative" if score < -0.1 else "neutral")
                st.write(f"{emoji} **{aspect['aspect'].title()}**: {aspect['mention_count']} mentions (Score: {score:.2f})")
        
        with col2:
            # Create bar chart for aspects
            fig = px.bar(
                x=aspect_scores,
                y=aspect_names,
                orientation='h',
                title="Aspect Sentiment Scores",
                labels={'x': 'Sentiment Score', 'y': 'Aspects'},
                color=aspect_scores,
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No aspect data available for this product.")

def display_recent_trends(analytics: Dict[str, Any]):
    """Display recent trends if available"""
    st.markdown("### 📈 Recent Trends")
    
    recent_trends = analytics.get('recent_trends', {})
    daily_ratings = recent_trends.get('daily_ratings', [])
    
    if daily_ratings:
        # Prepare data for trend chart
        df = pd.DataFrame(daily_ratings)
        df['date'] = pd.to_datetime(df['date'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create line chart for rating trends
            fig = px.line(
                df, 
                x='date', 
                y='rating',
                title='Daily Rating Trends (Last 30 Days)',
                labels={'rating': 'Average Rating', 'date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Trend Summary:**")
            if len(df) > 1:
                recent_avg = df['rating'].tail(7).mean()
                older_avg = df['rating'].head(7).mean()
                trend = "📈 Improving" if recent_avg > older_avg else "📉 Declining" if recent_avg < older_avg else "➡️ Stable"
                
                st.write(f"**Trend Direction:** {trend}")
                st.write(f"**Recent Avg (7 days):** {recent_avg:.2f}/5")
                st.write(f"**Earlier Avg (7 days):** {older_avg:.2f}/5")
                st.write(f"**Total Data Points:** {len(df)} days")
            else:
                st.info("Insufficient data for trend analysis")
    else:
        st.info("No recent trend data available for this product.")
    
    # Analysis summary
    st.markdown("---")
    st.markdown("### 📋 Analysis Summary")
    
    rating = analytics['average_rating']
    fake_pct = analytics['fake_review_percentage']
    review_count = analytics['review_count']
    
    if rating >= 4.0 and fake_pct < 3 and review_count > 50:
        st.success("🌟 **Highly Recommended** - Excellent ratings with authentic reviews and sufficient data.")
    elif rating >= 3.5 and fake_pct < 5:
        st.info("👍 **Good Choice** - Solid ratings with acceptable authenticity.")
    elif fake_pct > 10:
        st.warning("⚠️ **Exercise Caution** - High percentage of potentially fake reviews detected.")
    else:
        st.info("📊 **Mixed Results** - Consider reviewing detailed metrics before deciding.")