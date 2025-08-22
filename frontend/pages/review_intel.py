"""
Review Intelligence Page - Full Implementation
Individual review analysis with AI insights
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
from utils.styling import (
    create_info_card, create_success_message, create_warning_message,
    get_sentiment_emoji, get_risk_indicator, format_large_number
)

def show_page(api_client):
    """Display the review intelligence page"""
    st.markdown("## 🔍 Review Intelligence")
    
    # Review input section
    st.markdown("### 📝 Analyze Any Review")
    
    # Sample reviews for testing
    sample_reviews = {
        "Positive Food Review": "This coffee is absolutely amazing! The flavor is rich and smooth, perfect for morning brewing. I've been drinking coffee for 20 years and this is definitely in my top 3. The beans are freshly roasted and you can taste the quality. Highly recommend to anyone looking for premium coffee beans. Will definitely purchase again!",
        "Negative Food Review": "Terrible product. The coffee tastes burnt and bitter, nothing like the description. Packaging was damaged when it arrived and half the beans were spilled. Customer service was unhelpful when I complained. Save your money and buy something else. Very disappointed with this purchase.",
        "Neutral Food Review": "The coffee is okay, nothing special but not bad either. Taste is average for the price point. Packaging was fine and delivery was on time. It's decent if you just need basic coffee but there are probably better options available. Would consider other brands next time.",
        "Suspicious Review (Fake)": "Best product ever! Amazing quality and super fast shipping! Highly recommend! 5 stars! Will buy again! Great seller! Perfect! Excellent! Outstanding! Fantastic! Love it! Thank you! Best purchase ever made! Everyone should buy this!"
    }
    
    # Sample review selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_sample = st.selectbox(
            "Try a sample review:",
            ["Choose a sample..."] + list(sample_reviews.keys()),
            key="sample_review_selector"
        )
    
    with col2:
        if st.button("📝 Load Sample", use_container_width=True):
            if selected_sample != "Choose a sample...":
                st.session_state.review_text = sample_reviews[selected_sample]
                st.rerun()
    
    # Review text input
    review_text = st.text_area(
        "Enter review text for AI analysis:",
        value=st.session_state.get('review_text', ''),
        placeholder="Type or paste a product review here...",
        height=150,
        help="Our AI will analyze sentiment, detect fake patterns, and extract key aspects",
        key="review_input"
    )
    
    # Analysis options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🧹 Clear Text", use_container_width=True):
            st.session_state.review_text = ""
            st.rerun()
    
    with col3:
        include_aspects = st.checkbox("Include detailed aspect analysis", value=True)
    
    # Perform analysis
    if analyze_button and review_text.strip():
        perform_review_analysis(api_client, review_text.strip(), include_aspects)
    elif analyze_button and not review_text.strip():
        st.warning("Please enter review text to analyze")
    
    # Additional features
    display_review_tips()

def perform_review_analysis(api_client, review_text: str, include_aspects: bool = True):
    """Perform comprehensive review analysis"""
    
    with st.spinner("🤖 AI is analyzing the review..."):
        # Add some user and review metadata for more realistic analysis
        user_data = {"review_count": 15, "avg_rating_given": 4.2}
        review_data = {"helpful_votes": 5, "review_length": len(review_text)}
        
        analysis = api_client.analyze_review(
            review_text, 
            user_data=user_data, 
            review_data=review_data
        )
    
    if not analysis:
        st.error("Analysis failed. Please check your connection and try again.")
        return
    
    st.markdown("---")
    st.markdown("## 🤖 AI Analysis Results")
    
    # Display analysis results
    display_sentiment_results(analysis)
    display_fake_detection_results(analysis)
    display_processing_info(analysis)
    
    if include_aspects:
        display_aspect_results(analysis)
    
    display_analysis_summary(analysis, review_text)

def display_sentiment_results(analysis: Dict[str, Any]):
    """Display sentiment analysis results"""
    st.markdown("### 😊 Sentiment Analysis")
    
    sentiment = analysis['sentiment']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_label = sentiment['overall_sentiment']
        sentiment_emoji = get_sentiment_emoji(sentiment_label)
        st.metric(
            "Overall Sentiment",
            f"{sentiment_emoji} {sentiment_label}",
            help="AI-detected emotional tone"
        )
    
    with col2:
        confidence = sentiment['confidence']
        st.metric(
            "AI Confidence",
            f"{confidence:.1%}",
            help="How confident the AI is in its prediction"
        )
    
    with col3:
        score = sentiment['score']
        st.metric(
            "Sentiment Score",
            f"{score:.2f}",
            help="Numerical sentiment score (-1 to +1)"
        )
    
    with col4:
        # Sentiment strength indicator
        if abs(score) > 0.5:
            strength = "Strong"
        elif abs(score) > 0.2:
            strength = "Moderate"
        else:
            strength = "Mild"
        st.metric("Sentiment Strength", strength)
    
    # Sentiment gauge chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkgreen" if score > 0.2 else "darkred" if score < -0.2 else "orange"},
                'steps': [
                    {'range': [-1, -0.2], 'color': "lightcoral"},
                    {'range': [-0.2, 0.2], 'color': "lightyellow"},
                    {'range': [0.2, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Sentiment Scale:**")
        st.markdown("""
        - **1.0 to 0.5:** 😊 Very Positive
        - **0.5 to 0.2:** 🙂 Positive  
        - **0.2 to -0.2:** 😐 Neutral
        - **-0.2 to -0.5:** 🙁 Negative
        - **-0.5 to -1.0:** 😞 Very Negative
        """)

def display_fake_detection_results(analysis: Dict[str, Any]):
    """Display fake review detection results"""
    st.markdown("### 🕵️ Fake Review Detection")
    
    fake_data = analysis['fake_detection']
    fake_prob = fake_data.get('is_fake_probability', 0)
    confidence = fake_data.get('confidence', 0.5)
    risk_level = fake_data.get('risk_level', 'UNKNOWN')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_indicator = get_risk_indicator(risk_level, fake_prob)
        st.metric(
            "Authenticity Risk",
            risk_indicator,
            help="AI assessment of review authenticity"
        )
    
    with col2:
        st.metric(
            "Fake Probability",
            f"{fake_prob:.1%}",
            help="Likelihood this review is fake"
        )
    
    with col3:
        st.metric(
            "Detection Confidence",
            f"{confidence:.1%}",
            help="AI confidence in fake detection"
        )
    
    with col4:
        # Risk level indicator
        if fake_prob < 0.3:
            authenticity = "✅ Likely Authentic"
        elif fake_prob < 0.7:
            authenticity = "⚠️ Questionable"
        else:
            authenticity = "❌ Likely Fake"
        st.metric("Assessment", authenticity)
    
    # Fake detection details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fake probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fake_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fake Probability %"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if fake_prob > 0.7 else "orange" if fake_prob > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Detection Factors:**")
        
        # Provide insights into what makes a review seem fake
        review_length = len(analysis.get('review_text', ''))
        
        if fake_prob > 0.7:
            st.markdown("🚨 **High Risk Indicators:**")
            st.write("- Repetitive language patterns")
            st.write("- Generic or vague descriptions")
            st.write("- Extreme sentiment without specifics")
        elif fake_prob > 0.3:
            st.markdown("⚠️ **Moderate Risk Indicators:**")
            st.write("- Some pattern irregularities")
            st.write("- Mixed authenticity signals")
        else:
            st.markdown("✅ **Authentic Indicators:**")
            st.write("- Natural language patterns")
            st.write("- Specific product details")
            st.write("- Balanced sentiment expression")

def display_processing_info(analysis: Dict[str, Any]):
    """Display processing information"""
    st.markdown("### ⚡ Processing Information")
    
    processing_time = analysis.get('processing_time', 0)
    timestamp = analysis.get('timestamp', 'Unknown')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Processing Time",
            f"{processing_time:.3f}s",
            help="Time taken for AI analysis"
        )
    
    with col2:
        st.metric(
            "Analysis Type",
            "Real-time AI",
            help="Live analysis using trained models"
        )
    
    with col3:
        review_length = len(analysis.get('review_text', ''))
        st.metric(
            "Review Length",
            f"{review_length} chars",
            help="Character count of analyzed text"
        )

def display_aspect_results(analysis: Dict[str, Any]):
    """Display aspect-based sentiment analysis"""
    st.markdown("### 🔍 Aspect-Based Analysis")
    
    sentiment = analysis['sentiment']
    aspects = sentiment.get('aspects', {})
    
    if aspects and isinstance(aspects, dict):
        st.markdown("**Product Aspects Mentioned:**")
        
        # Prepare data for visualization
        aspect_names = []
        aspect_scores = []
        
        for aspect, data in aspects.items():
            if isinstance(data, dict) and 'sentiment' in data:
                aspect_names.append(aspect.title())
                aspect_scores.append(data['sentiment'])
        
        if aspect_names:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                for aspect, score in zip(aspect_names, aspect_scores):
                    emoji = get_sentiment_emoji("positive" if score > 0.1 else "negative" if score < -0.1 else "neutral")
                    st.write(f"{emoji} **{aspect}**: {score:.2f}")
            
            with col2:
                # Create horizontal bar chart for aspects
                if len(aspect_names) > 0:
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
            st.info("No specific aspects detected in this review.")
    else:
        st.info("Aspect analysis not available for this review.")

def display_analysis_summary(analysis: Dict[str, Any], review_text: str):
    """Display overall analysis summary"""
    st.markdown("---")
    st.markdown("### 📋 Analysis Summary")
    
    sentiment = analysis['sentiment']
    fake_data = analysis['fake_detection']
    
    sentiment_label = sentiment['overall_sentiment']
    sentiment_score = sentiment['score']
    fake_prob = fake_data.get('is_fake_probability', 0)
    
    # Generate summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Key Findings:**")
        
        # Sentiment summary
        if sentiment_score > 0.2:
            st.write("✅ **Positive customer experience** - Customer expresses satisfaction")
        elif sentiment_score < -0.2:
            st.write("❌ **Negative customer experience** - Customer expresses dissatisfaction") 
        else:
            st.write("😐 **Neutral customer experience** - Mixed or balanced sentiment")
        
        # Authenticity summary
        if fake_prob < 0.3:
            st.write("✅ **Appears authentic** - Natural language patterns detected")
        elif fake_prob < 0.7:
            st.write("⚠️ **Authenticity uncertain** - Mixed signals detected")
        else:
            st.write("❌ **Potentially fake** - Suspicious patterns detected")
        
        # Review quality
        review_length = len(review_text)
        if review_length > 100:
            st.write("📝 **Detailed review** - Provides substantial information")
        elif review_length > 50:
            st.write("📝 **Moderate detail** - Provides some information")
        else:
            st.write("📝 **Brief review** - Limited information provided")
    
    with col2:
        st.markdown("**Recommendation:**")
        
        if sentiment_score > 0.2 and fake_prob < 0.3:
            st.success("🌟 **High Value Review** - Positive and authentic")
        elif sentiment_score < -0.2 and fake_prob < 0.3:
            st.info("⚖️ **Critical Feedback** - Negative but authentic")
        elif fake_prob > 0.7:
            st.warning("🚫 **Treat with Caution** - Likely fake review")
        else:
            st.info("📊 **Standard Review** - Mixed characteristics")

def display_review_tips():
    """Display tips for understanding review analysis"""
    st.markdown("---")
    st.markdown("### 💡 Understanding Review Analysis")
    
    with st.expander("🎯 How does sentiment analysis work?"):
        st.markdown("""
        Our AI analyzes the emotional tone and opinion expressed in reviews using:
        - **Natural Language Processing** to understand context and meaning
        - **Trained models** on thousands of product reviews
        - **Aspect detection** to identify specific product features mentioned
        - **Confidence scoring** to indicate certainty of predictions
        """)
    
    with st.expander("🕵️ How does fake review detection work?"):
        st.markdown("""
        Our AI identifies potentially fake reviews by analyzing:
        - **Language patterns** - Repetitive or generic phrases
        - **Review structure** - Unusual formatting or length
        - **Sentiment extremes** - Overly positive/negative without specifics
        - **User behavior** - Review frequency and rating patterns
        
        **Note:** This is a probability assessment, not a definitive determination.
        """)
    
    with st.expander("📊 How to interpret the results?"):
        st.markdown("""
        **Sentiment Scores:**
        - Values range from -1 (very negative) to +1 (very positive)
        - Scores near 0 indicate neutral sentiment
        - Higher confidence percentages indicate more reliable predictions
        
        **Fake Detection:**
        - Low probability (<30%) suggests likely authentic review
        - High probability (>70%) suggests potentially fake review
        - Consider overall patterns, not individual reviews alone
        """)