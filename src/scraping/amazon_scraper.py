"""
Amazon Scraper Module for Product Review Intelligence System
Handles respectful web scraping of Amazon product reviews
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import sqlite3
from pathlib import Path
import re
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonScraper:
    """
    Respectful Amazon product review scraper
    """
    
    def __init__(self):
        self.db_path = DATABASE_URL.replace('sqlite:///', '')
        self.session = requests.Session()
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
        # Scraping configuration
        self.config = {
            'delay_min': 2,
            'delay_max': 5,
            'max_retries': 3,
            'timeout': 10,
            'max_reviews_per_product': 500,
            'respect_robots': True
        }
        
        # Sample Amazon product ASINs for testing
        self.sample_products = [
            'B08N5WRWNW',  # Echo Dot
            'B07XJ8C8F5',  # Fire TV Stick
            'B08F7PTF53',  # iPad Air
            'B085WCRS1J',  # MacBook Air
            'B07YNM3WNR'   # AirPods Pro
        ]
        
        # Initialize session
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with headers"""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _random_delay(self):
        """Add random delay between requests"""
        delay = random.uniform(self.config['delay_min'], self.config['delay_max'])
        time.sleep(delay)
    
    def _rotate_user_agent(self):
        """Rotate user agent"""
        self.session.headers['User-Agent'] = random.choice(self.user_agents)
    
    def scrape_product_reviews(self, asin: str, max_reviews: int = 100) -> List[Dict]:
        """
        Scrape reviews for a specific Amazon product
        """
        logger.info(f"Scraping reviews for ASIN: {asin}")
        
        reviews = []
        page = 1
        
        try:
            while len(reviews) < max_reviews:
                logger.info(f"Scraping page {page} for {asin}...")
                
                # Build URL for reviews page
                url = f"https://www.amazon.com/product-reviews/{asin}/"
                params = {
                    'ie': 'UTF8',
                    'reviewerType': 'all_reviews',
                    'pageNumber': page,
                    'sortBy': 'recent'
                }
                
                # Make request with retry logic
                page_reviews = self._scrape_reviews_page(url, params)
                
                if not page_reviews:
                    logger.info(f"No more reviews found on page {page}")
                    break
                
                reviews.extend(page_reviews)
                logger.info(f"Collected {len(page_reviews)} reviews from page {page}")
                
                # Add delay and rotate user agent
                self._random_delay()
                self._rotate_user_agent()
                
                page += 1
                
                # Safety limit
                if page > 10:
                    logger.warning("Reached maximum page limit (10)")
                    break
            
            logger.info(f"Total reviews collected for {asin}: {len(reviews)}")
            
            # Add metadata
            for review in reviews:
                review['product_asin'] = asin
                review['scrape_date'] = datetime.now()
                review['source'] = 'amazon_scraper'
            
            return reviews[:max_reviews]
            
        except Exception as e:
            logger.error(f"Failed to scrape reviews for {asin}: {e}")
            return []
    
    def _scrape_reviews_page(self, url: str, params: Dict) -> List[Dict]:
        """
        Scrape a single page of reviews
        """
        reviews = []
        
        for attempt in range(self.config['max_retries']):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config['timeout']
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    reviews = self._parse_reviews_from_soup(soup)
                    break
                elif response.status_code == 503:
                    logger.warning(f"Rate limited (503), waiting longer...")
                    time.sleep(random.uniform(10, 20))
                else:
                    logger.warning(f"HTTP {response.status_code}, attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
            if attempt < self.config['max_retries'] - 1:
                time.sleep(random.uniform(5, 10))
        
        return reviews
    
    def _parse_reviews_from_soup(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Parse reviews from BeautifulSoup object
        """
        reviews = []
        
        # Find review containers (Amazon's structure may change)
        review_containers = soup.find_all('div', {'data-hook': 'review'})
        
        if not review_containers:
            # Alternative selectors
            review_containers = soup.find_all('div', class_=re.compile(r'review'))
        
        for container in review_containers:
            try:
                review = self._extract_review_data(container)
                if review:
                    reviews.append(review)
            except Exception as e:
                logger.warning(f"Failed to parse review: {e}")
                continue
        
        logger.info(f"Parsed {len(reviews)} reviews from page")
        return reviews
    
    def _extract_review_data(self, container) -> Optional[Dict]:
        """
        Extract review data from a review container
        """
        try:
            review = {}
            
            # Review ID
            review_id = container.get('id', '')
            review['review_id'] = review_id
            
            # Rating
            rating_element = container.find('i', {'data-hook': 'review-star-rating'})
            if rating_element:
                rating_text = rating_element.get_text()
                rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if rating_match:
                    review['rating'] = float(rating_match.group(1))
            
            # Review title
            title_element = container.find('a', {'data-hook': 'review-title'})
            if title_element:
                review['review_title'] = title_element.get_text().strip()
            else:
                title_element = container.find('span', {'data-hook': 'review-title'})
                if title_element:
                    review['review_title'] = title_element.get_text().strip()
            
            # Review text
            text_element = container.find('span', {'data-hook': 'review-body'})
            if text_element:
                review['review_text'] = text_element.get_text().strip()
            
            # Reviewer name
            author_element = container.find('span', class_='a-profile-name')
            if author_element:
                review['reviewer_name'] = author_element.get_text().strip()
            
            # Review date
            date_element = container.find('span', {'data-hook': 'review-date'})
            if date_element:
                date_text = date_element.get_text().strip()
                review['review_date'] = self._parse_date(date_text)
            
            # Helpful votes
            helpful_element = container.find('span', {'data-hook': 'helpful-vote-statement'})
            if helpful_element:
                helpful_text = helpful_element.get_text()
                helpful_match = re.search(r'(\d+)', helpful_text)
                if helpful_match:
                    review['helpful_votes'] = int(helpful_match.group(1))
                else:
                    review['helpful_votes'] = 0
            else:
                review['helpful_votes'] = 0
            
            # Verified purchase
            verified_element = container.find('span', {'data-hook': 'avp-badge'})
            review['verified_purchase'] = verified_element is not None
            
            # Only return if we have essential data
            if 'rating' in review and 'review_text' in review:
                return review
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error extracting review data: {e}")
            return None
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """
        Parse Amazon date format
        """
        try:
            # Amazon format: "Reviewed in the United States on January 1, 2023"
            date_match = re.search(r'on (.+?)(?:\s|$)', date_text)
            if date_match:
                date_str = date_match.group(1).strip()
                return datetime.strptime(date_str, '%B %d, %Y')
        except:
            pass
        
        return datetime.now()
    
    def scrape_multiple_products(self, asins: List[str], max_reviews_per_product: int = 50) -> pd.DataFrame:
        """
        Scrape reviews for multiple products
        """
        logger.info(f"Scraping reviews for {len(asins)} products")
        
        all_reviews = []
        
        for i, asin in enumerate(asins):
            logger.info(f"Processing product {i+1}/{len(asins)}: {asin}")
            
            try:
                reviews = self.scrape_product_reviews(asin, max_reviews_per_product)
                all_reviews.extend(reviews)
                
                logger.info(f"Collected {len(reviews)} reviews for {asin}")
                
                # Longer delay between products
                if i < len(asins) - 1:
                    delay = random.uniform(10, 20)
                    logger.info(f"Waiting {delay:.1f} seconds before next product...")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to scrape {asin}: {e}")
                continue
        
        # Convert to DataFrame
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            df = self._standardize_scraped_data(df)
            logger.info(f"Total reviews collected: {len(df)}")
            return df
        else:
            logger.warning("No reviews collected")
            return pd.DataFrame()
    
    def _standardize_scraped_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize scraped data to match database schema
        """
        # Create standardized columns
        standardized = pd.DataFrame()
        
        standardized['product_id'] = df.get('product_asin', 'UNKNOWN')
        standardized['user_id'] = df.get('reviewer_name', 'UNKNOWN').astype(str)
        standardized['rating'] = df.get('rating', 3).astype(int)
        standardized['review_text'] = df.get('review_text', '').astype(str)
        standardized['review_title'] = df.get('review_title', '').astype(str)
        standardized['date'] = pd.to_datetime(df.get('review_date', datetime.now()))
        standardized['helpful_votes'] = df.get('helpful_votes', 0).astype(int)
        standardized['total_votes'] = standardized['helpful_votes']  # Approximation
        standardized['verified_purchase'] = df.get('verified_purchase', False)
        standardized['source'] = 'amazon_scraper'
        standardized['scrape_timestamp'] = datetime.now()
        
        # Clean text
        standardized['review_text'] = standardized['review_text'].str.strip()
        standardized['review_title'] = standardized['review_title'].str.strip()
        
        # Filter out empty reviews
        standardized = standardized[standardized['review_text'].str.len() > 10]
        
        logger.info(f"Standardized {len(standardized)} reviews")
        return standardized
    
    def save_scraped_reviews(self, df: pd.DataFrame, table_name: str = 'live_reviews') -> bool:
        """
        Save scraped reviews to database
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='append', index=False)
                logger.info(f"Saved {len(df)} scraped reviews to {table_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to save scraped reviews: {e}")
            return False
    
    def create_sample_scrape_data(self, num_reviews: int = 100) -> pd.DataFrame:
        """
        Create sample scraped data for testing (when live scraping isn't available)
        """
        logger.info(f"Creating {num_reviews} sample scraped reviews")
        
        np.random.seed(42)
        
        sample_data = {
            'product_id': np.random.choice(self.sample_products, num_reviews),
            'user_id': [f'ScrapedUser_{i}' for i in range(num_reviews)],
            'rating': np.random.choice([1, 2, 3, 4, 5], num_reviews, p=[0.05, 0.05, 0.15, 0.35, 0.40]),
            'review_text': [f'This is a scraped review {i}. ' + 
                           ('Great product! ' if np.random.random() > 0.5 else 'Could be better. ') +
                           'Would recommend to others.' for i in range(num_reviews)],
            'review_title': [f'Review Title {i}' for i in range(num_reviews)],
            'date': pd.date_range('2023-01-01', periods=num_reviews, freq='D'),
            'helpful_votes': np.random.randint(0, 20, num_reviews),
            'total_votes': np.random.randint(0, 25, num_reviews),
            'verified_purchase': np.random.choice([True, False], num_reviews, p=[0.8, 0.2]),
            'source': 'amazon_scraper_sample',
            'scrape_timestamp': datetime.now()
        }
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample scraped data: {len(df)} reviews")
        
        return df
    
    def validate_scraping_setup(self) -> Dict:
        """
        Validate scraping setup and connectivity
        """
        logger.info("Validating scraping setup...")
        
        validation = {
            'session_ready': False,
            'connectivity': False,
            'user_agents': len(self.user_agents) > 0,
            'config_valid': True,
            'database_accessible': False
        }
        
        # Test session
        try:
            response = self.session.get('https://httpbin.org/headers', timeout=5)
            validation['session_ready'] = response.status_code == 200
        except:
            pass
        
        # Test connectivity to Amazon
        try:
            response = self.session.get('https://www.amazon.com', timeout=10)
            validation['connectivity'] = response.status_code == 200
        except:
            pass
        
        # Test database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('SELECT 1')
                validation['database_accessible'] = True
        except:
            pass
        
        validation['overall_ready'] = all([
            validation['session_ready'],
            validation['user_agents'],
            validation['database_accessible']
        ])
        
        logger.info(f"Scraping validation: {validation}")
        return validation


def main():
    """
    Main function for testing scraper
    """
    scraper = AmazonScraper()
    
    # Validate setup
    validation = scraper.validate_scraping_setup()
    
    if validation['overall_ready']:
        logger.info("✅ Scraper validation passed")
        
        # Test with sample data (since live scraping may be blocked)
        logger.info("Creating sample scraped data for testing...")
        sample_df = scraper.create_sample_scrape_data(50)
        
        # Save to database
        success = scraper.save_scraped_reviews(sample_df)
        
        if success:
            logger.info("✅ Sample scraping data created and saved successfully")
            print(f"Sample data shape: {sample_df.shape}")
            print(f"Columns: {list(sample_df.columns)}")
            print(f"Rating distribution: {sample_df['rating'].value_counts().to_dict()}")
        else:
            logger.error("❌ Failed to save sample data")
    else:
        logger.warning("⚠️ Scraper validation failed")
        logger.info("Creating sample data instead...")
        
        sample_df = scraper.create_sample_scrape_data(50)
        scraper.save_scraped_reviews(sample_df)
        
        print("Sample scraping data created for testing")


if __name__ == "__main__":
    main()