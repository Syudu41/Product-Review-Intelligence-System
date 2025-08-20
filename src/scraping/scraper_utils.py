"""
Scraper Utilities Module for Product Review Intelligence System
Rate limiting, proxy management, and scraping utilities
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import threading
from collections import defaultdict, deque
import json
from pathlib import Path
import requests
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Thread-safe rate limiter for web scraping
    """
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
        
        logger.info(f"Rate limiter initialized: {max_requests} requests per {time_window}s")
    
    def wait_if_needed(self):
        """
        Wait if rate limit would be exceeded
        """
        with self.lock:
            now = datetime.now()
            
            # Remove old requests outside time window
            while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
                self.requests.popleft()
            
            # Check if we need to wait
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = self.time_window - (now - oldest_request).total_seconds()
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    
                    # Remove the oldest request after waiting
                    self.requests.popleft()
            
            # Record this request
            self.requests.append(now)
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        with self.lock:
            now = datetime.now()
            
            # Clean old requests
            while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
                self.requests.popleft()
            
            return {
                'current_requests': len(self.requests),
                'max_requests': self.max_requests,
                'time_window': self.time_window,
                'utilization': len(self.requests) / self.max_requests,
                'next_available': self.requests[0] + timedelta(seconds=self.time_window) if self.requests else now
            }


class DomainRateLimiter:
    """
    Rate limiter that manages different limits per domain
    """
    
    def __init__(self):
        self.domain_limiters = {}
        self.default_config = {'max_requests': 10, 'time_window': 60}
        
        # Domain-specific configurations
        self.domain_configs = {
            'amazon.com': {'max_requests': 5, 'time_window': 60},
            'bestbuy.com': {'max_requests': 8, 'time_window': 60},
            'target.com': {'max_requests': 6, 'time_window': 60}
        }
    
    def get_domain_limiter(self, url: str) -> RateLimiter:
        """Get or create rate limiter for domain"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        if domain not in self.domain_limiters:
            config = self.domain_configs.get(domain, self.default_config)
            self.domain_limiters[domain] = RateLimiter(**config)
            logger.info(f"Created rate limiter for {domain}")
        
        return self.domain_limiters[domain]
    
    def wait_for_url(self, url: str):
        """Wait if needed before making request to URL"""
        limiter = self.get_domain_limiter(url)
        limiter.wait_if_needed()


class ScrapingSession:
    """
    Enhanced requests session with automatic rate limiting and rotation
    """
    
    def __init__(self, rate_limiter: Optional[DomainRateLimiter] = None):
        self.session = requests.Session()
        self.rate_limiter = rate_limiter or DomainRateLimiter()
        
        # User agent rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Request statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': datetime.now()
        }
        
        self._setup_session()
    
    def _setup_session(self):
        """Setup session with default headers"""
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self._rotate_user_agent()
    
    def _rotate_user_agent(self):
        """Rotate user agent"""
        self.session.headers['User-Agent'] = random.choice(self.user_agents)
    
    def get(self, url: str, max_retries: int = 3, **kwargs) -> requests.Response:
        """
        Make GET request with rate limiting and retry logic
        """
        self.stats['total_requests'] += 1
        
        # Rate limiting
        self.rate_limiter.wait_for_url(url)
        
        # Rotate user agent occasionally
        if random.random() < 0.3:
            self._rotate_user_agent()
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, **kwargs)
                
                if response.status_code == 200:
                    self.stats['successful_requests'] += 1
                    return response
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = random.uniform(10, 30)
                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                elif response.status_code in [403, 503]:
                    # Blocked or service unavailable
                    wait_time = random.uniform(30, 60)
                    logger.warning(f"Blocked or unavailable, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = random.uniform(2, 8) * (attempt + 1)
                time.sleep(wait_time)
        
        self.stats['failed_requests'] += 1
        raise Exception(f"Failed to fetch {url} after {max_retries} attempts")
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'success_rate': (
                self.stats['successful_requests'] / self.stats['total_requests'] 
                if self.stats['total_requests'] > 0 else 0
            ),
            'requests_per_minute': (
                self.stats['total_requests'] / (runtime / 60) 
                if runtime > 0 else 0
            )
        }


class ProxyRotator:
    """
    Proxy rotation manager (placeholder - would need actual proxy service)
    """
    
    def __init__(self, proxy_list: Optional[List[str]] = None):
        self.proxy_list = proxy_list or []
        self.current_index = 0
        self.failed_proxies = set()
        
    def get_next_proxy(self) -> Optional[Dict]:
        """Get next working proxy"""
        if not self.proxy_list:
            return None
        
        attempts = 0
        while attempts < len(self.proxy_list):
            proxy = self.proxy_list[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxy_list)
            
            if proxy not in self.failed_proxies:
                return {'http': proxy, 'https': proxy}
            
            attempts += 1
        
        return None
    
    def mark_proxy_failed(self, proxy: str):
        """Mark proxy as failed"""
        self.failed_proxies.add(proxy)


class ScrapingMonitor:
    """
    Monitor scraping operations and detect blocking
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.status_codes = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
    def record_request(self, response_time: float, status_code: int):
        """Record request metrics"""
        now = datetime.now()
        
        self.response_times.append(response_time)
        self.status_codes.append(status_code)
        self.timestamps.append(now)
    
    def detect_blocking(self) -> Dict:
        """Detect if we're being blocked"""
        if len(self.status_codes) < 10:
            return {'blocked': False, 'confidence': 0}
        
        recent_codes = list(self.status_codes)[-20:]
        
        # Check for high error rates
        error_rate = sum(1 for code in recent_codes if code >= 400) / len(recent_codes)
        
        # Check for response time increases
        avg_response_time = sum(self.response_times) / len(self.response_times)
        recent_avg = sum(list(self.response_times)[-10:]) / 10
        
        # Blocking indicators
        high_error_rate = error_rate > 0.5
        slow_responses = recent_avg > avg_response_time * 2
        many_429s = sum(1 for code in recent_codes if code == 429) > 3
        
        blocking_score = sum([high_error_rate, slow_responses, many_429s]) / 3
        
        return {
            'blocked': blocking_score > 0.5,
            'confidence': blocking_score,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'recent_avg_response_time': recent_avg,
            'recommendations': self._get_recommendations(blocking_score, error_rate)
        }
    
    def _get_recommendations(self, blocking_score: float, error_rate: float) -> List[str]:
        """Get recommendations based on metrics"""
        recommendations = []
        
        if blocking_score > 0.7:
            recommendations.append("Consider longer delays between requests")
            recommendations.append("Rotate user agents more frequently")
            recommendations.append("Use proxy rotation if available")
        
        if error_rate > 0.3:
            recommendations.append("Reduce request rate")
            recommendations.append("Check for IP blocking")
        
        if blocking_score > 0.5:
            recommendations.append("Switch to sample data generation")
        
        return recommendations


class ScrapingUtils:
    """
    Utility functions for web scraping
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean scraped text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract numbers from text"""
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(num) for num in numbers]
    
    @staticmethod
    def is_valid_review(review_text: str, min_length: int = 10) -> bool:
        """Check if review text is valid"""
        if not review_text or len(review_text.strip()) < min_length:
            return False
        
        # Check for spam indicators
        spam_indicators = [
            'click here',
            'visit our website',
            'buy now',
            'limited time',
            '!!!!'
        ]
        
        text_lower = review_text.lower()
        spam_count = sum(1 for indicator in spam_indicators if indicator in text_lower)
        
        return spam_count == 0
    
    @staticmethod
    def normalize_rating(rating_text: str) -> Optional[int]:
        """Normalize rating from various formats"""
        if not rating_text:
            return None
        
        # Extract first number found
        numbers = ScrapingUtils.extract_numbers(rating_text)
        if numbers:
            rating = numbers[0]
            
            # Convert to 1-5 scale
            if rating <= 1:
                return max(1, int(rating * 5))
            elif rating <= 5:
                return int(rating)
            elif rating <= 10:
                return int(rating / 2)
            elif rating <= 100:
                return int(rating / 20)
        
        return None
    
    @staticmethod
    def save_scraping_log(stats: Dict, filename: str = None):
        """Save scraping statistics to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraping_log_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Scraping log saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save scraping log: {e}")


# Convenience function for quick setup
def create_scraping_session(requests_per_minute: int = 10) -> ScrapingSession:
    """
    Create a configured scraping session
    """
    # Convert requests per minute to rate limiter config
    max_requests = max(1, requests_per_minute // 6)  # Requests per 10 seconds
    time_window = 10
    
    rate_limiter = DomainRateLimiter()
    session = ScrapingSession(rate_limiter)
    
    logger.info(f"Created scraping session: {requests_per_minute} req/min")
    
    return session


def main():
    """
    Test scraping utilities
    """
    logger.info("Testing scraping utilities...")
    
    # Test rate limiter
    rate_limiter = RateLimiter(max_requests=3, time_window=10)
    
    print("Testing rate limiter...")
    for i in range(5):
        print(f"Request {i+1}")
        rate_limiter.wait_if_needed()
        time.sleep(1)
    
    # Test scraping session
    session = create_scraping_session(requests_per_minute=12)
    
    try:
        # Test with a safe URL
        response = session.get('https://httpbin.org/headers', timeout=5)
        print(f"Test request status: {response.status_code}")
    except:
        print("Test request failed (expected if no internet)")
    
    # Display stats
    stats = session.get_stats()
    print(f"Session stats: {stats}")
    
    print("✅ Scraping utilities test completed")


if __name__ == "__main__":
    main()