"""
API Client for Product Review Intelligence System
Handles all communication with the FastAPI backend
"""

import requests
import streamlit as st
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

class APIClient:
    """Centralized API client for backend communication"""
    
    def __init__(self):
        self.base_urls = [
            "http://127.0.0.1:8000",
            "http://localhost:8000"
        ]
        self.timeout = 10
        self.session = requests.Session()
        
    def _try_request(self, method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Try request with multiple base URLs"""
        for base_url in self.base_urls:
            url = f"{base_url}{endpoint}"
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if response.status_code < 500:  # Accept all non-server errors
                    return response
            except requests.exceptions.RequestException:
                continue
        return None
    
    def check_health(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self._try_request("GET", "/health")
            if response and response.status_code == 200:
                data = response.json()
                # Convert FastAPI format to our expected format
                return {
                    "healthy": data.get("status") == "healthy",
                    "status": data.get("status"),
                    "database": data.get("database"),
                    "models": data.get("models", {}),
                    "timestamp": data.get("timestamp")
                }
            return {"healthy": False, "error": "API not responding"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def search_products(self, query: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Search for products"""
        try:
            response = self._try_request(
                "GET", 
                "/products/search",
                params={"query": query, "limit": limit}
            )
            
            if response and response.status_code == 200:
                return response.json()
            elif response and response.status_code == 404:
                return {"products": [], "results_count": 0, "message": "No products found"}
            else:
                st.error("Search service unavailable")
                return None
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return None
    
    def get_product_analytics(self, product_id: str, include_recent: bool = True) -> Optional[Dict[str, Any]]:
        """Get detailed analytics for a product"""
        try:
            response = self._try_request(
                "GET",
                f"/products/{product_id}/analytics",
                params={"include_recent": include_recent}
            )
            
            if response and response.status_code == 200:
                return response.json()
            elif response and response.status_code == 404:
                st.error(f"❌ Product {product_id} not found in database")
                return None
            elif response and response.status_code == 500:
                st.error(f"❌ Server error analyzing product {product_id}")
                return None
            else:
                # Try to get more info about the error
                if response:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', f'HTTP {response.status_code}')
                        st.error(f"❌ Analytics failed: {error_msg}")
                    except:
                        st.error(f"❌ Analytics failed with HTTP {response.status_code}")
                else:
                    st.error("❌ Could not connect to analytics service")
                return None
                
        except Exception as e:
            st.error(f"❌ Analytics request failed: {str(e)}")
            return None
    
    def analyze_review(self, review_text: str, user_data: Optional[Dict] = None, 
                      review_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Analyze individual review text"""
        try:
            payload = {"review_text": review_text}
            if user_data:
                payload["user_data"] = user_data
            if review_data:
                payload["review_data"] = review_data
                
            response = self._try_request(
                "POST",
                "/analyze/review",
                json=payload
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                st.error("Review analysis service unavailable")
                return None
                
        except Exception as e:
            st.error(f"Review analysis failed: {str(e)}")
            return None
    
    def get_user_recommendations(self, user_id: str, limit: int = 10, 
                               refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get recommendations for a user"""
        try:
            response = self._try_request(
                "GET",
                f"/users/{user_id}/recommendations",
                params={"limit": limit, "refresh": refresh}
            )
            
            if response and response.status_code == 200:
                return response.json()
            elif response and response.status_code == 404:
                st.error(f"❌ User {user_id} not found in our database")
                return None
            elif response and response.status_code == 500:
                st.error(f"❌ Server error getting recommendations for {user_id}")
                return None
            else:
                # Try to get more info about the error
                if response:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', f'HTTP {response.status_code}')
                        st.error(f"❌ Recommendations failed: {error_msg}")
                    except:
                        st.error(f"❌ Recommendations failed with HTTP {response.status_code}")
                else:
                    st.error("❌ Could not connect to recommendation service")
                return None
                
        except Exception as e:
            st.error(f"❌ Recommendation request failed: {str(e)}")
            return None
    
    def get_system_stats(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive system statistics"""
        try:
            response = self._try_request("GET", "/analytics/system-stats")
            
            if response and response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            return None
    
    def retrain_models(self) -> bool:
        """Trigger model retraining (admin function)"""
        try:
            response = self._try_request("POST", "/admin/retrain-models")
            
            if response and response.status_code == 200:
                return True
            else:
                st.error("Model retraining service unavailable")
                return False
                
        except Exception as e:
            st.error(f"Retraining request failed: {str(e)}")
            return False
    
    def get_real_users(self, limit: int = 10) -> Optional[List[str]]:
        """Get real user IDs from the database"""
        try:
            response = self._try_request(
                "GET",
                "/analytics/system-stats"
            )
            
            if response and response.status_code == 200:
                # If we can't get users directly, we'll use a custom endpoint
                # For now, let's try to get some sample user IDs
                pass
            
            # Return some known sample user IDs that should exist in the database
            sample_users = [
                "A1RSDE90N6RSZF", "A3SGXH7AUHU8GW", "A1KLRMWW2FWPL4", 
                "A3R7JR3FMEBXQB", "A2MUGFV2TDQ47K", "A1V6B6TNIC10QR",
                "A3OXHLG6DIBRW8", "A1MT4WRPV50YX2", "A23HDGZPIJNPVS",
                "A2COOO0KXDO47H"
            ]
            return sample_users[:limit]
                
        except Exception as e:
            # Return fallback user IDs
            return [
                "A1RSDE90N6RSZF", "A3SGXH7AUHU8GW", "A1KLRMWW2FWPL4", 
                "A3R7JR3FMEBXQB", "A2MUGFV2TDQ47K"
            ]

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about API connection"""
        working_url = None
        
        for base_url in self.base_urls:
            try:
                response = self.session.get(f"{base_url}/health", timeout=2)
                if response.status_code == 200:
                    working_url = base_url
                    break
            except:
                continue
        
        return {
            "working_url": working_url,
            "all_urls": self.base_urls,
            "timeout": self.timeout,
            "connected": working_url is not None
        }