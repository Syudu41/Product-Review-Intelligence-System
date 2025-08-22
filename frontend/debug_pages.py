"""
Debug script to check page imports
"""

try:
    print("Testing page imports...")
    
    from pages import home
    print("✅ home imported successfully")
    
    from pages import analytics  
    print("✅ analytics imported successfully")
    
    from pages import review_intel
    print("✅ review_intel imported successfully")
    
    from pages import recommendations
    print("✅ recommendations imported successfully")
    
    from pages import insights
    print("✅ insights imported successfully")
    
    from components.api_client import APIClient
    print("✅ APIClient imported successfully")
    
    from utils.styling import load_custom_css
    print("✅ styling imported successfully")
    
    print("\n🎉 All imports working!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()