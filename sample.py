import sqlite3
import pandas as pd

# Connect to your existing database
conn = sqlite3.connect('./database/review_intelligence.db')

# Check what products you have
products_sample = pd.read_sql_query("SELECT * FROM products LIMIT 10", conn)
print("PRODUCTS SAMPLE:")
print(products_sample)

# Check review content
reviews_sample = pd.read_sql_query("SELECT review_text FROM reviews LIMIT 5", conn)
print("\nREVIEW SAMPLES:")
for i, review in enumerate(reviews_sample['review_text']):
    print(f"{i+1}. {review[:100]}...")

conn.close()