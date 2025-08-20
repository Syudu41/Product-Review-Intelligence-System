import sqlite3
import pandas as pd

# Connect to your database
conn = sqlite3.connect('./database/review_intelligence.db')

# Check what you have
print("Tables:", pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist())
print("Review count:", pd.read_sql("SELECT COUNT(*) as count FROM reviews", conn)['count'][0])
print("Rating dist:", pd.read_sql("SELECT rating, COUNT(*) as count FROM reviews GROUP BY rating", conn))

conn.close()