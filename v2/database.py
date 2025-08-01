from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env

# Read variables
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
user_collection = db["users"]
