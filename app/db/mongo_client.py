from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_mongo_connection():
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)  
    db_name = os.getenv("MONGO_DB")
    return client[db_name]
