import pymongo
import os 
import dotenv

dotenv.load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")

client = pymongo.MongoClient(MONGODB_URI)
db = client[DATABASE]
collection = db[COLLECTION]


