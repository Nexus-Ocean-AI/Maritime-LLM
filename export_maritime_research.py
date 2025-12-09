import json
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# Configuration
MONGO_URI = "mongodb+srv://doadmin:3482nswUH5KC61p7@klavness-uat-97109a4d.mongo.ondigitalocean.com/admin"
DB_NAME = "maritime_research"
COLLECTIONS = ["parsed_papers", "parsed_books", "parsed_technical_manager"]

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB types."""
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

def export_collections():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        
        for col_name in COLLECTIONS:
            print(f"Starting export for collection: {col_name}")
            collection = db[col_name]
            output_file = f"data/{col_name}.jsonl"
            
            # Ensure directory exists
            import os
            os.makedirs("data", exist_ok=True)
            
            doc_count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                # Use a cursor to iterate over all documents
                cursor = collection.find({})
                
                for doc in cursor:
                    # Determine if we need to filter any fields or keep all
                    # Currently keeping all fields as per request
                    
                    line = json.dumps(doc, cls=JSONEncoder, ensure_ascii=False)
                    f.write(line + "\n")
                    doc_count += 1
                    
            print(f"Finished exporting {col_name}. Total documents: {doc_count}")
            print(f"Output saved to: {output_file}\n")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    export_collections()
