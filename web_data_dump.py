
"""
Paginate through a MongoDB collection and dump documents
into a single JSONL file.

File name: <OUTPUT_PREFIX>.jsonl
Each line in a file is a JSON object with at least:
  - _id
  - url
  - category
  - text
"""

from pymongo import MongoClient
import json

# ====== CONFIG: SET THESE ====== #
MONGO_URI = "mongodb+srv://doadmin:3482nswUH5KC61p7@klavness-uat-97109a4d.mongo.ondigitalocean.com/admin"
DB_NAME = "Agentic_frontend_chat_history_Shubham"
COLLECTION_NAME = "scraped_public_raw"
OUTPUT_PREFIX = "maritime_web_text"               # prefix for output files
BATCH_LIMIT = 50
# ================================= #


def dump_all_docs_to_jsonl():
    """
    Connects to MongoDB, paginates over the collection using skip+limit,
    and writes docs (with metadata + text) into a single JSONL file.

    Each line in a file is a single JSON object.
    """

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    skip = 0
    total_docs = 0

    # projection: choose what you want to keep for training
    projection = {
        "_id": 1,
        "url": 1,
        "category": 1,
        "text": 1,
    }

    output_filename = f"{OUTPUT_PREFIX}.jsonl"

    with open(output_filename, "w", encoding="utf-8") as f:
        while True:
            docs = list(
                coll.find(
                    {},                 # all docs
                    projection
                ).skip(skip).limit(BATCH_LIMIT)
            )

            if not docs:
                break  # no more documents

            written_in_batch = 0
            for d in docs:
                # Convert ObjectId and datetimes to strings for JSON
                if "_id" in d:
                    d["_id"] = str(d["_id"])

                # Only dump docs that actually have text
                if not d.get("text"):
                    continue

                line = json.dumps(d, ensure_ascii=False)
                f.write(line + "\n")
                written_in_batch += 1
                total_docs += 1

            print(f"Processed batch starting at skip {skip}, wrote {written_in_batch} docs to {output_filename}")
            skip += BATCH_LIMIT
            if skip > 500:
                break

    client.close()
    print(f"Done. Total documents written: {total_docs}")


if __name__ == "__main__":
    dump_all_docs_to_jsonl()

