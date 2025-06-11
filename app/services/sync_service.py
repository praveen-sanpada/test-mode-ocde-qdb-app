from app.db.mysql_client import get_mysql_connection
from app.db.mongo_client import get_mongo_connection
from app.core.enrich import enrich_record
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Initialize Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Allowed tables and collections
ALLOWED_MYSQL_TABLES = ['users']
ALLOWED_MONGO_COLLECTIONS = ['ps_match_data', 'ps_venue_data']

def sync_all():
    sync_mysql()
    sync_mongo()

def safe_record_id(raw_id=None):
    if raw_id is None:
        return str(uuid.uuid4())
    if isinstance(raw_id, int):
        return raw_id
    try:
        return str(uuid.UUID(str(raw_id)))
    except Exception:
        return str(uuid.uuid4())


def sync_mysql():
    db = get_mysql_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SHOW TABLES")
    tables = [list(row.values())[0] for row in cursor.fetchall()]
    tables = [t for t in tables if t in ALLOWED_MYSQL_TABLES]

    for table in tables:
        cursor.execute(f"SELECT * FROM `{table}`")
        rows = cursor.fetchall()
        print(f"📥 Found {len(rows)} rows in MySQL table: {table}")

        if not rows:
            print(f"⚠️ Skipped {table} (no data)")
            continue

        qdrant.recreate_collection(
            collection_name=table,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )

        for row in rows:
            raw_id = row.get("id") or row.get("uid")
            record_id = raw_id if isinstance(raw_id, int) else str(uuid.uuid4())

            enriched = enrich_record(row, table)
            vector = model.encode(enriched).tolist()

            if not vector or not isinstance(vector, list):
                print(f"⚠️ Skipped row with ID {record_id} — vector is invalid")
                continue

            print(f"📤 MySQL → Qdrant | Table: {table} | ID: {record_id}")
            print(f"    ↪️ Vector Dim: {len(vector)} | Summary: {enriched[:80]}...")

            try:
                result = qdrant.upsert(
                    collection_name=table,
                    points=[
                        PointStruct(
                            id=record_id,
                            vector=vector,
                            payload={**row, "text": enriched}
                        )
                    ]
                )
                print(f"✅ Qdrant acknowledged ID: {record_id}")
            except Exception as e:
                print(f"❌ Failed to upsert ID {record_id} → {str(e)}")

        print(f"✅ Finished syncing MySQL table: {table}")


def sync_mongo():
    db = get_mongo_connection()
    collections = db.list_collection_names()
    collections = [c for c in collections if c in ALLOWED_MONGO_COLLECTIONS]

    for collection_name in collections:
        documents = list(db[collection_name].find({}, {"_id": 0}))

        print(f"📥 Found {len(documents)} docs in MongoDB collection: {collection_name}")

        if not documents:
            print(f"⚠️ Skipped {collection_name} (no documents)")
            continue

        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )

        for doc in documents:
            record_id = str(uuid.uuid4())

            enriched = enrich_record(doc, collection_name)
            vector = model.encode(enriched).tolist()

            if not vector or not isinstance(vector, list):
                print(f"⚠️ Skipped Mongo doc ID {record_id} — invalid vector")
                continue

            print(f"📤 MongoDB → Qdrant | Collection: {collection_name} | ID: {record_id}")
            print(f"    ↪️ Vector Dim: {len(vector)} | Summary: {enriched[:80]}...")

            try:
                result = qdrant.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=record_id,
                            vector=vector,
                            payload={**doc, "text": enriched}
                        )
                    ]
                )
                print(f"✅ Qdrant acknowledged Mongo ID: {record_id}")
            except Exception as e:
                print(f"❌ Failed to upsert Mongo ID {record_id} → {str(e)}")

        print(f"✅ Finished syncing MongoDB collection: {collection_name}")
