import os
import time
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
 
 
load_dotenv()
 
# Qdrant connection settings
#QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
#QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = "https://483b08fc-720a-468f-8a76-6d9bfe2cc4b2.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.A866DHV8BNercMWiKhP0p8DEy47xwvq8aK6BnH1G2hE"
# Use a multilingual embedding model so texts in languages like Hindi are
# indexed correctly. Update the vector size to match the model's dimension.
VECTOR_SIZE = 384
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
 
 
def _extract_vector_size(info) -> int | None:
    """Best-effort extraction of vector size from `get_collection` result."""
    try:
        # Try new API structure first
        if hasattr(info, "config"):
            if hasattr(info.config, "params"):
                return info.config.params.vectors.size
            return info.config.vectors_config.size
        
        # Fall back to old structure
        if hasattr(info, "vectors_config"):
            return info.vectors_config.size
        
        # Last resort - try dictionary access
        if isinstance(info, dict):
            if "config" in info:
                cfg = info["config"]
                if "params" in cfg:
                    return cfg["params"]["vectors"]["size"]
                return cfg["vectors_config"]["size"]
            return info.get("vectors", {}).get("size")
            
        return None
    except (AttributeError, KeyError, TypeError):
        return None
 
def create_source_index(client: QdrantClient, collection_name: str, max_retries: int = 3) -> bool:
    """
    Create indexes for metadata fields using proper schema types.
    """
    index_fields = [
        ("source", "keyword"),  
        ("access_level", "keyword"),
        ("owner_id", "keyword"),
        ("team_id", "keyword"),
        ("page_number", "keyword"),
    ]
    all_success = True
 
    for field_name, schema_type in index_fields:
        for attempt in range(max_retries):
            try:
                field_schema = models.TextIndexParams(type=models.TextIndexType.TEXT)
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
                print(f"✅ Created '{field_name}' ({schema_type}) index")
                break
            except UnexpectedResponse as e:
                if "already exists" in str(e).lower() or e.status_code == 409:
                    print(f"ℹ️ Index '{field_name}' exists")
                    break
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                time.sleep(1)
        else:
            print(f"❌ Failed '{field_name}' after {max_retries} attempts")
            all_success = False
 
    return all_success
 
 
def ensure_collection_compatible(collection_name: str, vector_size: int) -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
 
    try:
        info = client.get_collection(collection_name)
        stored_size = _extract_vector_size(info)
        
        if stored_size is None:
            print(f"⚠️ Warning: Could not determine vector size for collection '{collection_name}'. Recreating.")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            # Wait a moment for collection to be ready
            time.sleep(2)
            # Create the index after recreation
            create_source_index(client, collection_name)
        elif stored_size != vector_size:
            print(
                f"⚠️ Warning: Collection '{collection_name}' has wrong size "
                f"(expected {vector_size}, found {stored_size}). Recreating."
            )
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            # Wait a moment for collection to be ready
            time.sleep(2)
            # Create the index after recreation
            create_source_index(client, collection_name)
        else:
            # Collection exists with correct size, ensure index exists
            create_source_index(client, collection_name)
            
    except Exception as exc:
        print(f"ℹ️ Creating collection '{collection_name}' with size {vector_size}.")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        # Wait a moment for collection to be ready
        time.sleep(2)
        # Create the index after creation
        create_source_index(client, collection_name)
 
 
def store_uploaded_file_to_qdrant(
    documents,
    collection_name: str = "chunks",
    recreate_collection: bool = False,
):
    if not documents:
        print("❌ No documents to store in Qdrant.")
        return
 
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
 
    if recreate_collection:
        print(f"🧹 Recreating Qdrant collection: {collection_name}")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        # Wait for collection to be ready
        time.sleep(2)
        # Create the index
        create_source_index(qdrant_client, collection_name)
    else:
        ensure_collection_compatible(collection_name, VECTOR_SIZE)
 
    print(f"📤 Uploading {len(documents)} documents to Qdrant...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
 
    Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=False  # Since we already conditionally recreate
    )
 
    print(f"✅ Successfully stored {len(documents)} chunks in Qdrant.")
 
 
def store_document_summary_to_qdrant(
    summary: str,
    metadata: dict,
    collection_name: str = "summaries",
    recreate_collection: bool = False,
):
    """Store a single document summary in Qdrant."""
    print(f"[DEBUG] Attempting to store summary: {repr(summary)}")
    if not summary or not summary.strip():
        print("❌ Empty summary, skipping storage.")
        return
 
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
 
    if recreate_collection:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        # Wait for collection to be ready
        time.sleep(2)
        # Create the index
        create_source_index(qdrant_client, collection_name)
    else:
        ensure_collection_compatible(collection_name, VECTOR_SIZE)
 
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    doc = Document(page_content=summary, metadata=metadata)
    Qdrant.from_documents(
        documents=[doc],
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=False,
    )
    print("✅ Stored document summary in Qdrant.")