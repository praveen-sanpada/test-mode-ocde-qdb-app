from fastapi import APIRouter
from app.services.sync_service import sync_all

router = APIRouter()

@router.post("/sync-all")
def sync_all_endpoint():
    sync_all()
    return {"status": "Sync complete"}
