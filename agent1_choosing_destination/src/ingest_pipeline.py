import json
import chromadb
import torch
import pickle
import warnings
import os
from model import TripMindEncoder
from utils import get_semantic_vector
warnings.filterwarnings("ignore")

# Config
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy Ingest trên thiết bị: {DEVICE}")

ASSETS_PATH = "TripMind/agent/weights/assets.pkl"
WEIGHTS_PATH = "TripMind/agent/weights/encoder_weights.pth"
DB_PATH = "TripMind/agent/tripmind_vector_db"
DATA_PATH = "TripMind/data/cleaned_data.jsonl"

def load_encoder():
    with open(ASSETS_PATH, "rb") as f:
        assets = pickle.load(f)
    
    vocab_size = assets['vocab_size']
    num_categories = len(assets['cat_encoder'].classes_)
    
    embed_dim = assets.get('embed_dim', 256)
    hidden_size = assets.get('hidden_size', 128)
    num_layers = assets.get('num_layers', 2)
    
    encoder = TripMindEncoder(
        vocab_size=vocab_size,
        num_categories=num_categories,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0 # Chạy inference không cần dropout
    )
    
    encoder.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    encoder.to(DEVICE).eval()
    return encoder, assets

def ingest_data():
    encoder, assets = load_encoder()
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection("tripmind_reviews")
        print("--- Đã xóa collection cũ ---")
    except:
        pass

    collection = client.create_collection(
        name="tripmind_reviews", 
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100
    ids, docs, metas, embs = [], [], [], []

    print(f"Bắt đầu nạp dữ liệu từ: {DATA_PATH}")

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                name = data.get('name', 'Unknown').strip()
                review_text = data.get('text', '').strip()
                p_id = str(data.get("province_id", "")).zfill(2) 
                
                if not review_text and not name:
                    continue

                # Processing Categories and Trip Type for Metadata
                cats = data.get('categories', [])
                cat_name = cats[0].get('name', 'Khác') if isinstance(cats, list) and cats else "Khác"
                
                trip_raw = data.get("trip", "{}")
                trip_data = json.loads(trip_raw) if isinstance(trip_raw, str) else (trip_raw or {})
                trip_type = str(trip_data.get("trip_type", "any")).lower()

                # Build rich text for boosting semantic search
                # Combine Name + Category + Review 
                rich_text = f"Địa danh: {name}. Loại hình: {cat_name}. Đánh giá: {review_text}".lower()

                vector = get_semantic_vector(rich_text, encoder, assets, DEVICE)

                metadata = {
                    "province_id": p_id,
                    "destination_id": str(data.get("destination_id")),
                    "name": name,
                    "category": cat_name,
                    "trip_type": trip_type,
                    "rating": float(data.get("rating_x", 0))
                }

                ids.append(str(data.get('id_review', f"rev_{i}")))
                docs.append(review_text) 
                metas.append(metadata)
                embs.append(vector)

                if len(ids) >= batch_size:
                    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                    if (i + 1) % 500 == 0:
                        print(f"Đã nạp {i+1} bản ghi...")
                    ids, docs, metas, embs = [], [], [], []

            except Exception as e:
                continue

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    
    print(f"Hoàn thành! Tổng cộng: {collection.count()} bản ghi đã sẵn sàng cho Agent 1.")

if __name__ == "__main__":
    ingest_data()
