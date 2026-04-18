import json
import chromadb
import torch
import logging
from utils import get_semantic_vector
import torch
import chromadb
import logging
from underthesea import word_tokenize

logger = logging.getLogger(__name__)

DB_PATH = "TripMind/agent/tripmind_vector_db" #fixed data path 

def get_provinces_stats():
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        target = "tripmind_reviews"

        collection = client.get_collection(name=target)
        
        results = collection.get(include=['metadatas'])
        metadatas = results.get('metadatas', [])
        
        if not metadatas:
            return {}

        stats = {}

        for meta in metadatas:

            p_id = str(meta.get('province_id', '')).zfill(2)
            d_id = meta.get('destination_id')
            
            if p_id:
                if p_id not in stats:
                    stats[p_id] = {"total_reviews": 0, "unique_places": set()}
                
                stats[p_id]["total_reviews"] += 1
                if d_id:
                    stats[p_id]["unique_places"].add(str(d_id))

        final_result = {

            pid: {
                "total_reviews": data["total_reviews"],
                "unique_places": len(data["unique_places"])
            } for pid, data in stats.items()
        }
        
        logger.info(f"Thống kê xong: {len(final_result)} tỉnh thành.")
        return final_result

    except Exception as e:
        logger.error(f"Lỗi khi tính stats: {str(e)}")
        return {}
    
def ingest_to_chromadb(file_path, model, assets, device, batch_size=100):
    client = chromadb.PersistentClient(path="TripMind/agent/tripmind_vector_db")
    
    try:
        client.delete_collection("tripmind_reviews")
        logger.info("Đã xóa collection cũ để làm mới hoàn toàn")
    except:
        pass
    
    collection = client.create_collection(
        name="tripmind_reviews",
        metadata={"hnsw:space": "cosine"}
    )
    
    model.eval()
    
    valid_count = 0
    skipped_count = 0
    embeddings_batch, documents_batch, metadatas_batch, ids_batch = [], [], [], []
    
    detected_provinces = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                text = record.get("text", "").strip()
                
                if not text or len(text) < 10:
                    skipped_count += 1
                    continue

                p_id = str(record.get("province_id", ""))
                d_id = str(record.get("destination_id", ""))
                
                # FOR DEBUG, cuz province_id has only 2 index so p_id > 7, data is highly likely that the data is misaligned in columns.
                if not p_id or not d_id or len(p_id) > 7:
                    skipped_count += 1
                    continue

                detected_provinces.add(p_id)

                # Processing trip_type
                trip_raw = record.get("trip", "{}")
                try:
                    trip_data = json.loads(trip_raw) if isinstance(trip_raw, str) else trip_raw
                    trip_type = trip_data.get("trip_type") if trip_data else "any"
                    trip_type = trip_type.lower().strip() if trip_type else "any"
                    if trip_type in ["", "none", "null"]: trip_type = "any"
                except:
                    trip_type = "any"

                # Processing categories
                cat_raw = record.get("categories", "[]")
                try:
                    cat_list = json.loads(cat_raw) if isinstance(cat_raw, str) else cat_raw
                    if isinstance(cat_list, list) and len(cat_list) > 0:
                        cat_id = str(cat_list[0].get('id', '0'))
                        cat_name = cat_list[0].get('name', 'Chưa phân loại')
                    else:
                        cat_id = "0"
                        cat_name = "Chưa phân loại"
                except:
                    cat_id = "0"
                    cat_name = "Chưa phân loại"

                # Build semantic vector 
                vector = get_semantic_vector(text, model, assets, device)


                metadatas_batch.append({
                    "province_id": str(p_id).zfill(2), 
                    "destination_id": d_id,
                    "name": record.get("name", "").strip(),
                    "rating": float(record.get("rating_x", 0)),
                    "trip_type": trip_type,
                    "category": cat_name, 
                    "place_key": f"{p_id}_{d_id}"
                })
                
                embeddings_batch.append(vector)
                documents_batch.append(text)
                ids_batch.append(str(record.get("id_review", f"rev_{idx}")))
                
                valid_count += 1

                if len(embeddings_batch) >= batch_size:
                    collection.add(embeddings=embeddings_batch, documents=documents_batch, 
                                   metadatas=metadatas_batch, ids=ids_batch)
                    embeddings_batch, documents_batch, metadatas_batch, ids_batch = [], [], [], []
                
            except Exception as e:
                skipped_count += 1
                continue
    
    # Insert final batch
    if embeddings_batch:
        collection.add(embeddings=embeddings_batch, documents=documents_batch, 
                       metadatas=metadatas_batch, ids=ids_batch)

    logger.info(f"Hoàn tất! Phát hiện thực tế: {len(detected_provinces)} tỉnh.") #for debugging
    logger.info(f"Danh sách ID tỉnh: {sorted(list(detected_provinces))}") #for debugging
    return collection

def agent_1_output(
    user_query, 
    model, 
    word2idx, 
    assets, 
    device, 
    province_id, 
    trip_type='any', 
    n_places=15, 
    max_reviews_per_place=5
):
    try:
        from database import DB_PATH 
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("tripmind_reviews")
        
        # Filter by province_id (hard filter)
        p_id_str = str(province_id).zfill(2) 
        where_filter = {"province_id": p_id_str}
        logger.info(f"Agent 1 đang tìm kiếm tại tỉnh: {p_id_str}")

        model.eval()
        pred_cat_name = "Khác" 
        
        tokens = word_tokenize(user_query.lower())
        indices = [word2idx.get(t, 1) for t in tokens[:100]]
        indices += [0] * (100 - len(indices))
        input_tensor = torch.LongTensor([indices]).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                query_vector = output[0].cpu().numpy()[0].tolist()
                cat_logits = output[1]
                
                if assets and 'cat_encoder' in assets:
                    pred_cat_idx = torch.argmax(cat_logits, dim=1).item()
                    pred_cat_name = assets['cat_encoder'].inverse_transform([pred_cat_idx])[0]
                    print(f"Hệ thống dự đoán bạn đang tìm: {pred_cat_name}")
            else:
                query_vector = output.cpu().numpy()[0].tolist()

        # Search Vector database with rich query after hard filter
        rich_query = f"{user_query} {pred_cat_name}".lower()
        final_query_vector = get_semantic_vector(rich_query, model, word2idx, device)

        results = collection.query(
            query_embeddings=[final_query_vector],
            n_results=100, 
            where=where_filter 
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        # Grouping and scoring by sub-criteria
        places_dict = {}
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]

        target_trip_type = trip_type.lower().strip()

        for i in range(len(docs)):
            meta = metas[i]
            d_id = str(meta.get("destination_id", f"unknown_{i}"))
            db_cat_name = meta.get("category", "Khác")
            db_trip_type = str(meta.get("trip_type", "any")).lower()
            
            final_score = dists[i]
            
            # Bonus 1: Match predicted category (-0.15)
            if db_cat_name == pred_cat_name and pred_cat_name != "Khác":
                final_score -= 0.15

            # Bonus 2: Match trip_type (-0.10)
            if target_trip_type != "any" and db_trip_type == target_trip_type:
                final_score -= 0.10

            # Grouping reviews by destination_id
            if d_id not in places_dict:
                places_dict[d_id] = {
                    "destination_id": d_id,
                    "name": meta.get("name", "Địa danh chưa xác định"),
                    "category": db_cat_name,
                    "trip_type": db_trip_type,
                    "reviews": [docs[i]],
                    "min_score": final_score
                }
            else:
                if len(places_dict[d_id]["reviews"]) < max_reviews_per_place:
                    places_dict[d_id]["reviews"].append(docs[i])
                if final_score < places_dict[d_id]["min_score"]:
                    places_dict[d_id]["min_score"] = final_score

        # Select top N places based on min_score
        sorted_places = sorted(places_dict.values(), key=lambda x: x['min_score'])
        final_output = sorted_places[:n_places]

        print(f"Xử lý xong: Trích xuất được {len(final_output)} địa danh hàng đầu.")
        return final_output

    except Exception as e:
        logger.error(f"Lỗi Agent 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return []