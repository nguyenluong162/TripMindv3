import chromadb

DB_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/tripmind_vector_db"

def debug_database():
    print("="*50)
    print("🔍 ĐANG KIỂM TRA DATABASE TẠI:", DB_PATH)
    print("="*50)
    
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("tripmind_reviews")
        print("so collection",collection.count())
        
        total_count = collection.count()
        print(f"✅ Tổng số bản ghi trong DB: {total_count}")
        
        if total_count == 0:
            print("❌ Database rỗng! Hãy chạy lại ingest_pipeline.py")
            return

        # Lấy thử một vài bản ghi để soi metadata
        results = collection.get(limit=5, include=['metadatas'])
        
        print("\n--- KIỂM TRA KIỂU DỮ LIỆU METADATA ---")
        for i, meta in enumerate(results['metadatas']):
            p_id = meta.get('province_id')
            p_id_type = type(p_id).__name__
            print(f"Mẫu {i+1}: province_id = '{p_id}' | Kiểu dữ liệu: {p_id_type}")
            
        # Thống kê danh sách các tỉnh thực tế đang có
        all_data = collection.get(include=['metadatas'])
        provinces_in_db = set(str(m.get('province_id')) for m in all_data['metadatas'])
        
        print("\n--- DANH SÁCH PROVINCE_ID ĐANG CÓ TRONG DB ---")
        print(sorted(list(provinces_in_db)))
        print(f"Tổng cộng: {len(provinces_in_db)} tỉnh.")

        # Kiểm tra thử một lệnh query lọc
        test_id = list(provinces_in_db)[0] if provinces_in_db else "None"
        print(f"\n--- TEST THỬ LỆNH LỌC VỚI ID: {test_id} ---")
        test_query = collection.get(where={"province_id": test_id}, limit=1)
        if len(test_query['ids']) > 0:
            print(f"✅ Thành công: Tìm thấy dữ liệu khi lọc bằng chuỗi '{test_id}'")
        else:
            print(f"❌ Thất bại: Không tìm thấy dữ liệu khi lọc bằng chuỗi '{test_id}'")

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    debug_database()

# import chromadb

# DB_PATH = "/Users/trannguyenmyanh/Documents/TripMind/agent/tripmind_vector_db"
# client = chromadb.PersistentClient(path=DB_PATH)

# # Lệnh này sẽ liệt kê tất cả các collection đang có
# collections = client.list_collections()
# print("Danh sách các collection đang có trong DB của bạn:")
# for col in collections:
#     print(f"- {col.name}")