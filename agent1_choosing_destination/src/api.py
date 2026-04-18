from flask import Flask, request, jsonify
import requests
import torch
import pickle
import os
import logging
from together import Together 
from model import TripMindEncoder
from database import get_provinces_stats, agent_1_output

# CONFIG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TripMind-Gateway")

app = Flask(__name__)

MODEL = None
WORD2IDX = None
ASSETS = None
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
PROVINCE_STATS = None

AGENT_2_URL = "http://localhost:8000/ranking"
AGENT_3_URL = "http://localhost:9000/optimize"

TOGETHER_CLIENT = Together(api_key="TOGETHER_AI_API_KEY")

def load_system():
    global MODEL, ASSETS, WORD2IDX, PROVINCE_STATS
    try:
        logger.info(f"Khởi tạo hệ thống trên thiết bị: {DEVICE}...")
        weights_path = "TripMind/agent1_choosing_destination/weights" # Cập nhật lại đúng đường dẫn đã lưu khi train
        
        with open(os.path.join(weights_path, "assets.pkl"), "rb") as f:
            ASSETS = pickle.load(f)
        
        WORD2IDX = ASSETS.get('word2idx')
        vocab_size = ASSETS.get('vocab_size')
        num_categories = len(ASSETS['cat_encoder'].classes_)
        
        # Đọc các siêu tham số của kiến trúc BiLSTM
        embed_dim = ASSETS.get('embed_dim', 256)
        hidden_size = ASSETS.get('hidden_size', 128)
        num_layers = ASSETS.get('num_layers', 2)
        
        # Khởi tạo mô hình với các tham số mới
        MODEL = TripMindEncoder(
            vocab_size=vocab_size,
            num_categories=num_categories,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0 # Thiết lập dropout = 0 trong quá trình suy luận (inference)
        ).to(DEVICE)
        
        weights_file = os.path.join(weights_path, "encoder_weights.pth")
        state_dict = torch.load(weights_file, map_location=DEVICE)
        MODEL.load_state_dict(state_dict)
        MODEL.eval()
        
        PROVINCE_STATS = get_provinces_stats()
        
        logger.info("Hệ thống Gateway tích hợp 4 Agent đã sẵn sàng!")
        
    except Exception as e:
        logger.error(f"Lỗi khởi động hệ thống: {str(e)}")
        raise e

def generate_storytelling(itinerary_data, user_query):
    """
    Agent 4: Using LLM to generate natural travel advice.
    """
    try:
        places_summary = ""
        for i, p in enumerate(itinerary_data, 1):
            score_pct = round(p.get('final_score', 0) * 100, 1)
            places_summary += f"{i}. {p['name']} (Độ hài lòng: {score_pct}%)\n"
            if p.get('reviews'):
                places_summary += f"   - Review: {p['reviews'][0][:150]}...\n"

        prompt = f"""
                Bạn là chuyên gia tư vấn du lịch của TripMind. 
                Người dùng hỏi: "{user_query}"

                Chúng tôi đã tìm ra và tối ưu lộ trình di chuyển cho 5 địa điểm tuyệt vời nhất:
                {places_summary}

                Hãy viết một đoạn phản hồi ngắn gọn (khoảng 150-200 chữ), thân thiện để chào mừng người dùng.
                Yêu cầu:
                - Giải thích rằng lộ trình đã được sắp xếp theo thứ tự di chuyển tối ưu nhất.
                - Nhấn mạnh vào chất lượng dịch vụ dựa trên điểm số hài lòng.
                - Trình bày bằng tiếng Việt, giọng văn chuyên nghiệp nhưng gần gũi.
                """

        response = TOGETHER_CLIENT.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Agent 4 LLM Error: {e}")
        return "Chúc bạn có một hành trình tuyệt vời với lộ trình di chuyển tối ưu mà chúng tôi đã chuẩn bị!"

@app.route('/api/v1/recommend', methods=['POST'])
def recommend_places():
    try:
        data = request.get_json()
        query = data.get('query')
        province_id = data.get('province_id')
        
        if not query or province_id is None:
            return jsonify({"success": False, "error": "Missing query or province_id"}), 400
        
        p_id_str = str(province_id).zfill(2) 
        trip_type = data.get('trip_type', 'any')
        n_places = min(int(data.get('n_places', 5)), 10)
        
        # Step 1: AGENT 1 (Recall) 
        candidates = agent_1_output(
            user_query=query, model=MODEL, word2idx=WORD2IDX, assets=ASSETS,
            device=DEVICE, province_id=p_id_str, trip_type=trip_type,
            n_places=15, max_reviews_per_place=5
        )
        
        if not candidates:
            return jsonify({"success": True, "data": [], "message": "Không tìm thấy kết quả"}), 200

        # Step 2: AGENT 2 (Sentiment Ranking) 
        try:
            res2 = requests.post(AGENT_2_URL, json=candidates, timeout=10)
            ranked_places = res2.json() if res2.status_code == 200 else candidates
        except:
            ranked_places = candidates

        top_candidates = ranked_places[:n_places]

        # Step 3: AGENT 3 (Route Optimization) 
        try:
            res3 = requests.post(AGENT_3_URL, json=top_candidates, timeout=10)
            final_itinerary = res3.json() if res3.status_code == 200 else top_candidates
        except:
            final_itinerary = top_candidates

        # Step 4: AGENT 4 (Together.ai Storytelling)
        logger.info("Đang khởi tạo Agent 4 để tổng hợp câu trả lời...")
        ai_message = generate_storytelling(final_itinerary, query)

        return jsonify({
            "success": True,
            "recommendation_text": ai_message, # Message from LLM
            "data": final_itinerary,           
            "metadata": {
                "province_id": p_id_str,
                "status": "Success",
                "agents_active": 4
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Pipeline Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "agents": {"A2": AGENT_2_URL, "A3": AGENT_3_URL, "A4": "Together.ai Active"}
    }), 200

if __name__ == "__main__":
    load_system()
    app.run(host='0.0.0.0', port=5000, debug=False)
