from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid

# Qdrant 연결
qdrant = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer("intfloat/e5-large")

# 💾 예제 데이터
product_texts = [
    {"text": "이 티셔츠는 땀을 빠르게 흡수하고 건조시킵니다.", "tag": "흡습속건"},
    {"text": "폴리에스터 소재로 통기성과 흡습성이 뛰어납니다.", "tag": "기능성 소재"},
]

web_texts = [
    {"text": "요즘 운동할 때는 땀 배출이 잘 되는 옷을 입어요.", "tag": "운동복", "summary": "운동 시 땀 배출 중요", "age_group": "20대"},
    {"text": "더운 날엔 땀이 금방 마르는 옷이 최고예요.", "tag": "여름 패션", "summary": "흡습속건 니즈", "age_group": "30대"},
]

# 🧱 컬렉션 생성
def create_collection(name):
    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"✅ 컬렉션 생성됨: {name}")

# 📌 업로드 함수
def upload_points(data_list, collection_name):
    enriched = []
    for doc in data_list:
        enriched_text = doc["text"]
        if collection_name == "web_data":
            enriched_text += f"\n태그: {doc.get('tag')}\n요약: {doc.get('summary')}\n연령대: {doc.get('age_group')}"
            payload = {
                "text": doc["text"],
                "tag": doc.get("tag", ""),
                "summary": doc.get("summary", ""),
                "age_group": doc.get("age_group", "")
            }
        else:
            enriched_text += f"\n태그: {doc.get('tag')}"
            payload = {
                "text": doc["text"],
                "tag": doc.get("tag", "")
            }
        vector = embed_model.encode(enriched_text).tolist()
        enriched.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    qdrant.upsert(collection_name=collection_name, points=enriched)
    print(f"📦 {len(enriched)}건 저장 완료 ({collection_name})")

# ✅ 실행
if __name__ == "__main__":
    create_collection("product_feature_data")
    create_collection("web_data")
    upload_points(product_texts, "product_feature_data")
    upload_points(web_texts, "web_data")
