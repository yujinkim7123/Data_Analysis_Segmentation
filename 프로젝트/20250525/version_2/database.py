from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid

# Qdrant 연결
qdrant = QdrantClient(host="localhost", port=6333)

# 임베딩 모델
embed_model = SentenceTransformer("intfloat/e5-large")

# ✅ 콜렉션 생성
def create_collection(collection_name, vector_size=1024):
    if not qdrant.collection_exists(collection_name):
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: {collection_name}")

# ✅ 문서 업로드 (임베딩에 메타데이터 포함)
def upload_documents(text_list, collection_name, tags=None, age_groups=None, summaries=None):
    points = []

    for i, text in enumerate(text_list):
        tag = tags[i] if tags else ""
        age = age_groups[i] if age_groups else ""
        summary = summaries[i] if summaries else ""

        # 메타데이터까지 포함해 임베딩할 텍스트 생성
        text_for_embedding = f"{text}\n태그: {tag}\n요약: {summary}\n연령대: {age}"
        vector = embed_model.encode(text_for_embedding).tolist()

        # Qdrant payload에는 원본 필드 저장
        payload = {
            "text": text,
            "tag": tag,
            "age_group": age,
            "summary": summary
        }

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        ))

    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"📌 Uploaded {len(points)} documents to '{collection_name}'")

# ✅ 검색 함수
def search_documents(query, collection_name, top_k=5):
    # 사용자가 메타데이터 포함 검색을 원한다면 여기서 확장 가능
    vector = embed_model.encode(query).tolist()

    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    return [
        {
            "text": h.payload.get("text"),
            "tag": h.payload.get("tag"),
            "age_group": h.payload.get("age_group"),
            "summary": h.payload.get("summary"),
            "score": round(h.score, 4)
        }
        for h in hits
    ]

# ✅ 테스트용 실행 코드
if __name__ == "__main__":
    # 예시 데이터
    web_texts = [
        "요즘 사람들은 땀을 많이 흘리는 여름철에 기능성 티셔츠를 선호한다.",
        "흡습속건 기술이 적용된 옷은 운동 후에도 쾌적함을 유지해준다."
    ]
    web_tags = ["여름의류", "운동복"]
    web_ages = ["20대", "30대"]
    web_summaries = ["여름철 땀 관리에 대한 수요", "운동 후 쾌적함에 대한 니즈"]

    # 1. 컬렉션 생성
    create_collection("web_data")

    # 2. 데이터 업로드
    upload_documents(web_texts, "web_data", web_tags, web_ages, web_summaries)

    # 3. 검색 테스트
    query = "여름에 덥지 않고 쾌적한 셔츠"
    print(f"\n🔍 검색: '{query}'\n")
    results = search_documents(query, "web_data")
    for r in results:
        print(f"문장   : {r['text']}")
        print(f"태그   : {r['tag']}")
        print(f"요약   : {r['summary']}")
        print(f"연령대 : {r['age_group']}")
        print(f"유사도 : {r['score']}\n")
