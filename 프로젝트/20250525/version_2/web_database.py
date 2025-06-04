# upload_web_data_batch.py

import os
import glob
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# 설정
FOLDER_PATH = "./web_csvs"  # 웹 크롤링 csv들이 들어있는 폴더
COLLECTION_NAME = "web_data"

# Qdrant & 모델 초기화
qdrant = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer("intfloat/e5-large")

# 컬렉션 생성
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"✅ 컬렉션 생성됨: {COLLECTION_NAME}")

# 폴더 내 모든 CSV 불러오기
csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError("❌ CSV 파일이 없습니다.")

total_uploaded = 0

for file in csv_files:
    print(f"\n📂 처리 중: {file}")
    df = pd.read_csv(file)

    if "text" not in df.columns:
        print(f"❌ 'text' 컬럼이 없어 건너뜀: {file}")
        continue

    texts = df["text"].dropna().astype(str).tolist()
    tags = df["tag"].astype(str).tolist() if "tag" in df.columns else ["" for _ in texts]
    ages = df["age_group"].astype(str).tolist() if "age_group" in df.columns else ["" for _ in texts]
    summaries = df["summary"].astype(str).tolist() if "summary" in df.columns else ["" for _ in texts]

    for i, text in enumerate(texts):
        # 중복 체크
        existing, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="text", match=MatchValue(value=text))]),
            limit=1,
            with_payload=False
        )
        if existing:
            print(f"⚠️ 중복 문서 건너뜀: {text[:30]}...")
            continue

        enriched_text = f"{text}\n태그: {tags[i]}\n요약: {summaries[i]}\n연령대: {ages[i]}"
        vector = embed_model.encode(enriched_text).tolist()

        payload = {
            "text": text,
            "tag": tags[i],
            "age_group": ages[i],
            "summary": summaries[i]
        }

        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
        total_uploaded += 1
        print(f"✅ 문서 저장됨: {text[:30]}...")

print(f"\n🎉 총 {total_uploaded}개 문서가 Qdrant에 업로드되었습니다.")