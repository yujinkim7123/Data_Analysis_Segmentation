# qdrant_admin_tool.py

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct, VectorParams, Distance
import uuid
import pandas as pd
import io

# Qdrant 연결
client = QdrantClient(host="localhost", port=6333)

st.set_page_config(page_title="Qdrant Admin Tool", layout="wide")
st.title("🧩 Qdrant 문서 관리 도구")

# ✅ 컬렉션 선택
with st.sidebar:
    st.header("📁 컬렉션 선택 및 필터")
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    selected_collection = st.selectbox("컬렉션을 선택하세요", collection_names)

    query_text = st.text_input("🔍 포함 키워드", placeholder="예: 흡습속건")
    filter_tag = st.text_input("🔖 태그 필터", placeholder="예: 여름")
    filter_age = st.selectbox("🎯 연령대 필터", ["", "10대", "20대", "30대", "40대", "50대 이상"])
    delete_text = st.text_input("🗑️ 삭제할 정확한 텍스트 입력", placeholder="문서 전체 내용이 정확히 일치해야 함")

st.markdown("---")

# ✅ 데이터 조회
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔎 문서 조회"):
        filter_conditions = []
        if query_text:
            filter_conditions.append(FieldCondition(key="text", match=MatchValue(value=query_text)))
        if filter_tag:
            filter_conditions.append(FieldCondition(key="tag", match=MatchValue(value=filter_tag)))
        if filter_age:
            filter_conditions.append(FieldCondition(key="age_group", match=MatchValue(value=filter_age)))

        filter_condition = Filter(must=filter_conditions) if filter_conditions else None

        results, _ = client.scroll(
            collection_name=selected_collection,
            scroll_filter=filter_condition,
            limit=100,
            with_payload=True
        )

        st.session_state["last_results"] = results

# 결과 출력
if "last_results" in st.session_state:
    st.subheader(f"🔍 조회 결과 ({len(st.session_state['last_results'])}건)")
    for doc in st.session_state["last_results"]:
        with st.expander(f"📄 문서 ID: {doc.id}"):
            for k, v in doc.payload.items():
                st.markdown(f"- **{k}**: {v}")

# ✅ 데이터 삭제
st.markdown("---")
if st.button("🗑️ 삭제 실행") and delete_text:
    delete_filter = Filter(
        must=[FieldCondition(key="text", match=MatchValue(value=delete_text))]
    )
    result = client.delete(collection_name=selected_collection, points_selector=delete_filter)
    st.success("✅ 삭제 요청 완료됨")
    st.json(result.dict())

# ✅ 데이터 추가
st.markdown("---")
st.subheader("➕ 문서 수동 추가")
new_text = st.text_area("문서 내용 입력", height=100)
col_tag, col_summary, col_age = st.columns(3)
with col_tag:
    new_tag = st.text_input("태그")
with col_summary:
    new_summary = st.text_input("요약")
with col_age:
    new_age_group = st.selectbox("연령대", ["", "10대", "20대", "30대", "40대", "50대 이상"])

if st.button("⬆️ 문서 추가") and new_text:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("intfloat/e5-large")

    if selected_collection == "product_feature_data":
        enriched_text = f"{new_text}\n태그: {new_tag}"
        payload = {
            "text": new_text,
            "tag": new_tag
        }
    else:
        enriched_text = f"{new_text}\n태그: {new_tag}\n요약: {new_summary}\n연령대: {new_age_group}"
        payload = {
            "text": new_text,
            "tag": new_tag,
            "summary": new_summary,
            "age_group": new_age_group
        }

    vector = embed_model.encode(enriched_text).tolist()
    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
    client.upsert(collection_name=selected_collection, points=[point])
    st.success("✅ 문서 추가 완료")

# ✅ CSV 업로드로 데이터 추가
st.markdown("---")
st.subheader("📁 CSV 파일 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (text, tag, summary, age_group 컬럼 필요)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("❌ 'text' 컬럼이 있어야 합니다.")
    else:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("intfloat/e5-large")

        texts = df["text"].astype(str).tolist()
        tags = df["tag"].astype(str).tolist() if "tag" in df.columns else ["" for _ in texts]
        summaries = df["summary"].astype(str).tolist() if "summary" in df.columns else ["" for _ in texts]
        ages = df["age_group"].astype(str).tolist() if "age_group" in df.columns else ["" for _ in texts]

        points = []
        for i, text in enumerate(texts):
            if selected_collection == "product_feature_data":
                enriched_text = f"{text}\n태그: {tags[i]}"
                payload = {
                    "text": text,
                    "tag": tags[i]
                }
            else:
                enriched_text = f"{text}\n태그: {tags[i]}\n요약: {summaries[i]}\n연령대: {ages[i]}"
                payload = {
                    "text": text,
                    "tag": tags[i],
                    "summary": summaries[i],
                    "age_group": ages[i]
                }

            vector = embed_model.encode(enriched_text).tolist()
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        client.upsert(collection_name=selected_collection, points=points)
        st.success(f"✅ 총 {len(points)}개 문서가 업로드되었습니다.")
