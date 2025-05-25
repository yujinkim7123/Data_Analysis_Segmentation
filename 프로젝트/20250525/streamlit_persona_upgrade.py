import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import sys

sys.stdout.reconfigure(encoding='utf-8')

client = OpenAI(api_key="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")  # 🔐 실제 키로 교체
embed_model = SentenceTransformer("intfloat/e5-large")
qdrant = QdrantClient(host="localhost", port=6333)

# 🔍 키워드 확장
def expand_keywords(keyword):
    prompt = f"""
당신은 소비자 언어 분석 전문가이자, 마케팅 인사이트 분석가입니다.

아래는 기획자가 정의한 제품 기능 키워드입니다:
🔧 "{keyword}"

하지만 소비자는 일상 대화에서 해당 기능을 직접적으로 언급하지 않을 수 있습니다.
예: "살균" → "깨끗한", "세균 걱정 없는" 등

아래 조건을 만족하는 자연스러운 소비자 표현 10가지 제안해주세요:

✅ 기능을 몰라도 일상에서 쓰는 말  
✅ 감정/상황/니즈를 담은 표현  
✅ 광고 문구가 아닌 진짜 사용자 말투  

결과는 리스트 형식으로:
- 땀이 나도 시원한
- 하루 종일 뽀송한 느낌
    """
    res = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return [line.strip("-• ") for line in res.choices[0].message.content.split("\n") if line.strip()]

# 🔎 Qdrant 문서 검색
def search_context(keywords, collection, age_filters=None, tag=None, top_k=3, min_score=0.75):
    results = []
    seen = set()
    for kw in keywords:
        vector = embed_model.encode(kw)
        must = []
        if collection == "web_data" and age_filters:
            must.append(FieldCondition(key="age_group", match=MatchValue(value=age_filters[0])))
        if collection == "product_feature_data" and tag:
            must.append(FieldCondition(key="tag", match=MatchValue(value=tag)))
        query_filter = Filter(must=must) if must else None

        hits = qdrant.search(
            collection_name=collection,
            query_vector=vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter
        )
        for hit in hits:
            text = hit.payload.get("text")
            if text and text not in seen and hit.score >= min_score:
                seen.add(text)
                results.append({
                    "text": text,
                    "tag": hit.payload.get("tag"),
                    "summary": hit.payload.get("summary"),
                    "age_group": hit.payload.get("age_group"),
                    "score": round(hit.score, 4)
                })
    return results if results else [{"text": "[데이터 부족] 의미 있는 문장이 검색되지 않았습니다."}]

# 🤖 페르소나 생성
def generate_personas(expanded_keywords, context_web, context_product, user_input):
    def extract_texts(contexts): return [i['text'] for i in contexts if "text" in i]
    prompt = f"""
당신은 고객 인사이트 분석가입니다.
너는 소비자 인사이트 분석가야. 아래 문장과 키워드를 참고해서, 우리 제품을 인식하거나 아직 인식하지 못한 고객 중에서, 기획자가 제시한 기능 키워드를 필요로 할 수 있는 마이크로 페르소나 10명을 도출해줘.

각 페르소나는 아래 항목을 포함합니다:
1. 🧍‍♀️ 페르소나 이름 또는 유형
2. 🔍 제품 인식 상태 (인식/미인식)
3. 🧠 주요 관심사 또는 생활 속 니즈
4. ✅ 기능이 왜 필요한지 (상황 중심 설명)

---

📌 제품 기능 키워드
{user_input}

📌 기능 키워드 및 확장:
{', '.join(expanded_keywords)}

🌐 웹 문서 문장:
{chr(10).join(extract_texts(context_web))}

🏗️ 기능 문서 문장:
{chr(10).join(extract_texts(context_product))}
---
"""
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return res.choices[0].message.content, prompt

# ✅ UI
st.set_page_config("마이크로 페르소나 생성기", layout="wide")
st.title("🧠 마이크로 페르소나 생성기")

age_filters = st.multiselect("🎯 웹 문서 연령 필터", ["10대", "20대", "30대", "40대", "50대 이상"])
user_input = st.text_input("📝 제품 기능 키워드 입력", placeholder="예: 흡습속건 티셔츠")

if st.button("🚀 페르소나 생성하기") and user_input:
    with st.spinner("🔄 확장 키워드 및 문서 검색 중..."):
        expanded = expand_keywords(user_input)
        context_web = search_context(expanded, "web_data", age_filters=age_filters)
        context_product = search_context([user_input], "product_feature_data", tag=user_input)

        if all("데이터 부족" in x.get("text", "") for x in (context_web + context_product)):
            st.warning("📉 관련 문장이 충분하지 않습니다.")
        else:
            personas, used_prompt = generate_personas(expanded, context_web, context_product, user_input)
            st.success("✅ 페르소나 생성 완료")

            st.markdown("### 📌 확장된 키워드")
            st.markdown(" ".join([f"`{kw}`" for kw in expanded]))

            with st.expander("📂 Qdrant에서 추출된 문서"):
                for doc in context_web + context_product:
                    st.markdown(f"""
**문장**: {doc.get("text")}  
- 태그: {doc.get("tag")}
- 요약: {doc.get("summary")}
- 연령대: {doc.get("age_group")}
- 유사도: {doc.get("score")}
""")

            st.markdown("### 🎯 마이크로 페르소나 카드")
            for i, block in enumerate(personas.strip().split("\n\n")):
                st.markdown(f"#### 🧠 페르소나 {i+1}")
                st.markdown(block)

            with st.expander("🧠 LLM에게 전달된 프롬프트"):
                st.code(used_prompt, language="markdown")
