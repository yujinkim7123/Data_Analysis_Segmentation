import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import sys
import uuid

# 인코딩 설정 (Windows 대응)
sys.stdout.reconfigure(encoding='utf-8')

# OpenAI 클라이언트
client = OpenAI(api_key="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")  # 실제 키로 대

# 모델 로딩
embed_model = SentenceTransformer("intfloat/e5-large")
qdrant = QdrantClient(host="localhost", port=6333)

# 키워드 확장 함수
def expand_keywords(keyword):
    prompt = f"""
다음은 기획자가 입력한 제품 기능 키워드야: "{keyword}"

이 키워드는 소비자가 일상에서 직접적으로 사용하지 않을 수 있어.
예를 들어 "살균"이라는 기능 키워드는 일반인이 "깨끗한", "위생적인", "세균 걱정 없는" 같은 말로 표현할 수 있지.

아래 기능 키워드를 바탕으로, **기능 그 자체를 직접 말하지 않더라도**,
일반 소비자가 실제로 사용할 법한 자연스러운 표현이나 어휘 10가지를 제안해줘.

형식은 간결하게 리스트 형식으로 제시해줘.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return [line.strip("-\u2022 ") for line in response.choices[0].message.content.split("\n") if line.strip()]

# Qdrant 문서 검색 함수 (문장 + 메타데이터 포함)
def search_context(keywords, collection, age_filter=None, top_k=3, min_score=0.75):
    results = []
    for kw in keywords:
        vector = embed_model.encode(kw)

        query_filter = None
        if age_filter:
            query_filter = Filter(
                must=[FieldCondition(key="age_group", match=MatchValue(value=age_filter))]
            )

        hits = qdrant.search(
            collection_name=collection,
            query_vector=vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter
        )
        for hit in hits:
            if hasattr(hit, "score") and hit.score >= min_score:
                payload = hit.payload
                results.append({
                    "text": payload.get("text"),
                    "tag": payload.get("tag"),
                    "summary": payload.get("summary"),
                    "age_group": payload.get("age_group"),
                    "score": round(hit.score, 4)
                })
    if len(results) == 0:
        return [{"text": "[데이터 부족] 의미 있는 문장이 검색되지 않았습니다."}]
    return results

# 페르소나 생성 함수
def generate_personas(expanded_keywords, context_web, context_product):
    def extract_texts(contexts):
        return [item['text'] for item in contexts if "text" in item]

    prompt = f"""
너는 소비자 인사이트 분석가야. 아래 문장과 키워드를 참고해서, 우리 제품을 인식하거나 아직 인식하지 못한 고객 중에서, 기획자가 제시한 기능 키워드를 필요로 할 수 있는 마이크로 페르소나 10명을 도출해줘.

각 페르소나는 아래 요소를 포함해야 해:
1. 페르소나 이름 또는 유형
2. 해당 페르소나가 제품을 알고 있는지 여부 (인식/미인식)
3. 주요 관심사 또는 일상 속 니즈
4. 제품 기능 키워드가 해당 페르소나에게 왜 필요하다고 판단되는지 (상황 중심 설명)

---

📌 제품 기능 키워드:
{', '.join(expanded_keywords)}

📂 웹 크롤링 문서에서 추출된 문장:
{chr(10).join(extract_texts(context_web))}

📦 제품 기능 문서에서 추출된 문장:
{chr(10).join(extract_texts(context_product))}

위 정보를 참고해서, 서로 다른 유형의 마이크로 페르소나 10명을 리스트 형식으로 제안해줘.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content, prompt

# Streamlit UI
st.title("🧠 마이크로 페르소나 생성 챗봇")

if "history" not in st.session_state:
    st.session_state.history = []

# ✅ 나이 필터 선택 UI
age_filter = st.selectbox("🎯 타겟 연령대를 선택하세요 (필터링)", ["", "10대", "20대", "30대", "40대", "50대 이상"], index=0)

user_input = st.chat_input("제품 기능 키워드를 입력하세요 예: 흡습속건 티셔츠")

if user_input:
    with st.spinner("페르소나 생성 중..."):
        st.session_state.history.append(("user", user_input))

        # 1. 키워드 확장
        expanded = expand_keywords(user_input)

        # 2. 문장 검색 (연령 필터 반영)
        context_web = search_context(expanded, "web_data", age_filter=age_filter)
        context_product = search_context(expanded, "product_feature_data", age_filter=age_filter)

        # 3. 데이터 부족 체크
        if all("데이터 부족" in c.get("text", "") for c in (context_web + context_product)):
            st.warning("🔍 키워드와 관련된 문장이 충분하지 않아 페르소나를 생성할 수 없습니다.")
        else:
            # 4. 페르소나 생성
            personas, used_prompt = generate_personas(expanded, context_web, context_product)
            st.session_state.history.append(("bot", personas))

            # 확장 키워드 표시
            with st.expander("📌 확장된 키워드"):
                for kw in expanded:
                    st.markdown(f"- {kw}")

            # 문장 + 메타데이터 출력
            def display_docs(context_list, title):
                st.markdown(f"#### {title}")
                for doc in context_list:
                    st.markdown(f"""
- **문장**: {doc.get('text')}
  - 태그: {doc.get('tag')}
  - 요약: {doc.get('summary')}
  - 연령대: {doc.get('age_group')}
  - 유사도: {doc.get('score')}
                    """)

            with st.expander("🔍 Qdrant에서 추출된 문서 보기"):
                display_docs(context_web, "🌐 웹 데이터")
                display_docs(context_product, "🏗️ 제품 기능 문서")

            # 프롬프트 보기
            with st.expander("🧠 LLM에게 전달된 실제 Prompt 보기"):
                st.code(used_prompt, language="markdown")

# 히스토리 출력
for sender, msg in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(msg)
