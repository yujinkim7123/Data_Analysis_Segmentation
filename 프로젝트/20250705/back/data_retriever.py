from datetime import datetime
from collections import defaultdict
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range, SearchRequest, NamedVector
from .utils import get_embedding_models, get_qdrant_client, get_openai_client, parse_natural_date


def expand_keywords(keyword: str, product_type: str = None):
    """
    LLM을 사용하여 키워드를 확장합니다.
    1. 상황/경험 기반의 문장
    2. 유사/연관어 기반의 문장
    두 종류를 모두 생성하도록 고도화되었습니다.
    """
    client = get_openai_client(async_client=False)
    product_context = f"🧰 제품 카테고리: {product_type}\n이 맥락을 반영하여 아래 내용을 생성해주세요." if product_type else ""
    
    if product_type == None:
        product_context = ""

    # [수정된 프롬프트]
    prompt = f"""
    당신은 소비자 언어와 제품의 기술 용어를 모두 이해하는 소비자 인사이트 전문가입니다.
    아래 주어진 기능 키워드와 관련하여, 다음 두 가지 종류의 소비자 표현을 합쳐서 10~12개 생성해주세요.

    **기능 키워드: "{keyword}"**
    {product_context}
    ---

    ### 1. 상황/경험/니즈를 표현하는 문장 (5~6개)
    - 소비자는 "{keyword}"라는 단어를 직접 사용하지 않습니다.
    - 해당 기능이 **필요한 특정 상황, 겪고 있는 불편함, 또는 얻고 싶은 가치**를 중심으로 문장을 만들어주세요.
    - 예시 ('살균' 키워드): "아이가 아토피가 있어서 옷을 매번 삶아 입히는데 너무 번거로워요."

    ### 2. 키워드를 다른 용어로 표현하는 문장 (4~5개)
    - 소비자는 "{keyword}" 대신, 광고나 제품 상세페이지에서 본 **유사어, 연관 기술/마케팅 용어**를 사용하여 말하기도 합니다.
    - 아래 예시처럼, "{keyword}"의 핵심 가치를 전달하는 다른 표현을 사용한 문장을 만들어주세요.
    - 예시 ('살균' 키워드): "스팀으로 99.9% 세균을 박멸해준다니 안심돼요.", "UV 램프로 위생적으로 관리할 수 있어서 마음에 들어요."
    
    ---
    **[공통 제약 조건]**
    - 단순 칭찬("좋아요")이나 감정 표현은 지양해주세요.
    - 실제 사용자가 남긴 후기나 커뮤니티 게시글처럼 자연스러운 말투여야 합니다.
    - 리스트 형식으로, 각 항목은 1문장으로 출력해주세요.
    """
    
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.7)
        expanded_list = [line.strip("-• ") for line in res.choices[0].message.content.split("\n") if line.strip() and "###" not in line]
        
        # --- [핵심 수정] ---
        # 1. 원본 키워드를 리스트의 맨 앞에 추가합니다.
        # 2. set으로 변환했다가 다시 list로 만들어 혹시 모를 중복을 제거합니다.
        final_keywords = [keyword] + expanded_list
        return list(set(final_keywords))
    except Exception as e:
        print(f"키워드 확장 중 오류 발생: {e}")
        return [keyword]

def summarize_text(text_to_summarize: str):
    """LLM을 사용하여 텍스트를 요약합니다."""
    client = get_openai_client(async_client=False)
    prompt = f"""
    당신은 소비자 언어 분석 전문가입니다. 다음은 소비자의 글 원문입니다.
    이 글에서 **잠재고객의 니즈, 불편, 상황, 행동**이 드러나는 핵심 문장을 중심으로,
    원문 표현을 최대한 살려 3~5문장으로 간결하게 요약해주세요.
    원문: {text_to_summarize}
    """
    try:
        # 비동기 호출을 동기적으로 실행
        res = client.chat.completions.create(client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        ))
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"텍스트 요약 중 오류 발생: {e}")
        return text_to_summarize

def run_rrf_search(keywords: list, date_range: tuple | None = None, top_k=2000, score_threshold=0.5):
    """RRF 기반 하이브리드 검색"""
    meaning_model, topic_model =get_embedding_models()
    qdrant = get_qdrant_client()
    all_hits_map = {}
    rrf_scores = defaultdict(float)
    K_RRF = 60

    # --- [신규] 날짜 필터 생성 로직 ---
    must_conditions = []
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        print(f"🌀 Applying date filter: {start_date} ~ {end_date}")
        must_conditions.append(FieldCondition(
            key="date_timestamp", 
            range=Range(
                gte=int(datetime.combine(start_date, datetime.min.time()).timestamp()),
                lte=int(datetime.combine(end_date, datetime.max.time()).timestamp())
            )
        ))
    query_filter = Filter(must=must_conditions) if must_conditions else None


    for kw in keywords:
        meaning_vec = meaning_model.encode("query: " + kw)
        topic_vec = topic_model.encode(kw)
        search_results = qdrant.search_batch(
            collection_name="web_data",
            requests=[
                SearchRequest(vector=NamedVector(name="meaning", vector=meaning_vec.tolist()), limit=top_k, with_payload=True, filter=query_filter, score_threshold=score_threshold),
                SearchRequest(vector=NamedVector(name="topic", vector=topic_vec.tolist()), limit=top_k, with_payload=True, filter=query_filter, score_threshold=score_threshold)
            ]
        )
        for hits in search_results:
            for rank, hit in enumerate(hits):
                rrf_scores[hit.id] += 1 / (rank + K_RRF)
                if hit.id not in all_hits_map:
                    all_hits_map[hit.id] = hit

    sorted_hit_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)
    results = []
    seen_text = set()
    for hit_id in sorted_hit_ids:
        if len(results) >= top_k: break
        hit = all_hits_map[hit_id]
        original_sentence = hit.payload.get("sentence", "")
        if original_sentence and original_sentence not in seen_text:
            result_payload = hit.payload.copy()
            result_payload['id'] = str(hit.id)
            result_payload['original_text'] = original_sentence
            result_payload['score'] = round(rrf_scores[hit.id], 4)
            result_payload['text'] =  summarize_text(original_sentence) if len(original_sentence) > 150 else original_sentence
            result_payload['sentence_nouns'] = hit.payload.get("sentence_nouns", "")
            results.append(result_payload)
            seen_text.add(original_sentence)
    return results

#기능정보
def fetch_product_context(product_type: str = None, top_k: int = 20):
    qdrant = get_qdrant_client()
    query_filter = None

    print(f"🔍 [Debug] fetch_product_context called with product_type: '{product_type}'")

    # product_type이 None이 아니고 빈 문자열이 아니거나 "(선택 안함)"이 아닐 때만 필터 적용
    if product_type:
        query_filter = Filter(must=[FieldCondition(key="product_type", match=MatchValue(value=product_type))])
    
    try:
        records, _ = qdrant.scroll(
            collection_name="product_data",
            scroll_filter=query_filter,
            limit=top_k,
            with_payload=True,  # 👈 이 부분을 True로 변경합니다.
        )
        return [record.payload for record in records]
    except Exception as e:
        print(f"제품 데이터 검색 중 오류 발생: {e}")
        return []
#센서정보
def fetch_sensor_context(product_type: str | None, top_k: int = 20): # product_type에 None 허용
    """
    Qdrant에서 특정 'Product Category'에 해당하는 센서 데이터 샘플을 가져옵니다.
    """
    qdrant = get_qdrant_client()
    print(f"SENSOR_SEARCH 🔍 [Debug] Fetching sensor data for Product Category: '{product_type}'")

    # 🚨 product_type이 None이거나 빈 문자열일 경우, 데이터 조회를 건너뜁니다.
    if not product_type or product_type == '':
        print("SENSOR_SEARCH_SKIP ⚠️ product_type이 제공되지 않아 센서 데이터 조회를 건너뜁니다.")
        return []

    try:
        # 'Product Category' 필드를 기준으로 필터 생성
        # MatchValue의 value는 Qdrant에 저장된 실제 값과 정확히 일치해야 합니다.
        query_filter = Filter(
            must=[
                FieldCondition(key="Product", match=MatchValue(value=product_type))
            ]
        )

        records, _ = qdrant.scroll(
            collection_name="sensor_data",
            scroll_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        
        return [record.payload for record in records]
    except Exception as e:
        # 오류 메시지를 좀 더 명확하게 변경
        print(f"SENSOR_SEARCH_ERROR ❌ 센서 데이터 검색 중 오류 발생: {e}. 'sensor_data' 컬렉션의 'Product Category' 필드 값과 '{product_type}' 일치 여부를 확인하세요.")
        return []
    

#컬럼정보
def get_columns_for_product(product_type: str,top_k: int = 20):
    """Qdrant에서 특정 제품군의 상세 필드 정보를 조회합니다."""
    print(f"🔩 Getting column info for product_type='{product_type}'...")
    qdrant = get_qdrant_client()
    
    search_filter = Filter(
    must=[
        FieldCondition(
            key="product_type",
            match=MatchValue(value=product_type)
        )
    ]
)
    
    found_points, _ = qdrant.scroll(
        collection_name="product_metadata",
        scroll_filter=search_filter,
        limit=top_k,
        with_payload=True
    )
    
    if found_points:
        # 페이로드에서 'fields' 딕셔너리를 반환합니다.
        return found_points[0].payload.get("fields", {})
    else:
        # 일치하는 제품 정보가 없으면 빈 딕셔너리를 반환합니다.
        return {}


def conext_change(workspace: dict, product_type: str):
        # 4. 내부 제품 데이터 검색
    product_results = fetch_product_context(product_type)

    # 5. 센서 데이터 검색 (product_type이 있을 경우)
    sensor_data_results = fetch_sensor_context(product_type)

    #컬럼 정보 검색
    columns_product = get_columns_for_product(product_type)

    workspace["artifacts"]["columns_product"] = columns_product
    workspace["artifacts"]["sensor_data"] = sensor_data_results
    workspace["artifacts"]["product_data"] = product_results
    workspace["artifacts"]["product_type"] = product_type

    return {
        "columns_product": columns_product,
        "sensor_data": sensor_data_results,
        "product_data": product_results,
        "product_type": product_type,
    }
    

def run_data_retriever(workspace: dict, keyword: str, date_range_str: str | None, product_type: str | None):
    """
    Data Retriever 에이전트의 전체 작업을 오케스트레이션합니다.
    프론트엔드에서 명시적으로 전달된 키워드, 기간, 제품군을 사용합니다.
    """
    print(f"✅ [Agent Called] run_data_retriever: keyword='{keyword}', date_range_str='{date_range_str}', product_type='{product_type}'")
    
    # 1. 날짜 범위 파싱
    parsed_date_range = parse_natural_date(date_range_str) if date_range_str else None

    # 2. 키워드 확장 (이제 LLM으로 키워드 추출할 필요 없이 받은 키워드를 바로 확장합니다)
    # 여러 키워드를 입력받을 수 있으므로, 쉼표로 구분된 문자열을 리스트로 변환
    keywords_list = [k.strip() for k in keyword.split(',') if k.strip()]
    if not keywords_list: # 키워드 리스트가 비어있으면 원본 키워드 자체를 사용
        keywords_list = [keyword]

    all_expanded_keywords = []
    for kw in keywords_list:
        all_expanded_keywords.extend(expand_keywords(kw, product_type))
    # 중복 제거
    all_expanded_keywords = list(set(all_expanded_keywords))
    
    print(f"✅ Extracted & Expanded Keywords: {all_expanded_keywords}")
    print(f"✅ Parsed Date Range: {parsed_date_range}")
    print(f"✅ Product Type: {product_type}")

    # 3. 웹/소비자 데이터 검색 (RRF)
    web_results = run_rrf_search(all_expanded_keywords, date_range=parsed_date_range)

    # 4. 내부 제품 데이터 검색
    product_results = fetch_product_context(product_type)

    # 5. 센서 데이터 검색 (product_type이 있을 경우)
    sensor_data_results = fetch_sensor_context(product_type)

    #컬럼 정보 검색
    columns_product = get_columns_for_product(product_type)

    retrieved_data = {
        "query": keyword, # 사용자 입력 원본 키워드를 query로 저장
        "expanded_keywords": all_expanded_keywords, # 확장된 키워드도 저장
        "web_results": web_results,
    }


           # 수정: 워크스페이스에 결과 저장
    workspace["artifacts"]["retrieved_data"] = retrieved_data
    workspace["artifacts"]["columns_product"] = columns_product
    workspace["artifacts"]["sensor_data"] = sensor_data_results
    workspace["artifacts"]["product_data"] = product_results
    workspace["artifacts"]["product_type"] = product_type
    workspace["artifacts"]["analysis_results"] = (
    f"데이터 검색이 완료되었습니다. 키워드 '{keyword}'로 {len(web_results)}개의 웹 결과를 수집했습니다."
    )
    # 6. 워크스페이스에 저장할 형식으로 결과 가공
    return {
        "retrieved_data": {
            "query": keyword, # 사용자 입력 원본 키워드를 query로 저장
            "expanded_keywords": all_expanded_keywords, # 확장된 키워드도 저장
            "web_results": web_results,
        },
        "columns_product": columns_product,
        "sensor_data": sensor_data_results,
        "product_data": product_results,
        "product_type": product_type,
    }
