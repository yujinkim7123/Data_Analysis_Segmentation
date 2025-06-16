# agents/utils.py
import torch
from sentence_transformers import SentenceTransformer, models
from qdrant_client import QdrantClient
from openai import OpenAI, AsyncOpenAI # AsyncOpenAI 임포트도 중요합니다!
import os 
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from qdrant_client.http.models import Filter, FieldCondition, MatchValue 
sentiment_analyzer = None 

# 모델과 클라이언트를 저장할 전역 변수
meaning_model = None
topic_model = None
qdrant_client = None
openai_client = None


def get_sentiment_analyzer():
    """
    [신규] 사전 학습된 한국어 감성 분류 모델을 로드합니다.
    Hugging Face의 pipeline을 사용하여 쉽게 구현합니다.
    """
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("🌀 Loading pre-trained sentiment analysis model...")
        
        # Define the local path to your model files
        local_model_path = "C:/Users/User/DIC_Project/persona_mcp_server/agents/models/bert-nsmc" 
        
        try:
            # Load the tokenizer from your local path
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

            # 2. pipeline에 로컬 모델과 tokenizer 전달
            sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        except Exception as e:
            print(f"❌ Error loading sentiment analysis model from local path: {e}")
            sentiment_analyzer = None # Ensure it remains None if loading fails
    return sentiment_analyzer

def get_embedding_models():
    """
    의미 검색(e5-large) 모델과 주제 검색(ko-sbert) 모델을 모두 로드합니다.
    이미 로드되었다면 기존 객체를 반환합니다.
    """
    global meaning_model, topic_model
    if meaning_model is None or topic_model is None:
        print("🌀 Loading embedding models (meaning & topic)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Meaning Model 로드 (기존 코드와 동일)
        meaning_model = SentenceTransformer(
            modules=[models.Transformer("intfloat/e5-large"), models.Pooling(1024, pooling_mode_mean_tokens=True)],
            device=device
        )
        
        # 2. Topic Model 로드 (기존 코드와 동일)
        topic_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)
        
        print("✅ All embedding models loaded.")
        
    return meaning_model, topic_model

def get_qdrant_client():
    """Qdrant 클라이언트를 생성합니다."""
    global qdrant_client
    if qdrant_client is None:
        print("🌀 Initializing Qdrant client...")
        qdrant_client = QdrantClient(host="localhost", port=6333)
        print("✅ Qdrant client initialized.")
    return qdrant_client

def get_openai_client(async_client=False):
    """
    [개선됨] OpenAI 클라이언트를 반환합니다.
    최초 호출 시, 환경 변수에서 API 키를 읽어 객체를 생성합니다.
    """
    global openai_client
    if openai_client is None:
        print("🌀 Initializing OpenAI client...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("'.env' 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        openai_client = OpenAI(api_key=api_key) # 항상 동기 클라이언트를 생성하고 전역 변수에 저장
    return openai_client # 저장된 동기 클라이언트를 반환


def get_columns_for_product(product_type: str):
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
        limit=1,
        with_payload=True
    )
    
    if found_points:
        # 페이로드에서 'fields' 딕셔너리를 반환합니다.
        return found_points[0].payload.get("fields", {})
    else:
        # 일치하는 제품 정보가 없으면 빈 딕셔너리를 반환합니다.
        return {}
    
def parse_natural_date(text: str | None) -> tuple[date, date] | None:
    """'최근 3개월', '올해' 등 자연어 기간을 파싱하여 (시작일, 종료일) 튜플을 반환합니다."""
    if not text:
        return None
    
    today = date.today()
    
    # "최근 X개월" 형식
    match = re.search(r'최근\s*(\d+)\s*개월', text)
    if match:
        months = int(match.group(1))
        start_date = today - relativedelta(months=months)
        return start_date, today

    # "최근 X일" 형식
    match = re.search(r'최근\s*(\d+)\s*일', text)
    if match:
        days = int(match.group(1))
        start_date = today - timedelta(days=days)
        return start_date, today
        
    # "올해"
    if "올해" in text or "금년" in text:
        return date(today.year, 1, 1), today

    # "작년"
    if "작년" in text:
        last_year = today.year - 1
        return date(last_year, 1, 1), date(last_year, 12, 31)
    
    # if "최신" in text or "최근" in text:
    #     # 정규식으로 개월/일을 찾지 못하고 '최신'이라는 단어만 있으면 기본값(예: 1개월)으로 처리
    #     match_month = re.search(r'(\d+)\s*개월', text)
    #     match_day = re.search(r'(\d+)\s*일', text)
    #     if not match_month and not match_day:
    #         print("🌀 Interpreting '최신' as 'last 1 month'.")
    #         return today - relativedelta(months=1), today
    # --- 신규 로직 끝 ---

    # "최근 X개월" 형식
    match = re.search(r'최근\s*(\d+)\s*개월', text)
    # TODO: "YYYY-MM-DD ~ YYYY-MM-DD" 등 더 다양한 형식 추가 가능
    
    return None # 매칭되는 형식이 없음