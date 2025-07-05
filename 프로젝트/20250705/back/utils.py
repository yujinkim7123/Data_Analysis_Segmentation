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
import redis
import json
import logging
from redis.lock import Lock
from retry import retry
from typing import Any
sentiment_analyzer = None 

# 모델과 클라이언트를 저장할 전역 변수
meaning_model = None
topic_model = None
qdrant_client = None
sync_openai_client = None
async_openai_client = None
redis_client = None


MODEL_NAME = "gpt-4o-mini"


#---logging 하기 위한 과정 setup--------
def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("mcp_server.log")
        ]
    )
    return logging.getLogger(__name__)


#-----redis 세팅--------------------

def get_redis_client():
    global redis_client
    if redis_client is None:
        print("🌀 Initializing Redis client...")
        try:
            redis_client = redis.StrictRedis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            redis_client.ping()
            print("✅ Redis client initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize Redis client: {e}")
            redis_client = None
    return redis_client

@retry(tries=3, delay=1, backoff=2)
def save_workspace_to_redis(session_id: str, workspace: dict):
    r = get_redis_client()
    if r:
        try:
            with Lock(r, f"lock:session:{session_id}", timeout=10):
                serializable_workspace = _convert_datetime_to_str(workspace)
                # TTL을 사용자 활동에 따라 동적으로 설정 (예: 24시간)
                r.setex(f"session:{session_id}:workspace", 86400, json.dumps(serializable_workspace, ensure_ascii=False))
                r.expire(f"session:{session_id}:workspace", 86400)  # 요청 시 TTL 갱신
                print(f"💾 Workspace saved for session: {session_id}")
        except Exception as e:
            print(f"❌ Failed to save workspace to Redis for session {session_id}: {e}")
            logging.error(f"Redis save error: {e}")
            raise
    else:
        logging.error("Redis client not available")
        raise Exception("Redis connection unavailable")

@retry(tries=3, delay=1, backoff=2)
def load_workspace_from_redis(session_id: str) -> dict | None:
    r = get_redis_client()
    if r:
        try:
            with Lock(r, f"lock:session:{session_id}", timeout=10):
                workspace_json = r.get(f"session:{session_id}:workspace")
                if workspace_json:
                    loaded_workspace = json.loads(workspace_json)
                    deserialized_workspace = _convert_str_to_datetime(loaded_workspace)
                    r.expire(f"session:{session_id}:workspace", 86400)  # 로드 시 TTL 갱신
                    print(f"✅ Workspace loaded for session: {session_id}")
                    return deserialized_workspace
                print(f"ℹ️ No workspace found for session: {session_id}")
                return None
        except Exception as e:
            print(f"❌ Failed to load workspace from Redis for session {session_id}: {e}")
            logging.error(f"Redis load error: {e}")
            return None
    return None

# ✨ 새로 추가할 함수 시작
def _convert_datetime_to_str(obj: Any) -> Any:
    """Recursively converts datetime and tool_calls objects to serializable format."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, '__dict__') and hasattr(obj, 'id') and hasattr(obj, 'function'):  # tool_calls 처리
        return {
            'id': obj.id,
            'type': getattr(obj, 'type', 'function'),
            'function': {
                'name': obj.function.name,
                'arguments': obj.function.arguments
            }
        }
    if isinstance(obj, list):
        return [_convert_datetime_to_str(elem) for elem in obj]
    if isinstance(obj, dict):
        return {k: _convert_datetime_to_str(v) for k, v in obj.items()}
    return obj

def _convert_str_to_datetime(obj: Any) -> Any:
    """Recursively converts ISO strings and tool_calls back to original format."""
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj)
        except ValueError:
            try:
                return date.fromisoformat(obj)
            except ValueError:
                pass
    if isinstance(obj, dict) and 'id' in obj and 'function' in obj:
        from openai.types.chat import ChatCompletionMessageToolCall
        return ChatCompletionMessageToolCall(
            id=obj['id'],
            type=obj.get('type', 'function'),
            function=obj['function']
        )
    if isinstance(obj, list):
        return [_convert_str_to_datetime(elem) for elem in obj]
    if isinstance(obj, dict):
        return {k: _convert_str_to_datetime(v) for k, v in obj.items()}
    return obj


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
    async_client=True 이면 비동기 클라이언트를, False(기본값)이면 동기 클라이언트를 반환합니다.
    최초 호출 시, 환경 변수에서 API 키를 읽어 객체를 생성합니다.
    """
    global sync_openai_client, async_openai_client
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("'.env' 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

    if async_client:
        # 비동기 클라이언트가 필요한 경우
        if async_openai_client is None:
            print("🌀 Initializing Async OpenAI client...")
            async_openai_client = AsyncOpenAI(api_key=api_key)
        return async_openai_client
    else:
        # 동기 클라이언트가 필요한 경우 (기본값)
        if sync_openai_client is None:
            print("🌀 Initializing Sync OpenAI client...")
            sync_openai_client = OpenAI(api_key=api_key)
        return sync_openai_client

def parse_natural_date(text: str | None) -> tuple | None:
    """
    [최종 수정] '1년간', '3개월간' 등의 표현도 인식하도록 정규표현식을 수정합니다.
    """
    if not text:
        return None

    today = datetime.now()
    text = text.lower()
    
    match = re.search(r'(?:최근|지난)\s*(\d+)\s*(개월|달|년|주|일)간?', text)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        end_date = today.date()
        if '년' in unit:
            start_date = (today - timedelta(days=num*365)).date()
        elif '주' in unit:
            start_date = (today - timedelta(days=num*7)).date()
        elif '일' in unit:
            start_date = (today - timedelta(days=num)).date()
        else: # 개월 또는 달
            start_date = (today - timedelta(days=num*30)).date()
        return start_date, end_date

    # 다른 패턴들은 그대로 유지
    if '올해' in text or '이번 년도' in text:
        return datetime(today.year, 1, 1).date(), today.date()
    if '작년' in text:
        last_year = today.year - 1
        return datetime(last_year, 1, 1).date(), datetime(last_year, 12, 31).date()

    if '이번 달' in text:
        return today.replace(day=1).date(), today.date()
    if '지난 달' in text:
        first_day_of_current_month = today.replace(day=1)
        last_day_of_last_month = first_day_of_current_month - timedelta(days=1)
        first_day_of_last_month = last_day_of_last_month.replace(day=1)
        return first_day_of_last_month.date(), last_day_of_last_month.date()

    if '어제' in text:
        yesterday = (today - timedelta(days=1)).date()
        return yesterday, yesterday
    if '오늘' in text:
        return today.date(), today.date()
        
    return None