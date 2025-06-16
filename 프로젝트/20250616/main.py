import os
from dotenv import load_dotenv
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. .env 파일 로드 및 초기화 ---
load_dotenv()

# --- 2. [수정] 명시적 import로 변경 ---
from agents.utils import (
    get_embedding_models, get_qdrant_client, get_openai_client, 
    get_sentiment_analyzer, parse_natural_date
)
from agents import (
    run_data_retriever,
    run_ward_clustering,
    run_semantic_network_analysis,
    run_topic_modeling_lda,
    create_customer_action_map,
    calculate_opportunity_scores,
    create_personas,
    create_service_ideas,
    create_service_ideas_from_manual_input,
    create_data_plan_for_service, # 신규 에이전트 함수 추가
    create_cdp_definition,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [수정됨] FastAPI 앱의 시작 시, .env 파일을 가장 먼저 로드하도록 수정합니다.
    """
    # --- 서버가 시작될 때 실행될 코드 ---
    print("\n" + "="*50)
    print("🚀 [Lifespan] 서버 시작 프로세스에 진입합니다...")
    
    # 1. ❗️가장 먼저 .env 파일을 로드합니다.
    load_dotenv()
    print("   - .env 파일 로드를 시도했습니다.")

    # 2. ❗️(디버깅용) 키가 제대로 로드되었는지 즉시 확인합니다.
    api_key_check = os.getenv("OPENAI_API_KEY")
    print(f"   - 로드된 OPENAI_API_KEY: {api_key_check[:5]}..." if api_key_check else "   - ❗️ ERROR: 키를 찾을 수 없습니다!")
    
    # 3. 모델과 클라이언트를 초기화합니다.
    get_embedding_models()
    get_qdrant_client()
    get_openai_client() # 이제 인자 없이 호출합니다.
    get_sentiment_analyzer()
    print("✅ [Lifespan] 모든 준비 완료. 요청을 받을 수 있습니다.")
    print("="*50 + "\n")

    yield
    
    # --- 서버가 종료될 때 실행될 코드 ---
    print("\n" + "="*50)
    print(" gracefully [Lifespan] 서버를 종료합니다.")
    print("="*50)

PERSONA_INPUT_GUIDE = """
---
**[페르소나 입력 추천 가이드]**
* **핵심 설명 (Who):** "저는 `[30대 직장인]`입니다."
* **목표와 니즈 (Goal):** "주로 `[간편한 저녁 식사를 원]합니다."`
* **가장 큰 불편함 (Pain Point):** "**가장 불편한 점은** `[퇴근 후 요리할 에너지가 없는 것]`**입니다.**"
* **(선택) 제품 연계 (Product):** "`[디오스 냉장고]`와 연계하고 싶습니다."
---
"""

SERVICE_INPUT_GUIDE = """
---
**[서비스 아이디어 수동 입력 가이드]**
* **서비스 이름 (What):** "제가 생각한 서비스는 `[서비스 이름]`입니다."
* **핵심 기능 (How):** "이 서비스는 `[사용자에게 제공하는 핵심 기능]`을 합니다."
* **해결 문제 (Why):** "이를 통해 `[사용자의 어떤 불편함이나 니즈를 해결]`할 수 있습니다."
* **(선택) 연관 제품 (Product):** "`[디오스 냉장고]`와 연계하면 좋을 것 같아요."
---
"""

# [3. SYSTEM_PROMPT 수정]
SYSTEM_PROMPT = f"""
당신은 사용자가 더 나은 제품과 서비스를 기획할 수 있도록 돕는 전문 AI 어시스턴트입니다.
당신은 '데이터 검색', 'CX 분석', '페르소나 생성', '서비스 아이디어 생성', '데이터 기획' 등 다양한 분석 도구를 사용할 수 있습니다.

# 주요 행동 지침
- 항상 사용자의 최종 목표를 파악하고, 목표 달성에 가장 적합한 다음 단계를 제안해주세요.
- 만약 사용자가 "서비스 아이디어 만들어줘" 와 같이 **어떤 페르소나를 기반으로 할지 명확하지 않게** 요청하면, 도구를 바로 호출하지 마세요.
- 대신, 사용자에게 두 가지 선택지를 제시하며 질문해주세요.
  1. **데이터 기반 분석으로 진행하기**: 특정 키워드에 대한 실제 데이터를 분석하여 페르소나를 도출하는 방법.
  2. **페르소나 직접 설명하기**: 사용자가 머릿속에 있는 페르소나를 직접 설명하는 방법. 이 경우, 아래의 '페르소나 입력 추천 가이드'를 함께 제시하여 사용자가 좋은 정보를 제공하도록 유도해야 합니다.
- 만약 사용자가 "데이터 기획해줘" 와 같이 **어떤 서비스 아이디어를 기반으로 할지 명확하지 않게** 요청하면, 도구를 바로 호출하지 마세요.
- 대신, 사용자에게 두 가지 선택지를 제시하며 질문해주세요.
  1. **기존 아이디어 선택하기**: 이전에 생성된 서비스 아이디어 목록을 보여주고 선택하게 하는 방법.
  2. **서비스 아이디어 직접 설명하기**: 사용자가 머릿속에 있는 서비스 아이디어를 직접 설명하는 방법. 이 경우, 아래의 '서비스 아이디어 수동 입력 가이드'를 함께 제시하여 사용자가 좋은 정보를 제공하도록 유도해야 합니다.

{PERSONA_INPUT_GUIDE}
{SERVICE_INPUT_GUIDE}
"""

# FastAPI 앱 객체를 생성합니다.
app = FastAPI(lifespan=lifespan, title="기획자 AI Agent MCP 서버")

SESSIONS = {}
tools = [
    # 1. 데이터 검색 에이전트 (이 부분은 이전과 동일)
    {
        "type": "function",
        "function": {
            "name": "run_data_retriever",
            "description": "사용자가 특정 키워드에 대한 데이터, 정보, 소비자 반응, VOC, 뉴스 기사 등의 검색을 요청할 때 사용되는 필수적인 첫 단계 도구입니다. **이후 STS 세그멘테이션 분석을 위한 기초 데이터를 수집합니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "사용자가 검색을 원하는 가장 핵심적인 키워드입니다. (예: 스타일러 살균 기능, 공기청정기 필터 성능)"
                    },
                    "product_type": {
                        "type": "string",
                        "description": "검색 대상을 특정 제품군으로 한정할 때 사용합니다. '스타일러', '트롬', '퓨리케어'와 같은 명확한 제품군, 카테고리, 브랜드 이름만 추출해주세요. 사용자가 '스타일러 살균 기능'이라고 말했다면, '스타일러'가 여기에 해당됩니다."
                    },
                    "query": {
                        "type": "string",
                        "description": "사용자가 검색을 요청한 전체 문장 또는 구절입니다. (예: '스타일러 살균 기능', '최근 1년간의 공기청정기 후기')"
                    },           
                    "date_range_str": {
                        "type": "string",
                        "description": "검색할 기간입니다. '최근 3개월', '2025년 상반기', '어제' 등 사용자가 말한 자연어 형식의 기간을 그대로 전달해주세요."
                    }
                },
                "required": ["keyword", "query", "date_range_str"],
            },
        },
    },
    
    # 2. Ward Clustering (Segmentation의 'S' - 숲 파악)
    {
        "type": "function",
        "function": {
            "name": "run_ward_clustering",
            "description": "📊 **[STS Segmentation - S (Segmentation) 1단계: 고객 그룹 분류 (숲 파악)]** 전체 고객의 목소리(VOC)에서 나타나는 거시적인 주제나 관심사 그룹을 발견합니다. 고객 대화의 '숲'을 먼저 파악하는 과정입니다. **이 단계는 STP 전략 수립을 위한 첫걸음이며, 고객의 니즈를 폭넓게 파악하지 않으면 비효율적인 마케팅으로 이어질 수 있습니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_clusters": {"type": "integer", "description": "나눌 그룹의 개수 (기본값: 5)", "default": 5}
                },
                "required": [],
            },
        },
    },
    
    # 3. Semantic Network Analysis (Segmentation의 'S' - 나무 파악)
    {
        "type": "function",
        "function": {
            "name": "run_semantic_network_analysis",
            "description": "🔍 **[STS Segmentation - S (Segmentation) 2단계: 고객 생각 연결 구조 분석 (나무 파악)]** 특정 주제 그룹 내부의 핵심 키워드 간의 연결 구조를 분석합니다. 이를 통해 고객의 생각이 어떤 세부적인 개념들로 구성되어 있는지, 즉 '나무'들을 자세히 들여다봅니다. **이 단계를 통해 세그먼트의 구체적인 니즈를 파악하지 못하면, 추상적인 전략에 머물러 실행력이 떨어집니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "분석할 고객 그룹의 ID 번호"}
                },
                "required": ["cluster_id"],
            },
        },
    },
    
    # 4. Topic Modeling LDA (Segmentation의 'S' - 행동 식별)
    {
        "type": "function",
        "function": {
            "name": "run_topic_modeling_lda",
            "description": "🎯 **[STS Segmentation - S (Segmentation) 3단계: 고객 행동 식별 (액션 파악)]** 고객의 목소리에서 구체적인 '고객 행동(Customer Action)' 또는 '사용 시나리오'를 식별합니다. 고객들이 실제로 무엇을 '하는지'에 대한 주제들을 찾아냅니다. **이 단계를 통해 고객의 실제 행동을 파악하지 못하면, 고객의 문제 상황에 딱 맞는 솔루션 기획이 어려워집니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "토픽을 분석할 고객 그룹의 ID 번호"},
                    "num_topics": {"type": "integer", "description": "추출할 토픽의 개수 (기본값: 3)", "default": 3}
                },
                "required": ["cluster_id"],
            },
        },
    },
    
    # 5. Calculate Opportunity Scores (Targeting의 'T' & Positioning의 'P' - 사업 기회 우선순위)
    # CAM보다 먼저 오도록 순서 변경 및 설명 업데이트
    {
        "type": "function",
        "function": {
            "name": "calculate_opportunity_scores",
            "description": "📈 **[STS Targeting & Positioning - T/P 1단계: 사업 기회 점수 계산]** 도출된 모든 '고객 행동'과 'Pain Point'에 대해, 언급량(중요도)과 고객 만족도(감성)를 종합하여 사업적 '기회 점수(Opportunity Score)'를 계산합니다. 어떤 문제에 집중해야 할지 정량적으로 우선순위를 결정합니다. **이 단계를 통해 리소스 투입의 우선순위를 정량적으로 확보하지 못하면, 어떤 Pain Point에 집중할지 모호해져 STP 전략 실행의 효율성이 떨어집니다.**", # 설명 수정 및 강화
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # 6. Customer Action Map (Targeting의 'T' & Positioning의 'P' - 고통과 목표 심층 분석, 이제 최종 단계)
    # Opportunity Scores 다음에 오도록 순서 변경 및 설명 업데이트
    {
        "type": "function",
        "function": {
            "name": "create_customer_action_map",
            "description": "🗺️ **[STS Targeting & Positioning - T/P 2단계: 고객 액션맵(CAM) 완성 (최종 분석 단계)]** 식별된 '고객 행동(Action)'에 대해, 고객이 궁극적으로 원하는 'Goal'과 그 과정에서 겪는 'Pain Point'를 심층적으로 분석하여 고객 액션맵(CAM)을 완성합니다. 고객의 숨은 의도와 불편함을 파악하는 핵심 단계입니다. **이 단계는 STP 중 타겟 고객의 '진짜 문제'를 정의하고 포지셔닝할 '가치'를 발굴하는 데 필수적이며, 기회 점수를 통해 우선순위가 높은 행동에 대해 더욱 깊이 있는 이해를 돕습니다.**", # 설명 수정 및 강화
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "string", "description": "분석할 토픽(Action)의 ID 번호 (예: '0-1')"}
                },
                "required": ["topic_id"],
            },
        },
    },

       {
        "type": "function",
        "function": {
            "name": "create_personas",
            "description": "데이터 검색 및 CX 분석 결과를 바탕으로 고객 페르소나를 생성합니다. 검색된 고객의 목소리(VOC)는 페르소나 생성을 위한 필수 입력값입니다. 더 정교한 페르소나를 위해 기존에 분석된 고객 행동(Action/Topic)을 선택적으로 활용할 수 있습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_personas": {
                        "type": "integer",
                        "description": "생성할 페르소나의 개수입니다. (예: 3)",
                        "default": 3
                    },
                    "focus_topic_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "페르소나 생성 시 특히 집중해서 분석할 고객 행동(Action/Topic)의 ID 목록입니다. (예: ['1-0', '2-1']) CX 분석을 수행했을 경우에만 유효한 값입니다."
                    }
                },
                "required": ["num_personas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_ideas",
            "description": "지정된 페르소나의 Pain Point를 해결하기 위한 새로운 서비스 아이디어를 생성합니다. 어떤 페르소나를 기반으로 아이디어를 만들지 'persona_name'으로 명확히 지정해야 합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona_name": {
                        "type": "string",
                        "description": "서비스 아이디어 생성의 기반이 될 페르소나의 이름입니다. (예: '박서준')"
                    },
                    "num_ideas": {
                        "type": "integer",
                        "description": "생성할 아이디어의 개수입니다. (기본값: 3)",
                        "default": 3
                    }
                },
                "required": ["persona_name"],
            },
        },
    },
       {
        "type": "function",
        "function": {
            "name": "create_service_ideas_from_manual_input",
            "description": "사용자가 사전에 분석된 페르소나 없이, 직접 페르소나 정보를 텍스트로 설명하여 서비스 아이디어를 얻고 싶을 때 사용합니다. 제품군도 선택적으로 지정할 수 있습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona_description": {
                        "type": "string",
                        "description": "사용자가 생각하는 페르소나에 대한 상세한 설명입니다. (예: '저는 30대 1인 가구 직장인이고, 퇴근 후 요리할 시간이 부족한 게 고민이에요.')"
                    },
                    "product_type": {
                        "type": "string",
                        "description": "서비스 아이디어를 연계하고 싶은 특정 LG 제품군 이름입니다. (예: '디오스', '스타일러')"
                    },
                    "num_ideas": {
                        "type": "integer",
                        "description": "생성할 아이디어의 개수입니다. (기본값: 3)",
                        "default": 3
                    }
                },
                "required": ["persona_description"],
            },
        },
    },
    # [4. tools 리스트에 신규 도구 추가]
    {
        "type": "function",
        "function": {
            "name": "create_data_plan_for_service",
            "description": "💡 **[서비스 구체화]** 도출되거나 직접 입력한 서비스 아이디어를 실현하기 위한 상세 데이터 기획안을 생성합니다. 기존 제품 데이터, 센서 데이터, 외부 데이터를 어떻게 활용하고 수집할지 구체적인 전략을 수립하는 최종 단계입니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": { "type": "string", "description": "데이터 기획의 기반이 될, 이전에 생성된 서비스 아이디어의 이름입니다. (예: 'AI 육아 위생 컨설턴트')" },
                    "service_description": { "type": "string", "description": "사용자가 직접 새로운 서비스 아이디어를 텍스트로 입력할 때 사용합니다. '서비스 수동 입력 가이드'를 참고하여 상세히 설명해야 합니다." },
                    "product_type": { "type": "string", "description": "서비스 아이디어와 연계할 특정 LG 제품군 이름입니다. (예: '디오스', '스타일러')" }
                },
            },
        },
    },
      {
        "type": "function",
        "function": {
            "name": "create_cdp_definition",
            "description": "📑 **[최종 산출물]** 지금까지 분석하고 기획한 모든 내용(페르소나, 서비스 아이디어, 데이터 기획안)을 종합하여 최종 C-D-P(Customer-Data-Performance) 정의서를 생성합니다. 모든 기획 프로세스의 마지막 단계입니다.",
            "parameters": {
                "type": "object",
                "properties": {}, # 이 함수는 워크스페이스의 데이터를 사용하므로 사용자 입력 인자가 필요 없음
            },
        },
    }

]

LG_PRODUCT_KEYWORDS = [
    "스타일러", "트롬", "휘센", "퓨리케어", "디오스", "그램", 
    "올레드", "코드제로", "틔운", "시네빔", "울트라기어"
]

available_functions = {
    "run_data_retriever": run_data_retriever,
    "run_ward_clustering": run_ward_clustering,
    "run_semantic_network_analysis": run_semantic_network_analysis,
    "run_topic_modeling_lda": run_topic_modeling_lda,
    "create_customer_action_map": create_customer_action_map,
    "calculate_opportunity_scores": calculate_opportunity_scores,
    "create_personas": create_personas,
    "create_service_ideas": create_service_ideas,
    "create_service_ideas_from_manual_input": create_service_ideas_from_manual_input,
    "create_data_plan_for_service": create_data_plan_for_service,
    "create_cdp_definition": create_cdp_definition,
    #"run_pain_point_analysis_workflow": run_pain_point_analysis_workflow
}




# 루트 URL("/")로 GET 요청이 오면 이 함수를 실행합니다.
@app.get("/")
def read_root():
    # JSON 형식으로 메시지를 반환합니다.
    return {"message": "MCP 서버가 성공적으로 실행되었습니다."}

# --- 워크스페이스 생성 함수 ---
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response_message: str
    workspace: dict

def create_new_workspace():
    """새로운 세션의 워크스페이스 뼈대를 생성합니다."""
    return {
        "conversation_state": None,
        "pending_action": None, 
        "history": [],
        "artifacts": {
            "product_type":None,
            "retrieved_data": None,
            "analysis_results": None,
            "cx_lda_results": [], 
            "cx_opportunity_scores": [], 
            "cx_cam_results": [], 
            "cx_ward_clustering_results": None, 
            "cx_sna_results": [], 
            "personas": [],
            "selected_persona": None,
            "selected_service_idea": None,
            "service_ideas": None,
            "superset_service_datas": None,
            "cej_plan": None,
            "cdp_definition": None,
            "data_plan_for_service": None,
            "sensor_data" : None,
            "product_data": None,
        }
    }

MODEL_NAME = "gpt-4o-mini"

def summarize_and_reset_history(workspace: dict, completed_tool_name: str, result_artifact: dict):
    """
    주요 작업(섹션)이 끝났을 때, history를 요약된 시스템 메시지로 교체하여
    LLM의 컨텍스트를 정리하고 다음 단계에 집중할 수 있도록 합니다.
    """
    print(f"🌀 Context Reset: '{completed_tool_name}' 작업 완료 후 히스토리를 리셋합니다.")
    
    # 다음 단계에 필요한 최소한의 요약 정보를 생성합니다.
    summary_text = f"이전 단계 작업인 '{completed_tool_name}'이 성공적으로 완료되었습니다."
    
    if completed_tool_name == 'run_data_retriever' and result_artifact:
        num_web_results = len(result_artifact.get('web_results', []))
        num_product_results = len(result_artifact.get('product_results', []))
        summary_text += f" 그 결과로 {num_web_results}개의 웹 문서와 {num_product_results}개의 제품 기능 정보를 찾았습니다."
        summary_text += " 이제 이 데이터를 기반으로 다음 작업을 수행할 준비가 되었습니다."
    
    # history를 요약된 시스템 메시지로 교체합니다.
    workspace['history'] = [
        {"role": "system", "content": summary_text}
    ]
    
    return workspace


def interpret_and_suggest_next_step(tool_name: str, result_artifact: dict, workspace: dict) -> str:
    """
    분석 결과를 비전문가인 기획자에게 쉽게 설명하고, 다음 행동을 안내합니다.
    """
    # get_openai_client()는 이제 동기 클라이언트를 반환합니다.
    client = get_openai_client()

    # LLM이 대화의 전체 맥락을 파악할 수 있도록 history를 전달합니다.
    history_context = json.dumps(workspace.get("history", []), ensure_ascii=False)

    # LLM에게 보낼 기본 프롬프트 구조
    prompt = f"""
    당신은 데이터 분석 결과를 비전문가인 기획자에게 아주 쉽게 설명해주는 친절한 CX 분석 컨설턴트입니다.
    항상 분석 단계의 '비즈니스적 의미'를 먼저 설명하고, 기술 용어는 최소화하여 대화해주세요.
    전체 대화의 맥락은 다음과 같습니다: {history_context}

    방금 '{tool_name}' 분석을 마쳤고, 그 결과는 다음과 같습니다:
    {json.dumps(result_artifact, ensure_ascii=False, indent=2)}

    ---
    위 분석 결과를 바탕으로, 아래 지시에 따라 사용자에게 다음 행동을 안내하는 메시지를 생성해주세요.
    """

    # 각 분석 단계별로 다른 지시사항을 추가합니다.
    if tool_name == "run_data_retriever":
        prompt += """
        [지시사항]
        1. 데이터 검색이 완료되었음을 먼저 알려주세요.
        2. 이제 본격적으로 고객을 이해하는 첫걸음인 **'STS 세그멘테이션 분석'** 단계에 진입할 것을 제안해주세요.
        3. STS의 첫 단계인 **S(Segmentation: 고객 분류)**를 위해, 고객들의 목소리를 큰 그룹으로 나누는 **'고객 관심사 그룹 분석(Ward Clustering)'**을 진행할 것을 제안합니다.
        4. 이 단계를 건너뛰면 모든 고객에게 동일한 마케팅을 하게 되어 효과가 떨어질 수 있다는 점을 간략히 언급하여 필요성을 강조해주세요.
        5. 사용자에게 `CX 분석 시작해줘` 또는 `고객 그룹 분석 시작해줘` 와 같이 다음 행동을 명확히 제시해주세요.
        """

    elif tool_name == "run_ward_clustering":
        num_clusters = result_artifact.get("cx_ward_clustering_results", {}).get("num_clusters", "N/A")
        cluster_summaries = result_artifact.get("cx_ward_clustering_results", {}).get("cluster_summaries", {})

        summary_text = ""
        for cluster_id, summary in cluster_summaries.items():
            keywords_preview = ', '.join(summary.get('keywords', [])[:3]) # 상위 3개 키워드만 표시
            summary_text += f"\n- {cluster_id}번 그룹: '{keywords_preview}...' 등"

        workspace["artifacts"]["cx_ward_clustering_results"] = result_artifact.get("cx_ward_clustering_results")

        prompt += f"""
        [지시사항]
        1. 첫 번째 S(Segmentation) 단계인 **'고객 관심사 그룹 분석(Ward Clustering)'**이 완료되었음을 알려주세요.
        2. 고객들의 목소리가 **{num_clusters}개의 큰 주제 그룹**으로 나뉘었음을 설명하고, 각 그룹의 특징(대표 키워드)을 간략히 요약해주세요:
        {summary_text}
        3. 이제 각 그룹 내부를 더 깊이 들여다보는 **'의미 연결망 분석(SNA)'**을 통해, 특정 그룹 고객들의 생각이 어떤 세부적인 개념들로 연결되어 있는지 파악할 차례임을 설명해주세요.
        4. 이 단계를 건너뛰면 세그먼트의 니즈를 정확히 파악하기 어려워 구체적인 타겟팅이 힘들다는 점을 언급하여 필요성을 강조해주세요.
        5. 사용자에게 "어떤 그룹을 분석할까요?" 라고 물으며, 다음 행동을 명확히 제시해주세요. (예: `1번 그룹 SNA 분석해줘` 또는 `클러스터 개수를 바꿔 다시 분석해줘`)
        """

    elif tool_name == "run_semantic_network_analysis":
        cluster_id = result_artifact.get("cx_sna_results", {}).get("cluster_id")
        micro_segments = result_artifact.get("cx_sna_results", {}).get("micro_segments", [])

        core_keywords_preview = ', '.join([seg.get('core_keyword', '') for seg in micro_segments[:3]])

        workspace["artifacts"]["cx_sna_results"] = result_artifact.get("cx_sna_results")

        prompt += f"""
        [지시사항]
        1. **{cluster_id}번 그룹**에 대한 **'의미 연결망 분석(SNA)'**이 완료되었음을 알려주세요.
        2. 이 그룹 고객들의 생각은 '{core_keywords_preview}...' 등의 핵심 개념들을 중심으로 연결되어 있음을 설명해주세요.
        3. 이제 다음 S(Segmentation) 단계로, 이 그룹 고객들이 실제로 어떤 **'행동(Customer Action)'**을 하는지 파악하는 **'토픽 모델링(LDA)'**을 진행할 차례임을 설명해주세요.
        4. 이 단계를 건너뛰면 추상적인 니즈에 머물러 구체적인 제품/서비스 기획이 어렵다는 점을 언급하여 필요성을 강조해주세요.
        5. 사용자에게 "이 그룹의 고객 행동을 분석해볼까요?" 라고 물으며, 다음 행동을 명확히 제시해주세요. (예: `{cluster_id}번 그룹 LDA 분석` 또는 `다른 그룹 SNA 분석해줘`)
        """

    elif tool_name == "run_topic_modeling_lda":
        # result_artifact는 {"success": True, "message": ..., "newly_identified_topics_preview": [...] } 형태
        cluster_id_from_topic = result_artifact.get("newly_identified_topics_preview", [{}])[0].get("topic_id", "").split('-')[0] if result_artifact.get("newly_identified_topics_preview") else "N/A"
        topics_preview = result_artifact.get("newly_identified_topics_preview", [])

        topics_summary = ""
        for topic in topics_preview:
            topics_summary += f"\n- 토픽 {topic.get('topic_id')}: '{', '.join(topic.get('action_keywords', [])[:3])}...' 등의 행동"

        prompt += f"""
        [지시사항]
        1. **{cluster_id_from_topic}번 그룹**의 고객들이 보이는 주요 '행동(Customer Action)'들을 **'토픽 모델링(LDA)'**을 통해 식별했음을 알려주세요.
        2. 식별된 주요 행동들은 다음과 같습니다:{topics_summary}
        3. 이 정보는 이제 우리가 어떤 고객(Target)에게 집중하고, 어떤 문제를 해결할지 정량적으로 우선순위를 정하는 데 중요합니다. 다음 단계로, 모든 고객 행동과 Pain Point들을 종합하여 사업적 **'기회 점수(Opportunity Score)'**를 계산합니다.
        4. 이 단계를 건너뛰면 어디에 집중해야 할지 명확한 근거 없이 결정하게 되어 STP 전략 수립에 어려움을 겪을 수 있다는 점을 언급하여 필요성을 강조해주세요.
        5. 사용자에게 `기회 점수 계산해줘` 라고 명확히 다음 행동을 제시해주세요.
        """

    elif tool_name == "calculate_opportunity_scores":
        # calculate_opportunity_scores의 반환값은 {"cx_opportunity_scores": scores}
        opportunity_scores_list = result_artifact.get("cx_opportunity_scores", [])

        top_3_opportunities = ""
        if opportunity_scores_list:
            for i, score_item in enumerate(opportunity_scores_list[:3]):
                action_keywords = score_item.get("action_keywords", [])
                score = score_item.get("opportunity_score", 0)
                top_3_opportunities += f"\n- {i+1}순위: '{', '.join(action_keywords[:2])}...' (점수: {score})"
        else:
            top_3_opportunities = "\n- (아직 도출된 기회 영역이 없습니다.)"

        prompt += f"""
        [지시사항]
        1. '기회 점수' 계산 결과, 가장 점수가 높은 **상위 3개의 기회 영역(토픽)**은 다음과 같습니다:
        {top_3_opportunities}
        2. 이 점수는 STP 중 타겟 고객에게 어떤 문제(Pain Point)를 해결해줄 것인지(Positioning) 정량적으로 우선순위를 정하는 데 매우 중요한 근거가 됩니다.
        3. 이제 이 결과를 바탕으로, 가장 중요한 기회 영역에 대한 **'고객 액션맵(CAM) 분석'**을 진행하여 고객의 목표와 불편함을 심층적으로 파악할 차례입니다. 이는 우리가 어떤 Pain Point에 집중할지 최종적으로 결정하는 핵심 단계입니다.
        4. 이 단계를 건너뛰면 우선순위만 확인하고 실제 고객의 고통을 해결하기 위한 구체적인 전략을 세우기 어렵다는 점을 강조해주세요.
        5. 사용자에게 "어떤 행동(토픽 ID)에 대한 CAM 분석을 할까요?" 라고 물으며, 예를 들어 `1-0번 토픽 CAM 분석해줘`와 같이 다음 행동을 명확히 제시해주세요.
        """

    elif tool_name == "create_customer_action_map":
        # create_customer_action_map의 반환값은 {"cx_cam_results": existing_cams} (전체 누적 리스트)
        # 여기서는 방금 생성된 CAM 하나를 대상으로 설명해야 합니다.
        last_cam_result = workspace.get("artifacts", {}).get("cx_cam_results", [])[-1] if \
                          workspace.get("artifacts", {}).get("cx_cam_results") else {}
        action_name = last_cam_result.get("action_name", "N/A")
        pain_points_preview = ', '.join(last_cam_result.get("pain_points", [])[:2])

        prompt += f"""
        [지시사항]
        1. '{action_name}' 행동에 대한 **'고객 액션맵(CAM) 분석'**이 완료되었음을 알려주세요.
        2. 이 분석을 통해 핵심적인 Pain Point와 Goal이 명확히 파악되었습니다. (주요 Pain Point 예시: '{pain_points_preview}...')
        3. 모든 **STS 세그멘테이션 분석**이 성공적으로 완료되었습니다! 고객들의 다양한 관심사, 행동, 그리고 그들의 고통까지 깊이 이해할 수 있었네요.
        
        ### 👇 [수정됨] 다음 행동으로 '페르소나 생성'을 제안합니다. ###
        4. 이제 이 분석 결과를 바탕으로 '**핵심 고객 페르소나를 생성**'하여 전략을 구체화할 수 있습니다. 예를 들어, `페르소나 3명 생성해줘` 또는 `1-0 토픽 중심으로 페르소나 만들어줘` 와 같이 요청하여 다음 단계를 진행해보세요.
        5. 또는 다른 토픽(행동)에 대한 '고객 액션맵(CAM)'을 다시 생성하여 상세한 고객의 목표와 불편함을 더 깊이 이해할 수도 있습니다. (예: `1-0번 토픽 CAM 분석해줘`)
        """
    elif tool_name == "create_personas":
        personas = result_artifact.get("personas_result", {}).get("personas", [])
        if not personas:
            return "페르소나 생성에 실패했거나, 생성된 페르소나가 없습니다. 다시 시도해주세요."
        
        num_personas = len(personas)
        # 페르소나 목록을 보여주기 위한 문자열 생성
        persona_list_str = "\n".join(
            [f"* **{p.get('name')} ({p.get('title')})**" for p in personas]
        )
        
        # 첫 번째 페르소나 이름을 예시에 사용
        example_persona_name = personas[0].get('name')

        # 사용자에게 보여줄 최종 메시지
        return f"""✅ 페르소나 생성이 완료되었습니다. 분석 결과를 바탕으로 다음과 같은 {num_personas}명의 핵심 고객 페르소나를 도출했습니다.

{persona_list_str}

이제 이 중 한 명을 선택하여 맞춤형 서비스 아이디어를 구체화해볼까요?
예를 들어, **"{example_persona_name} 페르소나를 위한 서비스 아이디어 3개 제안해줘"** 와 같이 요청해주세요."""

   
    elif tool_name == "create_service_ideas" or tool_name == "create_service_ideas_from_manual_input":
        service_ideas = result_artifact.get("service_ideas_result", {}).get("service_ideas", [])
        if service_ideas:
            num_ideas = len(service_ideas)
            persona_type = "직접 입력해주신 페르소나를 기반으로" if "manual_input" in tool_name else "분석 결과를 바탕으로"
            response = f"✅ 좋습니다! {persona_type} {num_ideas}개의 새로운 서비스 아이디어를 생성했습니다. 워크스페이스에서 상세 내용을 확인해보세요."
            response += "\n\n이제 마음에 드는 아이디어를 바탕으로 구체적인 **데이터 기획**을 시작해볼까요? 예를 들어, `'[서비스 이름] 데이터 기획안 만들어줘'` 와 같이 요청해보세요."
            return response
        else:
            return "서비스 아이디어 생성에 실패했거나, 생성된 아이디어가 없습니다."

    # [8. 데이터 기획 에이전트 완료 후 안내 메시지 추가]
    elif tool_name == "create_data_plan_for_service":
        plan = result_artifact.get("data_plan_result", {}).get("data_plan", {})
        if plan:
            service_name = plan.get('service_name', '해당 서비스')
            return f"""✅ **'{service_name}'**에 대한 데이터 기획안 생성이 완료되었습니다. 워크스페이스의 'data_plan_for_service' 항목에서 상세 내용을 확인해보세요.
            
이제 모든 준비가 끝났습니다!
마지막으로, 지금까지의 모든 내용을 종합하여 최종 산출물인 **'C-D-P 정의서'**를 만들어볼까요?
`"C-D-P 정의서 만들어줘"` 라고 요청해주세요.
"""
        else:
            return "데이터 기획안 생성에 실패했거나, 생성된 기획안이 없습니다."

    # 👈 [신규 안내 메시지 추가]
    elif tool_name == "create_cdp_definition":
        return """
        📑 **모든 기획 과정이 완료되었습니다!**
        최종 산출물인 **C-D-P 정의서**가 성공적으로 생성되었습니다. 워크스페이스의 'cdp_definition' 항목에서 상세 내용을 확인해보세요.
        이 문서를 바탕으로 실제 제품 및 서비스 개발을 시작할 수 있습니다. 수고하셨습니다!
        """

    else:
        return f"✅ '{tool_name}' 작업이 완료되었습니다."




@app.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    # 1. 세션 준비 및 사용자 메시지 기록
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    if session_id not in SESSIONS: SESSIONS[session_id] = create_new_workspace()
    workspace = SESSIONS[session_id]

    history_with_system_prompt = [{"role": "system", "content": SYSTEM_PROMPT}]

    if workspace["history"]:
        history_with_system_prompt.extend(workspace["history"])

    history_with_system_prompt.append({"role": "user", "content": request.message})

    workspace["history"].append({"role": "user", "content": request.message})
    
    response_to_user = "죄송합니다. 요청을 처리하는 데 실패했습니다."

    try:
        # 2. LLM 호출
        client = get_openai_client(async_client=True)
        response =  client.chat.completions.create(
            model=MODEL_NAME, 
            messages=history_with_system_prompt, 
            tools=tools, 
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        
        # 3. LLM 응답에 따라 분기
        if response_message.tool_calls:
            # [A. 도구 호출 시]
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            workspace["history"].append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type, # 'function'
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments, # arguments는 이미 JSON 문자열
                        }
                    }
                ]
            })
            
            # 4. 의존성 검사
            dependency_check_passed = True
            if function_name.startswith("run_cx_") or function_name == "run_ward_clustering":
                if not workspace["artifacts"].get("retrieved_data"):
                    dependency_check_passed = False
                    response_to_user = "분석을 위해서는 먼저 '데이터 검색'을 수행해야 합니다."
            
            if not dependency_check_passed:
                workspace["history"].append({"role": "assistant", "content": response_to_user})
                return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)

            # 5. 에이전트 실행
            function_to_call = available_functions[function_name]
            
            # [사용자 요청 반영] run_data_retriever 호출 전 특별 처리
            if function_name == "run_data_retriever":
                query = function_args.get("query")
                product_type = function_args.get("product_type")
                tip_message = ""

                # 프로그래밍 방식으로 product_type 검사 및 보강
                for p_keyword in LG_PRODUCT_KEYWORDS:
                    if p_keyword in query.lower():
                        product_type = p_keyword
                        break
                
                # 팁 메시지 생성
                if not product_type:
                    tip_message = "\n\n💡 **(팁)** 검색 키워드가 LG전자 제품 관련이라면, 제품군을 명시하면 더 정확한 결과를 얻을 수 있습니다."
                
                parsed_date_range = parse_natural_date(function_args.get("date_range_str"))
                
                result_artifact =  function_to_call(
                    workspace=workspace, keyword=function_args.get("keyword"), product_type=product_type, date_range=parsed_date_range
                )
            else:
                result_artifact =  function_to_call(workspace=workspace, **function_args)

            # 6. 결과 처리
            if result_artifact and "error" not in result_artifact:
                workspace["artifacts"].update(result_artifact)
                workspace["history"].append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": json.dumps(result_artifact, ensure_ascii=False, default=str)})
                
                if function_name == 'run_data_retriever':
                    response_to_user = "✅ 데이터 검색이 완료되었습니다."
                    recommendation = "\n\n---\n**이제 무엇을 해볼까요?**\n* `\"CX 분석 시작해줘\"`\n* `\"검색된 데이터로 페르소나 만들어줘\"`"
                    response_to_user += recommendation + tip_message
                    workspace = summarize_and_reset_history(workspace, function_name, result_artifact)
                else:
                    response_to_user = interpret_and_suggest_next_step(function_name, result_artifact, workspace)
            else:
                response_to_user = result_artifact.get("error", "알 수 없는 오류가 발생했습니다.")
        
        else: # [B. LLM이 직접 답변 시]
            response_to_user = response_message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        response_to_user = f"요청 처리 중 심각한 오류가 발생했습니다: {e}"

    workspace["history"].append({"role": "assistant", "content": response_to_user})
    return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)
