#---외부라이브러리--
import os
import asyncio
import json
import hashlib
import tiktoken
import gzip

#--웹 서버와 api 요청/응답 처리 지원--
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware # 1. 이 줄을 추가합니다.

#--데이터 모델 정의
from pydantic import BaseModel
from typing import List, Dict, Union


#--openai api 응답 처리
from openai.types.chat import ChatCompletionMessage

from dotenv import load_dotenv
import uuid
from agents.tools import tools, available_functions,suggest_next_step,create_new_workspace
from pydantic import BaseModel

#--내부 모듈 함수
from agents.utils import ( get_openai_client,
    save_workspace_to_redis, load_workspace_from_redis,MODEL_NAME, setup_logging
)



# --- 파일 로드 및 초기화 ---
load_dotenv()
#-----로그 준비------
logger = setup_logging()


#----AI 에이전트 동작 가이드라인을 정의함
SYSTEM_PROMPT = """
        당신은 최고의 AI 컨설턴트이자 비즈니스 코치입니다. 사용자의 요청을 처리하기 위해 다음의 단계별 작업 흐름과 도구 사용 가이드라인을 엄격히 따릅니다. 사용자의 요청은 두 가지 형식으로 들어옵니다: (1) 프론트엔드에서 전송된 JSON 형식의 구조화된 요청, (2) 자연어로 작성된 발화 형식의 요청. 이를 구분하여 적절히 처리해야 합니다.

        **워크스페이스 상태**:
        - Artifacts: {artifacts_summary}
        - VOC 데이터 존재 여부: {has_retrieved_data}
        - 마지막 요청 타입: {last_request_type}

        **지침**:
        1. Artifacts에 'retrieved_data'가 존재하면 클러스터링 요청을 처리할 수 있습니다.
        2. 클러스터링 요청 시 'retrieved_data'를 기반으로 결과를 제공하세요.
        3. 'retrieved_data'가 없으면 사용자에게 데이터 검색을 요청하세요.
        4. 도구 호출 결과(예: run_ward_clustering)를 명확히 반영하세요.
        5. 사용자에게 다음 단계 제안을 항상 포함하세요.

        **[핵심 작업 흐름 및 도구 사용 가이드라인]**

        1. **요청 형식 분석 및 분류**
           - **JSON 요청 처리**:
             - 메시지가 JSON 형식인지 확인합니다. JSON 요청은 `type` 필드를 포함하며, 예: `{{ "type": "data_retriever_request", "keyword": "...", "date_range": "...", "product_type": "..." }}`.
             - `type`에 따라 도구 호출:
               - `"data_retriever_request"`: `run_data_retriever(keyword, date_range_str, product_type)` 호출.
               - `"manual_persona_request"`: `create_personas(workspace, persona_data)` 호출.
               - `"manual_service_request"`: `create_service_ideas_from_manual_input(workspace, service_data)` 호출.
               - `"change_product_type_request"`: `fetch_product_context(workspace, product_type)`, `fetch_sensor_context(workspace, product_type)`, `get_columns_for_product(product_type)` 호출.
             - 필수 파라미터 누락 시, 사용자에게 명확한 오류 메시지와 함께 필요한 정보를 요청.
           - **자연어 요청 처리**:
             - `type: "chat_message"`인 경우, 메시지 내용을 분석하여 의도를 파악하고, 워크스페이스의 `artifacts`와 `history`를 참조하여 적절한 도구 호출 또는 직접 응답.
             - 모호한 경우, 추가 정보를 요청하거나 워크스페이스 상태를 기반으로 다음 단계 제안.

        2. **데이터 검색**
           - JSON 요청 `type: "data_retriever_request"` 또는 자연어로 "데이터 검색", "VOC 분석" 등의 의도가 확인되면 `run_data_retriever(keyword, date_range_str, product_type)` 호출.
           - 필수 파라미터: `keyword` (문자열), `date_range_str` (선택, 문자열), `product_type` (선택, 문자열).
           - **전제 조건**: 이 단계는 가장 먼저 수행되어야 하며, VOC 데이터 없이는 분석 불가.
           - 예: `{{ "type": "data_retriever_request", "keyword": "살균", "date_range": "최근 1년", "product_type": "스타일러" }}`

        3. **고객 인사이트 분석**
           - **전제 조건**: `run_data_retriever`로 VOC 데이터가 워크스페이스의 `artifacts.retrieved_data`에 저장된 경우.
           - 도구 호출 순서:
             - `run_ward_clustering(workspace)`: 데이터 검색 후 첫 분석.
             - `run_semantic_network_analysis(workspace, cluster_id)`: `cluster_id` 지정 시.
             - `run_topic_modeling_lda(workspace, cluster_id)`: `cluster_id` 지정 시.
             - `calculate_opportunity_scores(workspace)`: 토픽 모델링 완료 후.
             - `create_customer_action_map(workspace, topic_id)`: `topic_id` 지정 시.
           - 자연어 요청 예: "클러스터링 해줘" → `run_ward_clustering`.
           - JSON 요청 예: `{{ "type": "chat_message", "content": "0번 클러스터에 대해 SNA 분석해줘" }}` → `run_semantic_network_analysis`.

        4. **페르소나 정의**
           - **전제 조건**: 고객 인사이트 분석 완료 (`artifacts.cx_ward_clustering_results` 또는 `cx_lda_results` 존재).
           - JSON 요청: `{{ "type": "manual_persona_request", "persona_data": {{ "name": "...", "title": "...", ... }} }}` → `create_personas`.
           - 자연어 요청: "페르소나 만들어줘" → 분석 결과 확인 후 `create_personas`.
           - 수정 요청: `modify_personas(workspace, persona_data)`.

        5. **서비스 아이디어 도출**
           - **전제 조건**: 페르소나 정의 완료 (`artifacts.personas` 존재).
           - JSON 요청: `{{ "type": "manual_service_request", "service_data": {{ "service_name": "...", ... }} }}` → `create_service_ideas_from_manual_input`.
           - 자연어 요청: "서비스 아이디어 제안해줘" → `create_service_ideas(workspace)`.
           - 수정 요청: `modify_service_ideas(workspace, service_data)`.

        6. **데이터 기획**
           - **전제 조건**: 서비스 아이디어 도출 완료 (`artifacts.service_ideas` 존재).
           - 도구: `create_data_plan_for_service(workspace)`, `modify_data_plan(workspace, data_plan)`.

        7. **C-D-P 정의서 작성**
           - **전제 조건**: 모든 단계 완료 (`artifacts.data_plan_for_service` 존재).
           - 도구: `create_cdp_definition(workspace)`, `modify_cdp_definition(workspace, cdp_data)`.

        **[오류 처리 및 사용자 안내 원칙]**

        - **JSON 요청 오류**: 필수 파라미터 누락 시, 어떤 파라미터가 필요한지 명확히 안내 (예: "keyword가 누락되었습니다. 검색어를 입력해 주세요.").
        - **자연어 요청 모호성**: 의도가 불분명하면 추가 질문을 하거나 워크스페이스 상태를 기반으로 다음 단계 제안.
        - **도구 실행 실패**: 오류 메시지 전달 및 누락된 단계 안내 (예: "VOC 데이터가 없습니다. 먼저 데이터 검색을 수행해 주세요.").
        - **워크스페이스 활용**: 최신 `artifacts`를 참조하여 문맥에 맞는 응답 생성. 아티팩트 요약Zm은 `retrieved_data`, `cx_ward_clustering_results`, `cx_lda_results`, `cx_opportunity_scores`, `cx_cam_results`, `cx_sna_results`, `personas`, `selected_persona`, `service_ideas`, `selected_service_idea`, `data_plan_for_service`, `selected_data_plan_for_service`, `cdp_definition`, `sensor_data`, `product_data`, `columns_product`, `data_plan_recommendation_message`, `selected_cdp_definition`을 포함.

        **[응답 형식]**
        - `{{ "response_message": "...", "workspace": {{...}} }}`
        - 모든 응답은 `ChatResponse` 모델 준수.

        **[현재 워크스페이스 아티팩트 요약]**
        {artifacts_summary}

        **[작업 수행]**
        1. 입력 메시지를 분석하여 JSON인지 자연어인지 판단.
        2. JSON 요청이면 `type`과 파라미터를 파싱하여 도구 호출. `run_data_retriever`는 `keyword`, `date_range_str`, `product_type`으로 호출, 다른 도구는 `workspace` 포함.
        3. 자연어 요청이면 의도를 파악하여 도구 호출 또는 직접 응답.
        4. 도구 호출 전 전제 조건 확인 (예: VOC 데이터 존재 여부).
        5. 도구 호출 후 결과를 `artifacts`에 저장하고, 다음 단계 제안.
        6. 최종 응답은 `response_message`와 `workspace` 포함.
    """

#--------------------- FastAPI 앱 객체 설정--------------------------------
app = FastAPI(title="기획자 AI Agent MCP 서버")

# CORS 설정 프론트엔드 통신 허용 기준
origins = [
    "http://localhost:3001", # 프론트엔드 서버의 주소
    "http://localhost:3000",
]

# X-Session-ID 헤더를 통해 세션 ID를 전달하여 워크스페이스 지속성을 유지
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"]
)

#-----클래스 정의-------------------------------------
#사용자 요청 데이터 모델
class UserRequest(BaseModel):
    session_id: str | None = None
    message: str
#서버 응답 데이터 모델
class ChatResponse(BaseModel):
    response_message: str
    workspace: dict
    user_history: list
    artifacts: dict
    error: str | None = None

#internal_history의 메시지를 검증하여 tooll 메시지가 유효한 tool_call_id를 가지는지 확인합니다.---
def validate_messages(messages: List[Dict]) -> List[Dict]:
    """Validate messages to ensure 'tool' messages follow 'tool_calls'."""
    validated_messages = []
    tool_call_ids = set()
    
    for i, msg in enumerate(messages):
        if msg.get('role') == 'tool':
            if not msg.get('tool_call_id'):
                print(f"Warning: Tool message without tool_call_id at index {i}")
                continue  # Skip invalid tool message
            if msg['tool_call_id'] not in tool_call_ids:
                print(f"Warning: Tool message with invalid tool_call_id {msg['tool_call_id']} at index {i}")
                continue
        validated_messages.append(msg)

        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                tool_call_ids.add(tool_call['id'])
    
    return validated_messages

#워크스페이스에 internal_history, user_history에 메시지를 추가하고, 최대 50개로 제한한다.
def append_to_history(workspace, message):
    workspace["internal_history"].append(message)
    if message["role"] in ["user", "assistant"]:
        workspace["user_history"].append(message)
    max_history_length = 50
    if len(workspace["internal_history"]) > max_history_length:
        workspace["internal_history"] = workspace["internal_history"][-max_history_length:]
    if len(workspace["user_history"]) > max_history_length:
        workspace["user_history"] = workspace["user_history"][-max_history_length:]

#internal_history, user_history에 토큰 수 제한
def trim_history(history: List[Union[Dict, ChatCompletionMessage]]):
    """
    대화 기록을 관리하고, 특정 길이를 초과하면 오래된 메시지를 제거합니다.
    OpenAI ChatCompletionMessage 객체를 JSON 직렬화 가능하도록 변환합니다.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    # 각 메시지를 JSON 직렬화 가능한 형태로 변환하면서 토큰 수를 계산합니다.
    for msg_item in history: # 'msg' 대신 'msg_item'으로 변수명 변경 (가독성 목적)
        json_serializable_msg = msg_item
        
        # ChatCompletionMessage나 ChatCompletionMessageToolCall 같은 Pydantic 모델인지 확인하고 딕셔너리로 변환
        if hasattr(msg_item, 'model_dump') and callable(msg_item.model_dump):
            json_serializable_msg = msg_item.model_dump()
        elif hasattr(msg_item, 'dict') and callable(msg_item.dict): # Pydantic v1 호환성
            json_serializable_msg = msg_item.dict()
        
        # 직렬화 가능한 메시지를 JSON으로 덤프하고 토큰 수를 계산합니다.
        total_tokens += len(encoding.encode(json.dumps(json_serializable_msg, ensure_ascii=False))) # ensure_ascii=False 추가하여 한글/이모지 처리 개선

    return history # 수정된 history를 반환

#----워크스페이스의 내용들 요약생성 system_prompt에 전달된다 ------
def summarize_artifact(artifacts: dict) -> str:
    """워크스페이스의 아티팩트를 요약하여 LLM 프롬프트에 포함할 문자열을 생성합니다."""
    summary_parts = []

    if not artifacts:
        return "현재 워크스페이스에 저장된 아티팩트가 없습니다."

    for key, value in artifacts.items():
        if key == "retrieved_data" and value and value.get("web_results"):
            summary_parts.append(f"- 검색된 VOC 데이터: {len(value['web_results'])}건")
        elif key == "cx_ward_clustering_results" and value and value.get("cluster_summaries"):
            summary_parts.append(f"- 워드 클러스터링: {len(value['cluster_summaries'])}개 클러스터")
        elif key == "cx_lda_results" and value and value.get("topics"):
            summary_parts.append(f"- 토픽 모델링: {len(value['topics'])}개 토픽")
        elif key == "cx_cam_results" and value:
            summary_parts.append(f"- 고객 행동 맵: {len(value)}개 생성됨")
        elif key == "cx_opportunity_scores" and value:
            summary_parts.append(f"- 기회 점수 분석: {len(value)}개 완료됨")
        elif key == "cx_sna_results" and value:
            summary_parts.append(f"- 의미 네트워크 분석: {len(value)}개 완료됨")
        elif key == "personas" and value and isinstance(value, list):
            names = ", ".join([p.get("name", "이름 없음") for p in value])
            summary_parts.append(f"- 페르소나: {len(value)}개 ({names})")
        elif key == "selected_persona" and value and value.get("name"):
            summary_parts.append(f"- 현재 선택된 페르소나: {value['name']}")
        elif key == "service_ideas" and value and isinstance(value, list):
            names = ", ".join([s.get("service_name", "이름 없음") for s in value])
            summary_parts.append(f"- 서비스 아이디어: {len(value)}개 ({names})")
        elif key == "selected_service_idea" and value and value.get("service_name"):
            summary_parts.append(f"- 현재 선택된 서비스 아이디어: {value['service_name']}")
        elif key == "data_plan_for_service" and value and isinstance(value, list):
            names = ", ".join([p.get("service_name", "이름 없음") for p in value])
            summary_parts.append(f"- 데이터 기획안: {len(value)}개 ({names})")
        elif key == "selected_data_plan_for_service" and value and value.get("service_name"):
            summary_parts.append(f"- 현재 선택된 데이터 기획안: {value['service_name']}")
        elif key == "cdp_definition" and value:
            summary_parts.append(f"- C-D-P 정의서: {len(value)}개 생성됨")
        elif key == "sensor_data" and value:
            summary_parts.append(f"- 센서 데이터: {len(value)}건")
        elif key == "product_data" and value:
            summary_parts.append(f"- 제품 데이터: {len(value)}건")
        elif key == "columns_product" and value:
            summary_parts.append(f"- 제품 메타데이터: {len(value)}개 필드")
        elif key == "data_plan_recommendation_message" and value:
            summary_parts.append(f"- 데이터 기획 추천 메시지: 저장됨")
        elif key == "selected_cdp_definition" and value:
            summary_parts.append(f"- 현재 선택된 C-D-P 정의서: 저장됨")
        # conversation_state는 artifacts 외부에서 관리하므로 여기서 제외

    if not summary_parts:
        return "현재 워크스페이스에 저장된 아티팩트가 없습니다."

    return "현재 워크스페이스에는 다음 아티팩트가 저장되어 있습니다:\n" + "\n".join(summary_parts)


#---openai api용 메시지 리스트 internal_histoty와 artifacts 상태를 포함한 메시지 생성---
def prepare_openai_messages(workspace: dict, system_message_content: str) -> list:
    """Prepare messages for OpenAI API with explicit artifacts state."""
    messages = [{"role": "system", "content": system_message_content}]
    messages.extend(workspace.get("internal_history", []))
        # Artifacts 상태를 메시지에 추가
    artifacts_summary = summarize_artifact(workspace.get("artifacts", {}))
    messages.append({
        "role": "system",
        "content": f"Current artifacts state: {artifacts_summary}"
    })
    return messages

#------json 사용자 요청 대응 ---------------------
async def handle_json_request(message_dict: dict, workspace: dict, session_id: str) -> tuple[str, dict]:
    logger = setup_logging()
    
    #agent 함수 매칭
    function_mapping = {
        "data_retriever_request": {"func": available_functions["run_data_retriever"], "required": ["keyword"]},
        "manual_persona_request": {"func": available_functions["create_persona_from_manual_input"], "required": ["persona_data"]},
        "manual_service_request": {"func": available_functions["create_service_ideas_from_manual_input"], "required": ["service_data"]},
        "change_product_type_request": {"func": available_functions["conext_change"], "required": ["product_type"]}
    }

    #함수 종류 type 추출
    message_type = message_dict.get("type")
    workspace["last_request_type"] = message_type

    #------사용자 요청 함수가 function_mapping에 존재하는지 확인
    if message_type not in function_mapping:
        response_to_user = f"🚨 오류: 알 수 없는 요청 타입: {message_type}"
        return response_to_user, workspace


    #----매핑되는 함수 존재할 경우-------------
    func_info = function_mapping[message_type]
    #--------func 실제 함수 이름에 접근---------
    function_name = func_info["func"].__name__
    #-----함수 필수 인자 -------
    required_params = func_info["required"]
    #----함수에 대한 필수 인자 존재 여부 확인 ------
    function_args = {k if k != "date_range" else "date_range_str": message_dict.get(k) for k in ["keyword", "date_range", "product_type", "persona_data", "service_data"] if message_dict.get(k)}


    #--------필수 파라미터 확인 및 누락 시 오류 반환 -----------------
    if not all(k in function_args and function_args[k] for k in required_params):
        missing_params = [k for k in required_params if k not in function_args or not function_args[k]]
        response_to_user = f"⚠️ 요청에 필수 파라미터가 누락되었습니다: {', '.join(missing_params)}"
        return response_to_user, workspace
    
    
    #------------------필수 인자 확인 완료 + 함수 존재 => 함수 호출 --------
    try:
        
        logger.debug(f"Calling {function_name} with args: {function_args}")


        #------비동기 쓰레드로 함수 호출------------
        result_artifact = await asyncio.to_thread(func_info["func"], workspace=workspace, **function_args)
        
        #logger.debug(f"{function_name} result: {result_artifact}")

        #함수 호출 결과물이 error인 경우,실행실패
        if "error" in result_artifact:
            response_to_user = f"⚠️ {function_name} 실행 실패: {result_artifact['error']}"
        #함수가 잘 호출이 된 경우,값 저장 후 반환
        else:
            save_workspace_to_redis(session_id, workspace)
            # 다음 단계 제안
            next_step = suggest_next_step(workspace)
            response_to_user = f"{function_name} 작업이 완료되었습니다. 다음 단계를 진행하시겠습니까?\n\n📌 다음 단계 제안: {next_step}"

        return response_to_user, workspace
    
    ###진행 중 에러가 난 경우
    except Exception as e:
        logger.error(f"{function_name} 실행 중 오류 발생: {e}")
        response_to_user = f"🚨 {function_name} 실행 중 오류 발생: {e}"
        return response_to_user, workspace

#----사용자 json 요청 자연어로 변환
def transform_user_resquest(user_message: str, message_dict: dict, message_type: str):

    if message_type == "data_retriever_request":
        keyword = message_dict.get("keyword", "")
        date_range = message_dict.get("date_range", "지정되지 않음")
        product_type = message_dict.get("product_type", "지정되지 않음")
        natural_language_content = f"{product_type} 제품에 대해 '{keyword}' 키워드로 {date_range} 데이터를 검색해 주세요."
    elif message_type == "manual_persona_request":
        persona_name = message_dict.get("persona_data", {}).get("name", "지정되지 않음")
        natural_language_content = f"'{persona_name}' 페르소나를 생성해 주세요."
    elif message_type == "manual_service_request":
        service_name = message_dict.get("service_data", {}).get("service_name", "지정되지 않음")
        natural_language_content = f"'{service_name}' 서비스 아이디어를 제안해 주세요."
    elif message_type == "change_product_type_request":
        product_type = message_dict.get("product_type", "지정되지 않음")
        natural_language_content = f"제품군을 '{product_type}'로 변경해 주세요."
    else:
        natural_language_content = user_message

    return natural_language_content


async def run_agent_and_get_response(user_message: str, workspace: dict, session_id: str) -> tuple[str, dict]:
    logger = setup_logging()
    client = get_openai_client(async_client=True)
    
    #json 형태의 사용자 요청인지 확인하기
    try:
        message_dict = json.loads(user_message)
        #json인 경우 type 파싱해서 요청한 함수 확인하기
        message_type = message_dict.get("type")
        #사용자 요청을 자연어로 변환
        natural_language_content = transform_user_resquest(user_message, message_dict, message_type)
    except json.JSONDecodeError:
    #json 요청이 아닌 경우,일반 자연어 요청
        message_type = "chat_message"
        message_dict = {"type": "chat_message", "content": user_message}
        natural_language_content = user_message

    #변환된 자연어가 있는 경우 user_history로 저장 사용자 요청 답변으로 사용
    if user_message:
        append_to_history(workspace, {"role": "user", "content": natural_language_content})

    #json 요청인 경우
    if message_type != "chat_message":
        #json 요청 처리하는 함수 호출 
        response_to_user, workspace = await handle_json_request(message_dict, workspace, session_id)
        append_to_history(workspace, {"role": "assistant", "content": response_to_user})
        workspace["internal_history"] = trim_history(workspace["internal_history"])
        workspace["user_history"] = trim_history(workspace["user_history"])
        save_workspace_to_redis(session_id, workspace)
        return response_to_user, workspace


    #요약 workspace 상황 저장하기 
    current_artifacts_summary = summarize_artifact(workspace.get("artifacts", {}))

    #검색 결과값 있는지 확인
    has_retrieved_data = bool(workspace.get("artifacts", {}).get("retrieved_data"))

    if not workspace["last_request_type"]:
        workspace["last_request_type"] = "없음"
    print(workspace["last_request_type"])

    #검색 결과값 여부와 last_request_type 정보, 통계요약 정보를 시스템 프롬포트에 대입해서 
    ######최종 시스템 프롬포트 도출 ###############
    system_message_content = SYSTEM_PROMPT.format(
        artifacts_summary=current_artifacts_summary,
        has_retrieved_data=str(has_retrieved_data),
        last_request_type=workspace.get("last_request_type")
    )

    ###openai 보낼 최종 문장 생성 -> tool 호출을 잘하기 위해서 
    messages = prepare_openai_messages(workspace, system_message_content)

    ##tools 변수에 담기
    tools_for_openai = tools

    try:
        ###tools에서 선택해줘(사용자의 발화가 원하는 함수)
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools_for_openai,
            tool_choice="auto",
            stream=False,
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        llm_initial_content = response_message.content

        append_to_history(workspace, response_message.model_dump(exclude_none=True))

        if tool_calls:
            print("--- 🛠️ Tool Calls Requested ---")
            
            tool_outputs_to_append = []
            collected_error_messages = []

            #tools_calls 에 호출된 함수 순회
            for tool_call in tool_calls:
                #선택된 함수 이름 추출/함수 가져오기
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)

                #-----호출된 함수가 있는 경우---------------
                if function_to_call:
                    #--------------호출된 함수 인자 추출------------
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"🛠️ Executing tool: {function_name} with args: {function_args}")

                    #------------호출된 함수의 종류에 따른 분기점----------------
                    try:
                        result_artifact = await asyncio.to_thread(function_to_call, workspace=workspace, **function_args)

                       
                        #결과값에 에러가 가 있는 경우
                        if "error" in result_artifact:
                            tool_summary_content["error"] = result_artifact["error"]
                            collected_error_messages.append(f"도구 '{function_name}' 실행 실패: {result_artifact['error']}")
                        # else:
                        #     #결과값 정상인 경우, 결과값 업데이트하기 
                        #     workspace["artifacts"].update(result_artifact)

                        artifact_summary = summarize_artifact(workspace.get("artifacts", {}))

                        ####tool------------openai 답변 생성 --------------------------------------------------
                        #요약 생성 호출된 함수에 대한
                        tool_summary_content = {
                            "tool_name": function_name,
                            "success": "error" not in result_artifact,
                            "details": artifact_summary
                            }
                        #최종 결과 정리해서 tool role로 openai용 생성
                        tool_outputs_to_append.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(tool_summary_content, ensure_ascii=False)
                        })

                    # 해당 단계에서 에러가 나오면 예외 처리    
                    except Exception as e:
                        error_message = f"🚨 도구 '{function_name}' 실행 중 오류 발생: {e}"
                        collected_error_messages.append(error_message)
                        tool_outputs_to_append.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": error_message}, ensure_ascii=False)
                        })

                ###호출된 함수가 없는경우 (사용자가 요청하는 함수가 무엇인지 모르겠음)     
                else:
                    error_message = f"🚨 오류: 알 수 없는 함수 호출 시도: {function_name}"
                    collected_error_messages.append(error_message)
                    tool_outputs_to_append.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"error": error_message}, ensure_ascii=False)
                    })

            #최종 호출된 함수들의 interal_history 업데이트
            append_to_history(workspace, tool_outputs_to_append if len(tool_outputs_to_append) > 1 else tool_outputs_to_append[0])
            workspace["internal_history"] = trim_history(workspace["internal_history"])
            workspace["user_history"] = trim_history(workspace["user_history"])
           
            #모든 답변과 함께 업데이트된 workspace 저장하기 
            save_workspace_to_redis(session_id, workspace)

            #에러가 있는 경우, 답변 생성 에러관련
            if collected_error_messages:
                response_to_user = "⚠️ 다음 문제로 인해 요청을 완료할 수 없습니다:\n" + "\n".join(collected_error_messages)
                append_to_history(workspace, {"role": "assistant", "content": response_to_user})
                return response_to_user, workspace

            #두 번째 LLM 호출 전 artifacts 상태 재확인 및 system prompt 생성하기
            current_artifacts_summary = summarize_artifact(workspace.get("artifacts", {}))
            has_retrieved_data = bool(workspace.get("artifacts", {}).get("retrieved_data"))
            system_message_content = SYSTEM_PROMPT.format(
                artifacts_summary=current_artifacts_summary,
                has_retrieved_data=str(has_retrieved_data),
                last_request_type=workspace.get("last_request_type", "없음")
            )

            #open ai에게 보낼 message 최종 생성
            messages = prepare_openai_messages(workspace, system_message_content)
            
            #최종 결과에 대한 ai 답변 생성
            final_response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_for_openai,
                tool_choice="auto",
                stream=False,
            )
            final_response_message = final_response.choices[0].message
            final_llm_content = final_response_message.content

            append_to_history(workspace, final_response_message.model_dump(exclude_none=True))

            workspace["internal_history"] = trim_history(workspace["internal_history"])
            workspace["user_history"] = trim_history(workspace["user_history"])
            save_workspace_to_redis(session_id, workspace)

            response_to_user = final_llm_content
            return response_to_user, workspace

        else:
            ### 도구 호출이 없더라도 llm에 응답 뒤에 다음 단계 제안하기
            if llm_initial_content:
                append_to_history(workspace, {"role": "assistant", "content": llm_initial_content})
                next_step = suggest_next_step(workspace)
                response_to_user = llm_initial_content + f"\n\n📌 다음 단계 제안: {next_step}"
            else:
                response_to_user = "어떤 도움을 드릴까요?"
                append_to_history(workspace, {"role": "assistant", "content": response_to_user})
            ###저장 최종으로 
                workspace["internal_history"] = trim_history(workspace["internal_history"])
                workspace["user_history"] = trim_history(workspace["user_history"])
                save_workspace_to_redis(session_id, workspace)
                return response_to_user, workspace
        
    ## 이 모든 과정에서 생성되는 오류 
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        error_message = f"🚨 에이전트 실행 중 오류 발생: {str(e)}"
        append_to_history(workspace, {"role": "assistant", "content": error_message})
        save_workspace_to_redis(session_id, workspace)
        return error_message, workspace
    

@app.get("/")
def read_root():
    return {"message": "MCP 서버가 성공적으로 실행되었습니다."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_request: UserRequest, response: Response):
    print("--- 💬 /chat 엔드포인트 호출됨 ---")
    session_id = user_request.session_id or str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}")


    ##session_id에 따른 wroksapce 가져오기
    workspace = load_workspace_from_redis(session_id)
    if not workspace or not isinstance(workspace, dict):
        workspace = create_new_workspace()
        logger.info(f"New workspace initialized for session: {session_id}")

    try:
        assistant_response_content, updated_workspace = await run_agent_and_get_response(
            user_message=user_request.message,
            workspace=workspace,
            session_id=session_id
        )
        response.headers["X-Session-ID"] = session_id
        save_workspace_to_redis(session_id, updated_workspace)
        return {
            "response_message": assistant_response_content,
            "workspace": updated_workspace,
            "user_history": updated_workspace.get("user_history", []),
            "artifacts": updated_workspace.get("artifacts", {}),
            "error": None
        }
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        error_message = f"🚨 서버 오류: {str(e)}"
        append_to_history(workspace, {"role": "assistant", "content": error_message})
        save_workspace_to_redis(session_id, workspace)
        return {
            "response_message": error_message,
            "workspace": workspace,
            "user_history": workspace.get("user_history", []),
            "artifacts": workspace.get("artifacts", {}),
            "error": str(e)
        }