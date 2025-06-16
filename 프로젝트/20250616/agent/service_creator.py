# agents/service_creator.py

import json
from .utils import get_openai_client, get_columns_for_product
from .data_retriever import fetch_product_context
from qdrant_client.http.models import Filter, FieldCondition, MatchValue 
def _get_json_format_prompt(product_type: str | None) -> str:
    # ... (이 함수는 기존과 동일, 변경 없음) ...
    tip_field = ""
    if not product_type:
        tip_field = ',\n      "tip": "팁: 특정 LG 제품군을 지정하면 해당 제품에 더 최적화된 서비스 아이디어를 얻을 수 있습니다."'

    return f"""
    ```json
    {{
      "service_ideas": [
        {{
          "service_name": "AI 육아 위생 컨설턴트",
          "description": "페르소나의 아이 연령과 건강 상태(예: 아토피)에 맞춰, 의류, 장난감, 식기 등의 최적 살균 주기와 방법을 알려주고 가전제품(세탁기, 건조기 등)을 자동으로 제어해주는 구독형 서비스입니다.",
          "solved_pain_points": [
            "살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다",
            "매번 옷을 삶는 것은 번거롭고 옷감이 상할까 걱정된다"
          ],
          "service_scalability": "초기에는 ThinQ 앱의 기능으로 제공하고, 추후 영유아 건강 데이터를 연동한 프리미엄 유료 구독 모델로 확장할 수 있습니다. 또한, 축적된 데이터는 새로운 영유아 전문 가전 개발의 기반이 될 수 있습니다."
        }}
      ]{tip_field}
    }}
    ```
    """

# [리팩토링] 중복되는 LLM 프롬프트 생성 로직을 공통 함수로 분리
def _build_service_creation_prompt(persona: dict, product_type: str | None, device_columns: dict, feature_docs: list, num_ideas: int) -> str:
    """서비스 아이디어 생성을 위한 LLM 프롬프트를 동적으로 구성합니다."""
    prompt_header = f"""
    당신은 LG전자의 신사업 기획을 총괄하는 최고의 서비스 전략가입니다.
    고객 데이터에 기반하여, 기존의 틀을 깨는 혁신적이면서도 실현 가능한 서비스 아이디어를 만드는 데 특화되어 있습니다.
    """
    
    persona_data_prompt = f"""
    ### [분석 대상 페르소나 정보]
    - 이름: {persona.get('name')} ({persona.get('title')})
    - 인구통계: {persona.get('demographics')}
    - 핵심 니즈 및 목표: {persona.get('needs_and_goals')}
    - **핵심 불편함 (Pain Points): {persona.get('pain_points')}**
    - 동기부여 문구: "{persona.get('motivating_quote')}"
    """
    
    product_context_prompt = ""
    if product_type:
        product_context_prompt = f"""
    ### [기존 제품 및 기능 정보 (제품군: {product_type})]
    - 제품 상세 데이터 필드: {json.dumps(device_columns, ensure_ascii=False)}
    - 관련 기능 문서 요약: {json.dumps(feature_docs, ensure_ascii=False)}
    """
    else:
        product_context_prompt = """
    ### [기존 제품 및 기능 정보]
    - (지정된 제품군 정보가 없습니다.)
    """

    instructions_prompt = f"""
    ### [지시사항]
    위 정보를 바탕으로, 다음 요구사항을 반드시 만족하는 **새로운 서비스 아이디어 {num_ideas}개**를 제안해주세요.

    1.  **Pain Point 해결**: 각 아이디어는 페르소나의 Pain Point 중 하나 이상을 명확하고 직접적으로 해결해야 합니다.
    2.  **고객 관점**: 서비스가 고객에게 어떤 가치를 주는지, 고객이 어떻게 경험하게 될지를 중심으로 서술해주세요.
    3.  **혁신성**: 기존에 없던 새롭고 창의적인 아이디어를 제시해주세요. (만약 기존 제품 정보가 있다면, 해당 제품을 '확장'하거나 '보완'하는 서비스에 집중하세요.)
    4.  **서비스 확장성 (Scalability)**: 제안하는 서비스가 일회성 기능에 그치지 않고, 미래에 어떻게 성장하고 확장될 수 있는지 구체적인 방안을 반드시 포함해주세요. (예: 다른 제품 연동, 구독 모델 발전, 데이터 기반 개인화, 플랫폼화 등)
    5.  **결과 형식**: 아래 JSON 구조를 반드시 준수하여 다른 설명 없이 결과만 반환해주세요.
    {_get_json_format_prompt(product_type)}
    """
    
    return prompt_header + persona_data_prompt + product_context_prompt + instructions_prompt

# [수정] 기존 함수를 리팩토링된 구조에 맞게 수정
def create_service_ideas(workspace: dict, persona_name: str, num_ideas: int = 3):
    """(워크스페이스에 저장된) 지정된 페르소나의 Pain Point를 해결하는 새로운 서비스 아이디어를 생성합니다."""
    print(f"✅ [Service Creator] Running Service Idea Generation for '{persona_name}'...")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    all_personas = artifacts.get("personas")
    if not all_personas:
        return {"error": "서비스를 생성하려면 먼저 '페르소나 생성'을 통해 고객 페르소나를 만들어야 합니다."}

    selected_persona = next((p for p in all_personas if p.get("name") == persona_name), None)
    if not selected_persona:
        available_names = ", ".join([f"'{p.get('name')}'" for p in all_personas])
        return {"error": f"'{persona_name}' 페르소나를 찾을 수 없습니다. 사용 가능한 페르소나: [{available_names}]"}

    workspace["artifacts"]["selected_persona"] = selected_persona
    print(f"📌 Persona '{persona_name}' has been set as the selected persona.")

    retrieved_data = artifacts.get("retrieved_data", {})
    artifacts = workspace.get("artifacts", {})
    product_type = retrieved_data.get("product_type")
    
    device_columns = {}
    feature_docs = []
    if product_type:
        print(f"🔍 제품군 '{product_type}'에 대한 기존 정보 활용 중...")
        device_columns = get_columns_for_product(product_type)
        workspace["artifacts"]["product_data"] = device_columns
        feature_docs = retrieved_data.get("product_results", []) 
        
    final_prompt = _build_service_creation_prompt(selected_persona, product_type, device_columns, feature_docs, num_ideas)

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        service_idea_results = json.loads(res.choices[0].message.content)
        workspace["artifacts"]["service_ideas"] = service_idea_results
        return {"service_ideas_result": service_idea_results}
    except Exception as e:
        print(f"❌ 서비스 아이디어 생성 중 오류 발생: {e}")
        return {"error": f"서비스 아이디어 생성 중 오류가 발생했습니다: {e}"}

# [신규 추가] 수동 입력을 위한 새로운 에이전트 함수
def create_service_ideas_from_manual_input(workspace: dict, persona_description: str, product_type: str = None, num_ideas: int = 3):
    """사용자가 직접 입력한 페르소나 설명 텍스트를 기반으로 서비스 아이디어를 생성합니다."""
    print(f"✅ [Service Creator] Running Service Idea Generation from manual input...")
    client = get_openai_client()

    parser_prompt = f"""
    다음 텍스트는 사용자가 생각하는 고객 페르소나에 대한 설명입니다. 이 설명에서 '이름', '제목', '인구통계', '핵심 니즈 및 목표', '핵심 불편함(Pain Points)', '동기부여 문구'를 추출하여 아래 JSON 형식으로 정리해주세요.
    만약 특정 필드에 대한 정보가 부족하면, 내용을 바탕으로 추론하거나 비워두세요.
    ---
    [사용자 입력 텍스트]: {persona_description}
    ---
    [JSON 형식]: {{ "name": "사용자 (가명)", "title": "페르소나의 특징을 나타내는 제목", "demographics": "추정되는 인구통계 정보", "needs_and_goals": ["니즈/목표 1"], "pain_points": ["불편함 1"], "motivating_quote": "페르소나를 한 문장으로 요약하는 인용구" }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": parser_prompt}],
            response_format={"type": "json_object"}
        )
        selected_persona = json.loads(res.choices[0].message.content)
        print(f"🔍 파싱된 페르소나 정보: {selected_persona}")

        workspace["artifacts"]["personas"].append(selected_persona)
        workspace["artifacts"]["selected_persona"] = selected_persona
        print(f"📌 Manually described persona has been created and set as the selected persona.")


    except Exception as e:
        return {"error": f"페르소나 설명 분석 중 오류 발생: {e}"}

    device_columns = {}
    feature_docs = []
    if product_type:
        print(f"🔍 제품군 '{product_type}'에 대한 정보 실시간 조회 중...")
        device_columns = get_columns_for_product(product_type)
        workspace["artifacts"]["product_data"] = device_columns
        keyword = selected_persona.get("pain_points", [""])[0] or persona_description
        feature_docs = fetch_product_context(keyword, product_type, top_k=20)
        workspace["artifacts"]["product_type"] = product_type
    final_prompt = _build_service_creation_prompt(selected_persona, product_type, device_columns, feature_docs, num_ideas)

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        service_idea_results = json.loads(res.choices[0].message.content)
        workspace["artifacts"]["service_ideas"] = service_idea_results
        return {"service_ideas_result": service_idea_results}
    except Exception as e:
        print(f"❌ 서비스 아이디어 생성 중 오류 발생: {e}")
        return {"error": f"서비스 아이디어 생성 중 오류가 발생했습니다: {e}"}