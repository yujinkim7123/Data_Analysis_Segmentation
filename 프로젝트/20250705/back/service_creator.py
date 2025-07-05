# agents/service_creator.py

import json
from .utils import get_openai_client
from .data_retriever import fetch_product_context, fetch_sensor_context, get_columns_for_product
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

def _build_service_creation_prompt(persona: dict, product_type: str | None, device_columns: dict, feature_docs: list, sensor_columns: list, num_ideas: int) -> str:
    """서비스 아이디어 생성을 위한 LLM 프롬프트를 동적으로 구성합니다."""
    prompt_header = f"""
    당신은 LG전자의 신사업 기획을 총괄하는 최고의 서비스 전략가입니다.
    고객 데이터에 기반하여, 단계별로 생각(Think step-by-step)해서 기존의 틀을 깨는 혁신적이면서도 실현 가능한 서비스 아이디어를 만드는 데 특화되어 있습니다.
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
    - **연관 센서 데이터 (샘플): {json.dumps(sensor_columns, ensure_ascii=False)}**
    """
    else:
        product_context_prompt = """
    ### [기존 제품 및 기능 정보]
    - (지정된 제품군 정보가 없습니다.)
    """
    instructions_prompt = f"""
    ### [지시사항]
    위 페르소나와 제품/센서 정보를 바탕으로, 다음 요구사항을 반드시 만족하는 **새로운 서비스 아이디어 {num_ideas}개**를 제안해주세요.

    1.  **Pain Point 해결**: 각 아이디어는 페르소나의 Pain Point 중 하나 이상을 명확하고 직접적으로 해결해야 합니다.
    2.  **데이터 활용**: 제안하는 서비스의 핵심 기능이 **어떤 제품 또는 센서 데이터**를 어떻게 활용하는지 구체적으로 설명해야 합니다. 특히, 센서 데이터를 조합하여 새로운 가치를 만드는 방안을 적극적으로 모색해주세요.
    3.  **서비스 확장성 (Scalability)**: 제안하는 서비스가 미래에 어떻게 성장하고 확장될 수 있는지 구체적인 방안을 반드시 포함해주세요. (예: 다른 제품 연동, 구독 모델 발전, 데이터 기반 개인화, 플랫폼화 등)
    4.  **결과 형식**: 아래 JSON 구조를 반드시 준수하여 다른 설명 없이 결과만 반환해주세요.
    {_get_json_format_prompt(product_type)}
    """
    
    return prompt_header + persona_data_prompt + product_context_prompt + instructions_prompt
# [수정] 기존 함수를 리팩토링된 구조에 맞게 수정
def create_service_ideas(workspace: dict, persona_name: str, num_ideas: int = 3):
    """(워크스페이스에 저장된) 지정된 페르소나의 Pain Point를 해결하는 새로운 서비스 아이디어를 생성합니다."""
   
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

    product_type = artifacts.get("product_type")
    device_columns = {}
    product_docs = []
    sensor_columns = []
    if product_type:
        print(f"🔍 제품군 '{product_type}'에 대한 기존 정보 활용 중...")
        if not artifacts.get("product_data"):
            product_docs = fetch_product_context(product_type)
            workspace["artifacts"]["product_data"] = product_docs
        else:   
            product_docs = artifacts.get("product_data")
        if not artifacts.get("product_data"):    
            device_columns = get_columns_for_product(product_type)
            workspace["artifacts"]["columns_product"] = device_columns
        else:
            device_columns = artifacts.get("columns_product")
        if not artifacts.get("sensor_data"):    
            sensor_columns = fetch_sensor_context(product_type)
            workspace["artifacts"]["sensor_data"] = sensor_columns
        else:
            sensor_columns = artifacts.get("sensor_data")
        
    final_prompt = _build_service_creation_prompt(selected_persona, product_type, device_columns, product_docs,sensor_columns, num_ideas)

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        service_idea_results = json.loads(res.choices[0].message.content)
        #workspace["artifacts"]["service_ideas"] = service_idea_results


        existing_ideas_list = artifacts.get("service_ideas", [])
        new_ideas_list = service_idea_results.get("service_ideas", [])
        
        # 3. 기존 리스트에 새로운 리스트를 합칩니다 (extend 사용).
        existing_ideas_list.extend(new_ideas_list)
        
        # 4. 합쳐진 리스트를 다시 올바른 객체 구조로 만들어 저장합니다.
        workspace["artifacts"]["service_ideas"] = {"service_ideas": existing_ideas_list}

        return {"service_ideas_result": service_idea_results}
    except Exception as e:
        print(f"❌ 서비스 아이디어 생성 중 오류 발생: {e}")
        return {"error": f"서비스 아이디어 생성 중 오류가 발생했습니다: {e}"}

def create_service_ideas_from_manual_input(workspace: dict, service_data: dict, num_ideas: int = 3) -> dict:
    """사용자가 직접 입력한 서비스 데이터를 기반으로 서비스 아이디어를 생성합니다."""
    print(f"✅ [Service Creator] Running Service Idea Generation from manual input...")
    
    artifacts = workspace.get("artifacts", {})

    # 입력 데이터 검증
    required_fields = ["service_name", "description", "solved_pain_points", "service_scalability"]
    missing_fields = [field for field in required_fields if field not in service_data or not service_data[field]]
    if missing_fields:
        return {"error": f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}"}
    
    # 워크스페이스에 서비스 데이터 저장
    if "service_ideas" not in artifacts:
        artifacts["service_ideas"] = {"service_ideas": []}

    
    # 기존 서비스 아이디어에 추가
    artifacts["service_ideas"].append({
        "service_name": service_data["service_name"],
        "description": service_data["description"],
        "solved_pain_points": service_data["solved_pain_points"],
        "service_scalability": service_data["service_scalability"]
    })

    return {"service_ideas_result": service_data}

    
def modify_service_ideas(workspace: dict, modification_request: str):
    """(수정) 기존에 생성된 서비스 아이디어를 사용자의 요청에 따라 수정합니다."""
    print(f"✅ [Service Creator] Running Service Idea Modification: '{modification_request}'")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    existing_ideas = artifacts.get("service_ideas")
    selected_persona = artifacts.get("selected_persona")
    if not existing_ideas or not selected_persona:
        return {"error": "수정할 서비스 아이디어가 없거나, 대상 페르소나가 선택되지 않았습니다."}

    product_type = artifacts.get("product_type")

    if product_type:
        print(f"🔍 제품군 '{product_type}'에 대한 기존 정보 활용 중...")
        if not artifacts.get("product_data"):
            product_docs = fetch_product_context(product_type)
            workspace["artifacts"]["product_data"] = product_docs
        else:   
            product_docs = artifacts.get("product_data")
        if not artifacts.get("product_data"):    
            device_columns = get_columns_for_product(product_type)
            workspace["artifacts"]["columns_product"] = device_columns
        else:
            device_columns = artifacts.get("columns_product")
        if not artifacts.get("sensor_data"):    
            sensor_columns = fetch_sensor_context(product_type)
            workspace["artifacts"]["sensor_data"] = sensor_columns
        else:
            sensor_columns = artifacts.get("sensor_data")

    # 수정용 프롬프트
    technical_context_prompt = ""
    if product_type:
        technical_context_prompt = f"""
    ### [참고용 기술 데이터]
    - 제품 상세 데이터 필드: {json.dumps(device_columns, ensure_ascii=False)}
    - 관련 기능 문서 요약: {json.dumps(product_docs, ensure_ascii=False)}
    - 연관 센서 데이터 (샘플): {json.dumps(sensor_columns, ensure_ascii=False)}
    - 제품군 : {product_type}
    """

    # 📌 [수정] 3. 수정용 프롬프트에 기술 컨텍스트와 지시사항 추가
    prompt = f"""
    당신은 최고의 서비스 전략가입니다. 단계별로 생각(Think step-by-step)하여, 아래 '기존 서비스 아이디어'를 '사용자 수정 요청'에 맞게 수정해주세요. 
    수정의 기반이 되는 페르소나 및 기술 데이터도 참고하세요.
    그리고 수정요청되지 않은 부분은 절대 변경하지 마세요.

    ### 기반 페르소나 정보
    {json.dumps(selected_persona, ensure_ascii=False, indent=2)}
    
    {technical_context_prompt}

    ### 기존 서비스 아이디어
    {json.dumps(existing_ideas, ensure_ascii=False, indent=2)}

    ### 사용자 수정 요청
    "{modification_request}"

    ### 지시사항
    - '사용자 수정 요청'을 완벽하게 반영하여 서비스 아이디어 전체를 다시 생성해주세요.
    - **(중요)** 수정사항을 반영할 때, 위에 제시된 **[참고용 기술 데이터]**를 적극적으로 활용하여 아이디어를 기술적으로 더 구체화하거나 보강해주세요.
    - **결과 형식**: 아래 JSON 구조를 반드시 준수하여 다른 설명 없이 결과만 반환해주세요.
    {_get_json_format_prompt(product_type)}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        service_idea_results = json.loads(res.choices[0].message.content)
        new_ideas = service_idea_results.get("service_ideas", [])

        # 1. 기존 서비스 아이디어 목록을 가져옵니다. 만약 없으면 빈 리스트로 시작합니다.
        existing_ideas_obj = workspace["artifacts"].get("service_ideas") or {"service_ideas": []}
        existing_ideas = existing_ideas_obj.get("service_ideas", [])

        # 2. 기존 아이디어를 'service_name'을 key로 사용하는 dict(ideas_map)으로 변환합니다.
        ideas_map = {idea["service_name"]: idea for idea in existing_ideas}

        # 3. 새로 생성된 아이디어 목록을 하나씩 확인하며 맵을 업데이트합니다.
        for idea in new_ideas:
            idea_name = idea.get("service_name")
            if idea_name:
                ideas_map[idea_name] = idea # 이름이 같으면 교체, 없으면 추가

        # 4. 업데이트된 dict를 다시 리스트로 변환하여 workspace에 저장합니다.
        #    이때 {"service_ideas": [ ... ]} 구조를 유지합니다.
        workspace["artifacts"]["service_ideas"] = {"service_ideas": list(ideas_map.values())}

        # 5. 함수 호출 결과로 새로 생성된 아이디어 정보를 그대로 반환합니다.
        return {"service_ideas_result": service_idea_results}
    except Exception as e:
        return {"error": f"서비스 아이디어 수정 중 오류: {e}"}