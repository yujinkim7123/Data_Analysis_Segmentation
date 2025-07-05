# agents/data_planner.py

import json
from .utils import get_openai_client
from .data_retriever import fetch_product_context, fetch_sensor_context, get_columns_for_product

def create_data_plan_for_service(workspace: dict, service_name: str = None, product_type: str = None):
    """
    서비스 아이디어를 기반으로 데이터 기획안을 생성합니다.
    """
    print(f"✅ [Data Planner] Running Data Plan Generation...")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    service_context_text = ""
    selected_idea_name = "사용자 정의 아이디어"

    if service_name:
        all_ideas = artifacts.get("service_ideas", {}).get("service_ideas", [])
        selected_idea = next((idea for idea in all_ideas if idea.get("service_name") == service_name), None)
        
        workspace["artifacts"]["selected_service_idea"] = selected_idea
        print(f"📌 Service idea '{service_name}' has been set as the selected service idea.")
          
        if not selected_idea:
            return {"error": f"'{service_name}' 이름의 서비스 아이디어를 찾을 수 없습니다."}
        
        service_context_text = json.dumps(selected_idea, ensure_ascii=False, indent=2)
        selected_idea_name = selected_idea.get('service_name')
    else:
        return {"error": "데이터 기획안을 생성하려면 'service_name' 또는 'service_description' 중 하나는 반드시 제공되어야 합니다."}


    final_product_type = product_type or artifacts.get("product_type")
    product_context_prompt = "연관된 특정 제품군 정보가 없습니다."

    if final_product_type:
        print(f"🔍 제품군 '{final_product_type}'에 대한 기존 정보 활용 중...")
        if not artifacts.get("product_data"):
            product_docs = fetch_product_context(final_product_type)
            workspace["artifacts"]["product_data"] = product_docs
        else:   
            product_docs = artifacts.get("product_data")
        if not artifacts.get("product_data"):    
            device_columns = get_columns_for_product(final_product_type)
            workspace["artifacts"]["columns_product"] = device_columns
        else:
            device_columns = artifacts.get("columns_product")
        if not artifacts.get("sensor_data"):    
            sensor_columns = fetch_sensor_context(final_product_type)
            workspace["artifacts"]["sensor_data"] = sensor_columns
        else:
            sensor_columns = artifacts.get("sensor_data")

        product_context_prompt = f"""
        ### [제품/센서 데이터 컨텍스트 (제품군: {final_product_type})]
        - **기존 제품 상세 데이터 필드:** {json.dumps(device_columns, ensure_ascii=False)}
        - **관련 제품 기능 문서 (요약):** {json.dumps(product_docs, ensure_ascii=False)}
        - **관련 센서 데이터 (샘플):** {json.dumps(sensor_columns, ensure_ascii=False)}
        """
    else:
        product_context_prompt += "\n💡 **팁:** 서비스와 연관될 LG 제품군(예: '스타일러', '디오스')을 지정하면, 더 구체적인 기획안을 받을 수 있습니다."

    prompt = f"""
    당신은 LG전자에서 신규 서비스의 데이터 전략을 수립하는 최고의 데이터 전략가(Data Strategist)입니다.
    주어진 서비스 아이디어와 관련 데이터를 바탕으로, 서비스를 성공시키기 위한 구체적이고 실행 가능한 데이터 기획안을 작성해주세요.

    ### [기획 대상 서비스 아이디어]
    {service_context_text}

    ### [제품/센서 데이터 컨텍스트]
    {product_context_prompt}

    ### [지시사항]
    아래 네 가지 관점에 따라, 상세한 데이터 기획안을 제시해주세요.

    1.  **기존 제품 데이터 활용 방안:** '기존 제품 상세 데이터 필드'를 조합/가공하여 서비스의 핵심 기능을 강화할 아이디어 2~3개를 제시해주세요.
    2.  **기존 센서 데이터 기반 신규 데이터 생성:** '관련 센서 데이터 (샘플)'을 참고하여, 기존 센서 데이터를 조합/분석하여 새로운 의미있는 데이터를 도출할 아이디어 2~3개를 제시해주세요.
    3.  **신규 센서 및 데이터 추천:** 이 서비스에 없는 새로운 센서를 1~2개 추천하고, 수집 데이터와 그 가치를 명확히 설명해주세요.
    4.  **외부 데이터 연동 및 활용:** 연동하면 좋을 외부 데이터를 1~2개 추천하고, 내부 데이터와 결합하여 새로운 가치를 제공할 방안을 설명해주세요.

    **[출력 형식]**
    결과는 반드시 아래의 JSON 형식으로만 반환해주세요.
    ```json
    {{
      "data_plan": {{
        "service_name": "{selected_idea_name}",
        "product_data_utilization": [
          {{"idea": "활용 아이디어 1", "details": "구체적인 활용 방안 설명", "required_data": ["필요한 기존 데이터 필드 1"]}}
        ],
        "new_data_from_sensors": [
          {{"idea": "신규 데이터/인사이트 아이디어 1", "details": "기존 센서 데이터 조합 및 분석 방법 설명", "required_sensors": ["사용될 기존 센서 1"]}}
        ],
        "new_sensor_recommendation": [
          {{"sensor_name": "추천 신규 센서 이름", "collectable_data": "수집 가능 데이터 설명", "value_proposition": "서비스 가치 증대 방안 설명"}}
        ],
        "external_data_integration": [
          {{"external_data_name": "추천 외부 데이터 이름", "integration_plan": "내/외부 데이터 결합 활용 방안 설명", "value_proposition": "결합을 통해 제공할 새로운 고객 가치 설명"}}
        ]
      }},
      "recommendation_message": "제품군을 지정하면 더 구체적인 결과를 얻을 수 있습니다."
    }}
    ```
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data_plan_result = json.loads(res.choices[0].message.content)

        new_plan = data_plan_result.get("data_plan")
        if new_plan:
    # 기존에 같은 이름의 기획안이 있으면 제거 (덮어쓰기 효과)
          workspace["artifacts"]["data_plan_for_service"] = [
          plan for plan in workspace["artifacts"]["data_plan_for_service"]
          if plan.get("service_name") != new_plan.get("service_name")
          ]
          workspace["artifacts"]["data_plan_for_service"].append(new_plan)
          print(workspace["artifacts"]["data_plan_for_service"])

        workspace["artifacts"]["selected_data_plan_for_service"] = data_plan_result
        workspace["artifacts"]["data_plan_recommendation_message"] = data_plan_result.get("recommendation_message", None)
        return {"data_plan_result": data_plan_result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"데이터 기획안 생성 중 오류 발생: {e}"}
    


def modify_data_plan(workspace: dict, modification_request: str):
    """
    기존에 생성된 데이터 기획안을 사용자의 요청에 따라 수정합니다.
    """
    print(f"✅ [Data Planner] Running Data Plan Modification...")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    # 1. 수정할 기존 데이터 기획안 및 컨텍스트 정보 가져오기
    existing_plan = artifacts.get("selected_data_plan_for_service")
    if not existing_plan:
        return {"error": "수정할 데이터 기획안이 없습니다. 먼저 데이터 기획안을 생성해주세요."}

    selected_idea = artifacts.get("selected_service_idea")
    if not selected_idea:
        return {"error": "데이터 기획의 기반이 되는 서비스 아이디어가 선택되지 않았습니다."}

    # 2. 프롬프트에 포함할 컨텍스트 준비
    service_context_text = json.dumps(selected_idea, ensure_ascii=False, indent=2)
    selected_idea_name = selected_idea.get('service_name', "알 수 없는 서비스")

    # 기존에 조회된 제품/센서 데이터 활용
    product_type = artifacts.get("product_type")
    product_context_prompt = "연관된 특정 제품군 정보가 없습니다."
    if product_type:
        device_columns = artifacts.get("columns_product", {})
        product_docs = artifacts.get("product_data", [])
        sensor_columns = artifacts.get("sensor_data", [])

        product_context_prompt = f"""
        ### [제품/센서 데이터 컨텍스트 (제품군: {product_type})]
        - **기존 제품 상세 데이터 필드:** {json.dumps(device_columns, ensure_ascii=False)}
        - **관련 제품 기능 문서 (요약):** {json.dumps(product_docs, ensure_ascii=False)}
        - **관련 센서 데이터 (샘플):** {json.dumps(sensor_columns, ensure_ascii=False)}
        """

    # 3. 수정을 위한 LLM 프롬프트 구성
    prompt = f"""
    당신은 LG전자에서 신규 서비스의 데이터 전략을 수립하는 최고의 데이터 전략가(Data Strategist)입니다.
    아래 주어진 '기존 데이터 기획안'을 '사용자 수정 요청'에 맞게 수정해주세요.
    수정 시에는 '기획 대상 서비스 아이디어'와 '제품/센서 데이터 컨텍스트'를 반드시 참고하여 더욱 구체적이고 완성도 높은 결과물을 만들어야 합니다.

    ### [기획 대상 서비스 아이디어]
    {service_context_text}

    ### [제품/센서 데이터 컨텍스트]
    {product_context_prompt}

    ---
    ### [기존 데이터 기획안]
    {json.dumps(existing_plan, ensure_ascii=False, indent=2)}

    ### [사용자 수정 요청]
    "{modification_request}"
    ---

    ### [수정 지시사항]
    '기존 데이터 기획안'의 내용을 기반으로, '사용자 수정 요청'을 완벽하게 반영하여 데이터 기획안를 다시 작성해주세요.
    수정 요청되지 않은 부분은 반드시 보존하여 주세요.
    결과는 반드시 아래의 JSON 형식으로만 반환해야 합니다.

    **[출력 형식]**
    ```json
    {{
      "data_plan": {{
        "service_name": "{selected_idea_name}",
        "product_data_utilization": [
          {{"idea": "수정된 활용 아이디어 1", "details": "...", "required_data": [...]}}
        ],
        "new_data_from_sensors": [
          {{"idea": "수정된 신규 데이터 아이디어 1", "details": "...", "required_sensors": [...]}}
        ],
        "new_sensor_recommendation": [
          {{"sensor_name": "수정된 추천 센서", "collectable_data": "...", "value_proposition": "..."}}
        ],
        "external_data_integration": [
          {{"external_data_name": "수정된 외부 데이터", "integration_plan": "...", "value_proposition": "..."}}
        ]
      }},
      "recommendation_message": "데이터 기획안 수정이 완료되었습니다."
    }}
    ```
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data_plan_result = json.loads(res.choices[0].message.content)
        modified_plan = data_plan_result.get("data_plan")
        # 수정된 결과를 워크스페이스에 덮어쓰기
        if modified_plan:
            # 📌 [수정] 저장 로직 수정 (덮어쓰기)
            workspace["artifacts"]["data_plan_for_service"] = [
                plan for plan in workspace["artifacts"].get("data_plan_for_service", []) 
                if plan.get("service_name") != modified_plan.get("service_name")
            ]
            workspace["artifacts"]["data_plan_for_service"].append(modified_plan)
            workspace["artifacts"]["selected_data_plan_for_service"] = modified_plan

        workspace["artifacts"]["data_plan_recommendation_message"] = data_plan_result.get("recommendation_message", None)
        return {"data_plan_result": data_plan_result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"데이터 기획안 수정 중 오류 발생: {e}"}