# agents/data_planner.py

import json
from .utils import get_openai_client, get_columns_for_product
from .data_retriever import fetch_product_context, fetch_sensor_context

def create_data_plan_for_service(workspace: dict, service_name: str = None, service_description: str = None, product_type: str = None):
    """
    서비스 아이디어를 기반으로 데이터 기획안을 생성합니다.
    """
    print(f"✅ [Data Planner] Running Data Plan Generation...")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    service_context_text = ""
    keyword_for_search = ""
    selected_idea_name = "사용자 정의 아이디어"

    if service_name:
        all_ideas = artifacts.get("service_ideas", {}).get("service_ideas", [])
        selected_idea = next((idea for idea in all_ideas if idea.get("service_name") == service_name), None)
        
        workspace["artifacts"]["selected_service_idea"] = selected_idea
        print(f"📌 Service idea '{service_name}' has been set as the selected service idea.")
          
        if not selected_idea:
            return {"error": f"'{service_name}' 이름의 서비스 아이디어를 찾을 수 없습니다."}
        
        service_context_text = json.dumps(selected_idea, ensure_ascii=False, indent=2)
        keyword_for_search = service_name + " " + selected_idea.get('description', '')
        selected_idea_name = selected_idea.get('service_name')
    elif service_description:
        service_context_text = f"사용자가 직접 설명한 서비스 아이디어: {service_description}"
        keyword_for_search = service_description
    else:
        return {"error": "데이터 기획안을 생성하려면 'service_name' 또는 'service_description' 중 하나는 반드시 제공되어야 합니다."}

    final_product_type = product_type or artifacts.get("product_type")
    product_context_prompt = "연관된 특정 제품군 정보가 없습니다."

    if final_product_type:
        print(f"🔍 '{final_product_type}' 제품군에 대한 정보 활용 중...")
        product_columns = get_columns_for_product(final_product_type)
        workspace["artifacts"]["product_data"] = product_columns
        feature_docs = fetch_product_context(keyword_for_search, final_product_type, top_k=5)
        workspace["artifacts"]["product_type"] = final_product_type
        # 👇 [수정] 키워드 대신 'final_product_type'을 전달합니다.
        sensor_docs = fetch_sensor_context(product_type=final_product_type, top_k=10) 
        workspace["artifacts"]["sensor_data"] = sensor_docs

        product_context_prompt = f"""
        ### [제품/센서 데이터 컨텍스트 (제품군: {final_product_type})]
        - **기존 제품 상세 데이터 필드:** {json.dumps(product_columns, ensure_ascii=False)}
        - **관련 제품 기능 문서 (요약):** {json.dumps(feature_docs, ensure_ascii=False)}
        - **관련 센서 데이터 (샘플):** {json.dumps(sensor_docs, ensure_ascii=False)}
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
        workspace["artifacts"]["data_plan_for_service"] = data_plan_result.get("data_plan")
        return {"data_plan_result": data_plan_result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"데이터 기획안 생성 중 오류 발생: {e}"}