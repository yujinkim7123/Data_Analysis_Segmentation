
# agents/creator.py

import json
from .utils import get_openai_client

def create_cdp_definition(workspace: dict):
    """
    워크스페이스의 페르소나, 서비스 아이디어, 데이터 기획안을 종합하여
    최종 C-D-P(Customer-Data-Performance) 정의서를 생성합니다.
    """
    print("✅ [Creator Agent] Running C-D-P Definition Generation...")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    # 1. 필수 데이터 추출 및 유효성 검사
    # 이 단계에서는 가장 최근(또는 첫 번째)의 데이터를 기반으로 문서를 생성한다고 가정합니다.
    persona = artifacts.get("selected_persona")
    service_idea = artifacts.get("selected_service_idea")
    data_plan = artifacts.get("data_plan_for_service")

    if not all([persona, service_idea, data_plan]):
        error_message = """
        C-D-P 정의서 생성을 위한 정보가 부족합니다.
        프로세스를 다시 진행하여 페르소나, 서비스 아이디어, 데이터 기획안을 먼저 생성해야 합니다.
        **만약 페르소나를 직접 만들고 싶으시다면, "내가 만들고 싶은 페르소나에 대해 설명" 후, 서비스 도출 및 데이터 기획안 작성을 요청해주세요.**
        """
        return {"error": error_message}

    # 2. LLM을 사용하여 추상적인 부분(고객감동목표, 트래커 지표) 생성
    prompt = f"""
    당신은 신규 서비스의 핵심 가치와 성과 지표를 정의하는 최고의 비즈니스 전략가입니다.
    아래에 제공된 페르소나, 서비스 아이디어, 데이터 기획안을 종합적으로 분석하여, 다음 두 가지 항목을 생성해주세요.

    ### 1. 페르소나 정보
    {json.dumps(persona, ensure_ascii=False, indent=2)}

    ### 2. 서비스 아이디어 정보
    {json.dumps(service_idea, ensure_ascii=False, indent=2)}

    ### 3. 데이터 기획안 정보
    {json.dumps(data_plan, ensure_ascii=False, indent=2)}

    ---
    ### 지시사항
    위 모든 정보를 바탕으로, 아래 JSON 형식에 맞춰 결과만 반환해주세요.

    1.  `customer_delight_goal`: 서비스가 고객에게 제공하는 궁극적인 가치를 한 문장의 슬로건으로 표현해주세요. (예: "내가 직접, 눈으로 확인할 수 있는 '완벽 안심 살균솔루션'")
    2.  `dx_tracker_items`: 서비스의 성공을 '고객 관점'에서 측정할 수 있는 핵심 경험 지표(UXP) 3-4개를 리스트 형태로 제안해주세요. 이 지표는 데이터 기획안의 내용과 긴밀하게 연결되어야 합니다.

    ```json
    {{
      "customer_delight_goal": "생성된 고객 감동 목표 슬로건",
      "dx_tracker_items": [
        "제안된 첫 번째 추적 지표",
        "제안된 두 번째 추적 지표",
        "제안된 세 번째 추적 지표"
      ]
    }}
    ```
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        llm_results = json.loads(res.choices[0].message.content)
    except Exception as e:
        return {"error": f"C-D-P 정의서의 일부 항목 생성 중 LLM 오류 발생: {e}"}

    # 3. 이미지 형식에 맞춰 최종 C-D-P 정의서 JSON 객체 조립
    cdp_definition = {
        "title": f"유첨. {service_idea.get('service_name', '')} C-D-P 정의서",
        "customer_delight_goal": llm_results.get("customer_delight_goal", "정의된 고객 감동 목표 없음"),
        "cx": {
            "target_definition": {
                "description": f"{persona.get('title', '')} ({persona.get('demographics', '')})",
                "quote": persona.get('motivating_quote', ''),
                "market_info": "대한민국 전체 가구의 핵심 니즈를 공략하는 주요 타겟 고객층"
            },
            "core_experience": {
                "title": "우리가 만드는 고객가치는?",
                "care": service_idea.get('description', ''),
                "customization": service_idea.get('solved_pain_points', []),
                "servitization": service_idea.get('service_scalability', '')
            }
        },
        "performance": {
            "concept": {
                "find": "살균된 가습을 안심하고 이용할 수 있는 경험",
                "unique": [
                    item.get("idea", "") + ": " + item.get("details", "") for item in data_plan.get("new_data_from_sensors", [])
                ] + [
                    item.get("sensor_name", "") + ": " + item.get("collectable_data", "") for item in data_plan.get("new_sensor_recommendation", [])
                ]
            },
            "competitiveness": {
                "lump_sum_sales": "일시불판매(대) OOO",
                "subscription_sales": "구독판매(대) OOO",
                "revenue": "매출액(억원) OOO"
            },
             "customer_value_graph": "고객가치 그래프 (시간에 따른 가치 변화, 예: '23.12, '24.1, '24.2...)"
        },
        "dx": {
            "trigger": {
                "title": "CX 기획 Data 기반 발굴",
                "items": [
                    "기존 제품 사용자 대상 VOC 및 온라인 커뮤니티 데이터 분석",
                    "CX 분석 파이프라인(Ward Clustering, SNA, LDA) 기반 핵심 고객 행동 및 Pain Point 도출",
                    "기회 점수(Opportunity Score) 분석을 통한 타겟 문제 우선순위 선정"
                ]
            },
            "accelerator": {
                "title": "CX 구현 솔루션 제공",
                "up_contents_service": [
                    item.get("idea", "") for item in data_plan.get("product_data_utilization", [])
                ],
                "data_driven_experience": [
                     item.get("idea", "") for item in data_plan.get("new_data_from_sensors", [])
                ] + [
                    item.get("integration_plan", "") for item in data_plan.get("external_data_integration", [])
                ]
            },
            "tracker": {
                "title": "CX검증 Data 기반 고객경험 모니터링",
                "items": llm_results.get("dx_tracker_items", [])
            }
        }
    }

    workspace["artifacts"]["cdp_definition"] = cdp_definition
    return {"cdp_definition": cdp_definition}