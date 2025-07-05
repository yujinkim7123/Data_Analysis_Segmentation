# agents/persona_generator.py

import json
from .utils import get_openai_client


def _call_persona_llm(prompt: str):
    client = get_openai_client()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content)

def create_personas(workspace: dict, num_personas: int = 3, focus_topic_ids: list[str] = None):
    """
    워크스페이스의 데이터를 기반으로 데이터 기반 페르소나를 생성합니다.
    이 함수는 독립적인 페르소나 생성 에이전트의 역할을 합니다.
    """

    print(f"✅ [Persona Agent] Running Persona Generation for {num_personas} personas...")
    artifacts = workspace.get("artifacts", {})

    # 1. 필수 데이터 (검색된 데이터) 확인
    retrieved_data = artifacts.get("retrieved_data")
    if not retrieved_data:
        return {"error": "페르소나를 생성하려면 먼저 '데이터 검색'을 통해 고객의 목소리를 수집해야 합니다."}

    # 2. 데이터 추출 및 샘플링
    web_results = retrieved_data.get("web_results", [])
    raw_texts_sample = "\n- ".join([d.get('original_text', '') for d in web_results if d.get('original_text')][:30])

    # 3. 선택적 데이터 (제품군, 분석 결과) 처리
    product_type = artifacts.get("product_type")
    lda_results = artifacts.get("cx_lda_results", {}).get("topics_summary_list", [])
    cam_results = artifacts.get("cx_cam_results", [])

    product_info_prompt = f"분석 대상 제품군은 '{product_type}' 입니다." if product_type else \
        "**[안내]** 제품군 정보가 없습니다. 신제품이 아닌 경우, 특정 제품군을 지정하면 해당 제품 사용 맥락을 더 잘 반영한 페르소나를 만들 수 있습니다."

    analysis_results_prompt = ""
    if lda_results:
        analysis_summary = []
        
        topics_to_focus = lda_results
        if focus_topic_ids:
            topics_to_focus = [t for t in lda_results if t.get("topic_id") in focus_topic_ids]
            analysis_results_prompt += f"**[핵심 분석 정보]** 아래 {len(topics_to_focus)}개의 고객 행동에 집중하여 페르소나를 구체화해주세요:\n"
        else:
            analysis_results_prompt += "**[전체 분석 정보]** 아래 고객 행동들을 종합적으로 고려하여 페르소나를 생성해주세요:\n"

        cam_map = {cam.get("topic_id"): cam for cam in cam_results}

        for topic in topics_to_focus:
            topic_id = topic.get("topic_id")
            keywords = ', '.join(topic.get('action_keywords', []))
            summary = f"- **행동 ID {topic_id}**: '{keywords}'"
            
            cam_data = cam_map.get(topic_id) 
            if cam_data:
                goals = ', '.join(cam_data.get('goals', []))
                pains = ', '.join(cam_data.get('pain_points', []))
                summary += f"\n  - 주요 목표: {goals}\n  - 주요 불편: {pains}"
            analysis_summary.append(summary)
        
        analysis_results_prompt += "\n".join(analysis_summary)
    else:
        analysis_results_prompt = "**[안내]** CX 분석 결과가 없습니다. 고객 행동(Action), 목표(Goal), 불편(Pain Point)을 먼저 분석하면, 페르소나의 행동 패턴과 니즈를 훨씬 더 깊이 있게 정의할 수 있습니다."

    # 4. 최종 프롬프트 구성
    # ✅ [수정] 아래 prompt 할당 블록의 들여쓰기를 수정하여 독립적으로 실행되도록 합니다.
    prompt = f"""
    당신은 소비자 데이터 분석 결과를 해석하여 생생하고 데이터 기반의 고객 페르소나를 도출하는 전문 UX 리서처입니다.
    주어진 정보를 바탕으로 단계별로 생각하여(Think step-by-step) 요청받은 과업을 수행하세요.
    아래 데이터를 바탕으로, 페르소나를 생성하거나 수정해주세요.

    ---
    ### 1. (필수) 고객 발화 원문 (샘플)
    - {raw_texts_sample}

    ### 2. (선택) 제품군 정보
    {product_info_prompt}

    ### 3. (선택) CX 분석 결과 요약
    {analysis_results_prompt}
    --

    ### 지시사항
    - 위 모든 정보를 종합적으로 해석하여, 각 페르소나의 인구 통계 정보, 핵심 행동, 니즈와 목표, 페인 포인트를 구체적으로 추론해주세요.
     - **서로 다른 핵심적인 특징과 동기를 가진 {num_personas}명의 페르소나**를 생성해주세요.
    - 결과는 반드시 아래의 JSON 형식으로만 반환해주세요. 다른 설명은 절대 추가하지 마세요.

    ```json
    {{
      "personas": [
        {{
          "name": "박서준 (가명)",
          "title": "꼼꼼한 위생관리맘",
          "demographics": "30대 후반, 맞벌이, 7세 아이 엄마",
          "key_behaviors": [ "아이 옷은 반드시 살균 기능으로 관리", "가전제품 구매 전 온라인 후기를 30개 이상 비교 분석" ],
          "needs_and_goals": [ "가족의 건강을 유해세균으로부터 지키고 싶다", "반복적인 가사 노동 시간을 줄이고 싶다" ],
          "pain_points": [ "매번 옷을 삶는 것은 번거롭고 옷감이 상할까 걱정된다", "살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다" ],
          "motivating_quote": "아이가 쓰는 건데, 조금 비싸더라도 확실한 걸로 사야 마음이 놓여요."
        }}
      ]
    }}
    ```
    """

    try:
        res = _call_persona_llm(prompt)
        # ✅ [수정] _call_persona_llm이 이미 json.loads를 수행했으므로, 결과를 바로 사용합니다.
        persona_results = res
        workspace["artifacts"]["personas"].extend(persona_results.get("personas", []))
        return {"personas_result": persona_results}
    except Exception as e:
        print(f"❌ 페르소나 생성 중 오류 발생: {e}")
        return {"error": f"페르소나 생성 중 오류가 발생했습니다: {e}"}
    

def create_persona_from_manual_input(workspace: dict, persona_data: dict) -> dict:
    """
    사용자가 수동으로 입력한 페르소나 데이터를 워크스페이스에 저장합니다.
    """
    print(f"📝 [Persona Agent] Saving manual persona: {persona_data.get('name', 'Unknown')}")
    artifacts = workspace.get("artifacts", {})

    # 필수 필드 검증
    required_fields = ["name", "title", "demographics", "key_behaviors", "needs_and_goals", "pain_points", "motivating_quote"]
    missing_fields = [field for field in required_fields if field not in persona_data or not persona_data[field]]
    if missing_fields:
        return {"error": f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}"}

    # 워크스페이스에 페르소나 저장
    if "personas" not in artifacts:
        artifacts["personas"] = []

    # 중복 이름 확인
    existing_names = {p["name"] for p in artifacts["personas"]}
    if persona_data["name"] in existing_names:
        return {"error": f"이미 존재하는 페르소나 이름: {persona_data['name']}"}

    # 페르소나 데이터 저장
    artifacts["personas"].append(persona_data)
    workspace["artifacts"]["selected_persona"] = persona_data
    workspace["artifacts"] = artifacts

    return {
        "personas_result": {"personas": [persona_data]},
        "message": f"페르소나 '{persona_data['name']}'이 성공적으로 저장되었습니다."
    }

# [신규] 2. 페르소나 수정 함수
def modify_personas(workspace: dict, modification_request: str):
    """(수정) 기존에 생성된 페르소나를 사용자의 요청에 따라 수정합니다."""
    print(f"✅ [Persona Agent] Running Persona Modification: '{modification_request}'")
    artifacts = workspace.get("artifacts", {})
    
    existing_personas = artifacts.get("personas")
    if not existing_personas:
        return {"error": "수정할 페르소나가 없습니다. 먼저 페르소나를 생성해주세요."}
    
    existing_personas_str = json.dumps(existing_personas, ensure_ascii=False, indent=2)

    prompt = f"""
    당신은 ... 전문 UX 리서처입니다. 단계별로 생각하여(Think step-by-step) 기존 페르소나를 사용자의 요청에 맞게 수정해주세요.

    ### 기존 페르소나 결과
    {existing_personas_str}

    ### 사용자 수정 요청사항
    "{modification_request}"

    ### 수정 지시사항
    '기존 페르소나 결과'를 바탕으로 '사용자 수정 요청'을 완벽하게 반영하여 페르소나 전체를 다시 생성해주세요.
    변경이 요청되지 않은 부분은 반드시 온전히 유지시켜야합니다.
    결과 형식은 반드시 기존과 동일한 JSON 구조를 따라야 합니다.

    ```json
      {{
      "personas": [
        {{
          "name": "박서준 (가명)",
          "title": "꼼꼼한 위생관리맘",
          "demographics": "30대 후반, 맞벌이, 7세 아이 엄마",
          "key_behaviors": [ "아이 옷은 반드시 살균 기능으로 관리", "가전제품 구매 전 온라인 후기를 30개 이상 비교 분석" ],
          "needs_and_goals": [ "가족의 건강을 유해세균으로부터 지키고 싶다", "반복적인 가사 노동 시간을 줄이고 싶다" ],
          "pain_points": [ "매번 옷을 삶는 것은 번거롭고 옷감이 상할까 걱정된다", "살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다" ],
          "motivating_quote": "아이가 쓰는 건데, 조금 비싸더라도 확실한 걸로 사야 마음이 놓여요."
        }}
      ]
    }}
    ```
    """
    try:
        res = _call_persona_llm(prompt)
        # ✅ [수정] 여기도 마찬가지로 결과를 바로 사용합니다.
        persona_results = res
        new_personas = persona_results.get("personas", [])

        # 1. 기존 페르소나 목록을 가져와 'name'을 key로 사용하는 dict(personas_map)으로 변환합니다.
        #    이렇게 하면 이름으로 중복 여부를 빠르게 확인할 수 있습니다.
        existing_personas = workspace["artifacts"].get("personas", [])
        personas_map = {p["name"]: p for p in existing_personas}
        
        for persona in new_personas:
            # 3. 같은 이름의 페르소나가 이미 맵에 있으면 교체하고, 없으면 새로 추가합니다.
            persona_name = persona.get("name")
            if persona_name:
                personas_map[persona_name] = persona

        # 4. 업데이트된 dict를 다시 리스트 형태로 변환하여 workspace에 저장합니다.
        workspace["artifacts"]["personas"] = list(personas_map.values())
        return {"personas_result": persona_results}
    except Exception as e:
        return {"error": f"페르소나 수정 중 오류: {e}"}
