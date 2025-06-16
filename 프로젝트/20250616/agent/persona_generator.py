# agents/persona_generator.py

import json
from .utils import get_openai_client

def create_personas(workspace: dict, num_personas: int = 3, focus_topic_ids: list[str] = None):
    """
    워크스페이스의 데이터를 기반으로 데이터 기반 페르소나를 생성합니다.
    이 함수는 독립적인 페르소나 생성 에이전트의 역할을 합니다.
    """
    print(f"✅ [Persona Agent] Running Persona Generation for {num_personas} personas...")
    client = get_openai_client()
    artifacts = workspace.get("artifacts", {})

    # 1. 필수 데이터 (검색된 데이터) 확인
    retrieved_data = artifacts.get("retrieved_data")
    if not retrieved_data:
        return {"error": "페르소나를 생성하려면 먼저 '데이터 검색'을 통해 고객의 목소리를 수집해야 합니다."}

    # 2. 데이터 추출 및 샘플링
    web_results = retrieved_data.get("web_results", [])
    raw_texts_sample = "\n- ".join([d.get('original_text', '') for d in web_results if d.get('original_text')][:30])

    # 3. 선택적 데이터 (제품군, 분석 결과) 처리
    # product_type은 retrieved_data에 함께 저장되었다고 가정
    product_type = artifacts.get("product_type")
    lda_results = artifacts.get("cx_lda_results", [])
    cam_results = artifacts.get("cx_cam_results", [])

    # 프롬프트에 삽입될 안내 문구 및 데이터 생성
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
    prompt = f"""
    당신은 소비자 데이터 분석 결과를 해석하여 생생하고 데이터 기반의 고객 페르소나를 도출하는 전문 UX 리서처입니다.
    아래 데이터를 바탕으로, {num_personas}개의 서로 다른 특징을 가진 고객 페르소나를 생성해주세요.

    ---
    ### 1. (필수) 고객 발화 원문 (샘플)
    - {raw_texts_sample}

    ### 2. (선택) 제품군 정보
    {product_info_prompt}

    ### 3. (선택) CX 분석 결과 요약
    {analysis_results_prompt}
    ---

    ### 지시사항
    - 위 모든 정보를 종합적으로 해석하여, 각 페르소나의 인구 통계 정보, 핵심 행동, 니즈와 목표, 페인 포인트를 구체적으로 추론해주세요.
    - 각 페르소나는 서로 다른 핵심적인 특징과 동기를 가져야 합니다.
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
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        persona_results = json.loads(res.choices[0].message.content)
        
        workspace["artifacts"]["personas"] = persona_results.get("personas", [])
        
        return {"personas_result": persona_results}
    except Exception as e:
        print(f"❌ 페르소나 생성 중 오류 발생: {e}")
        return {"error": f"페르소나 생성 중 오류가 발생했습니다: {e}"}