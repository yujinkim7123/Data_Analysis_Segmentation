# agents/cx_analyst.py

import json
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from .utils import get_sentiment_analyzer
from .utils import get_openai_client
from scipy.sparse import csr_matrix # 희소 행렬 변환 시 필요
from collections import defaultdict 

# --- 내부 헬퍼(보조) 함수들 ---
def _get_sentiment_score(text: str) -> float:
    """텍스트의 감성 점수를 계산합니다. (동기 버전)"""
    analyzer = get_sentiment_analyzer() # 동기 함수이므로 await 없음
    if analyzer is None:
        print("❌ 감성 분석 모델이 로드되지 않았습니다. 감성 점수 계산 불가.")
        return 0.0 # 모델 로드 실패 시 중립 점수 반환
    
    try:
        # Hugging Face pipeline 결과는 리스트 [{label: 'LABEL_0', score: 0.99}] 형태
        # bert-nsmc 모델은 LABEL_0 (부정)과 LABEL_1 (긍정)을 반환합니다.
        result = analyzer(text[:512])[0] # 모델 입력 길이 제한 (예: 512)
        
        # LABEL_1 (긍정)은 긍정 점수, LABEL_0 (부정)은 부정 점수로 매핑
        if result['label'] == 'LABEL_1': # 긍정
            return float(result['score'])
        elif result['label'] == 'LABEL_0': # 부정
            return -float(result['score'])
        else: # 중립 또는 알 수 없는 경우 (없을 가능성 높음)
            return 0.0
    except Exception as e:
        print(f"감성 점수 계산 중 오류 발생: {e}")
        return 0.0 # 오류 발생 시 중립 점수 반환

def _get_top_keywords(feature_names, topic_components, n_top_words):
    """
    LDA 토픽 모델의 컴포넌트(단어-토픽 분포)에서 각 토픽별 상위 N개 키워드를 추출합니다.
    Args:
        feature_names (list): TF-IDF 벡터라이저의 피처(단어) 이름 목록.
        topic_components (np.array): LDA 모델의 .components_ 속성 (토픽-단어 분포 행렬).
        n_top_words (int): 각 토픽에서 추출할 상위 키워드 개수.
    Returns:
        list of lists: 각 토픽별 상위 키워드 리스트.
    """
    top_keywords = []
    for topic_idx, topic in enumerate(topic_components):
        # topic은 해당 토픽의 모든 단어에 대한 가중치를 포함하는 NumPy 배열입니다.
        # argsort()[-n_top_words-1:-1:-1]는 내림차순으로 상위 n_top_words 개의 인덱스를 효율적으로 찾습니다.
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        
        # 해당 인덱스의 단어들을 feature_names에서 가져와 리스트로 만듭니다.
        keywords_for_topic = [feature_names[i] for i in top_words_indices]
        top_keywords.append(keywords_for_topic)
    return top_keywords

# --- Master Agent가 호출할 Tool 함수들 ---


def run_ward_clustering(workspace, num_clusters=5):
    """
    고객의 목소리(VOC) 데이터를 워드 클러스터링하여 주요 주제 그룹을 발견하고,
    각 클러스터의 대표 키워드를 추출합니다.
    """
    print(f"[CX Analysis] Running Ward Clustering with {num_clusters} clusters (동기 모드)")

    retrieved_data = workspace["artifacts"].get("retrieved_data")
    if not retrieved_data:
        return {"error": "데이터 클러스터링을 위한 검색된 데이터가 워크스페이스에 없습니다. 먼저 데이터 검색을 해주세요."}

    documents = [d.get('original_text', '') for d in retrieved_data.get('web_results', []) if d.get('original_text')]
    
    if not documents:
        return {"error": "클러스터링할 유효한 텍스트 문서가 없습니다. 검색 결과를 확인해주세요."}

    try:
        # 2. 텍스트 벡터화 (TF-IDF)
        # max_features는 단어의 수를 제한하고, min_df/max_df는 너무 적거나 너무 많은 문서에 나타나는 단어를 제외합니다.
        # stop_words='english'는 영어 불용어를 제거하지만, 한국어 데이터에는 적합하지 않으므로 None을 유지합니다.
        # 한국어 불용어 처리가 필요하다면, 직접 불용어 리스트를 제공해야 합니다.
        vectorizer = TfidfVectorizer(max_features=2000, min_df=0.01, max_df=0.9) # min_df, max_df 범위 조정
        X = vectorizer.fit_transform(documents)

        if X.shape[1] == 0: # 문서-단어 행렬에 유효한 피처(단어)가 없는 경우
            return {"error": "TF-IDF 벡터화 후 유효한 단어가 추출되지 않았습니다. 데이터를 확인하거나 TfidfVectorizer 설정을 조정하세요."}

        # 3. K-Means 클러스터링 수행
        # num_clusters가 데이터 포인트 수보다 크면 오류가 날 수 있으므로, 제한합니다.
        if num_clusters > X.shape[0]:
            num_clusters = X.shape[0] # 문서 수보다 클러스터 수가 많을 수 없음
            print(f"⚠️ 클러스터 개수가 문서 수보다 많아 {num_clusters}개로 조정되었습니다.")
        
        # 클러스터 개수가 1이면 클러스터링의 의미가 없거나 K-Means 오류 가능
        if num_clusters < 2:
            return {"error": "클러스터 개수는 최소 2개 이상이어야 합니다."}

        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_.tolist()

        # 4. 각 클러스터의 대표 키워드 추출
        feature_names = np.array(vectorizer.get_feature_names_out())
        cluster_centers = kmeans.cluster_centers_

        cluster_summaries = {}
        for i in range(num_clusters):
            # 클러스터에 속한 문서 인덱스 찾기
            cluster_docs_indices = np.where(kmeans.labels_ == i)[0]
            num_docs_in_cluster = len(cluster_docs_indices)

            # 해당 클러스터에 문서가 없으면 스킵
            if num_docs_in_cluster == 0:
                cluster_summaries[str(i)] = {"keywords": [], "description": f"{i}번 그룹 (0개 문서)에는 문서가 없습니다."}
                continue

            # 해당 클러스터의 TF-IDF 벡터 합계 계산 (가장 중요)
            # 클러스터 내 문서들의 단어 중요도를 합산하여 해당 클러스터의 전반적인 단어 분포를 파악합니다.
            cluster_tfidf_sum = X[cluster_docs_indices].sum(axis=0)

            # Sum된 벡터에서 가장 높은 TF-IDF 값을 가진 단어 인덱스 찾기
            # .flatten()으로 1D 배열로 만든 후 argsort 사용
            top_feature_indices = cluster_tfidf_sum.A.flatten().argsort()[-10:][::-1] # .A는 희소 행렬을 밀집 행렬로 변환

            top_keywords = feature_names[top_feature_indices].tolist()

            cluster_summaries[str(i)] = {
                "keywords": top_keywords,
                "description": f"{i}번 그룹 ({num_docs_in_cluster}개 문서)은 주로 '{', '.join(top_keywords[:5])}...' 등의 키워드를 포함합니다."
            }
        
        # 5. 워크스페이스에 임시 데이터 저장 (LDA, SNA를 위해)
        # X (TF-IDF 행렬)는 희소 행렬이므로, 직렬화 가능한 형태로 변환하여 저장
        # toarray().tolist()는 매우 큰 메모리를 사용할 수 있으므로 주의 (대용량 데이터의 경우 다른 방식 고려)
        tfidf_matrix_list = X.toarray().tolist()

        # feature_names는 리스트로 변환하여 저장
        feature_names_list = feature_names.tolist()

        cluster_docs_map = defaultdict(list)
        for doc_idx, label in enumerate(cluster_labels):
            cluster_docs_map[label].append(doc_idx)

        workspace["artifacts"]["_cx_temp_data"] = {
            "cluster_labels": cluster_labels,
            "tfidf_matrix": tfidf_matrix_list, # JSON 직렬화 가능한 형태로 저장
            "feature_names": feature_names_list, # JSON 직렬화 가능한 형태로 저장
             "documents": documents, # 👈 원본 문서 리스트 추가
             "cluster_docs_map": dict(cluster_docs_map) # 👈 defaultdict를 일반 dict로 변환하여 추가
        }

        # 6. 분석 결과 반환
        return {
            "cx_ward_clustering_results": {
                "num_clusters": num_clusters,
                "cluster_labels": cluster_labels,
                "cluster_summaries": cluster_summaries,
            },
            "analysis_results": "Ward clustering analysis complete. 각 클러스터의 대표 키워드를 확인해보세요. 특정 클러스터에 대해 더 깊은 분석(의미 연결망 분석)을 원하시면 클러스터 ID와 함께 요청해주세요."
        }

    except Exception as e:
        print(f"❌ 워드 클러스터링 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"워드 클러스터링 분석 중 오류가 발생했습니다: {e}"}


def run_semantic_network_analysis(workspace: dict, cluster_id: int):
    """PDF 2단계: 특정 클러스터에 대해 SNA를 수행하여 핵심 노드를 찾습니다."""
    print(f"✅ [CX Agent] Step 2: Running SNA for Cluster ID: {cluster_id} (동기 모드)")
    temp_data = workspace.get("artifacts", {}).get("_cx_temp_data", {})
    
    if not temp_data.get("cluster_labels"): return {"error": "군집화를 먼저 수행해야 합니다."}
    if not temp_data.get("tfidf_matrix"): return {"error": "TF-IDF 행렬이 워크스페이스에 없습니다."}
    if not temp_data.get("feature_names"): return {"error": "피처 이름이 워크스페이스에 없습니다."}

    try: # 전체 로직을 try-except로 감싸서 안정성 확보
        # 1. 특정 클러스터에 해당하는 문서들의 TF-IDF 행렬 추출
        docs_indices = [i for i, label in enumerate(temp_data["cluster_labels"]) if label == cluster_id]
        if not docs_indices: return {"error": f"ID가 {cluster_id}인 클러스터에 문서가 없습니다."}
        
        # temp_data["tfidf_matrix"]는 이제 일반 Python 리스트의 리스트이므로,
        # 다시 희소 행렬로 변환해야 합니다.
        cluster_matrix = csr_matrix(np.array(temp_data["tfidf_matrix"])[docs_indices])

        # 2. 단어 동시 출현 행렬 계산
        co_occurrence_matrix = (cluster_matrix.T * cluster_matrix)
        co_occurrence_matrix.setdiag(0) # 자기 자신과의 동시 출현은 0으로 설정

        # 3. 네트워크X 그래프 생성 및 중심성 계산
        G = nx.from_scipy_sparse_array(co_occurrence_matrix)
        # ⚠️ 디버깅 목적이 아니라면, G 객체나 co_occurrence_matrix를 직접 print()하지 마세요.
        # print(f"DEBUG: Graph G: {G}") # 이 줄처럼 객체 자체를 출력하는 것은 피해야 합니다.

        centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        # 4. 핵심 노드 (핵심 키워드) 추출
        feature_names = temp_data["feature_names"]

        micro_segments = []
        for node_idx, score in sorted_nodes[:10]:
            if node_idx < len(feature_names):
                micro_segments.append({
                    "core_keyword": feature_names[node_idx],
                    "centrality_score": round(score, 4)
                })
            else:
                print(f"⚠️ 경고: SNA 노드 인덱스 {node_idx}가 feature_names 범위를 벗어납니다.")

        # 5. 결과 반환 (직렬화 가능한 데이터만 포함)
        return {
            "cx_sna_results": {
                "cluster_id": cluster_id,
                "micro_segments": micro_segments,
                "analysis_description": f"{cluster_id}번 클러스터 내에서 가장 핵심적인 키워드들을 찾아 의미 연결망 분석을 수행했습니다."
            },
            "analysis_results": "Semantic network analysis complete." # 이 부분은 Pydantic이 직렬화 가능해야 함
        }

    except Exception as e:
        print(f"❌ SNA 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"SNA 분석 중 오류가 발생했습니다: {e}"}


def run_topic_modeling_lda(workspace: dict, cluster_id: int, num_topics: int = 3):
    """
    PDF 3단계: 특정 클러스터에 대해 LDA를 수행하여 구체적인 '고객 액션'을 식별합니다.
    """
    print(f"✅ [CX Agent] Step 3: Running LDA for Cluster ID: {cluster_id} (동기 모드)")
    artifacts = workspace.get("artifacts", {}) # artifacts를 먼저 가져옵니다.
    temp_data = artifacts.get("_cx_temp_data", {}) # _cx_temp_data는 artifacts 안에 있습니다.
    
    # 1. 필수 데이터 존재 여부 검사
    if not temp_data.get("cluster_labels"):
        return {"error": "토픽 모델링을 위해서는 군집화를 먼저 수행해야 합니다."}
    if not temp_data.get("tfidf_matrix"):
        return {"error": "토픽 모델링을 위한 TF-IDF 행렬이 워크스페이스에 없습니다."}
    if not temp_data.get("feature_names"):
        return {"error": "토픽 모델링을 위한 피처 이름이 워크스페이스에 없습니다."}

    try:
        # 2. 특정 클러스터에 해당하는 문서들의 TF-IDF 행렬 추출
        docs_indices = [i for i, label in enumerate(temp_data["cluster_labels"]) if label == cluster_id]
        if not docs_indices:
            return {"error": f"ID가 {cluster_id}인 클러스터에 문서가 없습니다."}
        
        # _cx_temp_data에 저장된 tfidf_matrix는 list of lists (Python 객체)이므로
        # 다시 희소 행렬로 변환해야 합니다.
        doc_term_matrix = csr_matrix(np.array(temp_data["tfidf_matrix"])[docs_indices])
        
        # 3. 피처 이름(단어 목록) 가져오기
        feature_names = temp_data["feature_names"] # TfidfVectorizer 객체 대신 저장된 리스트 사용

        # 4. LDA 모델 학습
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix) # 문서-단어 행렬로 LDA 학습

        # 각 문서의 토픽 분포를 계산 (calculate_opportunity_scores에서 사용될 데이터)
        doc_topic_dist_for_cluster = lda.transform(doc_term_matrix) # 👈 클러스터에 해당하는 문서들의 토픽 분포

        # 5. 토픽별 상위 키워드 추출
        topics_list = []
        top_keywords_per_topic = _get_top_keywords(feature_names, lda.components_, 7) # 상위 7개 키워드

        for i, keywords in enumerate(top_keywords_per_topic):
            topics_list.append({
                "topic_id": f"{cluster_id}-{i}", # 클러스터 ID와 토픽 인덱스를 조합
                "action_keywords": keywords,
                "description": f"주요 키워드: {', '.join(keywords[:3])}..." # 간단한 설명 추가
            })

    
        workspace["artifacts"]["_cx_temp_data"]["doc_topic_distribution"] = doc_topic_dist_for_cluster.tolist()
        workspace["artifacts"]["_cx_temp_data"]["lda_components_for_cluster"] = lda.components_.tolist() # 필요시 저장

        # 🚨🚨🚨 cx_lda_results 리스트에 토픽 결과 누적 저장 (main.py에서 []로 초기화됨)
        # 이미 이 로직은 잘 되어 있습니다.
        current_lda_results = artifacts.get("cx_lda_results", [])
        current_lda_results.extend(topics_list)
        artifacts["cx_lda_results"] = current_lda_results # artifacts 딕셔너리에 업데이트

        # 7. LLM에게 반환할 데이터 (간결하게)
        return {
            "success": True,
            "message": f"클러스터 {cluster_id}에 대해 {num_topics}개의 토픽을 성공적으로 식별했습니다.",
            "newly_identified_topics_preview": [
                {"topic_id": t["topic_id"], "action_keywords": t["action_keywords"]}
                for t in topics_list
            ]
        }

    except Exception as e:
        print(f"❌ 토픽 모델링(LDA) 중 오류 발생: {e}")
        import traceback
        traceback.print_exc() # 상세 에러 스택 트레이스 출력
        return {"error": f"토픽 모델링(LDA) 분석 중 오류가 발생했습니다: {e}"}

def create_customer_action_map(workspace: dict, topic_id: str):
    """
    [완성본] PDF 4단계: '분석된 결과'를 바탕으로 CAM(Pain Point 등)을 생성합니다.
    """
    print(f"✅ [CX Agent] Step 4: Creating CAM for Topic ID: {topic_id}...")
    client = get_openai_client()
    
    # --- 1. workspace에서 이 토픽에 대한 모든 분석 '결과'를 가져옵니다. ---
    artifacts = workspace.get("artifacts", {})
    lda_results = artifacts.get("cx_lda_results", [])
    opportunity_scores = artifacts.get("cx_opportunity_scores", [])
    
    # 해당 topic_id에 대한 정보를 찾습니다.
    topic_lda_data = next((item for item in lda_results if item.get("topic_id") == topic_id), None)
    topic_score_data = next((item for item in opportunity_scores if item.get("topic_id") == topic_id), None)

    if not topic_lda_data or not topic_score_data:
        return {"error": f"ID가 {topic_id}인 토픽에 대한 분석 결과가 부족합니다. LDA와 기회 점수 계산을 먼저 수행해주세요."}
    
    action_keywords = topic_lda_data.get('action_keywords', [])
    first_keyword = action_keywords[0] if action_keywords else topic_id # 리스트가 비어있으면 topic_id 사용

    # --- 2. LLM에게 전달할 '분석 요약 정보'를 구성합니다. ---
    prompt = f"""
    당신은 데이터 분석 결과를 해석하여 고객 액션맵(CAM)을 완성하는 최고의 CX 전략가입니다.
    아래는 특정 고객 행동(Action)에 대한 정량적/정성적 분석 요약 결과입니다.

    [분석 데이터 요약]
    - 행동(Action) ID: {topic_id}
    - 행동의 핵심 키워드: "{', '.join(topic_lda_data.get('action_keywords', []))}"
    - 이 행동에 대한 고객 만족도 점수: {topic_score_data.get('satisfaction')} (-1.0: 매우 부정, 1.0: 매우 긍정)
    - 이 행동의 중요도(언급량): {topic_score_data.get('importance')}

    위 분석 결과를 바탕으로, 이 행동을 하는 고객들의 'Goal(궁극적 목표)'과 'Pain Point(핵심 불편함)'를 각각 2~3가지씩 깊이 있게 추론해주세요.
    PDF의 CAM 프레임워크를 참고하여, 이 행동이 주로 발생하는 'Context(상황)'와 관련된 'Touchpoint/Artifact(사물/서비스)'도 함께 추론하여 제시해주세요.

    결과는 반드시 아래의 JSON 형식으로만 반환해주세요.
    {{
      "action_name": "{first_keyword}",
      "goals": ["추론된 목표 1", "추론된 목표 2"],
      "pain_points": ["추론된 불편함 1", "추론된 불편함 2"],
      "context": ["추론된 상황 1", "추론된 상황 2"],
      "touchpoint_artifact": ["관련된 사물 1", "관련된 사물 2"]
    }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        cam_results = json.loads(res.choices[0].message.content)
        
        existing_cams = workspace.get("artifacts", {}).get("cx_cam_results", [])
        existing_cams.append(cam_results)
        
        return {"cx_cam_results": existing_cams}
    except Exception as e:
        return {"error": f"고객 액션맵 생성 중 오류: {e}"}


def calculate_opportunity_scores(workspace: dict):
    """
    [완성본] PDF 5단계: 모든 토픽의 기회 점수를 계산하여 우선순위를 정합니다.
    """
    print("✅ [CX Agent] Step 5: Calculating Opportunity Scores...")
    client = get_openai_client()
    
    # --- 1. workspace에서 이전 단계의 결과물들을 가져옵니다. ---
    artifacts = workspace.get("artifacts", {})
    temp_data = artifacts.get("_cx_temp_data", {})
    lda_results = artifacts.get("cx_lda_results", []) # 모든 클러스터의 LDA 토픽 리스트 (main.py에서 []로 초기화됨)
    
    # 🚨🚨🚨 이 부분들을 수정합니다: lda_model, vectorizer 대신 직렬화 가능한 데이터를 가져옴
    documents = temp_data.get("documents") # 워드 클러스터링에서 저장됨
    doc_topic_distribution = temp_data.get("doc_topic_distribution") # LDA 토픽 모델링에서 저장됨 (list of lists)
    tfidf_matrix_all = temp_data.get("tfidf_matrix") # 워드 클러스터링에서 저장된 전체 TF-IDF 행렬 (list of lists)
    feature_names = temp_data.get("feature_names") # 워드 클러스터링에서 저장된 단어 목록
    cluster_docs_map = temp_data.get("cluster_docs_map") # 클러스터별 문서 인덱스 맵 (워드 클러스터링에서 저장됨)
    
    print(f"🔍 [Debug-Opportunity] documents: {'None' if documents is None else 'OK'}, "
          f"doc_topic_distribution: {'None' if doc_topic_distribution is None else 'OK'}, "
          f"tfidf_matrix_all: {'None' if tfidf_matrix_all is None else 'OK'}, "
          f"feature_names: {'None' if feature_names is None else 'OK'}, "
          f"lda_results: {'None' if not lda_results else 'OK'}, " # 빈 리스트도 False이므로 `not` 사용
          f"cluster_docs_map: {'None' if cluster_docs_map is None else 'OK'}")

    # 필수 데이터 누락 확인 (조건문 수정)
    if not all([documents, doc_topic_distribution is not None, tfidf_matrix_all is not None,
                 feature_names is not None, lda_results, cluster_docs_map]):
        return {"error": "기회 점수 계산을 위한 분석 결과(문서, 토픽 분포, TF-IDF 행렬, LDA 토픽, 클러스터 맵, 피처 이름)가 부족합니다. 이전 분석 단계를 모두 수행했는지 확인하세요."}
    
    # 넘파이 배열로 변환 (필요시)
    doc_topic_distribution_np = np.array(doc_topic_distribution) # list of lists -> numpy array

    scores = []
    
    # --- 2. 각 토픽(Action)별로 순회하며 점수를 계산합니다. ---
    for topic_item in lda_results: # lda_results는 이미 리스트 형태여야 함
        topic_id_str = topic_item.get("topic_id") # 예: "2-1" (클러스터ID-토픽인덱스)
        action_keywords = topic_item.get("action_keywords") # LDA에서 추출된 키워드
        
        # topic_id_str이 유효한 형태인지 확인 (예: "C-T" 형식)
        if '-' not in topic_id_str:
            print(f"⚠️ 경고: 유효하지 않은 topic_id 형식: {topic_id_str}. 스킵합니다.")
            continue
        
        try:
            cluster_id = int(topic_id_str.split('-')[0])
            topic_index_in_cluster = int(topic_id_str.split('-')[1])
        except ValueError:
            print(f"⚠️ 경고: topic_id {topic_id_str}에서 클러스터 ID 또는 토픽 인덱스를 파싱할 수 없습니다. 스킵합니다.")
            continue

        # 해당 클러스터에 속하는 원본 문서들의 인덱스
        cluster_docs_indices_in_all_docs = cluster_docs_map.get(cluster_id)
        if cluster_docs_indices_in_all_docs is None or not cluster_docs_indices_in_all_docs:
            print(f"⚠️ 경고: 클러스터 {cluster_id}에 대한 문서 인덱스가 없습니다. 스킵합니다.")
            continue

        # 🚨🚨🚨 이 부분들을 수정합니다: lda_model과 vectorizer 대신 저장된 데이터를 활용
        # doc_topic_distribution_np는 run_topic_modeling_lda에서 계산된 클러스터 문서들에 대한 분포입니다.
        # 따라서, 여기서 다시 transform 할 필요 없이, 바로 해당 토픽의 문서를 찾으면 됩니다.
        
        TOPIC_ASSIGNMENT_THRESHOLD = 0.05

        
        # 3. 임계값을 기반으로 문서 필터링
        docs_belonging_to_this_topic_indices_by_threshold = np.where(doc_topic_distribution_np[:, topic_index_in_cluster] >= TOPIC_ASSIGNMENT_THRESHOLD)[0]

        # 4. (선택 사항) 가장 높은 확률로 할당된 문서도 함께 확인 (디버깅용)
        docs_most_strongly_assigned_to_this_topic_indices = np.where(np.argmax(doc_topic_distribution_np, axis=1) == topic_index_in_cluster)[0]

        print(f"🔍 [Debug-TopicDocs] Topic {topic_id_str}: "
              f"Docs by Threshold ({TOPIC_ASSIGNMENT_THRESHOLD}): {len(docs_belonging_to_this_topic_indices_by_threshold)} / {doc_topic_distribution_np.shape[0]}, "
              f"Docs by Argmax: {len(docs_most_strongly_assigned_to_this_topic_indices)} / {doc_topic_distribution_np.shape[0]}")


        # 실제 원본 문서 리스트에서 해당 토픽의 문서들을 가져옵니다.
        # 임계값 기준으로 찾은 문서를 사용
        actual_original_docs_indices = [cluster_docs_indices_in_all_docs[i] for i in docs_belonging_to_this_topic_indices_by_threshold]
        topic_docs = [documents[i] for i in actual_original_docs_indices]
        
        if not topic_docs:
            # 경고 메시지에 Argmax 기준 문서 수도 추가하여 비교
            print(f"⚠️ 경고: 토픽 {topic_id_str}에 해당하는 문서가 없습니다. "
                  f"(임계값 {TOPIC_ASSIGNMENT_THRESHOLD} 기준: 0개, Argmax 기준: {len(docs_most_strongly_assigned_to_this_topic_indices)}개) 스킵합니다.")
            continue

        # 3. 중요도(언급량) 계산: 해당 토픽에 속하는 실제 문서의 수
        importance = len(topic_docs)
        
        # 4. 만족도(감성 점수) 계산 - 동기 처리 (샘플링)
        sample_size = min(len(topic_docs), 20)
        sentiments = [_get_sentiment_score(doc) for doc in topic_docs[:sample_size]]
        satisfaction = np.mean(sentiments) if sentiments else 0.0
        
        # 5. 기회 점수 계산
        opportunity_score = importance * max(0, -satisfaction) # 부정적 감성일수록 기회가 큼

        scores.append({
            "topic_id": topic_id_str,
            "action_keywords": action_keywords,
            "importance": importance, # 언급량
            "satisfaction": round(satisfaction, 2), # 감성 점수 (-1.0 ~ 1.0)
            "opportunity_score": round(opportunity_score, 2)
        })
        
    scores.sort(key=lambda x: x['opportunity_score'], reverse=True)

    # 6. 워크스페이스에 결과 저장
    workspace["artifacts"]["cx_opportunity_scores"] = scores # 이미 main.py에서 리스트로 정의됨

    return {"cx_opportunity_scores": scores}
