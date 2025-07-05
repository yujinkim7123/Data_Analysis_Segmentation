# agents/cx_analyst.py

import json
import numpy as np
import pandas as pd
import networkx as nx


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_sentiment_analyzer
from .utils import get_openai_client
from scipy.sparse import csr_matrix # 희소 행렬 변환 시 필요
from collections import defaultdict 
from sklearn.decomposition import PCA, LatentDirichletAllocation
from scipy.sparse import csr_matrix # 🚨 추가: csr_matrix 임포트
import community as co


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
        print(result)
        # LABEL_1 (긍정)은 긍정 점수, LABEL_0 (부정)은 부정 점수로 매핑
        if result['label'] == 'positive': # 긍정
            return float(result['score'])
        elif result['label'] == 'negative': # 부정
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


def run_ward_clustering(workspace, num_clusters=5):
    """
    고객의 목소리(VOC) 데이터를 워드 클러스터링하여 주요 주제 그룹을 발견하고,
    각 클러스터의 대표 키워드를 추출합니다.
    """
    print(f"[CX Analysis] Running Ward Clustering with {num_clusters} clusters (동기 모드)")

    retrieved_data = workspace["artifacts"].get("retrieved_data")
    if not retrieved_data:
        return {"error": "데이터 클러스터링을 위한 검색된 데이터가 워크스페이스에 없습니다. 먼저 데이터 검색을 해주세요."}

    documents = [d.get('sentence_nouns', '') for d in retrieved_data.get('web_results', []) if d.get('sentence_nouns')]
    
    if not documents:
        return {"error": "클러스터링할 유효한 텍스트 문서가 없습니다. 검색 결과를 확인해주세요."}

    # Comprehensive Korean Stopwords List for Web Content
    noun_stopwords = [
        # 1. 한 글자 명사 및 의존 명사
        '것', '수', '일', '점', '때', '곳', '분', '데', '중', '안', '앞', '뒤', '속', '위', '아래', '뿐', '만', '쪽', '편', '겸', '김', '낯', '이', '그', '저',
        # 2. 일반/추상 명사
        '문제', '경우', '생각', '이유', '부분', '사실', '내용', '상황', '사람', '정도', '가지', '결과', '과정', '방법', '사용', '기능', '제품', '정보', '느낌', '마음', '기분', '순간', '처음', '마지막', '시작', '하루', '오늘', '어제', '내일', '지금', '요즘', '최근', '이전', '이후', '현재', '미래', '세상', '시대', '사회',
        # 3. 대명사 (명사로 분류될 수 있는)
        '저', '나', '내', '제', '우리', '저희', '너', '당신', '그', '그녀', '그들', '누구', '무엇', '여기', '저기', '거기', '어디',
        # 4. 시간/장소/수량 관련 명사
        '하나', '둘', '한번', '두번', '이번', '다음', '일단', '먼저', '약간', '조금', '계속', '요새', '근래',
        # 5. 사용자 피드백 기반 추가 (웹 환경)
        '진짜', '완전', '정말', '최고', '그냥', '바로'
    ]
    try:
        # 2. 텍스트 벡터화 (TF-IDF)
        vectorizer = TfidfVectorizer(max_features=2000, min_df=0.01, stop_words=noun_stopwords)
        X = vectorizer.fit_transform(documents)

        if X.shape[1] == 0: # 문서-단어 행렬에 유효한 피처(단어)가 없는 경우
            return {"error": "TF-IDF 벡터화 후 유효한 단어가 추출되지 않았습니다. 데이터를 확인하거나 TfidfVectorizer 설정을 조정하세요."}

        # 3. K-Means 클러스터링 수행
        if num_clusters > X.shape[0]:
            num_clusters = X.shape[0]
            print(f"⚠️ 클러스터 개수가 문서 수보다 많아 {num_clusters}개로 조정되었습니다.")
        
        if num_clusters < 2:
            return {"error": "클러스터 개수는 최소 2개 이상이어야 합니다."}

        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_.tolist() # 🚨 클러스터 라벨 리스트로 변환

        # 4. 각 클러스터의 대표 키워드 추출
        feature_names = np.array(vectorizer.get_feature_names_out())
        cluster_centers = kmeans.cluster_centers_

        cluster_summaries = {}
        for i in range(num_clusters):
            cluster_docs_indices = np.where(kmeans.labels_ == i)[0]
            num_docs_in_cluster = len(cluster_docs_indices)

            if num_docs_in_cluster == 0:
                cluster_summaries[str(i)] = {"keywords": [], "description": f"{i}번 그룹 (0개 문서)에는 문서가 없습니다."}
                continue

            cluster_tfidf_sum = X[cluster_docs_indices].sum(axis=0)
            top_feature_indices = cluster_tfidf_sum.A.flatten().argsort()[-10:][::-1]
            top_keywords = feature_names[top_feature_indices].tolist()

            cluster_summaries[str(i)] = {
                "keywords": top_keywords,
                "description": f"{i}번 그룹 ({num_docs_in_cluster}개 문서)은 주로 '{', '.join(top_keywords[:5])}...' 등의 키워드를 포함합니다."
            }
        
        # 🚨 추가: 시각화를 위한 2D 데이터 축소 (PCA)
        # 데이터 포인트가 2개 미만일 경우 PCA 오류 방지
        if X.shape[0] < 2:
            # 데이터 포인트가 2개 미만이면 PCA를 적용할 수 없으므로, 더미 데이터 생성 또는 빈 리스트 반환
            reduced_features_2d = np.array([[0.0, 0.0]] * X.shape[0]).tolist() # 모든 점을 원점에
        else:
            pca = PCA(n_components=2, random_state=42)
            # X (TF-IDF 행렬)는 scipy.sparse.csr_matrix 타입이므로, toarray()로 변환하여 PCA에 전달
            reduced_features_2d = pca.fit_transform(X.toarray()).tolist() # 🚨 numpy 배열을 리스트로 변환

        # 5. 워크스페이스에 임시 데이터 저장 (LDA, SNA를 위해)
        tfidf_matrix_list = X.toarray().tolist()
        feature_names_list = feature_names.tolist()

        cluster_docs_map = defaultdict(list)
        for doc_idx, label in enumerate(cluster_labels):
            cluster_docs_map[label].append(doc_idx)

        workspace["artifacts"]["_cx_temp_data"] = {
            "cluster_labels": cluster_labels,
            "tfidf_matrix": tfidf_matrix_list,
            "feature_names": feature_names_list,
            "documents": documents,
            "cluster_docs_map": dict(cluster_docs_map),
        }

        workspace["artifacts"]["cx_ward_clustering_results"] = {
            "num_clusters": num_clusters,
            "cluster_labels": cluster_labels,
            "cluster_summaries": cluster_summaries,
            "visual_data": {
                "reduced_features_2d": reduced_features_2d,
                "cluster_labels": cluster_labels,
            }
        }
        workspace["artifacts"]["analysis_results"] = (
            "Ward clustering analysis complete. 각 클러스터의 대표 키워드를 확인해보세요. "
            "특정 클러스터에 대해 더 깊은 분석(의미 연결망 분석)을 원하시면 클러스터 ID와 함께 요청해주세요."
        )

        print("💾 Ward clustering results saved to workspace artifacts.")
        # 6. 분석 결과 반환
        return {
            "cx_ward_clustering_results": {
                "num_clusters": num_clusters,
                "cluster_labels": cluster_labels, # 🚨 클러스터 라벨 포함
                "cluster_summaries": cluster_summaries,
                "visual_data": { # 🚨 추가: 시각화 데이터 포함
                    "reduced_features_2d": reduced_features_2d,
                    "cluster_labels": cluster_labels,
                }
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

    try:
        docs_indices = [i for i, label in enumerate(temp_data["cluster_labels"]) if label == cluster_id]
        if not docs_indices: return {"error": f"ID가 {cluster_id}인 클러스터에 문서가 없습니다."}
        
        cluster_matrix = csr_matrix(np.array(temp_data["tfidf_matrix"])[docs_indices])

        co_occurrence_matrix = (cluster_matrix.T * cluster_matrix)
        co_occurrence_matrix.setdiag(0)

        G = nx.Graph()
        feature_names = temp_data["feature_names"]
        
        # 노드 추가: 노드 ID를 실제 키워드 이름으로 직접 할당
        for i, name in enumerate(feature_names):
            G.add_node(name, id=name, name=name) # 🚨 노드 ID를 키워드 이름으로 명시

        # 🚨 엣지 추가: 동시 출현 빈도 임계값 조정 (매우 중요!)
        # 0에 가까울수록 더 많은 엣지 생성. 데이터에 따라 적절한 값 찾기.
        threshold = 0.1 # 🚨 초기값 2에서 1.0으로 조정 (더 많은 엣지 생성 시도)
                        # 만약 여전히 그래프가 안 나오면 0.5, 0.1 등으로 더 낮춰보세요.
                        # 아주 드문 경우 0으로 설정하여 모든 연결을 시도할 수도 있습니다.
        
        # TF-IDF 기반 동시 출현 행렬의 값을 직접 사용하여 엣지를 추가
        # feature_names에 대한 인덱스를 기반으로 엣지를 추가하고, 실제 키워드 이름은 노드 자체의 'id'와 'name' 속성을 통해 프론트엔드로 전달됨
        for i in range(co_occurrence_matrix.shape[0]):
            for j in range(i + 1, co_occurrence_matrix.shape[1]):
                weight = co_occurrence_matrix[i, j]
                if weight > threshold:
                    # 🚨 엣지 소스/타겟을 인덱스 대신 실제 키워드 이름으로 변경 (네트워크X 엣지는 노드 ID로 연결)
                    G.add_edge(feature_names[i], feature_names[j], weight=float(weight)) # 🚨 키워드 이름으로 엣지 연결

        # 4. 핵심 노드 (핵심 키워드) 추출 및 Micro-segments, Graph Data 구성
        micro_segments = []
        graph_data = {} # 🚨 graph_data 변수를 여기서 초기화

        # 커뮤니티 감지 및 그래프 데이터 구성 (python-louvain 활용)
        try:
            # import community as co # 이미 상단에 임포트됨
            partitions = co.best_partition(G)
            
            for community_id in set(partitions.values()):
                community_nodes_names = [node_name for node_name, part_id in partitions.items() if part_id == community_id]
                
                if community_nodes_names:
                    # 서브그래프는 실제 노드 이름으로 구성
                    subgraph = G.subgraph(community_nodes_names)
                    if subgraph.nodes():
                        # 해당 커뮤니티에서 중심성이 가장 높은 노드를 핵심 키워드로
                        sub_centrality = nx.degree_centrality(subgraph)
                        core_keyword_name = max(sub_centrality, key=sub_centrality.get)
                    else:
                        core_keyword_name = community_nodes_names[0]

                    micro_segments.append({
                        "community_id": community_id,
                        "core_keyword": core_keyword_name,
                        "keywords": community_nodes_names # 커뮤니티에 속하는 모든 키워드 (이름)
                    })
            micro_segments = sorted(micro_segments, key=lambda x: (x['community_id'], x['core_keyword']))

            # 🚨 최종 graph_data 생성 및 노드에 커뮤니티 정보 추가
            graph_data_temp = nx.node_link_data(G)
            for node_data in graph_data_temp['nodes']:
                node_data['community'] = partitions.get(node_data['id'], None)
            centrality = nx.degree_centrality(G)    
            for node_data in graph_data_temp['nodes']:
                node_data['centrality'] = float(centrality.get(node_data['id'], 0.0))
            graph_data = graph_data_temp # 최종 graph_data에 할당

        except ImportError:
            print("Warning: 'python-louvain' library not found. Community detection skipped. Micro-segments based on top keywords.")
            # python-louvain이 없으면 중심성 높은 상위 N개 키워드로 구성
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            for node_name, score in sorted_nodes[:10]: # 노드 이름과 점수
                micro_segments.append({
                    "community_id": "N/A",
                    "core_keyword": node_name, # 노드 이름 직접 사용
                    "keywords": [node_name],
                    "centrality_score": round(score, 4)
                })
            graph_data = nx.node_link_data(G) # G의 노드 ID는 이미 키워드 이름

        except Exception as e:
            print(f"Community detection or graph processing failed: {e}. Falling back to simple segments and basic graph data.")
            import traceback; traceback.print_exc() # 오류 스택 트레이스 출력
            micro_segments = [] # 오류 시 빈 리스트
            
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            for node_name, score in sorted_nodes[:min(len(sorted_nodes), 10)]: # 최대 10개
                micro_segments.append({
                    "community_id": "Error",
                    "core_keyword": node_name,
                    "keywords": [node_name],
                    "centrality_score": round(score, 4)
                })
            graph_data = nx.node_link_data(G)



        # 수정: 워크스페이스에 결과 저장
        workspace["artifacts"]["cx_sna_results"] = {
            "cluster_id": cluster_id,
            "micro_segments": micro_segments,
            "graph_data": graph_data,
            "analysis_description": f"{cluster_id}번 클러스터 내에서 가장 핵심적인 키워드들을 찾아 의미 연결망 분석을 수행했습니다."
        }
        workspace["artifacts"]["analysis_results"] = "Semantic network analysis complete."

        # 수정: save_workspace_to_redis 호출 제거 (main.py에서 처리)
        print("💾 SNA results saved to workspace artifacts.")     

        # 5. 결과 반환 (직렬화 가능한 데이터만 포함)
        return {
            "cx_sna_results": {
                "cluster_id": cluster_id,
                "micro_segments": micro_segments,
                "graph_data": graph_data,
                "analysis_description": f"{cluster_id}번 클러스터 내에서 가장 핵심적인 키워드들을 찾아 의미 연결망 분석을 수행했습니다."
            },
            "analysis_results": "Semantic network analysis complete."
        }

    except Exception as e:
        print(f"❌ SNA 분석 중 예상치 못한 오류 발생: {e}") # 메시지 구체화
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

        # 🚨 추가: LDA 모델 학습 전 데이터 유효성 검사
        if doc_term_matrix.shape[0] < num_topics:
            return {"error": f"문서 개수({doc_term_matrix.shape[0]}개)가 토픽 개수({num_topics})보다 적습니다. 더 작은 토픽 개수로 다시 시도해주세요."}
        if doc_term_matrix.shape[1] == 0: # 유효한 단어가 없으면
            return {"error": "토픽 모델링을 수행할 단어가 부족합니다. CountVectorizer 설정을 조정하세요."}

        # 4. LDA 모델 학습
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix) # 문서-단어 행렬로 LDA 학습

        # 각 문서의 토픽 분포를 계산 (calculate_opportunity_scores에서 사용될 데이터)
        doc_topic_dist_for_cluster = lda.transform(doc_term_matrix) # 👈 클러스터에 해당하는 문서들의 토픽 분포
        assignments = np.argmax(doc_topic_dist_for_cluster, axis=1)
        # 5. 토픽별 상위 키워드 추출 및 요약 정보 구성 (기존 로직)
        topics_list = []
        top_keywords_per_topic = _get_top_keywords(feature_names, lda.components_, 7) # 상위 7개 키워드

        # 🚨 LDA 그래프 시각화 데이터 생성 시작 🚨
        topic_graph_data_points = [] # 각 토픽의 2D 위치 (그래프 점)
        topic_keywords_with_weights = [] # 각 토픽의 키워드 및 가중치 (툴팁용)

        # LDA 모델의 components_는 (num_topics, num_features) 형태의 토픽-단어 분포 행렬
        topic_embeddings = lda.components_

        # PCA를 사용하여 토픽 임베딩을 2D로 축소
        # 토픽 개수가 2개 미만이거나, 피처 개수가 2개 미만이면 PCA 적용 불가
        if topic_embeddings.shape[0] >= 2 and topic_embeddings.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            # 각 토픽의 2D 위치
            topic_positions_2d = pca.fit_transform(topic_embeddings).tolist()
        else: # PCA 적용 불가 시 임시 위치 할당
            topic_positions_2d = [[np.random.rand() * 10, np.random.rand() * 10] for _ in range(num_topics)]
            print("Warning: Not enough data for meaningful PCA for LDA topics. Using random positions for graph.")


        for i, keywords in enumerate(top_keywords_per_topic):
            topic_docs = [
                docs_indices[j]
                for j, assigned in enumerate(assignments)
                if assigned == i
            ]
            # 토픽별 상위 키워드의 가중치도 함께 추출 (확률로 정규화)
            current_topic_comp = lda.components_[i]
            keywords_and_weights = {
                kw: float(current_topic_comp[feature_names.index(kw)]) / current_topic_comp.sum()
                for kw in keywords if kw in feature_names
            }
            topic_keywords_with_weights.append({
                "topic_id": i,
                "keywords": keywords_and_weights
            })

            # LDA 그래프에 표시될 데이터 포인트
            topic_graph_data_points.append({
                "topic_id": i,
                "x": topic_positions_2d[i][0],
                "y": topic_positions_2d[i][1],
                "keywords_data": keywords_and_weights # 해당 토픽의 키워드 데이터 (툴팁용)
            })

            # 기존 topics_list (요약 메시지용)
            topics_list.append({
                "topic_id": f"{cluster_id}-{i}", # 클러스터 ID와 토픽 인덱스를 조합
                "action_keywords": keywords, # 상위 키워드
                "description": f"주요 키워드: {', '.join(keywords[:5])}...", # 간단한 설명 추가
                "document_indices": topic_docs
            })

        # 🚨 LDA 그래프 시각화 데이터 최종 구성
        lda_graph_data = {
            "topics": topic_graph_data_points,
            "num_topics": num_topics
      
        }
        # 수정: 워크스페이스에 결과 저장
        workspace["artifacts"]["cx_lda_results"] = {
            "cluster_id": cluster_id,
            "num_topics": num_topics,
            "topics_summary_list": topics_list,
            "graph_data": lda_graph_data
        }
        workspace["artifacts"]["analysis_results"] = (
            f"클러스터 {cluster_id}에 대해 {num_topics}개의 토픽을 성공적으로 식별했습니다."
        )     
        # 7. LLM에게 반환할 데이터 (간결하게)
        return {
            "cx_lda_results": { # ArtifactRenderer에서 이 키를 통해 접근
                "cluster_id": cluster_id,
                "num_topics": num_topics,
                "topics_summary_list": topics_list, # 기존 토픽 요약 메시지용 리스트
                "graph_data": lda_graph_data # 🚨 새롭게 추가된 그래프 데이터
            },
            "success": True, # 성공 여부
            "message": f"클러스터 {cluster_id}에 대해 {num_topics}개의 토픽을 성공적으로 식별했습니다.",
            "newly_identified_topics_preview": [ # LLM 메시지용 간략화된 미리보기
                {"topic_id": t["topic_id"], "action_keywords": t["action_keywords"]}
                for t in topics_list
            ]
        }

    except Exception as e:
        print(f"❌ 토픽 모델링(LDA) 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"토픽 모델링(LDA) 분석 중 오류가 발생했습니다: {e}"}

def create_customer_action_map(workspace: dict, topic_id: str):
    """
    [완성본] PDF 4단계: '분석된 결과'를 바탕으로 CAM(Pain Point 등)을 생성합니다.
    """
    print(f"✅ [CX Agent] Step 4: Creating CAM for Topic ID: {topic_id}...")
    client = get_openai_client()
    
    # --- 1. workspace에서 이 토픽에 대한 모든 분석 '결과'를 가져옵니다. ---
    artifacts = workspace.get("artifacts", {})
    #lda_results = artifacts.get("cx_lda_results", [])
    lda_results = artifacts.get("cx_lda_results", {}).get("topics_summary_list", [])
    opportunity_scores = artifacts.get("cx_opportunity_scores", [])
    
    # 해당 topic_id에 대한 정보를 찾습니다.
    topic_lda_data = next((item for item in lda_results if item.get("topic_id") == topic_id), None)
    topic_score_data = next((item for item in opportunity_scores if item.get("topic_id") == topic_id), None)

    if not topic_lda_data or not topic_score_data:
        return {"error": f"ID가 {topic_id}인 토픽에 대한 분석 결과가 부족합니다. LDA와 기회 점수 계산을 먼저 수행해주세요."}
    
        # --- 1.5. graph_data에서 keywords_data 추출 ---
    graph_topics = artifacts.get("cx_lda_results", {}).get("graph_data", {}).get("topics", [])
    # topic_id는 "cluster-topicIndex" 형태이니, 끝 숫자만 파싱
    try:
        idx = int(topic_id.split("-")[-1])
    except ValueError:
        idx = None
    keywords_data = {}
    if idx is not None:
        tg = next((t for t in graph_topics if t.get("topic_id")==idx), None)
        if tg:
            keywords_data = tg.get("keywords_data", {})



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
        cam_results["keywords_data"] = keywords_data
        existing_cams = workspace.get("artifacts", {}).get("cx_cam_results", [])
        existing_cams.append(cam_results)
        workspace["artifacts"]["cx_cam_results"] = existing_cams
        workspace["artifacts"]["analysis_results"] = (
            f"토픽 ID {topic_id}에 대한 고객 액션맵(CAM)이 성공적으로 생성되었습니다."
        )

        print("💾 CAM results saved to workspace artifacts.")

        return {"cx_cam_results": existing_cams}
    except Exception as e:
        return {"error": f"고객 액션맵 생성 중 오류: {e}"}


def calculate_opportunity_scores(workspace: dict):
    # 1) LDA 토픽 분석 결과 & 원문 꺼내오기
    lda_results = workspace["artifacts"]["cx_lda_results"]["topics_summary_list"]
    all_docs    = [d["original_text"] for d in workspace["artifacts"]["retrieved_data"]["web_results"]]

    # 2) 원시 중요도·감성 수집
    try:
        raw_importances, raw_sentiments = [], []
        for topic in lda_results:
            idxs = topic.get("document_indices", [])
            docs = [all_docs[i] for i in idxs if i < len(all_docs)]
            raw_importances.append(len(docs))
            scores = [_get_sentiment_score(doc) for doc in docs]
            raw_sentiments.append(float(np.mean(scores)) if scores else 0.0)

        # 3) 정규화
        imp_min, imp_max = min(raw_importances), max(raw_importances)
        sent_min, sent_max = min(raw_sentiments), max(raw_sentiments)
        def norm(x, lo, hi): return 0.0 if hi==lo else (x-lo)/(hi-lo)*10
        norm_imps = [norm(v, imp_min, imp_max) for v in raw_importances]
        norm_sats = [norm(v, sent_min, sent_max) for v in raw_sentiments]

        # 4) 기회 점수 계산
        opportunity_scores = []
        for i, topic in enumerate(lda_results):
            imp10 = norm_imps[i]
            sat10 = norm_sats[i]
            opp   = imp10 + (10 - sat10)  # 덧셈 모델
            opportunity_scores.append({
                "topic_id":          topic["topic_id"],
                "action_keywords":   topic["action_keywords"],
                "importance":        round(imp10, 2),
                "satisfaction":      round(sat10, 2),
                "opportunity_score": round(opp, 2)
            })

        # 5) 정렬 & 반환
        sorted_scores = sorted(opportunity_scores, key=lambda x: x["opportunity_score"], reverse=True)
        workspace["artifacts"]["cx_opportunity_scores"] = sorted_scores
        workspace["artifacts"]["analysis_results"] = (
            "기회 점수 계산이 완료되었습니다. 각 토픽의 중요도와 만족도를 확인해보세요."
        )

        # 수정: save_workspace_to_redis 호출 제거 (main.py에서 처리)
        print("💾 Opportunity scores saved to workspace artifacts.")

        return {"cx_opportunity_scores": sorted_scores}

    except Exception as e:
        print(f"❌ 기회 점수 계산 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"기회 점수 계산 중 오류가 발생했습니다: {e}"}
