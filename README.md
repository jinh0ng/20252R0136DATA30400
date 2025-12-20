# 20252R0136DATA30400
### DATA304 Final Project — Amazon Products Taxonomy Classification
BigDataAnalysis Lecture Final Project

이 레포는 Amazon products 텍스트(문서) 임베딩을 기반으로, taxonomy graph(계층 구조)를 활용한 GAT 모델 + EMA self-training으로 multi-label 분류를 수행하고 Kaggle 제출 파일(`submission.csv`)을 생성합니다.

본 프로젝트는 **임베딩 모델을 교체**(MPNet / GTE / BGE)하여 성능을 비교했고, 동일 파이프라인을 임베딩별로 자동 실행하는 `run_*.sh` 스크립트를 제공합니다.

---

## 1. Directory Structure

프로젝트 루트(`~/20252R0136DATA30400`) 기준 구조는 아래와 같습니다.
```
20252R0136DATA30400/
├─ src/
│  ├─ make_embeddings_mpnet.py   # 임베딩 생성 (MPNet/GTE/BGE 공통)
│  ├─ build_silver.py            # silver label + adj 생성
│  ├─ train_gat.py               # GAT 학습
│  ├─ self_train.py              # EMA self-train
│  ├─ submission.py              # test 추론 + dynamic top-k + csv 저장
│  └─ utils/
│     ├─ paths.py                # dataset_dir/data_dir/outputs_dir 규칙
│     ├─ embeddings.py           # 임베딩/ pid_list 로드
│     ├─ models_gat.py           # GATLayer, TaxoClassGAT
│     ├─ taxonomy.py             # taxonomy 경로/ancestor 처리
│     └─ io.py                   # np/json 저장 유틸
├─ project_release/
│  └─ Amazon_products/           # 데이터셋 루트
│     ├─ train/, test/
│     ├─ classes.txt, class_hierarchy.txt, class_related_keywords.txt
│     └─ embeddings_bge/         # (주 사용) BGE 임베딩 결과
├─ data_bge/                     # (주 사용) silver/graph 중간 산출물
│  ├─ silver/                    # y_silver.npy, y_refined_round*.npy, core_classes.json
│  └─ graph/                     # adj.npy
├─ outputs_bge/                  # (주 사용) 학습/추론 결과
│  ├─ checkpoints/               # gat.pt, ema_teacher.pt
│  └─ submissions/               # submission_bge.csv
└─ run_bge.sh                    # BGE 전체 파이프라인 실행 스크립트

```


---

## 2. End-to-End Pipeline

전체 파이프라인은 아래 Step 1 ~ Step 5로 구성됩니다.

### Step 1) Make Embeddings
- 스크립트: `src/make_embeddings_mpnet.py`
- 역할:
  - `train/train_corpus.txt`, `test/test_corpus.txt` 텍스트를 SentenceTransformer로 임베딩
  - 클래스 이름(`classes.txt`) 임베딩 생성
  - 결과를 `dataset_dir/embeddings_<model>/` 아래에 `.npy`로 저장
- 출력 예:
  - `train_doc_mpnet.npy`
  - `test_doc_mpnet.npy`
  - `class_name_mpnet.npy`
  - `pid_list_train.npy`
  - `pid_list_test.npy`

> 주의: 파일명은 `*_mpnet.npy` 형태지만, 실제 임베딩 모델은 `--model_name`으로 결정됩니다.  
> 모델별로 `embeddings_mpnet`, `embeddings_gte`, `embeddings_bge`처럼 **폴더를 분리**하여 충돌을 방지합니다.

---

### Step 2) Build Silver Labels + Graph Adjacency
- 스크립트: `src/build_silver.py`
- 역할:
  - 문서 임베딩과 클래스 임베딩 간 유사도 기반으로 core class를 선정
  - taxonomy 경로(ancestor 포함)로 positive set을 확장하여 **silver label** 생성
  - class hierarchy 기반 adjacency matrix 생성
- 출력:
  - `data_<model>/silver/y_silver.npy`
  - `data_<model>/silver/core_classes.json`
  - `data_<model>/graph/adj.npy`

---

### Step 3) Train GAT (TaxoClassGAT)
- 스크립트: `src/train_gat.py`
- 역할:
  - 문서 임베딩과 클래스 임베딩을 입력으로 GAT 기반 모델 학습
  - `y_silver.npy`를 pseudo ground-truth로 사용
- 출력:
  - `outputs_<model>/checkpoints/gat.pt`

---

### Step 4) EMA Self-Training
- 스크립트: `src/self_train.py`
- 역할:
  - Step 3의 student 모델을 초기값으로
  - EMA teacher로 pseudo label을 정제(refine)
  - 정제된 라벨로 student를 재학습하면서 EMA teacher 업데이트
- 출력:
  - `outputs_<model>/checkpoints/ema_teacher.pt`
  - `data_<model>/silver/y_refined_round*.npy`

---

### Step 5) Make Submission CSV
- 스크립트: `src/submission.py`
- 역할:
  - EMA teacher로 test 문서에 대한 label probability 예측
  - dynamic top-k 정책으로 labels 결정
  - `submission.csv` 생성
- 출력:
  - `outputs_<model>/submissions/submission_<model>.csv`

---

## 3. How to Run
### 3.1 One-time setup (make scripts executable)

```bash
cd ~/20252R0136DATA30400
chmod +x run_bge.sh
```


### 3.2 실행 (임베딩별 전체 파이프라인)

MPNet:
```./run_mpnet.sh```

GTE:
```./run_gte.sh```

BGE (추천 / 최고 성능):
```./run_bge.sh```

실행 완료 후 제출 파일은 아래에 저장됩니다.
outputs_mpnet/submissions/submission_mpnet.csv
outputs_gte/submissions/submission_gte.csv
outputs_bge/submissions/submission_bge.csv

## 4. Useful Overrides (Environment Variables)
실행 시 환경변수로 주요 파라미터를 간단히 변경할 수 있습니다.
임베딩 강제 재생성
```FORCE_EMB=1 ./run_bge.sh
submission top-k 정책 변경 (예: 더 많은 label 허용)
THRESHOLD=0.70 MAX_LABELS=4 OUT_NAME=submission_bge_t070_k24.csv ./run_bge.sh
```

## 5. Embedding Models Tested & Result Summary
동일한 downstream 파이프라인(GAT + EMA self-training)에서 임베딩 모델만 교체하여 성능을 비교했습니다.
```MPNet: sentence-transformers/all-mpnet-base-v2
GTE: thenlper/gte-base
BGE: BAAI/bge-base-en-v1.5
```
실험 결과, BGE가 가장 높은 Kaggle score(약 0.48610)를 기록하여 최종 제출에 사용했습니다.
(GTE는 약 0.474 수준으로 소폭 개선)
점수는 seed / threshold 등 제출 후처리 설정에 따라 소폭 변동할 수 있습니다.

## 6. Notes / Troubleshooting
(1) y_silver 파일명 대소문자
build_silver.py 실행 결과는 data_*/silver/y_silver.npy 입니다.
리눅스 환경에서는 대소문자 구분이 되므로, 스크립트 내부에서 해당 경로를 명시적으로 사용합니다.

(2) BGE 로딩 에러 발생 시
환경에 따라 SentenceTransformer 로딩 시 에러가 날 수 있습니다.
이 경우 src/make_embeddings_mpnet.py에서 trust_remote_code=True 옵션이 필요할 수 있습니다.

