# PRFNO 아키텍처 실험: Variant 1 (`step1__train_operators__variant1`)

본 디렉터리는 기본 PRFNO 모델에서 위치 인코딩(Positional Encoding) 및 파라미터 브랜치(Branch Layer)의 결합 방식을 변형한 **PRFNO Variant 1** 모델의 학습 및 비교 실험 모듈입니다.

---

## 1. 특징 및 구조 (`variants/variant1.py`)

- **`PRFNO_Variant1`**:
  - `pos_enc_layer`와 `branch_layer`를 통해 공간 좌표 격자와 스칼라 충돌 지수 $\alpha$의 상호작용 특징을 별도의 다층 퍼셉트론으로 학습한 후 푸리에 레이어의 채널에 주입합니다.
  - 기존 모델 대비 주파수 보존 능력 및 수렴 속도의 차이를 분석하기 위한 어블레이션 실험용 모델입니다.

---

## 2. 파일 구성

- **`train__prfno_variant1.py`**: Variant 1 모델의 학습 스크립트.
- **`variants/variant1.py`**: `PRFNO_Variant1` 신경망 아키텍처 정의.
- **`train_utils/`**: 데이터 로더, 손실 함수, 인자 파서.
- **`PRFNO2D/`**: 2D Variant 1 모델의 학습 체크포인트 및 로그.

---

## 3. 실행 방법

```bash
# 루트 디렉터리에서 자동화 스크립트 실행
python main__prfno_variant1.py --cuda_index 0

# 또는 직접 실행
python train__prfno_variant1.py --cuda_index 0
```
