# PRFNO 아키텍처 실험: Variant 2 (`step1__train_operators__variant2`)

본 디렉터리는 PRFNO 아키텍처의 파라미터 결합 방식 및 리프트/프로젝션 레이어의 구조적 변형을 적용한 **PRFNO Variant 2** 모델의 학습 및 비교 실험 모듈입니다.

---

## 1. 특징 및 구조 (`variants/variant2.py`)

- **`PRFNO_Variant2`**:
  - Variant 1과 다른 특징 맵 결합 메커니즘을 채택하여, radially symetric Fourier layer 전후에 non-linearity 와 channel projection을 추가한 변형 아키텍처입니다.

---

## 2. 파일 구성

- **`train__prfno_variant2.py`**: Variant 2 모델의 학습 스크립트.
- **`variants/variant2.py`**: `PRFNO_Variant2` 신경망 아키텍처 정의.
- **`train_utils/`**: 데이터 로더, 손실 함수, 인자 파서.
- **`PRFNO2D/`**: 2D Variant 2 모델의 학습 체크포인트 및 로그.

---

## 3. 실행 방법

```bash
# 루트 디렉터리에서 자동화 스크립트 실행
python main__prfno_variant2.py --cuda_index 0

# 또는 직접 실행
python train__prfno_variant2.py --cuda_index 0
```
