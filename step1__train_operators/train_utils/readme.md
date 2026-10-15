# 학습 유틸리티 (`step1__train_operators/train_utils`)

본 디렉터리는 충돌 연산자 학습에 필요한 데이터셋 로더, 커널 파라미터 변환 함수, 물리 보존 손실 함수 등을 제공합니다.

---

## 1. 모듈별 기능 설명

- **`dataloaders.py`**:
  - `datasets/` 경로에 저장된 궤적 파일로부터 입력 분포 $f$와 타겟 충돌항 $Q(f)$, 커널 지수 $\alpha$를 읽어들여 학습용 `TensorDataset` 및 `DataLoader`를 생성합니다.
- **`loss_functions.py` (`compute_loss_cons`)**:
  - 예측된 충돌항 $\hat{Q}(f)$가 만족해야 하는 이산 적분 보존 법칙을 강제합니다:
    - $\int \hat{Q}(f) dv = 0$ (질량 보존)
    - $\int v \hat{Q}(f) dv = 0$ (운동량 보존)
    - $\int |v|^2 \hat{Q}(f) dv = 0$ (에너지 보존)
- **`parameter_transformers.py`**:
  - 스칼라 지수 $\alpha$와 속도 격자 반경 $|v|$의 거듭제곱 조합($|v|^\alpha$) 또는 방사 기저 확장을 생성하여 신경망 입력 특징으로 가공합니다.
- **`parser.py` (`CustomParser`)**:
  - 모델 아키텍처 옵션, 학습 파라미터 및 경로 설정을 위한 명령행 인자 파서입니다.
- **`append_path.py`**:
  - 프로젝트 최상단 디렉터리를 `sys.path`에 안전하게 등록하여 어디서든 `core` 패키지를 임포트할 수 있도록 보장합니다.
