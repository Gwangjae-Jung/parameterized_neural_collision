# Step 1: 충돌 연산자 학습 모듈 (`step1__train_operators`)

본 디렉터리는 동차 볼츠만 방정식의 충돌 연산자 $Q(f, \alpha)$를 대리하는 신경 연산자(**PRFNO**)를 학습시키는 핵심 모듈입니다.

신경망은 분포 함수 $f$와 충돌 커널 지수 $\alpha$를 입력받아 충돌항 $Q(f)$를 산출하도록 훈련됩니다:
$$Q_\theta: (f, \alpha) \longmapsto Q(f)$$

---

## 1. 주요 학습 스크립트

- **`train__prfno.py`**:
  - **(표준 모델)** Parameterized Radial Fourier Neural Operator (PRFNO) 학습 스크립트입니다.
  - 방사 대칭 푸리에 믹싱 레이어(`RadialFourierLayer`)를 통해 회전 대칭성을 보존하고, 파라미터 $\alpha$를 조건부로 결합합니다.
  - 상대 $L_2$ 손실과 질량/운동량/에너지 보존 손실(`compute_loss_cons`)을 함께 최소화합니다.
- **`train__prfno__enhanced.py`**:
  - 푸리에 특징 임베딩 또는 활성화 함수/채널 확장이 적용된 개량형 PRFNO 학습 스크립트입니다.
- **`__old__train__pfno.py`**:
  - 방사 대칭성이 없는 기본 파라미터화 FNO(PFNO) 베이스라인 학습 스크립트입니다.
- **`train_utils/`**:
  - 데이터셋 로딩, 파라미터 변환기 및 물리 보존 손실 함수 모음입니다.

---

## 2. 학습 실행 방법

프로젝트 루트의 `main__prfno.py`를 통해 실행하거나 명령행 인자를 주어 단독 실행할 수 있습니다:

```bash
python train__prfno.py \
    --model_type prfno \
    --dim_domain 2 \
    --max_freq 8 \
    --n_layers 4 \
    --learning_rate 1e-4 \
    --batch_size 2048 \
    --num_epochs 1000 \
    --path_train ../datasets/train/Boltzmann_2D__coeff1.59e-01__res_t051_v032 \
    --path_val ../datasets/validation/Boltzmann_2D__coeff1.59e-01__res_t051_v032 \
    --cuda_index 0
```

---

## 3. 학습 결과물
- **`checkpoint__{DIR}/`**: 에포크별 모델 가중치(`state_dict.pth`) 및 최적 모델 가중치(`best__state_dict.pth`), 학습 곡선 히스토리(`train_history.pth`)가 저장됩니다.
- **`{DIR}.log`**: 에포크별 Train/Validation Loss 및 물리량 보존 오차 진행 로그가 기록됩니다.
