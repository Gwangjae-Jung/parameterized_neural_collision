# Step 1: 충돌 연산자 및 RK4 결합 평가 모듈 (`step1__test_operators`)

본 디렉터리는 학습된 신경망 충돌 연산자 $Q_\theta$를 로드하고, 이를 **고전적 4차 룽게-쿠타(RK4) 수치 적분기와 결합하여 시간 전진 시뮬레이션을 수행**한 후 참값 수치해(Fast Spectral Method)와 비교 검증하는 평가 환경입니다.

$$\frac{df}{dt} = Q_\theta(f, \alpha) \quad \xrightarrow{\text{RK4}} \quad f_{k+1} = f_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

---

## 1. 주요 노트북 및 스크립트 구성

- **`test_main__2d.ipynb` / `test_main__3d.ipynb`**:
  - **(핵심 종합 평가 노트북)**: 다양한 충돌 지수 $\alpha$를 갖는 복수 궤적에 대해 신경망 RK4 솔버를 실행하고, 전체 시간 구간에서의 상대 $L_2$ 오차 곡선 및 평균 오차율을 산출합니다.
- **`test_main__*__single_term.ipynb`**:
  - 시간 전진 없이, 특정 분포 $f$가 입력되었을 때 모델이 출력하는 충돌항 $Q_\theta(f)$ 자체의 정밀도를 FastSM 정해 충돌항과 직접 1:1 비교합니다.
- **`test_main__*__single_traj.ipynb`**:
  - 단일 테스트 궤적에 대한 장시간(Long-time, $t \in [0, 20]$) 시계열 진화 과정을 추적하고 시각화합니다.
- **`fsm__*__single_term.ipynb`**:
  - Fast Spectral Method의 단일 충돌항 계산 기준값을 산출하는 검증 노트북입니다.
- **`plot_errors.ipynb`**:
  - 해상도 및 커널 파라미터별 오차 통계를 종합 플롯으로 시각화합니다.
- **`loader.py`**:
  - `datasets/test/` 경로에서 평가용 궤적 및 파라미터를 편리하게 불러오는 `load_dataset` 함수를 제공합니다.

---

## 2. 사용 방법

1. Jupyter Notebook에서 `test_main__2d.ipynb` (또는 `test_main__3d.ipynb`)을 엽니다.
2. 설정 셀에서 디바이스와 모델 파라미터를 확인합니다:
   ```python
   DIMENSION   = 2 # 또는 3
   N_LAYERS    = 4
   MAX_FREQ    = 8
   MODEL_TYPE  = "prfno"
   DELTA_T     = 0.1
   MAX_T       = 5.0
   ```
3. 노트북을 실행하면 `pretrained/{MODEL_TYPE}_{DIM}D/model_and_history.pth` 가중치를 로드하여 RK4 시간 적분을 수행하고, FSM 정해 대비 상대 오차와 물리량 보존 상태를 그래프로 출력합니다.
