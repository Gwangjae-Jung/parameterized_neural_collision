# Parameterized Neural Collision (Runge-Kutta Method)

본 프로젝트는 homogeneous Boltzmann equation을 **operator learning** 기법으로 해결하는 딥러닝 연구 프로젝트입니다.  
충돌 커널(Collision Kernel)이 변하더라도 일관된 추론을 할 수 있는 단일 모델을 개발하였으며, 초기 조건 $f(0, \cdot)$이 주어졌을 때 FNO 계열 모델이 매 시간 스텝마다 **충돌항(Collision Term) $Q(f)$를 직접 예측하고, 이를 RK4와 결합**하여 해를 산출하는 방식을 사용합니다:
$$\frac{\partial f}{\partial t} = Q(f) \quad \Longrightarrow \quad f(t) \xrightarrow{\text{Model}} Q_\theta(f, \alpha) \xrightarrow{\text{RK4}} f(t + \Delta t)$$

---

## 1. 프로젝트 주요 특징

- **신경망 충돌 연산자 ($f \mapsto Q(f)$)**:
  - 분포 $f \mapsto f$를 통째로 근사하는 대신, 비선형 적분-미분 연산인 볼츠만 충돌 연산자 $Q(f, \alpha)$ 자체를 신경망으로 대리(Surrogate)합니다.
  - 시간 전진은 고전적인 수치해석 적분기(RK4)가 담당하므로, 긴 시간 롤아웃에서도 수치적 안정성이 극대화됩니다.
- **파라미터 조건부 방사 푸리에 연산자 (PRFNO)**:
  - 볼츠만 충돌 연산자의 핵심 물리 성질인 회전 불변성(Rotational Invariance)을 방사 대칭 푸리에 레이어로 엄밀히 보존합니다.
  - VHS 충돌 커널 지수 $\alpha \in [0, 1]$를 조건부 임베딩으로 전달하여 단일 신경망으로 모든 커널을 포괄합니다.
- **물리 보존 법칙 강제**:
  - 충돌 과정에서 질량, 운동량, 운동 에너지가 보존되어야 하는 성질($\int Q(f) \phi(v) dv = 0$)을 보존 손실 함수로 학습에 반영합니다.
- **독립 패키지 (`core`) 내장**: 외부 종속성 없이 자체 `core` 패키지를 통해 실행 가능합니다.

---

## 2. 디렉터리 구조

```
parameterized_neural_collision/
├── ablation__config.py             # 기본 PRFNO 모델 어블레이션 실험 파라미터 설정
├── ablation__config_variant1.py    # 아키텍처 Variant 1 실험 설정
├── ablation__config_variant2.py    # 아키텍처 Variant 2 실험 설정
├── main__prfno.py                  # PRFNO 학습 자동화 실행 진입점
├── main__prfno_variant1.py         # PRFNO Variant 1 학습 자동화 실행 진입점
├── main__prfno_variant2.py         # PRFNO Variant 2 학습 자동화 실행 진입점
├── __main__prfno__enhanced.py      # PRFNO Enhanced 모델 실행 진입점
├── runfile.sh                      # 멀티 GPU 백그라운드 학습 스크립트
├── core/                           # 신경망 레이어, 충돌 연산자, 수치 솔버 및 자동미분 패키지
├── data_generator/                 # 참값 $f$와 충돌항 $Q(f)$ 동시 생성을 위한 수치 생성기
├── step1__train_operators/         # Step 1: 충돌 연산자 $Q_\theta$ 학습 스크립트 모음
├── step1__train_operators__variant1/ # Variant 1 충돌 연산자 학습 모듈
├── step1__train_operators__variant2/ # Variant 2 충돌 연산자 학습 모듈
├── step1__test_operators/          # Step 1 평가: $Q_\theta$ + RK4 결합 시계열 롤아웃 검증 노트북
├── poisson/                        # 분리형 신경망(SPINN) 기반 2D Poisson 방정식 확장 실험
└── pretrained/                     # 사전 학습된 2D/3D PRFNO 모델 가중치 모음
```

---

## 3. 시작하기 (Quick Start)

### (1) 데이터셋 생성
충돌항 $Q(f)$와 궤적 $f(t)$를 포함하는 데이터셋을 생성합니다:
```bash
cd data_generator
python Boltzmann__bimaxwellian.py --config configs/config_generation__train_2D_032.yaml
```

### (2) 충돌 연산자 모델 학습
프로젝트 루트에서 메인 학습 스크립트를 실행합니다:
```bash
# PRFNO 모델 학습 실행 (예: CUDA 0번 장치)
python main__prfno.py --cuda_index 0

# 백그라운드 일괄 실행
bash runfile.sh
```

### (3) $Q_\theta$ + RK4 결합 시계열 추론 및 평가
`step1__test_operators/` 디렉터리의 노트북을 통해 학습된 충돌 연산자와 RK4 적분기를 결합하여 고속 스펙트럴 참값 해와 비교합니다:
- `step1__test_operators/test_main__2d.ipynb`: 2D 단일 및 다중 궤적 롤아웃 오차 검증
- `step1__test_operators/test_main__3d.ipynb`: 3D 대규모 텐서 롤아웃 오차 검증

---

## 4. 파일 입출력 및 환경 안내
- 모든 데이터 로더 및 모델 저장 경로는 프로젝트 루트(`_ROOT`)를 기준으로 자동 해석되어 실행 경로에 구애받지 않습니다.
- 사전 학습 가중치가 필요할 경우 `pretrained/PRFNO_2D/` 및 `pretrained/PRFNO_3D/` 디렉터리의 `model_and_history.pth`를 활용할 수 있습니다.