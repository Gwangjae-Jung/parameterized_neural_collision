# 코어 라이브러리 패키지 (`core`)

본 패키지는 외부 수치해석 패키지(`deep_numerical`)에 대한 의존성 없이 프로젝트가 자체적으로 완전하게 동작하도록 구축된 수치해석, 자동미분, 딥러닝 코어 라이브러리입니다.

---

## 1. 패키지 아키텍처

```
core/
├── __init__.py                 # EINSUM_STRING, repeat, ones, zeros 등
├── autograd/                   # 고계 도함수 및 벡터화 미분 모듈 (SPINN 연계)
│   ├── grad.py                 # compute_grad (torch.autograd.grad 기반)
│   ├── vmap.py                 # derivatives, jacobian, hessian (torch.func 기반)
│   └── __init__.py
├── fft/                        # 푸리에 주파수 그리드 생성 및 FFT 연산
├── neural/                     # 신경망 아키텍처 모듈
│   ├── collision_op/           # PFNO, PRFNO, ConservativeFNO, ConservativeRFNO
│   ├── layer/                  # MLP, RadialFourierLayer, RadialSpectralConv
│   ├── network/                # SeparableNet (차원 분리형 신경망)
│   ├── neural_collision.py     # PFNO/PRFNO 레거시 import 호환 인터페이스
│   ├── operator/               # Fourier Neural Operator (FNO, RadialFNO)
│   ├── utils/                  # Activations, count_parameters, initialize_weights
│   └── __init__.py             # BaseModule 추상 클래스
├── numerical/                  # 키네틱 방정식 수치해석 모듈
│   ├── distribution/           # 맥스웰 평형 분포 및 유체역학적 모멘트 연산
│   ├── integrate/              # 가우스-르장드르 수치적분
│   ├── math/                   # area_of_unit_sphere, sinc, phase
│   ├── solver/                 # FastSM_Boltzmann_VHS, FastSM_Landau_VHS, one_step_RK4_classic
│   └── __init__.py
└── utils/                      # velocity_grid, relative_error, type_as_real, get_time_str
```

---

## 2. 주요 모듈 및 임포트 예시

### (1) 신경망 및 충돌 연산자
```python
from core.neural.collision_op import PRFNO
from core.neural.network import SeparableNet
from core.neural.layer import RadialFourierLayer, MLP
from core.neural.utils import count_parameters, initialize_weights
```

### (2) 수치 솔버 및 물리량 계산
```python
from core.numerical.solver import FastSM_Boltzmann_VHS, one_step_RK4_classic, LAMBDA
from core.numerical.math import area_of_unit_sphere
from core.numerical.distribution import maxwellian_homogeneous, compute_moments_homogeneous
```

### (3) 자동미분 및 유틸리티
```python
from core.autograd import compute_grad, derivatives
from core.utils import velocity_grid, relative_error, type_as_real, get_time_str
```
