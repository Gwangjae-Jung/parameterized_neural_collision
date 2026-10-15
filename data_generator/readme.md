# 충돌항 데이터 생성 모듈 (`data_generator`)

본 디렉터리는 신경망 충돌 연산자($Q_\theta$)의 학습 및 테스트를 위해, 동차 볼츠만 방정식의 분포 함수 $f(t, v)$뿐만 아니라 **각 시점에서의 실제 충돌항 타겟 $Q(f(t, v))$를 고속 스펙트럴 솔버(FastSM)로 동시에 계산하여 저장**하는 모듈입니다.

---

## 1. 파일 및 디렉터리 구성

- **`Boltzmann__bimaxwellian.py`**:
  - Bi-Maxwellian 분포를 기반으로 초기 조건 $f_0$를 생성합니다.
  - 고속 스펙트럴 볼츠만 솔버(`FastSM_Boltzmann_VHS`)를 통해 각 시점 $t_k$에서:
    1. FFT 영역에서 충돌 적분 $Q(f)$를 연산하여 `target`으로 저장합니다.
    2. RK4 시간 적분을 1단계 수행하여 다음 시점의 분포 $f(t_{k+1})$를 구하여 `data`로 저장합니다.
  - 산출된 쌍 $(f, Q(f), \alpha)$을 텐서 딕셔너리로 묶어 `datasets/` 하위 경로에 저장합니다.
- **`configs/`**:
  - 차원(2D, 3D), 해상도(32, 64, 128), 용도(train, validation, test)별 YAML 설정 파일 모음입니다.
- **`kernel_modes/`**:
  - 스펙트럴 충돌 연산자 초기화 시 계산 비용이 큰 커널 모드 가중치 행렬을 캐시(`FSM_{DIM}D__res{RES}__coeff{COEFF}__exp_speed{ALPHA}.pth`)하여 다음 실행 시 재사용합니다.
- **`utils/`**:
  - YAML 설정 파서, 섭동 샘플링 및 밀도 정규화 도구 모음입니다.

---

## 2. 실행 방법

`configs/` 디렉터리에 준비된 설정 파일을 지정하여 실행합니다:

```bash
# 2D 학습용 데이터셋 생성 예시 (해상도 32)
python Boltzmann__bimaxwellian.py --config configs/config_generation__train_2D_032.yaml

# 2D 검증용 데이터셋 생성 예시
python Boltzmann__bimaxwellian.py --config configs/config_generation__validation_2D_032.yaml

# 3D 테스트용 데이터셋 생성 예시
python Boltzmann__bimaxwellian.py --config configs/config_generation__test_3D_064.yaml
```

---

## 3. 저장 텐서 구조 (`.pth`)

저장되는 파일(`alpha{ALPHA}__part{PART}.pth`)은 다음과 같은 키를 포함합니다:
- `'data'`: 시간 궤적에 따른 분포 함수 텐서 (형상: `(B, T, *V_grid, 1)`)
- `'target'`: 각 시점에서 연산된 참값 충돌항 $Q(f)$ 텐서 (형상: `(B, T, *V_grid, 1)`)
- `'vhs_alpha'`: 사용된 VHS 충돌 커널 지수 $\alpha$
- `'vhs_coeff'`: 볼츠만 충돌 연산자 계수
- `'delta_t'`, `'max_t'`, `'res_v'`, `'v_max'`: 계산 환경 파라미터
