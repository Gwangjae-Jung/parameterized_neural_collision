# 데이터 생성 보조 모듈 (`data_generator/utils`)

본 디렉터리는 데이터 생성 파이프라인에서 호출되는 설정 로더와 속도 공간 샘플링 함수를 포함하고 있습니다.

---

## 1. 모듈 구성

- **`config_loader.py` (`ConfigLoader`)**:
  - `configs/` 디렉터리의 YAML 설정 파일을 파싱하여 텐서 연산 타입(`dtype`), 계산 디바이스(`device`), 속도 격자 정보, 적분 스텝 및 저장소 경로 객체를 제공합니다.
  - 저장 경로(`storage`)가 주어질 때 프로젝트 루트 디렉터리와 결합하여 폴더가 없을 경우 자동 생성합니다.
- **`sampling.py`**:
  - Bi-Maxwellian 분포 구성을 위한 평균 속도 벡터($u_1, u_2$)와 온도($T_1, T_2$)를 설정된 섭동 비율(`v_perturb_rate`, `t_perturb_rate`)에 따라 무작위 생성합니다.
- **`normalizer.py`**:
  - 이산화된 속도 격자의 미소 부피($\Delta V$)를 고려하여 총 질량이 1이 되도록 분포 $f$를 수치 정규화합니다.
