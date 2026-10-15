# 데이터 생성 설정 모음 (`data_generator/configs`)

본 디렉터리는 차원, 해상도 및 데이터 용도에 따른 다양한 데이터셋 생성 설정(YAML) 파일들을 제공합니다.

---

## 1. 설정 파일 목록

| 파일명 | 차원 | 해상도 | 용도 | 샘플 개수 및 특징 |
| :--- | :---: | :---: | :---: | :--- |
| `config_generation__train_2D_032.yaml` | 2D | 32 | Train | $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$, 10개 파트, 배치 60 |
| `config_generation__validation_2D_032.yaml` | 2D | 32 | Validation | 검증용 독립 3개 파트 |
| `config_generation__test_2D_064.yaml` | 2D | 64 | Test | 고해상도(64) 제로샷 일반화 테스트용 |
| `config_generation__test_2D_128.yaml` | 2D | 128 | Test | 초고해상도(128) 제로샷 일반화 테스트용 |
| `config_generation__train_3D_032.yaml` | 3D | 32 | Train | 3차원 속도 공간 학습용 |
| `config_generation__validation_3D_032.yaml` | 3D | 32 | Validation | 3차원 속도 공간 검증용 |
| `config_generation__test_3D_032.yaml` | 3D | 32 | Test | 3차원 기본 테스트용 |
| `config_generation__test_3D_064.yaml` | 3D | 64 | Test | 3차원 고해상도(64) 테스트용 |

---

## 2. 사용 팁
- 모든 설정의 `storage` 필드는 `datasets/train`, `datasets/validation`, `datasets/test`와 같이 상대 경로로 지정되어 있어 프로젝트 루트 내 `datasets/` 경로에 자동 분배되어 저장됩니다.
- GPU 메모리 한계에 도달할 경우 각 YAML 파일 내 `batch_size`를 줄여서 실행하시기 바랍니다.
