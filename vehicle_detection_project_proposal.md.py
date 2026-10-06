# 1. 프로젝트 개요

## 1.1. 기본 정보

- 프로젝트 제목 : CNN 기반 차량·비차량 이미지 분류 및 오분류 분석
- 작성일자 : 2026년 10월 6일
- 학번 / 이름 : [20252416 /조성관]

# 2. 서론 및 배경

## 2.1. 배경

- 문제 정의 : 차량 이미지와 비차량 이미지를 입력으로 받아 이미지 전체가 `Vehicle`인지 `Non-Vehicle`인지 분류하는 이진 이미지 분류 문제이다. 이미지 내 차량의 위치를 찾는 객체 탐지(Object Detection)와는 구분한다.
- 제안 배경 : CNN의 이미지 특징 추출과 분류 과정을 실습하고, 직접 구성한 CNN과 전이학습 모델을 같은 조건에서 비교한다. Accuracy에 더해 Precision, Recall, F1-score, Confusion Matrix 및 오분류 사례를 분석해 모델의 성능과 한계를 함께 평가한다.

## 2.2. 기여점

- 참고 프로젝트 주제 : Kaggle Vehicle Detection Image Set을 이용한 차량/비차량 이진 이미지 분류. 기존 공개 구현에서는 CNN 및 InceptionV3, Xception, MobileNet, VGG, EfficientNet 계열을 활용한 분류 사례가 보고되어 있다.
- 기존 프로젝트와의 차별성 : 하나의 정확도 수치에 의존하지 않고 Baseline CNN과 전이학습 모델(ResNet18, MobileNetV3, EfficientNet-B0 후보)을 다중 지표로 비교한다. 64×64와 더 높은 입력 해상도, 데이터 증강 적용 여부를 검토하고, False Positive와 False Negative 이미지를 분석해 배경 의존성, 작은 차량, 가림, 해상도 저하 등 오류 원인을 구체화한다. 데이터 분할 전에 중복·유사 이미지에 따른 데이터 누수 가능성도 점검한다.

# 3. 상세 내용

## 3.1. 개발 목표 및 접근법

- 정량적/정성적 달성 목표 : 차량/비차량 분류 파이프라인과 Baseline CNN, 전이학습 모델을 구현한다. Accuracy, Precision, Recall, F1-score, Confusion Matrix를 산출하고 모델별 결과를 동일한 테스트 세트에서 비교한다. 성능 목표값은 실험 전 임의로 고정하지 않고 검증 결과와 자원 조건을 고려해 최종 모델을 선정한다. 정성적으로는 오분류 유형과 일반화 한계를 정리하고 재현 가능한 코드 및 결과 보고서를 제출한다.
- 핵심 알고리즘 및 접근법 : CNN의 합성곱층과 풀링층으로 이미지의 공간 특징을 추출한 뒤 이진 분류한다. 먼저 직접 구성한 CNN을 기준선으로 학습한다. 이후 ImageNet 등 대규모 데이터로 사전학습된 모델의 분류 계층을 두 클래스에 맞춰 교체하고 fine-tuning한다. 학습 데이터에만 회전, 좌우 반전, 밝기 변화, crop 등의 데이터 증강을 적용하고, 학습·검증 손실 및 정확도로 과적합을 확인한다. 후보 모델은 동일한 데이터 분할과 평가 절차를 기준으로 비교한다.

## 3.2. 시스템 아키텍처

- 파이프라인 : 이미지 데이터 수집 및 라벨 확인 → 중복·유사 이미지 점검 → 학습/검증/테스트 분할(클래스 비율 유지) → RGB 변환 및 크기 조정·정규화 → 학습 데이터 증강 → Baseline CNN 또는 전이학습 모델 학습 → 테스트 세트 예측 → 성능 지표·혼동행렬 산출 → 오분류 사례 시각화 및 분석 → 최종 결과 저장

```mermaid
flowchart LR
    A[Vehicle / Non-Vehicle 이미지] --> B[라벨 및 중복 점검]
    B --> C[Train / Validation / Test 분할]
    C --> D[RGB 변환·크기 조정·정규화]
    D --> E[Train 증강]
    E --> F{분류 모델}
    F --> G[Baseline CNN]
    F --> H[전이학습 모델]
    G --> I[클래스 예측 및 확률]
    H --> I
    I --> J[지표·혼동행렬 평가]
    I --> K[오분류 사례 분석]
    J --> L[최종 모델 및 보고서]
    K --> L
```

## 3.3. 입출력 인터페이스

- 입력 데이터 형태 : RGB 단일 이미지. 기본 실험 입력 크기는 64×64×3이며, 성능 비교를 위해 128×128 또는 224×224 입력도 검토한다. 원본 해상도는 이미지마다 다를 수 있으므로 모델 입력 전에 크기를 통일한다.
- 최종 출력 형태 : 이미지별 예측 클래스(`Vehicle` 또는 `Non-Vehicle`)와 두 클래스의 예측 확률. 평가 단계에서는 실제 라벨과 예측 라벨을 비교한 Accuracy, Precision, Recall, F1-score 및 Confusion Matrix를 출력한다. 바운딩 박스 좌표는 출력하지 않는다.

# 4. 개발 환경

## 4.1. 데이터셋

- 데이터셋 이름 : Vehicle Detection Image Set
- 데이터셋 출처 : Kaggle, Baris Dincer — https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set
- 데이터 규모 : 총 17,760장(차량 8,792장, 비차량 8,968장). 압축 파일 및 압축 해제 후 용량은 다운로드한 데이터 기준으로 측정해 기입한다.
- 데이터 특징 : RGB 컬러 이미지, 2개 클래스(`Vehicles`, `Non-Vehicles`)의 이진 라벨. 보고서에 기재된 기준 해상도는 64×64이며, 원본 이미지별 해상도와 파일 상태를 사전 확인한다. 클래스 수가 비교적 균형적이다.

## 4.2. 실행 환경

- 개발 언어 및 프레임워크 : Python 3.10 이상, PyTorch 또는 TensorFlow 중 하나를 선택
- 라이브러리 : TorchVision(또는 동등한 모델 라이브러리), NumPy, OpenCV, scikit-learn, Matplotlib
- 연산 환경 : Google Colab GPU 또는 CUDA 지원 GPU 환경을 기본 대안으로 사용한다. 권장 기준은 6~8코어 CPU, RAM 32GB, VRAM 8GB, SSD 여유 공간 20GB이며, 실제 요구량은 모델·입력 크기·배치 크기에 따라 조정한다. Apple Silicon 환경에서는 MPS 사용을 검토한다.

# 5. 수행 계획

## 5.1. 개발 일정

- 6~7주차 : 참고 프로젝트 분석, 제안서 작성 및 제출
- 8~9주차 : 데이터셋 확보·구조 확인, 중복 점검, 전처리 및 Baseline 코드 재현
- 10~13주차 : 모델 파이프라인 수정, CNN/전이학습 모델 비교 및 해상도·증강 실험
- 13~14주차 : 결과 분석, 오분류 사례 정리, 개선점 보완 및 기말 발표 자료 작성

```mermaid
gantt
    title Final Project Plan
    dateFormat  YYYY-MM-DD
    axisFormat  %m/%d

    section 1 설계
    참고 프로젝트 분석 : done, a1, 2026-09-21, 7d
    제안서 제출 : done, a2, 2026-09-28, 10d
    데이터셋 확보 및 베이스라인 코드 재현 : a3, 2026-10-12, 14d

    section 2 개발 및 실험
    모델 파이프라인 수정 및 실험 진행 : a4, 2026-11-02, 28d

    section 3 최종 보고
    결과 분석, 오류 보완 및 발표 자료 작성 : a5, 2026-11-23, 14d
```

# 6. 참고 문헌

- 논문 : 해당 프로젝트의 실험 설계와 직접 관련된 논문은 최종 모델 및 비교 범위를 확정한 뒤 추가한다.
- 오픈소스 저장소(Github URL) : https://github.com/Joseonggwan/Computer-vision-project
- 기타 데이터셋 출처 URL : Kaggle Vehicle Detection Image Set — https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set
- 참고 구현 및 데이터 설명 : Rational Matter, “Detecting road features” — https://navoshta.com/detecting-road-features/
