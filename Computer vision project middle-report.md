# CNN 기반 차량·비차량 이미지 분류 및 오분류 분석

> 컴퓨터 비전 기말 프로젝트 진행 보고서

## 초록

본 프로젝트에서는 Kaggle Vehicle Detection Image Set을 사용해 이미지 전체를 `vehicles`와 `non-vehicles`로 분류하는 이진 이미지 분류 모델을 구현하고 비교했다. 객체의 위치를 찾는 객체 탐지가 아니라 이미지 단위 분류 문제로 설정했다.

원본 데이터 17,760장을 확인하고 클래스별 이미지 수, 이미지 크기, 손상 여부를 점검했다. 완전히 동일한 이미지 사본을 제외한 뒤 데이터 분할을 만들었다. 이후 perceptual hash 기반 유사 이미지 그룹을 구성해 유사한 이미지가 train, validation, test에 나뉘지 않도록 grouped split을 생성했다.

grouped split에서 Baseline CNN의 테스트 정확도는 97.99%, ImageNet 사전학습 ResNet18의 테스트 정확도는 99.89%였다. ResNet18은 테스트 이미지 2,634장 중 2,631장을 맞혔다. 오분류 사례에서는 비차량으로 라벨된 이미지에 차량이 포함된 것으로 보이는 예가 있어 라벨 오류 가능성을 확인했다.

## 1. 프로젝트 목표

차량 또는 비차량 이미지를 입력받아 이미지 전체를 두 클래스 중 하나로 분류하는 모델을 구현한다. 직접 구성한 Baseline CNN과 ImageNet 사전학습 ResNet18을 같은 데이터 분할에서 비교한다. Accuracy, Precision, Recall, F1-score, Confusion Matrix를 이용해 성능을 평가하고 오분류 사례를 분석한다.

## 2. 개발 환경

- 컴퓨터: Apple Silicon MacBook Pro
- Python: 3.12 기반 가상환경
- PyTorch: 2.14.1
- TorchVision: 0.29.1
- 연산 장치: Apple MPS
- 주요 라이브러리: Pillow, scikit-learn, Matplotlib, SciPy

## 3. 데이터셋 및 사전 점검

### 3.1 데이터 출처 및 구조

데이터셋은 [Kaggle Vehicle Detection Image Set](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set)에서 받았다. 압축 해제 후 확인한 구조는 다음과 같다.

```text
data/raw/vehicle_detection/data/
├── non-vehicles/
└── vehicles/
```

모든 이미지는 RGB PNG이며 크기는 64×64 픽셀이다. Pillow로 이미지를 검사한 결과 열리지 않는 손상 이미지는 발견되지 않았다.

| 클래스 | 원본 이미지 수 |
|---|---:|
| non-vehicles | 8,968 |
| vehicles | 8,792 |
| **합계** | **17,760** |

### 3.2 완전 중복 제거

SHA-256 해시로 파일 내용이 완전히 같은 이미지를 조사했다. 동일한 이미지가 두 번씩 있는 중복 그룹 189개를 찾았다. 원본 파일은 보존하고 분할 목록을 만들 때 중복 사본을 제외했다.

| 클래스 | 중복 제외 후 이미지 수 |
|---|---:|
| non-vehicles | 8,795 |
| vehicles | 8,776 |
| **합계** | **17,571** |

## 4. 데이터 분할 및 유사 이미지 점검

처음에는 완전 중복만 제외하고 클래스 비율을 유지하는 train/validation/test 분할을 만들었다. 이후 perceptual hash로 분할 간 유사 이미지 후보를 조사했다. 서로 다른 분할에 매우 비슷한 이미지가 존재했고, 예시 이미지도 시각적으로 거의 같은 차량 장면이었다. 따라서 초기 무작위 분할의 성능은 최종 성능으로 사용하지 않았다.

저주파 DCT 기반 perceptual hash의 해밍 거리가 4 이하인 이미지들을 연결 그룹으로 묶었다. 17,571장 중 2,295장이 포함된 665개 유사 이미지 그룹이 확인됐으며, 구성한 그룹 안에 서로 다른 클래스 라벨은 없었다. `StratifiedGroupKFold`로 그룹을 분할해 한 그룹이 여러 split에 흩어지지 않도록 했다.

| 분할 | 전체 | non-vehicles | vehicles |
|---|---:|---:|---:|
| Train | 12,303 | 6,158 | 6,145 |
| Validation | 2,634 | 1,319 | 1,315 |
| Test | 2,634 | 1,318 | 1,316 |
| **합계** | **17,571** | **8,795** | **8,776** |

최종 분할 목록은 `data/split_manifest_grouped.csv`에 저장했다. 기존 분할 목록은 초기 실험 비교를 위해 유지했다.

## 5. 모델과 학습 방법

### 5.1 Baseline CNN

Baseline CNN은 64×64 RGB 이미지를 입력으로 사용했다. 32, 64, 128개 채널의 합성곱 층을 순서대로 배치하고, 각 층 뒤에 ReLU와 Max Pooling을 적용했다. 마지막 특징은 Adaptive Average Pooling으로 축약한 뒤 Dropout과 2-class 선형 분류층을 통과시켰다.

- 손실 함수: CrossEntropyLoss
- 최적화 알고리즘: Adam, 학습률 0.001
- 배치 크기: 32
- 학습 횟수: 5 epochs
- 모델 선택: validation accuracy가 가장 높은 체크포인트
- Grouped split 최고 검증 정확도: 97.72%
- 체크포인트: `outputs/baseline_cnn_grouped_best.pth`

### 5.2 ResNet18 전이학습

TorchVision의 ImageNet 사전학습 ResNet18을 사용했다. 입력 이미지를 224×224로 조정하고 ImageNet 평균·표준편차로 정규화했다. 사전학습된 특징 추출부는 고정하고 최종 분류층을 두 클래스에 맞게 교체해 학습했다. 따라서 전체 네트워크를 fine-tuning한 실험이 아니라 고정 특징 추출 방식의 전이학습이다.

- 손실 함수: CrossEntropyLoss
- 최적화 알고리즘: Adam, 학습률 0.001
- 배치 크기: 32
- 학습 횟수: 5 epochs
- 모델 선택: validation accuracy가 가장 높은 체크포인트
- Grouped split 최고 검증 정확도: 99.92%
- 체크포인트: `outputs/resnet18_grouped_best.pth`

두 모델 모두 학습 데이터 증강은 적용하지 않았다.

## 6. Grouped split 테스트 결과

각 모델의 최고 검증 체크포인트를 테스트 세트에서 평가했다.

| 모델 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 오분류 수 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 97.99% | 97.99% | 97.99% | 97.99% | 53 |
| ResNet18 | 99.89% | 99.89% | 99.89% | 99.89% | 3 |

ResNet18은 Baseline CNN보다 테스트 정확도가 약 1.90%p 높았다.

### 6.1 클래스별 지표

| 모델 | 클래스 | Precision | Recall | F1-score | Support |
|---|---|---:|---:|---:|---:|
| Baseline CNN | non-vehicles | 0.9854 | 0.9742 | 0.9798 | 1,318 |
| Baseline CNN | vehicles | 0.9745 | 0.9856 | 0.9800 | 1,316 |
| ResNet18 | non-vehicles | 1.0000 | 0.9977 | 0.9989 | 1,318 |
| ResNet18 | vehicles | 0.9977 | 1.0000 | 0.9989 | 1,316 |

### 6.2 혼동행렬

행은 실제 라벨, 열은 예측 라벨이며 클래스 순서는 `non-vehicles`, `vehicles`이다.

**Baseline CNN**

```text
[[1284,  34],
 [  19, 1297]]
```

비차량 34장을 차량으로, 차량 19장을 비차량으로 분류했다.

**ResNet18**

```text
[[1315,   3],
 [   0, 1316]]
```

비차량 3장을 차량으로 분류했으며, 차량을 비차량으로 놓친 사례는 없었다.

## 7. 초기 분할 결과에 대한 주의

유사 이미지 그룹화 전의 무작위 분할에서는 ResNet18 테스트 정확도가 99.96%였다. 이후 시각적으로 매우 비슷한 이미지가 train, validation, test에 나뉘어 들어간 사례를 확인했다. 이 점수는 유사 이미지 누수의 영향을 받을 수 있어 예비 결과로만 취급하고, 최종 비교에는 grouped split 결과를 사용했다.

## 8. 오분류 분석

Grouped split에서 ResNet18이 오분류한 세 이미지는 모두 데이터셋에서 `non-vehicles`로 라벨되었지만 모델은 `vehicles`로 예측했다.

- `data/raw/vehicle_detection/data/non-vehicles/image608.png`
- `data/raw/vehicle_detection/data/non-vehicles/image161.png`
- `data/raw/vehicle_detection/data/non-vehicles/extra2738.png`

오분류 이미지 중 일부는 흐리거나 차량과 유사한 형태가 포함된 것으로 보였다. 특히 `extra2738.png`는 차량이 실제로 포함된 것처럼 보여 데이터 라벨 오류 가능성이 있다. 테스트 예측을 확인한 뒤 라벨을 변경하지 않았으며, 지표는 데이터셋의 원래 라벨을 기준으로 계산했다.

## 9. 논의 및 한계

1. **유사 이미지 그룹화 기준**: perceptual hash 거리가 4 이하인 이미지를 같은 연결 그룹으로 묶었다. 유사 이미지 누수를 줄이기 위한 보수적 점검 방법이지만, 임계값에 따라 실제 중복이 아닌 비슷한 장면도 같은 그룹으로 묶일 수 있다.
2. **라벨 품질**: 오분류 이미지 중 차량이 포함된 것처럼 보이는 `non-vehicles` 사례가 있었다. 데이터 라벨 오류가 평가 결과에 영향을 줄 가능성이 있다.
3. **제한된 실험 범위**: 학습은 5 epochs로 진행했다. ResNet18은 최종 분류층만 학습했으며, 전체 네트워크 fine-tuning, 데이터 증강, MobileNetV3 등의 추가 모델 비교는 수행하지 않았다.
4. **입력 해상도**: 원본 이미지가 64×64 픽셀이다. ResNet18 입력에 맞춰 224×224로 조정했지만 크기 변경이 원본에 없는 세부 정보를 복원하지는 않는다.
5. **일반화 범위**: 높은 테스트 성능은 해당 데이터셋과 분할 기준에 대한 결과다. 다른 카메라, 환경, 차량 종류에 대한 성능을 보장하지 않는다.

## 10. 결론

차량·비차량 이진 이미지 분류를 위해 Baseline CNN과 ImageNet 사전학습 ResNet18을 구현했다. 완전 중복을 제외하고 perceptual hash 기반 유사 이미지 그룹을 동일 분할에 배치해 데이터 누수 가능성을 줄였다.

Grouped split 테스트에서 Baseline CNN은 97.99%, ResNet18은 99.89%의 정확도를 기록했다. ResNet18이 더 높은 성능을 보였으며, 세 건의 오분류는 모두 비차량 라벨 이미지에 대한 차량 예측이었다. 그중 한 이미지에는 실제 차량이 포함된 것으로 보여 데이터셋의 라벨 품질을 추가 검토할 필요가 있다.

## 11. 후속 작업

- 오분류 이미지의 라벨을 예측과 독립적인 기준으로 재검토하되, 현재 테스트 지표는 원래 라벨 기준으로 보존한다.
- grouped split과 동일한 조건에서 MobileNetV3 등 추가 모델을 비교한다.
- 필요하면 데이터 증강과 ResNet18 fine-tuning 실험을 추가한다.
- 최종 발표와 보고서에는 grouped split 결과를 주 결과로 사용하고, 초기 무작위 분할 점수는 예비 결과로 구분한다.
