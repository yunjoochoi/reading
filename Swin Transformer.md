# Swin Transformer

Hierarchical Vision Transformer using Shifted Windows

Swin Transformer는 새로운 트랜스포머 기반 비전 백본(범용성있게 활용 가능)

트랜스포머를 비전에 활용시 어려움 (언어와 다른 특성)

- 시각적 객체의 다양한 크기
- 픽셀 단위의 고해상도

**→ Hierarchical Transformer** 구조와 **Shifted Windows** 방식 도입으로 해결

- 윈도우 기반한 국소 self-attention연산으로 효율성 높임
- shift 통해 윈도우 간 연결성 보완

→ 다양한 스케일의 표현 다룸, **이미지 크기에 대해 연산 복잡도가 선형**으로 유지

결과

- **이미지 분류(ImageNet-1K):** Top-1 정확도 87.3%
- **객체 탐지(COCO):** Box AP 58.7, Mask AP 51.1
- **의미론적 분할(ADE20K):** mIoU 53.5

이전 SOTA 대비 크게 향상

- COCO 기준: Box AP +2.7, Mask AP +2.6 개선
- ADE20K 기준: mIoU +3.2 개선

3. Method

![image.png](attachment:b5154914-58da-4c51-b2e2-6f59d271f77a:image.png)

- **Patch Splitting (패치 분할)**
    - 입력 이미지를 **겹치지 않는 작은 패치**로 나눔.
    - Swin-T에서 패치 크기는 **4×4** → 각 패치의 차원은 4×4×3=484 \times 4 \times 3 = 484×4×3=48.
    - 이 패치가 **Transformer의 토큰(token)** 역할을 하게 됨.
- **Linear Embedding (선형 임베딩)**
    - 각 48차원 패치를 FC Layer를 통해 CCC-차원으로 변환.
    - 이렇게 변환된 (H/4×W/4)(H/4 \times W/4)(H/4×W/4)개의 토큰이 Stage 1의 입력이 됨.
- **Stage 1**
    - Linear Embedding 뒤에 여러 개의 **Swin Transformer Block**을 적용.
    - 토큰 개수는 그대로 유지됨 (H/4×W/4)(H/4 \times W/4)(H/4×W/4).
    - → 이 단계까지가 **Stage 1**.
- **Patch Merging (패치 병합)**
    - 계층적 표현(hierarchical representation)을 위해 단계가 깊어질수록 토큰 개수를 줄임.
    - 2×2 인접 패치 4개를 묶어 concat → 차원은 4C.
    - Linear Layer를 적용해 2-차원으로 줄임.
    - 해상도는 절반으로 줄어들고 (H/8×W/8H/8 \times W/8H/8×W/8), 채널은 2배로 증가.
- **Stage 2 ~ Stage 4**
    - Stage 2: 해상도 H/8×W/8H/8 \times W/8H/8×W/8, 차원 2C.
    - Stage 3: 해상도 H/16×W/16H/16 \times W/16H/16×W/16, 차원 4C.
    - Stage 4: 해상도 H/32×W/32H/32 \times W/32H/32×W/32, 차원 8C.
    - 각 스테이지는 "Patch Merging + 여러 개 Swin Transformer Block"으로 구성.

## Linear Embedding 과정

1. **Patch Partition (패치 분할)**
    - 원본 이미지 (H×W×3)(H \times W \times 3)(H×W×3)를 **작은 패치**로 나눕니다.
    - 예: 4×44 \times 44×4 픽셀 단위로 나누면, 각 패치는 (4×4×3)=48(4 \times 4 \times 3) = 48(4×4×3)=48 차원 벡터가 됩니다.
    - 따라서 Patch Partition 결과는 H4×W4\frac{H}{4} \times \frac{W}{4}4H×4W 개의 벡터(각각 48차원)가 만들어집니다.
2. **Linear Embedding (리니어 변환)**
    - 각 48차원 패치 벡터를 **선형 변환 (fully connected layer, 즉 Dense Layer)** 을 통해 C차원으로 매핑합니다.
    - 즉, R48→RC\mathbb{R}^{48} \to \mathbb{R}^{C}R48→RC 로 변환.
    - 여기서 CCC는 모델의 hidden dimension 크기이며, Stage 1의 채널 수를 의미합니다.
3. **출력 형태**
    - 최종 출력은 H4×W4×C\frac{H}{4} \times \frac{W}{4} \times C4H×4W×C.
    - 즉, 이미지가 패치 단위로 쪼개져 C-차원 임베딩으로 변환된 형태가 됩니다.
    - 이후 이 임베딩이 Swin Transformer Block으로 들어갑니다.

## Swin Transformer Block의 두 단계

하나의 Swin Transformer Block은 Figure 3(b)처럼 **두 개의 연속된 Attention 모듈**로 구성

1. **W-MSA (Window-based Multi-Head Self Attention)**
    - 이미지를 non-overlapping 윈도우(예: 7×7)로 나눠서,
    - 각 윈도우 안에서만 self-attention 계산.
    - **윈도우끼리 독립**이므로 cross-window 정보는 없음.
2. **SW-MSA (Shifted Window MSA)**
    - 모든 윈도우를 **가로·세로로 절반(혹은 일부)만큼 시프트**해서 다시 나눔.
    - 이때 원래 다른 윈도우에 있던 토큰들이 새 윈도우 안에서 함께 attention을 계산.
    - 따라서 **윈도우 간 상호작용이 발생**.

## 정보 섞이는순간

- **W-MSA 출력 → Residual → SW-MSA 입력**
    
    즉, **Shifted Window MSA를 거치는 순간** 서로 다른 윈도우에 있던 정보가 attention을 통해 합쳐집니다.
    
- SW-MSA 이후에는 다시 residual connection으로 더해지고, 이어지는 MLP에서 feature가 정제됩니다.

## 왜 이렇게 설계?

- 처음부터 글로벌 self-attention을 하면 계산량이 O(N2)O(N^2)O(N2)로 너무 큼.
- 윈도우 제한(W-MSA)으로 연산량을 줄이고,
- 시프트(SW-MSA)로 윈도우 경계의 정보도 주고받을 수 있게 해서 CNN처럼 **local + neighbor receptive field 확장** 효과

## Patch Merging 과정

토큰 수를 줄이고 차원을 늘리는 다운샘플링 단계

1. **입력 상태**
    - 예를 들어 Stage 1의 출력 feature map 크기가
        
        (H/4,W/4,C)라고 합시다.
        
    - 여기서 C는 채널 수(embedding dimension)입니다.
2. **2×2 인접 패치 묶기**
    - 공간 해상도를 줄이기 위해, **2×2 이웃 패치들을 하나의 패치로 합칩니다**.
    - 따라서 토큰 개수는 1/4로 줄어듭니다.
    - 하지만 각 토큰 벡터의 차원은 4배로 늘어납니다.
        - 즉, 크기: (H/8,W/8,4C)
3. **Linear Projection**
    - 차원이 너무 커지므로, FC Layer(Linear Layer)를 통해 2C차원으로 축소합니다.
    - 최종 출력 크기: (H/8,W/8,2C)
