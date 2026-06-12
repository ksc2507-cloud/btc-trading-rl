[README_A70040.md](https://github.com/user-attachments/files/28883538/README_A70040.md)
# 비트코인 트레이딩 환경에서 강화학습 알고리즘 성능 비교

REINFORCE와 DQN을 동일한 BTC 트레이딩 환경에서 학습. 단순 보유(Buy & Hold) 전략 대비 성능을 비교.

---

## 프로젝트 구조

```
├── Reinforcement_Learning_Project_A70040_F.ipynb   # 전체 실험 코드
└── README.md
```

---

## 실험 개요

**데이터**
- BTC-USD 일봉 (2020.01 ~ 2026.05, yfinance)
- Train : 2020.01 ~ 2024.12 / Test : 2025.01 ~ 2026.05

**환경 (MDP)**
- State : MA Ratio, RSI, 가격 변동률, 거래량 변동률, 포지션 (5차원)
- Action : Hold(0) / Buy(1) / Sell(2)
- Reward : 매도 시 실현수익률 + 보유 중 미실현 손익(×0.1) 반영

**알고리즘**
- REINFORCE : Monte Carlo Policy Gradient + 엔트로피 정규화 (coef=0.01)
- DQN : Replay Buffer + Target Network + ε-greedy (에피소드 단위 선형 감소)

---

## 최종 결과 (Test Set)

| 전략 | 평균 수익률 | 표준편차 |
|------|------------|---------:|
| REINFORCE | **+18.46%** | ±14.92% |
| DQN | -2.92% | ±3.59% |
| Buy & Hold | -21.89% | — |

> 테스트 구간(2025~2026)은 BTC 하락장. 두 알고리즘 모두 단순 보유 전략 상회.

---

## 실행 방법

Google Colab 기준.

**1. 노트북 열기**

Colab에 `Reinforcement_Learning_Project_A70040_F.ipynb` 업로드. 또는 GitHub에서 직접 열기.

**2. 패키지 설치**

첫 번째 셀 실행 시 자동 설치.

```bash
!pip install yfinance gymnasium torch matplotlib -q
```

**3. Google Drive 연결**

셀 실행 시 Drive 마운트 권한 허용. 저장 경로는 아래와 같음.

```python
SAVE_DIR = '/content/drive/MyDrive/RL_Project/'
```

경로 변경 시 `SAVE_DIR` 변수만 수정.

**4. 순서대로 실행**

셀을 위에서부터 순서대로 실행. 전체 학습(seed 3개 × 2 알고리즘) 소요 시간은 GPU 기준 약 20~30분.

---

## 학습된 모델 다운로드

> **[Google Drive에서 다운로드](https://drive.google.com/drive/folders/1q0eqv2MAxxy1wVy4PRJ1nO41EuxBSKu4?usp=drive_link)**

```
RL_Project/
├── reinforce_seed0.pth
├── reinforce_seed1.pth
├── reinforce_seed2.pth
├── dqn_seed0.pth
├── dqn_seed1.pth
├── dqn_seed2.pth
└── full_experiment.pkl   # 전체 실험 결과 (학습 이력 포함)
```

모델 로드 후 평가만 실행하려면 노트북 **비교 분석 및 시각화** 섹션의 `evaluate_reinforce` / `evaluate_dqn` 함수 사용.

---

## 주요 하이퍼파라미터

| 파라미터 | REINFORCE | DQN |
|---------|:---------:|:---:|
| Episodes | 300 | 300 |
| Learning Rate | 1e-3 | 1e-3 |
| Gamma (γ) | 0.99 | 0.99 |
| Hidden Dim | 64 | 64 |
| Entropy Coef | 0.01 | — |
| Batch Size | — | 64 |
| Buffer Size | — | 10,000 |
| Target Update | — | 10 ep마다 |
| ε (start→end) | — | 1.0 → 0.05 |

---

## 의존성

```
Python 3.10+
torch
gymnasium
yfinance
numpy
pandas
matplotlib
```
