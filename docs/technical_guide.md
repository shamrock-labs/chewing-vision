# chewing-vision 기술 가이드

> 팀원을 위한 실행 가이드. 기본 구조(데이터 흐름, 파일 스키마)는 `ml/PIPELINE_OVERVIEW.md`를 먼저 읽을 것.  
> 이 문서는 **vision 신호 파이프라인**과 **현재 실험 현황**에 집중한다.

---

## 목차

1. [Vision 신호 파이프라인](#1-vision-신호-파이프라인)
2. [주요 파라미터](#2-주요-파라미터)
3. [도구별 실행 방법](#3-도구별-실행-방법)
4. [현재 성능 수치 (LOSO)](#4-현재-성능-수치-loso)
5. [미결 사항](#5-미결-사항)

---

## 1. Vision 신호 파이프라인

### 왜 신호가 두 개인가

MediaPipe는 얼굴에서 두 가지 다른 성격의 데이터를 뽑아준다.

| 신호 | 출처 | 특성 |
|------|------|------|
| **MAR** (Mouth Aspect Ratio) | 랜드마크 좌표로 직접 계산 | 기하학적 측정. 미세 씹기에 민감하지만 고개 움직임에도 반응 |
| **jaw_open** | MediaPipe 블렌드쉐이프 | 신경망 출력. 머리 자세 보정이 내장되어 있지만 미세 씹기를 놓치는 경향 |

```
MAR    = (위 입술 ~ 아래 입술 거리) / (입 좌우 너비)
           → 픽셀 비율이라 얼굴 크기 무관
           
jaw_open = MediaPipe가 "턱이 얼마나 열렸나"를 0~1로 추정한 값
           → 고개를 돌려도 상대적으로 안정적
```

### 세 가지 signal_mode

`OursEngine`은 `signal_mode` 파라미터로 어떤 신호를 기준으로 씹기를 감지할지 선택한다.

```
"jaw_open"  → jaw_open 신호만 사용
"mar"       → MAR 신호만 사용
"composite" → mar_weight × norm(MAR) + (1 - mar_weight) × norm(jaw_open)
```

**composite**는 두 신호를 정규화해서 섞는다. `mar_weight=0.7`이면 MAR이 70%, jaw_open이 30%.

### motion gate — 고개 움직임 억제

composite/mar 모드에서만 적용되는 게이트. 프레임 간 코 위치 변화가 얼굴 높이의 3% 이상이면 해당 프레임의 신호를 약화시킨다.

```python
gate = _motion_gate(frames, threshold=0.03)
# 코 변위 / 얼굴 bbox 높이 > 0.03 → 신호 감쇠
```

고개를 크게 끄덕이거나 돌릴 때 발생하는 false positive를 줄이기 위한 장치다.

### 전체 흐름

```
영상 프레임
    │ MediaPipe
    ▼
MAR, jaw_open (raw, 30fps)
    │ Savitzky-Golay 스무딩
    ▼
smoothed 신호
    │ motion gate (composite/mar에만)
    ▼
primary signal 선택 (signal_mode에 따라)
    │ peak detection (0.8~2.5 Hz 범위)
    ▼
1초 윈도우 라벨링
  chewing event ≥ 1 → "chewing"
  chewing event = 0 → "rest"
  face quality < 0.5 → "bad_face"
    │
    ▼
labels_ours.csv (GT 라벨)
```

---

## 2. 주요 파라미터

| 파라미터 | 위치 | 현재값 | 의미 |
|---------|------|--------|------|
| `signal_mode` | `OursEngine.__init__` | 실험 중 | 씹기 감지 신호 선택 |
| `mar_weight` | `OursEngine.__init__` | `0.7` | composite에서 MAR 비중 |
| `WINDOW_SEC` | `ours.py:48` | `1.0` | GT 라벨 1개의 길이 (초) |
| motion gate threshold | `_motion_gate()` | `0.03` | 코 변위 / 얼굴 높이. 이 이상이면 신호 감쇠 |
| LOSO window_sec | `compare_sessions.py` | `2.0` | IMU 피처 추출 윈도우 |
| LOSO stride | `compare_sessions.py` | `0.5` | IMU 슬라이딩 윈도우 stride |

### 현재 실험 상태

```
labels_ours.csv 현재: composite 모드 (mar_weight=0.7)로 재분석한 상태

비교 결과:
  jaw_open 모드    → LOSO Pooled F1-chew = 0.763
  composite w=0.7  → LOSO Pooled F1-chew = 0.738  ← 현재 labels_ours.csv 기준
  composite w=0.3  → 신호 비교만 진행 (LOSO 미실행)
```

---

## 3. 도구별 실행 방법

> 모든 명령은 `chewing-vision/` 루트에서 실행.

### GT 라벨 재생성 (batch_reanalyze.py)

모든 세션의 `labels_ours.csv`를 재생성할 때 사용.

```bash
# composite w=0.7로 전체 재분석 (현재 설정)
python ml/batch_reanalyze.py

# jaw_open 모드로 변경하려면 batch_reanalyze.py 안에서:
# ENGINE = OursEngine(signal_mode="jaw_open")
```

출력: 각 세션 폴더의 `labels_ours.csv` 덮어쓰기

---

### 두 신호 비교 (compare_signals.py)

jaw_open vs composite를 한 세션에서 나란히 비교.

```bash
python ml/compare_signals.py sessions/20260515T125354_v1kr2w --mar-weight 0.3

# 출력: 씹기 횟수, 분당 횟수, chew 비율 비교표 + 비교 영상
```

```
Metric                    jaw_open   composite(w=0.3)
n_chews                        48             56
chews_per_min                27.9           32.5
chew_pct (%)                 38.5           45.2
```

---

### 불일치 구간 수동 라벨링 (annotate.py)

jaw_open과 composite가 다르게 판정한 윈도우만 뽑아서 직접 라벨링.  
어떤 엔진이 더 정확한지 검증하는 도구.

```bash
# 불일치 윈도우만 라벨링 (권장)
python ml/annotate.py sessions/20260515T125354_v1kr2w

# 모든 윈도우 라벨링
python ml/annotate.py sessions/20260515T125354_v1kr2w --all
```

키 조작:

| 키 | 동작 |
|----|------|
| `c` | chewing 라벨 부여 + 다음으로 이동 |
| `r` | rest 라벨 부여 + 다음으로 이동 |
| `b` | bad_face 라벨 부여 + 다음으로 이동 |
| `a` / `d` | 이전 / 다음 윈도우 (라벨 없이 이동) |
| `s` | 중간 저장 |
| `q` | 저장 후 종료 + 정확도 리포트 출력 |

종료 후 자동으로 정확도 비교 리포트가 출력된다:
```
jaw_open  matches human GT : X/N  (XX%)
composite matches human GT : X/N  (XX%)
```

> **BLIND 모드**: 라벨링 중 엔진 판정이 화면에 표시되지 않는다. 의도적인 설계 —  
> 영상만 보고 판단해야 엔진에 끌려가지 않는 공정한 비교가 된다.

---

### LOSO 교차 검증 (compare_sessions.py)

세션 하나씩 빼고 나머지로 학습 → 뺀 세션으로 예측.  
IMU 피처 → vision GT 라벨의 정렬 정도를 측정.

```bash
python ml/compare_sessions.py -o ml/outputs/loso_결과폴더/

# HTML 리포트 열기
open ml/outputs/loso_결과폴더/session_comparison.html
```

> **LOSO가 측정하는 것**: "vision 라벨이 얼마나 정확한가"가 아니라  
> "IMU 신호가 vision 라벨과 얼마나 잘 매핑되는가".  
> vision 라벨이 더 세밀해져도 IMU가 그 차이를 감지 못하면 LOSO가 내려갈 수 있다.

---

### 단일 세션 분석

```bash
python -m chewing.cli analyze \
  sessions/20260515T125354_v1kr2w/video_20260515T125354_v1kr2w.mp4 \
  -o /tmp/chewing_v1kr2w
```

---

## 4. 현재 성능 수치 (LOSO)

**6개 세션, LOSO 교차 검증 기준** (2026-05-15)

| 세션 (held out) | F1-chew | F1-rest | 비고 |
|----------------|---------|---------|------|
| 838cje | 0.827 | 0.489 | |
| a61a4c | 0.778 | 0.324 | |
| hz0mma | **0.936** | 0.610 | |
| n1xetu | 0.632 | 0.577 | chewing 비율 낮음 (39.7%) |
| uj2e92 | 0.730 | 0.414 | |
| v1kr2w | 0.255 | 0.485 | composite 추가 후 chewing% 급증 → 노이즈 의심 |
| **Pooled** | **0.738** | 0.487 | composite 라벨 기준 |

참고: jaw_open 라벨 기준 Pooled F1-chew = **0.763** (composite 전환 후 -0.025 하락)

### v1kr2w 세션이 낮은 이유

composite 모드에서 v1kr2w의 chewing 비율이 33.7% → 65.3%로 거의 두 배 뛰었다.

- 가능성 A: composite가 jaw_open이 놓친 **진짜 미세 씹기**를 잡은 것
- 가능성 B: 말하기/삼키기/고개 움직임이 씹기로 **잘못 분류**된 것

`annotate.py`로 수동 라벨링해야 결론을 낼 수 있다.

---

## 5. 미결 사항

### composite vs jaw_open 결정 대기 중

`labels_ours.csv`는 현재 composite(w=0.7)이지만 LOSO가 더 낮다.  
`annotate.py` 수동 라벨링 결과로 결정:

- composite가 사람 판단과 더 많이 일치 → composite 유지
- jaw_open이 더 많이 일치 → jaw_open으로 롤백

jaw_open으로 롤백하는 방법:
```bash
# batch_reanalyze.py 안에서
# ENGINE = OursEngine(signal_mode="jaw_open")
python ml/batch_reanalyze.py
python ml/compare_sessions.py -o ml/outputs/loso_jaw_open/
```

### 세션 수집 목표

- 현재: 6개 유효 세션
- 목표: 20개 이상
- 세션당 2~3분, 음식 종류·시간대·고개 각도 다양하게
- chewing 비율 50% 근처 유지 (의도적으로 쉬는 구간 포함)
