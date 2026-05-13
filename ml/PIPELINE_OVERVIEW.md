# AirPods IMU 씹기 감지 ML 파이프라인 개요

> 처음 보는 분을 위한 전체 흐름 + 데이터 보는 법 안내서

---

## 목차

1. [프로젝트가 하려는 것](#1-프로젝트가-하려는-것)
2. [시스템 구성 — 두 개의 레포](#2-시스템-구성--두-개의-레포)
3. [전체 데이터 흐름](#3-전체-데이터-흐름)
4. [수집 데이터 구조](#4-수집-데이터-구조)
5. [ML 파이프라인 상세](#5-ml-파이프라인-상세)
6. [분석 결과 보는 법](#6-분석-결과-보는-법)
7. [현재 성능 요약](#7-현재-성능-요약)
8. [한계 및 다음 단계](#8-한계-및-다음-단계)

---

## 1. 프로젝트가 하려는 것

**AirPods를 귀에 꽂기만 해도 씹기(chewing)를 자동으로 감지한다.**

AirPods Pro에는 6축 IMU(관성 측정 장치)가 내장되어 있다. 씹을 때 턱 근육이 움직이면 귀에 걸린 AirPods가 미세하게 진동한다. 이 진동 패턴을 머신러닝으로 분류하면, 카메라 없이도 씹기 횟수와 구간을 실시간으로 추적할 수 있다.

**최종 목표**: iPhone 앱에서 실시간으로 씹기 감지 → 씹기 횟수 표시 → 식습관 피드백

---

## 2. 시스템 구성 — 두 개의 레포

```
chewing-vision/                  chewing-imu-collector/
(Python, macOS)                  (Swift, iOS)
        │                                │
  영상 분석 + GT 라벨 생성         AirPods IMU 수집 + 실시간 추론
  ML 학습 + 모델 내보내기    →     ChewingClassifier.mlmodel 탑재
```

| 레포 | 역할 | 주요 기술 |
|------|------|----------|
| `chewing-vision` | 영상에서 입 움직임 추출 → GT 라벨 생성, ML 학습 | Python, MediaPipe, scikit-learn, Core ML |
| `chewing-imu-collector` | AirPods IMU 수집, 실시간 씹기 예측, 세션 저장 | Swift, CoreML, AVFoundation |

---

## 3. 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│  1. 데이터 수집 (iOS 앱)                                      │
│                                                               │
│  iPhone 카메라 ─────────────────────────────► video_*.mp4    │
│  AirPods IMU (50Hz, 6축) ───────────────────► imu_*.csv      │
│  타임스탬프 앵커 (mach clock) ─────────────► session_*.json  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. GT 라벨 생성 (chewing-vision CLI)                         │
│                                                               │
│  video_*.mp4                                                  │
│      │ MediaPipe FaceLandmarker                               │
│      │ MAR(입 열림 비율) 추출                                   │
│      ▼                                                        │
│  labels_ours.csv  ← 1초 단위 chewing / rest 라벨             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ML 학습 (ml/pipeline.py 또는 compare_sessions.py)         │
│                                                               │
│  imu_*.csv + labels_ours.csv                                  │
│      │ 타임스탬프 정렬 (session.json 기준)                      │
│      │ 슬라이딩 윈도우 (2초 / 0.5초 stride)                     │
│      │ 12차원 피처 추출 (6축 × RMS + std)                      │
│      │ Random Forest (n=20) 학습                               │
│      ▼                                                        │
│  ChewingClassifier.mlmodel  ← iOS 앱에 탑재                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 실시간 추론 (iOS 앱)                                       │
│                                                               │
│  AirPods IMU 스트림 (50Hz)                                    │
│      │ 100샘플 버퍼 (2초) / 25샘플마다(0.5초) 예측              │
│      ▼                                                        │
│  chewing / rest 실시간 표시 + 씹기 횟수 카운팅                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 수집 데이터 구조

### 파일 위치

```
chewing_collector_data/
├── video_20260513T145413_a61a4c.mp4      ← 영상 (30fps)
├── imu_20260513T145413_a61a4c.csv        ← IMU 원시 데이터
└── session_20260513T145413_a61a4c.json   ← 타임스탬프 앵커
```

파일명 형식: `{type}_{YYYYMMDDTHHMMSS}_{6자리 세션ID}.{ext}`

---

### imu_*.csv — IMU 원시 데이터

```
t_mach, t_rel_sec, attitude_roll, attitude_pitch, attitude_yaw,
rotation_x, rotation_y, rotation_z,
gravity_x, gravity_y, gravity_z,
user_accel_x, user_accel_y, user_accel_z
```

| 컬럼 | 단위 | 설명 |
|------|------|------|
| `t_rel_sec` | 초 | 녹화 시작 기준 상대 시간 |
| `rotation_x/y/z` | rad/s | 각속도 (자이로) — 씹기 신호 핵심 |
| `user_accel_x/y/z` | g | 중력 제거된 선형 가속도 |
| `attitude_roll/pitch/yaw` | rad | 자세각 (ML에서 미사용 — 자세 편향 큼) |
| `gravity_x/y/z` | g | 중력 벡터 (ML에서 미사용) |

샘플링 주파수: **50Hz** (20ms 간격)

---

### session_*.json — 타임스탬프 앵커

```json
{
  "clock": {
    "baseline_mach":        1234567890,
    "mach_timebase_numer":  125,
    "mach_timebase_denom":  3
  },
  "video": {
    "start_mach": 1234567891
  }
}
```

IMU의 `t_rel_sec`와 영상 GT 라벨의 시각(t_sec)을 정렬하기 위한 기준점.  
`video_start_sec = (video_start_mach - baseline_mach) × numer/denom / 1e9`  
`imu.t_vision = imu.t_rel_sec + video_start_sec`

---

### labels_ours.csv — GT 라벨 (chewing-vision 생성)

```
t_start, t_end, label, quality
0.0,     1.0,   rest,     0.95
1.0,     2.0,   chewing,  0.98
2.0,     3.0,   chewing,  0.97
...
```

| 컬럼 | 설명 |
|------|------|
| `t_start / t_end` | 1초 단위 구간 (영상 시작 기준) |
| `label` | `chewing` 또는 `rest` |
| `quality` | 얼굴 감지 품질 (0~1, 낮으면 신뢰도 낮음) |

MediaPipe가 얼굴을 감지하지 못한 구간은 라벨에서 제외됨.

---

## 5. ML 파이프라인 상세

### 피처 추출 — 12차원

슬라이딩 윈도우(2초, stride 0.5초) 안의 샘플들로 통계량 계산:

```
6개 축  ×  2개 통계량  =  12 features

rotation_x_rms,  rotation_x_std
rotation_y_rms,  rotation_y_std
rotation_z_rms,  rotation_z_std
user_accel_x_rms, user_accel_x_std
user_accel_y_rms, user_accel_y_std
user_accel_z_rms, user_accel_z_std
```

- **RMS**: 신호 에너지 (씹을 때 회전이 강해짐)
- **std**: 리듬 안정성 (규칙적인 씹기 패턴 포착)

### 라벨 부여

윈도우 [t_start, t_start+2s) 구간 내 GT 라벨 중 chewing 비율 ≥ 50% → **chewing(1)**, 아니면 **rest(0)**

### 모델

| 항목 | 값 |
|------|-----|
| 알고리즘 | Random Forest |
| 트리 수 | 20 |
| 평가 방식 | time-based 80/20 split (앞 80% 학습, 뒤 20% 평가) |
| 출력 | chewing(1) / rest(0) + 클래스 확률 |
| 모델 크기 | ~20KB (.mlmodel) |

### 씹기 횟수 추정

ML 예측 결과에서 연속된 chewing 윈도우를 **bout(구간)**으로 묶고 씹기 수를 추정:

```
gap ≤ 1.0초인 chewing 윈도우들 → 하나의 bout
bout 씹기 수 = bout 길이(초) × 1.2 Hz  (평균 씹기 주파수 기반)
```

---

## 6. 분석 결과 보는 법

### 세션 비교 리포트 열기

```bash
cd chewing-vision
open ml/session_comparison.html
```

HTML 파일 구성:
- **성능 비교 테이블**: 세션별 길이, 씹기 비율, 추정 씹기 수, F1 점수
- **시각화 이미지**: 4행 구성 (아래 설명)
- **Top-3 중요 피처**: 세션별 RF feature importance
- **어떻게 만들어졌나**: 파이프라인 재현 방법

---

### 시각화 이미지 (session_comparison.png) 읽는 법

```
Row 1: F1 바차트 (녹=chewing, 회=rest) | 씹기 비율 | 추정 씹기 횟수
       ─ 빨간 점선 = F1 기준치 0.70

Row 2: IMU rotation_y 신호 그래프 (세션별)
       ─ 파란 선  = AirPods 각속도 원시 신호
       ─ 녹색 배경 = GT에서 chewing으로 라벨된 구간
       ─ 하단 10% 컬러 스트립 = ML이 예측한 chewing(녹) / rest(회) 구간

Row 3: GT vs 예측 타임라인 (test set, 세션별)
       ─ 위 줄 = GT 라벨 / 아래 줄 = ML 예측
       ─ 두 줄이 일치할수록 F1이 높음

Row 4: Confusion Matrix (세션별 + 합산)
       ─ 대각선 숫자가 클수록 정확도 높음
```

---

### demo 영상 보기 (세션별)

```bash
open /tmp/chewing_uj2e92_demo/demo.mp4    # 영상 위에 씹기 감지 오버레이
open /tmp/chewing_uj2e92_demo/signals.png  # 신호 그래프 + 라벨 타임라인
```

demo 출력 파일 목록:

| 파일 | 내용 |
|------|------|
| `demo.mp4` | 원본 영상 + 씹기 감지 결과 오버레이 |
| `signals.png` | MAR 신호, 라벨, bout 시각화 |
| `labels_ours.csv` | 1초 단위 chewing/rest GT 라벨 |
| `bouts_ours.csv` | 씹기 구간(bout) 목록 |
| `summary.json` | 씹기 횟수, 분당 횟수, 엔진 비교 |

---

### 비교 스크립트 직접 실행

```bash
cd chewing-vision
.venv/bin/python ml/compare_sessions.py
```

전제 조건: 각 세션의 GT 라벨이 `/tmp/chewing_{id}/labels_ours.csv`에 있어야 함.  
없으면 먼저 아래 실행:

```bash
.venv/bin/python -m chewing.cli analyze \
  chewing_collector_data/video_20260513T145413_a61a4c.mp4 \
  -o /tmp/chewing_a61a4c
```

---

## 7. 현재 성능 요약

| 세션 | 길이 | 추정 씹기 수 | F1 (chewing) | 비고 |
|------|------|------------|--------------|------|
| a61a4c | 145초 | 127회 | **0.800** ✅ | 앱 배포 모델 학습 기반 |
| 838cje | 91초 | 74회 | **0.871** ✅ | |
| uj2e92 | 71초 | 52회 | 0.286 ❌ | 세션 짧아 test set 28개 |
| **합산 (3세션)** | — | — | **0.804** ✅ | |

앱에 탑재된 `ChewingClassifier.mlmodel`은 현재 **a61a4c 1세션** 기준으로 학습됨.

---

## 8. 한계 및 다음 단계

### 현재 한계

| 항목 | 현황 | 이유 |
|------|------|------|
| 학습 데이터 | 3세션 (1인) | 데이터 수집 초기 단계 |
| 검증 방식 | time-based 80/20 | LOSO 미도입 |
| 일반화 | 동일 인물만 | 다른 사람에게 동작 여부 미확인 |
| 클래스 균형 | chewing 65~71% | rest 구간이 부족 |

### 다음 단계

```
단기 (15~20 세션 수집 후)
├── LOSO 교차 검증 도입
│     각 세션을 test로 빼고 나머지로 학습 → 평균 F1 산출
│     → 모델이 "새 세션"에서도 동작하는지 검증
├── 전체 데이터로 최종 RF 재학습
└── Core ML 모델 교체 → 앱 배포

중기 (여러 사람 데이터 확보 후)
└── 개인화 모델 vs 범용 모델 비교
```

### 세션 수집 가이드

- **목표**: 세션당 2~3분, 총 15~20세션
- **다양성**: 다른 음식, 하루 다른 시간대, 고개 각도 변화
- **균형**: 식사 전후 쉬는 구간(rest)을 의도적으로 포함 → chewing 50% 근처 유지

---

*생성: `ml/compare_sessions.py` / 최종 업데이트: 2026-05-13*
