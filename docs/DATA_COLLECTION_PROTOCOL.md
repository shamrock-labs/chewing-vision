# 데이터 수집 프로토콜 — chewing-imu-collector

**버전**: v1.0  
**작성일**: 2026-05-14

---

## 목적

ML 모델 학습에 필요한 세 가지 활동을 명확히 분리해서 수집한다.

| 활동 | 목적 |
|------|------|
| **chewing** | 양성 샘플 (학습 대상) |
| **speaking** | 혼동 구간 분리 (false positive 제거) |
| **rest** | 음성 샘플 |
| **head movement** | attitude delta feature 검증 |

---

## 준비물

- AirPods Pro (또는 Pro 2 / 3rd / Max) 착용
- 음식 2종: **딱딱한 것** (견과류, 사과 등) + **부드러운 것** (바나나, 두부 등)
- 읽을 텍스트 (카톡 대화, 뉴스 기사 등 아무거나)
- chewing-imu-collector 앱 실행 준비

---

## 녹화 순서 (총 약 5분)

| 단계 | 행동 | 시간 | 목적 |
|------|------|------|------|
| 1 | 카메라 정면 보며 **가만히 있기** | 30초 | rest baseline |
| 2 | 텍스트 소리 내서 **읽기** (자연스럽게) | 30초 | speaking |
| 3 | 가만히 있기 | 10초 | rest 전환 |
| 4 | **씹기** — 딱딱한 음식 | 60초 | chewing (hard) |
| 5 | 가만히 있기 | 10초 | rest 전환 |
| 6 | 텍스트 소리 내서 **읽기** | 30초 | speaking |
| 7 | **씹기** — 부드러운 음식 | 60초 | chewing (soft) |
| 8 | 가만히 있기 | 20초 | rest |
| 9 | **고개 좌우/상하 반복 움직이기** | 20초 | attitude delta 검증 |

**총 녹화 시간**: 약 4분 40초

---

## 지켜야 할 것

### 필수
- 각 단계 사이 **2~3초 명확하게 멈추기** — 씹다가 바로 말하지 말 것
- 씹는 동안 **정면 유지** — 옆을 보거나 고개 숙이지 않기
- 말할 때 **자연스러운 속도** — 일부러 천천히 읽지 말 것

### 권장
- 충분한 조명 (MediaPipe face detection 품질)
- 배경 단색 또는 정적인 환경
- 세션당 **음식 한 입 분량** 준비 (씹다 멈추는 상황 방지)

---

## 녹화 후 분석

### 1. vision 분석

```bash
cd /Users/bohyeong/Desktop/공부/project/soma/chewing-vision

# 세션 ID 확인 (예: 20260514T120000_abc123)
SESSION_ID=20260514T120000_abc123
VIDEO_PATH=$(pwd)/chewing_collector_data/sessions/${SESSION_ID}/video.mp4

.venv/bin/chewing analyze ${VIDEO_PATH} \
  --out /tmp/chewing_${SESSION_ID: -6}/
```

### 2. 라벨 분포 확인

```bash
cut -d',' -f3 /tmp/chewing_${SESSION_ID: -6}/labels_ours.csv | sort | uniq -c
```

**기대 결과** (Phase 1 구현 후):
```
 N  chewing
 N  rest
 N  speaking   ← 이게 생겨야 정상
```

### 3. ML 파이프라인 추가

`ml/compare_sessions.py`의 `SESSIONS` 리스트에 추가:
```python
{
    "id": "20260514T120000_abc123",
    "label": "abc123",
    "labels_path": "/tmp/chewing_abc123/labels_ours.csv",
},
```

---

## 팀 수집 목표

| 목표 세션 수 | 효과 |
|------------|------|
| 현재 4세션 | LOSO fold당 3세션 학습 — 데이터 부족 |
| **10세션** | LOSO fold당 9세션 — F1 variance 안정화 시작 |
| **20세션+** | LOSO 신뢰도 확보, speaking 라벨 충분 |

팀원 3명이 각자 2~3세션씩 찍으면 9세션+ 확보 가능.

---

## 관련 문서

- `docs/SPEAKING_DETECTION_SPEC.md` — speaking 감지 알고리즘 설계
- `docs/SPEC.md` — chewing-vision 전체 스펙
- `ml/README.md` — ML 파이프라인 설명
