# chewing-vision — Vision/Model Spec (v2)

## 0.0 구현자 mandate (모든 ralph story 공통)

`prd.json`의 모든 user story 구현 시 다음 두 단계가 **무조건** 선행되어야 한다. 이건 권고가 아니라 실행 게이트다.

1. **카파시 가이드라인 로드** — 구현 직전 `Skill('andrej-karpathy-skills:karpathy-guidelines')` 호출. surgical change, 가정의 표면화, 검증 가능한 성공 기준, 과잉 추상화 회피라는 코딩 행동 baseline.
2. **advisor 검토** — 구현 직전 `advisor()` 호출로 접근 방식을 외부 reviewer에 검증. 구현 후 마무리 직전 한 번 더 `advisor()` 호출로 diff 점검.

이 두 단계를 거치지 않은 story는 `passes:true` 처리하지 않는다. 자세한 directive는 `prd.json` 의 `executionDirectives` 와 `perStoryRequiredActions` 필드 참조.

---

## 0. 모델·비전 제작자 관점의 첫 피드백

지금 Day 1 데모의 방향은 맞다. 영상에서 얼굴 landmark와 blendshape를 뽑고, MAR/jawOpen 신호의 peak를 세는 방식은 "씹기 신호가 보이는가"를 빠르게 확인하기에 좋다. 다만 이걸 실제 AirPods IMU 학습용 ground truth 라벨러로 쓰려면 스펙의 중심을 `chew count` 하나에서 `라벨 품질 관리 + 실패 케이스 분리 + 평가 가능한 시간축 데이터`로 옮겨야 한다.

핵심 판단은 다음과 같다.

- `씹기 1회 카운트`는 최종 지표가 아니라 파생 지표다. 모델 학습에는 frame/window/bout/event 라벨을 모두 남겨야 한다.
- 영상 자동 라벨은 정답이 아니라 weak label이다. 사람 라벨과 비교해 신뢰 가능한 구간만 GT 후보로 승격해야 한다.
- 비전만으로는 어금니 좌/우 chewing side를 안정적으로 확정하기 어렵다. orofac 엔진의 side 값은 참고 메타로 두고, v1 학습 라벨의 필수 정답으로 쓰지 않는다.
- 말하기, 웃기, 하품, 고개 움직임, 컵 들기, 손 가림은 chewing과 비슷한 신호를 만든다. negative/confounder 데이터를 의도적으로 넣어야 한다.
- 데모 오버레이는 발표용 산출물이고, 모델 학습 파이프라인은 별도 품질 기준을 가져야 한다.
- AirPods IMU 모델로 넘어가려면 처음부터 timestamp, fps, dropped frame, start/end trim, confidence, face quality를 엄격히 남겨야 한다.

따라서 v2 목표는 "영상 한 편에서 예쁜 카운트 영상을 만든다"가 아니라, **영상 기반 chewing weak-label generator + 검증 가능한 평가 CLI + 데모 오버레이**를 만드는 것이다.

---

## 1. 목적

`chewing-vision`은 로컬 영상 파일을 입력받아 식사 중 저작 운동을 시간축 데이터로 변환하는 CLI/파이썬 패키지다.

이 프로젝트의 최종 역할은 세 가지다.

1. AirPods IMU 모델 학습에 사용할 chewing ground-truth 후보 라벨을 만든다.
2. 비전 기반 저작 감지의 실패/성공 조건을 빠르게 실험한다.
3. 발표·검증용 오버레이 MP4와 그래프를 생성해 제품 가능성을 보여준다.

v1에서 의료 진단, 건강 개선 판단, 영양 추정은 하지 않는다. 이 도구는 "식사 동작과 씹기 리듬을 시간축으로 표시하는 연구/데모용 라벨러"다.

---

## 2. 현재 자산과 보존 원칙

| 파일/폴더 | 역할 | 처리 |
|---|---|---|
| `day1/chewing_demo.py` | MediaPipe Face Landmarker 기반 MAR/jawOpen 실험 코드 | 삭제하지 않고 reference로 보존 |
| `day1/README.md` | Day 1 데모 사용법과 영상 선택 기준 | 유지하되 패키지 README와 연결 |
| `prd.json` | 자동 구현용 user story와 acceptance criteria | SPEC v2 기준으로 점진 업데이트 |
| `SPEC.md` | 사람이 읽는 구현 스펙 | 이 문서를 canonical spec으로 사용 |

`prd.json`의 `project_root`와 acceptance path는 실제 폴더인 `/Users/bohyeong/Desktop/공부/project/soma/chewing-vision` 기준으로 맞춘다.

---

## 3. 범위

### 3.1 In Scope

- 로컬 비디오 파일 분석 CLI
- `ours` 엔진: MediaPipe Face Landmarker 기반 신호 추출
- `orofac` 엔진: `orofacIAnalysis==0.1.2` 래퍼
- 공통 `Result` 데이터 모델
- frame-level signal CSV
- chew event CSV
- window-level label CSV
- bout-level CSV
- summary JSON
- static signal plot PNG
- demo overlay MP4
- 사람 라벨 CSV와 자동 라벨 비교
- 두 엔진 간 agreement 비교
- confidence/quality gate 산출
- smoke test용 fixture 기반 end-to-end 검증

### 3.2 Out of Scope (v1)

- 실시간 웹캠 뷰어
- iOS 앱 통합
- AirPods IMU 동시 수집
- 서버 업로드/클라우드 처리
- 의료적 해석 또는 건강 상태 판정
- 음식 종류·칼로리·섭취량 추정
- 다인 얼굴 처리
- 완전 자동 고품질 GT 보장

### 3.3 v1 완료 조건

다음 한 줄이 안정적으로 동작해야 한다.

```bash
chewing demo tests/fixtures/sample_chewing_1.mp4 -o /tmp/cv_demo_out
```

생성물:

- `frame_signals_ours.csv`
- `labels_ours.csv`
- `events_ours.csv`
- `bouts_ours.csv`
- `labels_orofac.csv`
- `events_orofac.csv`
- `summary.json`
- `signals.png`
- `demo.mp4`

---

## 4. 주요 사용자 시나리오

### Scenario A — 연구자가 영상 1개를 분석한다

Given 먹는 장면이 포함된 1분 영상이 있다.  
When 연구자가 `chewing analyze video.mp4 --engine ours -o out/`를 실행한다.  
Then `out/`에는 frame signal, event, window, bout, summary 파일이 생성된다.  
And summary에는 face detection rate, usable duration, chew count, confidence 분포가 포함된다.

### Scenario B — 데모 담당자가 발표용 영상을 만든다

Given sample fixture 또는 본인 녹화 영상이 있다.  
When `chewing overlay video.mp4 --engine ours -o demo.mp4`를 실행한다.  
Then 원본 영상 위에 얼굴 landmark, 현재 chewing 상태, 누적 chew count, 최근 6초 signal trace가 표시된다.  
And MP4는 일반 플레이어에서 재생 가능하다.

### Scenario C — 모델 담당자가 사람 라벨과 비교한다

Given 사람이 만든 `human_labels.csv`와 자동 생성 `labels_ours.csv`가 있다.  
When `chewing eval --auto labels_ours.csv --human human_labels.csv`를 실행한다.  
Then event F1, window F1, bout IoU, count error가 출력된다.  
And face quality가 낮은 구간을 제외한 metric과 전체 metric이 따로 제공된다.

### Scenario D — 실패 케이스를 확인한다

Given 사람이 말하거나 웃거나 고개를 크게 움직이는 negative 영상이 있다.  
When `chewing analyze negative.mp4 --engine ours`를 실행한다.  
Then chewing count가 과도하게 발생하지 않아야 한다.  
And summary에는 false-positive 의심 구간과 quality warning이 포함된다.

### Scenario E — AirPods IMU 학습용 후보 라벨을 만든다

Given 같은 식사 세션의 영상과 IMU 로그가 있다.  
When 영상 분석 결과를 export한다.  
Then 모든 라벨은 절대 시간 또는 세션 기준 상대 시간으로 정렬 가능해야 한다.  
And confidence가 낮은 window는 학습에서 제외할 수 있도록 별도 flag가 있어야 한다.

---

## 5. 시스템 아키텍처

```text
video.mp4
  |
  v
VideoReader
  - fps, frame_count, duration
  - start/end trim
  - frame timestamp normalization
  |
  v
EngineBase.analyze()
  |
  +-- OursEngine
  |     - MediaPipe Face Landmarker
  |     - landmarks + blendshapes
  |     - MAR, jawOpen, chinY, headMotion
  |     - smoothing + peak detection
  |     - window/bout segmentation
  |
  +-- OrofacEngine
        - orofacIAnalysis wrapper
        - cycle/side extraction
        - shared schema mapping

Result
  |
  +-- labels.py      CSV/JSON output
  +-- viz.py         signals.png
  +-- overlay.py     demo.mp4
  +-- eval.py        auto vs human
  +-- compare.py     engine agreement
```

패키지 목표 구조:

```text
chewing/
  cli.py
  types.py
  video.py
  engines/
    base.py
    ours.py
    orofac.py
  signals.py
  smoothing.py
  peaks.py
  segmentation.py
  quality.py
  labels.py
  viz.py
  overlay.py
  eval.py
  compare.py
tests/
  fixtures/
  test_signals.py
  test_peaks.py
  test_eval.py
  test_smoke.py
```

---

## 6. 데이터 모델

### 6.1 공통 타입

```python
@dataclass
class FrameSignal:
    t_sec: float
    frame_index: int
    face_found: bool
    mar: float | None
    jaw_open: float | None
    chin_y: float | None
    head_motion: float | None
    quality: float

@dataclass
class ChewEvent:
    t_sec: float
    frame_index: int
    source_signal: str
    signal_value: float
    confidence: float
    side: str | None = None

@dataclass
class WindowLabel:
    t_start: float
    t_end: float
    label: str
    confidence: float
    quality: float
    n_events: int
    mar_mean: float | None
    jaw_open_mean: float | None

@dataclass
class Bout:
    t_start: float
    t_end: float
    n_events: int
    chews_per_min: float
    confidence: float

@dataclass
class Result:
    engine_name: str
    video_path: str
    duration_sec: float
    fps: float
    frame_count: int
    usable_duration_sec: float
    face_detection_rate: float
    n_chews: int
    chews_per_min: float
    frames: list[FrameSignal]
    events: list[ChewEvent]
    windows: list[WindowLabel]
    bouts: list[Bout]
    warnings: list[str]
    extra: dict[str, Any]
```

### 6.2 Label Vocabulary

`WindowLabel.label`은 v1에서 다음 값만 허용한다.

| label | 의미 |
|---|---|
| `chewing` | 씹기 동작이 주로 관찰됨 |
| `rest` | 씹기 동작 없음 |
| `speaking` | 말하기/입 움직임이 chewing과 섞임 |
| `drinking` | 컵/빨대/삼킴 중심 |
| `occluded` | 손, 컵, 마이크, 음식 등으로 입/턱 가림 |
| `bad_face` | face detection 또는 landmark 품질 낮음 |
| `unknown` | 자동/사람 모두 판단 보류 |

v1 모델 학습용 positive는 `chewing`, negative는 `rest`만 기본 사용한다. `speaking`, `drinking`, `occluded`, `bad_face`, `unknown`은 분석/필터링 메타로 남기고 학습에서 기본 제외한다.

---

## 7. 출력 스키마

### 7.1 frame_signals CSV

```csv
t_sec,frame_index,face_found,mar,jaw_open,chin_y,head_motion,quality
0.000,0,true,0.16,0.32,0.08,0.01,0.92
```

### 7.2 event CSV

```csv
t_sec,frame_index,signal,value,confidence,engine,side
0.340,10,jaw_open,0.48,0.81,ours,
```

### 7.3 window label CSV

기본 window는 1.0초 non-overlap이다. 옵션으로 0.5초 stride를 허용하되, 기본 산출물은 1초 단위로 고정한다.

```csv
t_start,t_end,label,confidence,quality,n_events,engine,mar_mean,jaw_open_mean
0.0,1.0,chewing,0.83,0.91,2,ours,0.18,0.42
1.0,2.0,rest,0.95,0.96,0,ours,0.06,0.05
```

### 7.4 bout CSV

Bout는 event 사이 간격이 `max_gap_sec <= 1.2`인 chewing event 묶음이다.

```csv
t_start,t_end,n_events,chews_per_min,confidence,engine
12.2,28.8,24,86.7,0.78,ours
```

### 7.5 summary JSON

```json
{
  "video": "sample_chewing_1.mp4",
  "duration_sec": 60.0,
  "fps": 30.0,
  "frame_count": 1800,
  "engines": {
    "ours": {
      "usable_duration_sec": 57.2,
      "face_detection_rate": 0.96,
      "n_chews": 72,
      "chews_per_min": 75.5,
      "n_bouts": 3,
      "quality_mean": 0.91,
      "warnings": []
    },
    "orofac": {
      "n_chews": 69,
      "chews_per_min": 72.1,
      "left": 28,
      "right": 31,
      "middle": 10
    }
  },
  "agreement": {
    "window_f1": 0.84,
    "event_f1_300ms": 0.78,
    "bout_iou": 0.71,
    "count_diff_pct": 4.2
  }
}
```

---

## 8. Vision Pipeline 상세

### 8.1 Video Ingest

- OpenCV `VideoCapture`로 로컬 파일을 읽는다.
- `fps <= 0` 또는 frame count 미확인 시 `fps=30.0` fallback을 사용하되 warning에 기록한다.
- `--start`, `--end`는 초 단위이며, frame index로 변환한다.
- 모든 결과 timestamp는 trim 이후 0초가 아니라 원본 영상 기준 시간을 기본으로 둔다.
- 옵션 `--relative-time`을 주면 trim 시작점을 0초로 export한다.

### 8.2 Face Tracking

- MediaPipe Face Landmarker Tasks API를 사용한다.
- `num_faces=1`로 고정한다.
- 여러 얼굴이 보이는 영상은 v1에서 지원하지 않는다.
- 얼굴 미검출 frame은 `face_found=false`, signal 값은 `NaN`, quality는 0으로 둔다.

### 8.3 Signal Extraction

필수 신호:

- `MAR`: mouth aspect ratio
- `jaw_open`: MediaPipe blendshape `jawOpen`
- `chin_y`: 코 기준 턱 y displacement
- `head_motion`: nose/chin/face bbox 중심 움직임 기반 head motion proxy

권장 파생 feature:

- `mar_delta`
- `jaw_open_delta`
- `chin_y_delta`
- rolling std/mean
- spectral energy around 0.8-2.5 Hz
- peak interval variability

### 8.4 Normalization

MAR는 입 좌우 거리로 normalize한다.  
chin_y/head_motion은 얼굴 bbox 높이 또는 눈 사이 거리로 normalize한다.  
픽셀 단위 threshold는 v1 `ours` 엔진에서 금지한다.

### 8.5 Smoothing

기본 smoothing:

- `jaw_open`: Savitzky-Golay 또는 Gaussian
- `mar`: median + Gaussian
- 결측 구간: 0.5초 이하만 linear interpolation
- 0.5초 초과 결측은 segment break로 처리

### 8.6 Peak Detection

기본 chewing frequency 가정:

- min chewing frequency: 0.8 Hz
- max chewing frequency: 2.5 Hz
- peak distance: `int(fps / max_freq)`
- prominence: `prominence_std * nanstd(signal)`

`jaw_open`을 primary signal로 사용하고, MAR는 보조 확인 신호로 사용한다. 둘 다 강하게 동의하면 confidence를 올리고, 한쪽만 반응하면 중간 confidence로 둔다.

### 8.7 Window Segmentation

1초 window에서 다음 규칙을 적용한다.

- quality < 0.5 또는 face_found 비율 < 0.6이면 `bad_face`
- event count >= 1이고 quality >= 0.6이면 `chewing`
- event count == 0이고 quality >= 0.6이면 `rest`
- head_motion이 과도하고 jaw_open peak만 많은 구간은 warning 추가
- speaking/drinking/occluded는 v1 자동 분류가 어렵기 때문에 사람 라벨 또는 future classifier가 있을 때만 사용

### 8.8 Confidence

confidence는 0-1 범위이며 다음 요소를 조합한다.

- face detection/landmark quality
- peak prominence z-score
- jaw_open과 MAR peak 동의도
- peak interval이 생리적으로 가능한 범위인지
- window 내부 결측률
- head motion 과다 여부

v1에서는 해석 가능한 heuristic score로 충분하다. 단, 산식은 `quality.py`에 고정해 재현 가능해야 한다.

---

## 9. 모델 전략

### 9.1 Baseline v1 — Heuristic Weak Labeler

목표는 "사람 라벨 없이도 대략적인 chewing 후보 구간을 뽑는 것"이다.

- 입력: video frame signals
- 출력: event/window/bout labels
- 장점: 빠르고 설명 가능
- 단점: speaking/drinking/head motion에 취약

Baseline이 통과해야 하는 최소 기준:

- sample chewing 영상에서 `n_chews > 0`
- negative rest 영상에서 분당 false chew < 10
- 30fps/60fps synthetic signal count 차이 <= 10%
- 같은 영상 재실행 결과 동일

### 9.2 Model v0 — Feature Classifier

사람 라벨이 30-60분 이상 쌓이면 window-level classifier를 만든다.

권장 모델:

- Logistic Regression 또는 Random Forest
- LightGBM은 선택사항
- 입력 feature는 1초 window의 signal statistics
- 출력 class는 `chewing/rest/bad_quality`

이 단계에서 deep learning은 아직 이르다. 데이터가 작고 라벨 품질이 핵심이기 때문이다.

### 9.3 Model v1 — Temporal Sequence Model

사람 라벨이 5-10시간 이상 쌓이고 confounder가 충분해지면 temporal model을 검토한다.

후보:

- TCN
- 1D CNN + BiLSTM
- 작은 Transformer encoder

입력:

- jaw_open, MAR, chin_y, head_motion time series
- optional facial landmark PCA features

출력:

- per-frame 또는 per-window chewing probability
- post-processing으로 event/bout 산출

### 9.4 IMU 전환 전략

영상 모델은 최종 모델이 아니다. AirPods IMU 모델 학습을 위한 teacher/labeler 역할이다.

전환 단계:

1. 영상 라벨과 IMU 로그를 같은 세션 timestamp로 동기화한다.
2. confidence 높은 chewing/rest window만 학습 데이터로 사용한다.
3. IMU-only 모델은 영상 feature를 보지 않는다.
4. 영상 라벨은 weak label임을 metadata에 남긴다.
5. 최종 앱 지표는 IMU 모델의 window-level chewing probability와 meal pace summary로 제한한다.

---

## 10. 사람 라벨링 규칙

### 10.1 라벨 단위

v1 라벨링은 1초 window 단위를 기본으로 한다. event-level 수동 라벨은 선택사항이다.

필수 컬럼:

```csv
t_start,t_end,label,annotator,notes
0.0,1.0,chewing,bh,
1.0,2.0,rest,bh,
```

### 10.2 Chewing 판정 기준

`chewing`:

- 입/턱의 반복 운동이 보인다.
- 음식물을 씹는 맥락이 명확하다.
- 말하는 입 움직임이 주된 행동이 아니다.

`rest`:

- 입/턱 반복 운동이 없다.
- 음식 대기, 가만히 있음, 카메라 보기 등.

`speaking`:

- 말하기가 주된 입 움직임이다.
- chewing과 섞이면 `speaking`으로 두고 학습에서 제외한다.

`drinking`:

- 컵/빨대/삼킴이 중심이다.

`occluded`:

- 입 또는 턱이 손/컵/마이크/자막/음식으로 가려졌다.

`bad_face`:

- 얼굴이 프레임 밖이거나 landmark가 신뢰하기 어렵다.

### 10.3 라벨 품질

- 첫 10개 영상은 한 명이 라벨링해도 된다.
- 평가 기준을 세울 때는 최소 2명 라벨을 비교한다.
- annotator agreement 목표:
  - window-level Cohen's kappa >= 0.75
  - event-level은 ±300ms tolerance F1 >= 0.75

---

## 11. 평가 지표

### 11.1 Window-level Metrics

기본 지표:

- precision
- recall
- F1
- confusion matrix

`bad_face`, `unknown`, `occluded`는 기본 metric에서 제외한 버전과 포함한 버전을 둘 다 출력한다.

### 11.2 Event-level Metrics

자동 event와 사람 event의 매칭 허용 오차:

- 기본 tolerance: ±300ms
- strict tolerance: ±150ms
- loose tolerance: ±500ms

한 사람 event는 최대 한 자동 event와만 matching된다.

### 11.3 Count Metrics

- absolute count error
- count error percent
- chews/min error
- per-bout count error

count error percent:

```text
abs(auto_count - human_count) / max(human_count, 1) * 100
```

### 11.4 Bout Metrics

- bout IoU
- number of bouts diff
- chewing duration diff

Bout IoU는 시간 구간 intersection/union으로 계산한다.

### 11.5 MVP Quality Gate

v1을 "쓸 만한 weak labeler"로 인정하는 기준:

- clean chewing fixture window F1 >= 0.80
- clean chewing fixture count error <= 20%
- negative rest/speaking clips false positive windows <= 15%
- face detection rate < 0.8 영상은 자동으로 low-quality warning
- 동일 입력 재실행 결과 byte-level 동일 또는 metric 동일

---

## 12. CLI Spec

### 12.1 analyze

```bash
chewing analyze VIDEO.mp4 \
  --engine ours|orofac|both \
  --start 0 \
  --end 60 \
  --window-sec 1.0 \
  --relative-time false \
  -o out/
```

생성 파일:

- `frame_signals_{engine}.csv`
- `labels_{engine}.csv`
- `events_{engine}.csv`
- `bouts_{engine}.csv`
- `summary.json`

### 12.2 plot

```bash
chewing plot VIDEO.mp4 --engine ours -o signals.png
```

`plot`은 내부적으로 analyze를 실행하거나 기존 result JSON/cache를 읽을 수 있다. v1에서는 내부 analyze 실행으로 충분하다.

### 12.3 overlay

```bash
chewing overlay VIDEO.mp4 --engine ours -o demo.mp4
```

레이아웃:

- canvas: 1600x840
- left video: 1280x720
- right sidebar: 320x720
- bottom trace: 1600x120
- chewing border: green
- rest border: gray
- peak marker: red

### 12.4 eval

```bash
chewing eval --auto labels_ours.csv --human human_labels.csv
```

출력:

- JSON stdout
- optional `--out eval.json`

### 12.5 compare

```bash
chewing compare --a labels_ours.csv --b labels_orofac.csv
```

출력:

- window F1
- event F1 if events provided
- count diff
- bout IoU

### 12.6 demo

```bash
chewing demo tests/fixtures/sample_chewing_1.mp4 -o /tmp/cv_demo_out
```

`demo`는 다음을 순서대로 실행한다.

1. analyze `--engine both`
2. plot `ours`
3. overlay `ours`
4. summary print

---

## 13. 실패 케이스와 방어 규칙

| 실패 케이스 | 위험 | v1 방어 |
|---|---|---|
| 말하기 | jawOpen peak가 chewing처럼 보임 | speaking fixture를 negative로 평가 |
| 웃음/리액션 | 큰 입 움직임 | peak interval과 bout consistency 확인 |
| 고개 끄덕임 | chin/head motion이 턱 신호에 섞임 | head_motion feature로 confidence 낮춤 |
| 손/컵 가림 | landmark jitter | face/landmark quality warning |
| 옆얼굴 | MAR 왜곡 | face detection rate와 bbox/aspect warning |
| 낮은 조명 | jitter 증가 | quality 낮춤 |
| 영상 fps 오류 | timestamp 불일치 | fps fallback warning |
| 씹기 없이 삼킴 | 단발 jawOpen peak | bout로 묶이지 않으면 low confidence |

---

## 14. 테스트 전략

### 14.1 Unit Tests

- `signals.py`: synthetic landmarks로 MAR/jawOpen/chin_y 검증
- `peaks.py`: 30fps vs 60fps synthetic chewing count 안정성
- `labels.py`: CSV round-trip
- `eval.py`: known confusion matrix
- `quality.py`: 결측/낮은 face rate에서 quality 하락

### 14.2 Smoke Tests

```bash
pytest tests/test_smoke.py -v
```

검증:

- sample video 분석 성공
- `n_chews > 0`
- output files 존재
- overlay MP4 크기 > 1MB
- MP4 resolution = 1600x840

### 14.3 Regression Fixtures

최소 fixture 세트:

- clean chewing 1개
- rest 1개
- speaking 1개
- drinking 1개
- occlusion 1개

초기에는 외부 배포가 어려우면 fixture path만 로컬로 두고, CI에서는 synthetic tests 중심으로 둔다.

---

## 15. 라이선스와 데이터 정책

- 우리 코드: MIT
- `orofacIAnalysis==0.1.2`: MIT attribution 필수
- 외부 영상은 repo에 커밋하지 않는다.
- 사람 얼굴 영상은 민감 데이터로 취급한다.
- 데모/학습용 영상은 가능하면 본인 촬영 또는 명시적 사용 허가가 있는 자료를 사용한다.
- export CSV에는 이름, 계정, 원본 URL 같은 PII를 넣지 않는다.

---

## 16. 구현 순서

### Phase 0 — Skeleton

- `pyproject.toml`
- package layout
- dataclasses
- CLI shell
- MIT license/attribution

### Phase 1 — OursEngine

- Day 1 코드 패키지화
- frame signal export
- jawOpen primary peak detection
- window/event/bout generation

### Phase 2 — Evaluation Core

- human label CSV schema
- window F1
- event F1 tolerance matching
- bout IoU
- quality-filtered metrics

### Phase 3 — OrofacEngine

- orofac wrapper
- shared schema mapping
- side count는 `extra`로만 보존
- cross-engine agreement

### Phase 4 — Visualization

- signals.png
- overlay.mp4
- demo command

### Phase 5 — Robustness

- negative fixtures
- confidence tuning
- fps/resolution regression
- README quickstart

---

## 17. Open Questions

- 사람 라벨링 도구를 CSV 수기 작성으로 시작할지, 간단한 Streamlit/Gradio UI를 만들지?
- event-level 사람 라벨을 v1부터 요구할지, window-level만 먼저 갈지?
- AirPods IMU 로그와 영상의 timestamp 동기화 방식은 무엇으로 할지?
- 실제 사용자 영상 저장 정책은 어떻게 할지?
- `orofacIAnalysis` sample fixture를 repo에 포함해도 라이선스/용량상 괜찮은지?
- CLI 이름을 `chewing`으로 유지할지, PyPI 충돌을 피하려고 `chewing-vision` 명령으로 갈지?

---

## 18. 구현자가 지켜야 할 원칙

- count 숫자가 좋아 보여도 라벨 품질을 숨기지 않는다.
- confidence 낮은 구간은 학습용 GT로 쓰지 않는다.
- 픽셀 절대 threshold를 쓰지 않는다.
- 모든 threshold는 fps/face scale에 상대적으로 둔다.
- 동일 입력은 동일 출력을 내야 한다.
- 영상 처리 실패는 조용히 넘어가지 말고 summary warning에 남긴다.
- 데모용 보기 좋은 화면과 학습용 export를 분리한다.
- 의료적 표현을 코드/문서/README에 넣지 않는다.

---

## 19. Definition of Done

v1은 다음을 모두 만족하면 완료다.

1. `pip install -e .` 후 `chewing --help`가 동작한다.
2. `chewing demo tests/fixtures/sample_chewing_1.mp4 -o /tmp/cv_demo_out`가 exit 0.
3. demo output에 CSV/JSON/PNG/MP4가 모두 존재한다.
4. `pytest tests/ -v` 통과.
5. `summary.json`에 quality, warnings, agreement가 포함된다.
6. `chewing eval --auto labels_ours.csv --human labels_ours.csv`가 F1 1.0을 반환한다.
7. clean chewing fixture에서 `n_chews > 0`.
8. negative fixture에서 false positive가 품질 기준을 넘지 않는다.
9. README에 install, quickstart, output schema, attribution이 있다.
10. `ATTRIBUTION.md`에 orofacIAnalysis MIT attribution이 있다.

---

## 20. 다음 액션

바로 구현에 들어간다면 첫 커밋은 다음 범위로 끊는다.

```text
feat(core): scaffold chewing vision package
```

포함:

- `pyproject.toml`
- `chewing/types.py`
- `chewing/engines/base.py`
- `chewing/cli.py`
- `LICENSE`
- `ATTRIBUTION.md`
- import smoke test

그 다음 커밋은 Day 1 코드를 `OursEngine`으로 옮기는 것이다.

```text
feat(vision): add mediapipe chewing signal engine
```
