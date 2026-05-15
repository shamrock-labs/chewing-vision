# chewing-vision

비디오에서 씹기 신호를 추출하고 Firebase Storage 세션을 관리하는 CLI 도구입니다.  
AirPods IMU 씹기 GT 워크플로의 weak-label 생성기로 설계되었습니다.

## 설치 (팀원용)

```bash
git clone <repo-url>
cd chewing-vision
bash setup.sh
```

`setup.sh`가 자동으로 처리합니다:
- Python 3.10+ 확인
- `.venv` 가상환경 생성
- 패키지 및 Firebase 의존성 설치
- `chewing-vision` 전역 명령어 등록 (`/usr/local/bin`)
- `.env` 템플릿 생성

### Firebase 설정

Firebase fetch 기능을 사용하려면 `.env`에 서비스 계정 JSON 경로를 설정하세요:

```
CHEWING_FIREBASE_CREDENTIALS=path/to/serviceaccount.json
```

서비스 계정 키 발급: Firebase Console → Project Settings → Service Accounts → Generate new private key

---

## 사용법

### 인터랙티브 메뉴

```bash
chewing-vision
```

화살표 키로 커맨드를 선택하면 필요한 옵션을 순서대로 입력할 수 있습니다.

### CLI 직접 실행

```bash
chewing-vision <command> [options]
chewing-vision <command> --help
```

---

## 커맨드 레퍼런스

### `analyze` — 씹기 분석

비디오에서 씹기 이벤트를 감지하고 CSV/JSON을 출력합니다.

```bash
chewing-vision analyze VIDEO.mp4 --engine ours|orofac|both -o out/
```

출력: `frame_signals_{engine}.csv`, `labels_{engine}.csv`, `events_{engine}.csv`, `bouts_{engine}.csv`, `summary.json`

### `plot` — 신호 시각화

MAR / jaw_open 시계열 그래프 PNG를 생성합니다.

```bash
chewing-vision plot VIDEO.mp4 --engine ours -o signals.png
```

### `overlay` — 오버레이 렌더링

씹기 감지 결과를 원본 비디오에 오버레이한 MP4를 생성합니다.

```bash
chewing-vision overlay VIDEO.mp4 --engine ours -o demo.mp4
```

### `eval` — 라벨 평가

자동 라벨 CSV와 수동 라벨 CSV를 비교해 precision · recall · F1을 출력합니다.

```bash
chewing-vision eval --auto AUTO_LABELS.csv --human HUMAN_LABELS.csv
# 이벤트 레벨 비교 포함
chewing-vision eval --auto AUTO.csv --human HUMAN.csv \
  --auto-events AUTO_EVENTS.csv --human-events HUMAN_EVENTS.csv
```

### `compare` — 엔진 비교

두 엔진(ours / orofac)이 출력한 CSV를 교차 비교해 agreement 점수를 산출합니다.

```bash
chewing-vision compare --a LABELS_A.csv --b LABELS_B.csv
```

### `demo` — 풀 파이프라인

analyze + plot + overlay를 한 번에 실행합니다.

```bash
chewing-vision demo VIDEO.mp4 -o out/
```

### `fetch` — Firebase 세션 다운로드

Firebase Storage에서 IMU/비디오 세션을 조회하고 다운로드합니다.

```bash
# 세션 목록 보기
chewing-vision fetch --list

# 특정 세션 다운로드
chewing-vision fetch SESSION_ID -o ./sessions

# 모든 세션 다운로드
chewing-vision fetch --all -o ./sessions
```

---

## 출력 스키마

`frame_signals_{engine}.csv`
```
t_sec,frame_index,face_found,mar,jaw_open,chin_y,head_motion,quality
```

`labels_{engine}.csv`
```
t_start,t_end,label,confidence,quality,n_events,engine,mar_mean,jaw_open_mean
```

`events_{engine}.csv`
```
t_sec,frame_index,signal,value,confidence,engine,side
```

`bouts_{engine}.csv`
```
t_start,t_end,n_events,chews_per_min,confidence,engine
```

`summary.json` — 비디오 메타데이터, 엔진별 카운트, 품질/경고, agreement 메트릭 포함. 상세 스키마는 `SPEC.md` §7 참조.

---

## 라이선스

MIT. 자세한 내용은 `LICENSE` 참조.  
orofacIAnalysis (MIT, Cameron Maloney) 사용 — `ATTRIBUTION.md` 참조.

### `loso` — LOSO 교차 검증

`sessions/` 디렉토리를 자동 탐색해 Leave-One-Session-Out CV를 실행하고 PNG + HTML 리포트를 생성합니다.

```bash
# 기본값 (--sessions-dir ./sessions, -o ml/outputs)
chewing-vision loso

# 경로 직접 지정
chewing-vision loso --sessions-dir /path/to/sessions -o ml/outputs
```

각 세션 디렉토리에 다음 세 파일이 있어야 탐색 대상이 됩니다:
- `imu*.csv` — IMU 데이터 (fetch로 다운로드)
- `session*.json` — 세션 메타데이터 (fetch로 다운로드)
- `labels_ours.csv` — 씹기 라벨 (analyze로 생성)

실행할 때마다 타임스탬프 폴더가 생성되어 히스토리를 추적할 수 있습니다:
```
ml/outputs/
├── 20260515T113400/
│   ├── session_comparison.png
│   └── session_comparison.html
├── 20260516T093000/
│   └── ...
└── latest -> 20260516T093000   # 항상 최신 결과 가리킴
```

---

## 전체 파이프라인 (Firebase → LOSO)

```bash
# 1. Firebase에서 세션 다운로드
chewing-vision fetch --all -o ./sessions

# 2. 각 세션 비디오 분석 (라벨 생성)
#    인터랙티브 메뉴에서 analyze 선택 시 출력 경로가 자동으로 세션 폴더로 설정됨
chewing-vision analyze ./sessions/{session_id}/video.mp4 -o ./sessions/{session_id}/

# 3. LOSO CV 실행
chewing-vision loso
```
