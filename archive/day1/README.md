# Day 1 — Chewing 데모

영상 한 편 → MediaPipe FaceMesh 로 입·턱 신호 추출 → 봉우리(=씹기 1회) 카운트 + 그래프.

## 1. 설치

```bash
cd ~/chewing-vision/day1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> macOS Apple Silicon 에서 `mediapipe` 휠이 없는 Python 버전(3.13 등)이면 설치 실패할 수 있다.
> 안전한 선택: **Python 3.11**.

## 2. 영상 준비

영상 자체는 직접 받아서 `~/chewing-vision/day1/` 안에 넣는다.

**왜 코드에 URL 을 하드코딩 안 했는지**
- 특정 크리에이터 채널을 ML 처리 대상으로 콕 집어 추천하는 건 IP 측면에서 깔끔하지 않다. 본인 판단으로 한 명 골라야 한다.
- 어떤 영상이 우리 파이프라인에 잘 맞는지는 *눈으로 미리 본 사람*만 안다.

**"chewing 잘 보이는" 영상의 조건**
| 조건 | 이유 |
|---|---|
| 얼굴 정면 (yaw 30° 이내) | MediaPipe 가 정면에서 가장 안정 |
| 입이 손/마이크에 안 가려짐 | 입술 landmark 가 잡혀야 MAR 계산 가능 |
| 1분 이상 *주로 씹기*인 구간 | 말하기/리액션 비중 큰 영상은 신호 섞임 |
| 자연 조명 또는 균일 조명 | 어두우면 landmark jitter ↑ |
| 1080p 이상 권장 | 미세한 턱 움직임 해상도 |

후보 장르 (직접 검색해서 1명만 고르기): mukbang, 요리/먹방 리뷰, 짧은 food review.
정한 영상에서 **차분히 씹는 1분 구간**을 골라 `--start`, `--end` 로 자른다.

다운로드 도구 예 (직접 결정):
```bash
# yt-dlp 가 있으면 (개인 학습 용도)
yt-dlp -f "bv*[height<=1080]+ba/b[height<=1080]" -o "%(title).50s.%(ext)s" "<URL>"
```

원본 영상 자체는 git 에 커밋하지 않을 것. `.gitignore` 에 `*.mp4`, `*.mkv`, `*.webm` 추가.

## 3. 실행

```bash
python chewing_demo.py mukbang_clip.mp4 --start 120 --end 180
```

처음 실행 시 `face_landmarker.task` 모델(약 5MB) 을 한 번 받아옴.

## 4. 출력 해석

콘솔 예:
```
[stats] face detection rate: 96.3%
[result] duration       : 60.0 s
[result] MAR     chews  :   78   rate 78.0 chews/min
[result] jawOpen chews  :   72   rate 72.0 chews/min
```

`chewing_signal.png` — 두 신호(MAR, jawOpen) 시계열 + 빨간 점이 자동 검출된 봉우리.

### 봐야 할 것
1. **face detection rate ≥ 80%** 인가? 낮으면 영상이 부적합.
2. 두 그래프의 봉우리가 **시각적으로 씹기 동작과 매칭**되는가? (영상 함께 보면서)
3. MAR vs jawOpen 의 count 차이가 크면 어느 쪽이 더 깨끗한지 본인 눈으로 결정.
4. 1초당 봉우리 ≒ 1~2개 범위인가? (정상 chewing 빈도)

### 흔한 실패 패턴
| 증상 | 가능한 원인 | 다음 액션 |
|---|---|---|
| chews=0 또는 매우 적음 | landmark 잘 안 잡힘 | face detection rate 확인, 영상 변경 |
| chews 가 분당 수백 | 작은 흔들림까지 봉우리로 잡힘 | `detect_chews(..., prominence_ratio=0.1)` 로 키우기 |
| MAR 평탄, jawOpen 만 출렁임 | 입 살짝 벌리고 어금니로만 씹는 경우 | 정상. jawOpen 신뢰. |
| 두 신호 모두 잡음만 | 영상 조명/얼굴 가림 | 다른 구간 시도 |

## 5. 다음 단계

이 데모가 정상 동작하면 Day 1 게이트 G1 통과.
- Day 2: 1초 window 자동 라벨 + confounder 영상으로 false positive 측정
- Day 3: 본인 셀프 1분 영상으로 동일 파이프라인 — *실제* signal 검증

자세한 단계는 `~/.claude/plans/vision-serialized-stardust.md` 참고.
