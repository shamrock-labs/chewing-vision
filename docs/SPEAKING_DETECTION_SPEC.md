# Speaking Detection Spec — chewing-vision

**버전**: v0.1 (draft)  
**작성일**: 2026-05-14  
**상태**: 리서치 기반 설계안. 구현 전 팀 리뷰 필요.

---

## 1. 문제 정의

### 현재 상태

`engines/ours.py` `_build_windows()`의 라벨 로직:

```
face_rate < 0.6  → bad_face
n_events >= 1    → chewing   ← 문제
else             → rest
```

말하는 동안 입이 열리면 `jaw_open` peak가 감지되어 **chewing으로 잘못 라벨링**된다.  
결과: ML 모델이 "speaking 중 턱 움직임 = chewing"으로 학습 → false positive 증가.

### 실제 관찰 (현재 4세션)

`labels_ours.csv`에 speaking 라벨이 **0건** — speaking이 chewing 또는 rest로 흡수되고 있음.

---

## 2. 리서치 요약 (판단 근거)

### 선행 연구 speaking 처리 방식

| 시스템 | 방식 | 성능 |
|--------|------|------|
| EarBit (2017, IMU) | 35초 이동 평균 후처리. 명시적 분류기 없음 | F1=0.80. "speaking이 주요 미해결 false positive" 명시 |
| IMChew (2024, AirPods IMU) | non-chewing 활동으로 묶어 학습 | 세부 미공개 |
| Cadavid (2012, vision) | AAM + 스펙트럼 분석 → SVM | chewing quasi-periodicity vs speaking 비주기성으로 구분 |
| OCOsense (2024, 스마트 안경) | HMM으로 temporal regularity 측정 | "eating이 talking보다 더 규칙적이고 강함" |

### 핵심 물리 수치

| 활동 | Jaw 움직임 주파수 | 특성 |
|------|-----------------|------|
| Chewing | **0.8–1.2 Hz** (일반), < 3 Hz (최대) | 규칙적, quasi-sinusoidal |
| Speaking | **> 2 Hz**, 복수 주파수 성분 | 불규칙, inter-peak interval 변동 큼 |
| **겹침 구간** | **2–3 Hz** | 완전 분리 불가 |

### 분야 공통 결론

1. Speaking이 chewing 감지 분야 **전체에서 가장 큰 미해결 혼동 원인**으로 반복 지목됨
2. Vision/IMU 기반 연구에서 speaking을 **별도 클래스로 학습**한 사례는 없음 (오디오 기반만 가능)
3. 구분 방법은 두 가지로 수렴:
   - **주기성 분석**: 스펙트럼 / inter-peak interval CV
   - **시간 후처리**: 짧은 burst 무시, 긴 bout만 인정

---

## 3. 우리 파이프라인에서 가능한 접근 옵션

### Option A: 주기성 기반 (inter-peak interval CV)

윈도우 내 jaw_open 이벤트 간격의 **변동계수(CV)**로 판단.

```
이벤트 간격 평균 < 0.5s (>2Hz)  → speaking 후보
CV > 0.8 (불규칙)               → speaking 후보
둘 다 해당                      → speaking
```

- 장점: 추가 신호 없이 구현 가능. 리서치에서 근거 있음 (Cadavid 2012).
- 단점: 1초 윈도우에 이벤트 2개 이하면 CV 계산 불안정. 겹침 구간(2–3Hz) 오분류 가능.
- 필요 최소 이벤트 수: **≥ 3** (CV 계산 신뢰도 확보).

### Option B: 주파수 필터 기반 (후처리)

이벤트 기반이 아닌 **신호 자체에 bandpass filter** 적용.

```
jaw_open 시계열에 0.5–2.0 Hz bandpass filter 적용
→ speaking 주파수 성분 제거 후 peak 재검출
```

- 장점: 윈도우 레벨이 아닌 신호 레벨에서 처리 → 더 깨끗한 chewing 신호.
- 단점: 현재 `apply_smoothing()`이 low-pass만 적용. bandpass 추가 필요. 2Hz 경계에서 정상 chewing도 일부 제거될 수 있음.

### Option C: MAR + jaw_open 조합 판단

Speaking 시 jaw_open은 높지만 **MAR(Mouth Aspect Ratio)는 낮은 경향**이 있음.  
씹을 때는 입 폭이 좁아지고 세로로 열리지만, 말할 때는 입 모양이 더 다양함.

```
jaw_open peak 감지 AND MAR 동시 상승 없음 → speaking 후보
```

- 장점: 이미 두 신호 모두 계산 중. 추가 연산 최소.
- 단점: MAR과 jaw_open이 씹기에서 항상 동조하는지 세션 데이터로 검증 필요. 식품 종류에 따라 달라질 수 있음.

### Option D: 라벨 없이 — ML 쪽에서 speaking 구간 제외 (None masking)

Vision 라벨 변경 없이, ML `assign_label()`에서 speaking이 감지된 구간은 학습에서 제외.

- 단점: 현재 4세션 labels_ours.csv에 speaking 라벨이 0건 → 즉시 효과 없음.  
  Vision이 speaking을 라벨링하기 전까지는 사실상 무의미.

---

## 4. 권장 구현 방향 (v1)

**2단계 접근**:

### Phase 1 — Option A (즉시 구현 가능)

`_build_windows()`에 inter-peak interval 분석 추가.  
이벤트 ≥ 3개인 윈도우에서 CV와 평균 간격을 계산해 speaking 판단.

```python
# 판단 조건 (임계값은 세션 데이터 보고 조정)
SPEAKING_MAX_INTERVAL = 0.5   # 초 — 이보다 짧으면 너무 빠름 (>2Hz)
SPEAKING_MIN_CV       = 0.7   # 변동계수 — 이보다 높으면 불규칙
MIN_EVENTS_FOR_CV     = 3     # CV 계산 최소 이벤트 수
```

**임계값 결정 방법**: 현재 4세션 frame_signals_ours.csv에서 말하는 구간(수동 식별)의 peak interval 분포를 먼저 시각화한 후 결정. 코드 수정 전 데이터 분석 선행 필요.

### Phase 2 — Option B (세션 늘어난 후)

데이터 30세션+ 확보되면 bandpass filter 적용 후 성능 비교.

---

## 5. 라벨 변경 사항

### `LABEL_VOCAB` (labels.py) — 변경 없음

이미 `speaking`이 정의되어 있음. 코드 변경 불필요.

### `_build_windows()` (engines/ours.py) — 변경

```python
# 기존
elif n_events >= 1:
    label = "chewing"

# 변경 후 (Phase 1)
elif n_events >= 1:
    label, confidence = _classify_jaw_events(event_times_in_window)
```

```python
def _classify_jaw_events(event_times: list[float]) -> tuple[str, float]:
    """jaw_open 이벤트 시간 목록으로 chewing vs speaking 판단."""
    if len(event_times) < MIN_EVENTS_FOR_CV:
        return "chewing", 0.6  # 이벤트 적으면 기본 chewing
    intervals = np.diff(sorted(event_times))
    mean_interval = float(np.mean(intervals))
    cv = float(np.std(intervals) / (mean_interval + 1e-6))
    if mean_interval < SPEAKING_MAX_INTERVAL and cv > SPEAKING_MIN_CV:
        return "speaking", 0.6
    return "chewing", min(1.0, 0.5 + 0.1 * len(event_times))
```

### `assign_label()` (ml/utils.py) — Phase 1 이후 추가

```python
def assign_label(t_start, t_end, labels, chewing_threshold=0.5):
    overlap = labels[(labels["t_end"] > t_start) & (labels["t_start"] < t_end)]
    if overlap.empty:
        return 0
    # speaking/drinking 구간은 None 반환 → 학습 제외
    ambiguous_labels = {"speaking", "drinking", "occluded"}
    if overlap["label"].isin(ambiguous_labels).any():
        return None
    chewing_count = (overlap["label"] == "chewing").sum()
    return 1 if chewing_count / len(overlap) >= chewing_threshold else 0
```

`make_windows_with_times()`에서 `None` 반환 윈도우 skip:

```python
label = assign_label(t, t + window_sec, labels)
if label is None:
    t += stride_sec
    continue
```

---

## 6. 검증 계획

### 구현 전 (데이터 분석 선행)

1. 현재 4세션 중 **말하는 구간을 수동으로 특정** (영상 직접 확인)
2. 해당 구간의 `frame_signals_ours.csv`에서 jaw_open peak interval 분포 시각화
3. chewing 구간 peak interval 분포와 비교 → 임계값 결정

### 구현 후 (재분석)

```bash
# 기존 세션 재분석
chewing analyze /path/to/session.mp4

# labels_ours.csv에 speaking 라벨 확인
cut -d',' -f3 /tmp/chewing_*/labels_ours.csv | sort | uniq -c

# LOSO 재실행 → F1 변화 확인
.venv/bin/python ml/compare_sessions.py
```

**성공 기준**:
- labels_ours.csv에 speaking 라벨 > 0건 생성
- Pooled F1(chewing) 유지 또는 개선 (현재 0.733)
- Pooled F1(rest) 개선 (현재 0.594 — speaking 오분류 제거 효과)

---

## 7. 미결 질문 (구현 전 결정 필요)

1. **임계값**: `SPEAKING_MAX_INTERVAL=0.5s`, `SPEAKING_MIN_CV=0.7`은 가설값. 데이터 분석 후 조정 필요.
2. **1초 윈도우의 한계**: 1초 안에 이벤트 3개 이상 나오는 경우가 얼마나 되는가? 이벤트 부족하면 CV 계산 불가 → chewing 기본값 fallback.
3. **MAR 활용 (Option C)**: Phase 1 이후 추가 검토. jaw_open과 MAR의 상관관계를 세션 데이터로 먼저 측정.
4. **세션 수집 프로토콜 변경**: 앞으로 수집 세션에 **말하는 구간을 의도적으로 포함**해서 speaking 라벨 학습 데이터 확보.

---

## 8. 관련 파일

| 파일 | 역할 |
|------|------|
| `chewing/engines/ours.py` | `_build_windows()` — speaking 판단 로직 추가 위치 |
| `chewing/labels.py` | `LABEL_VOCAB` — 이미 speaking 포함, 변경 불필요 |
| `ml/utils.py` | `assign_label()` — None masking 추가 위치 |
| `ml/compare_sessions.py` | `make_windows_with_times()` — None skip 처리 위치 |
