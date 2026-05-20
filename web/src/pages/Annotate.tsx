import { useEffect, useRef, useState, useCallback } from 'react'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { fetchWindows, saveLabel, batchSaveLabels, videoUrl, signalsUrl, imuUrl, fetchLatestImuPreds, matchImuPred } from '../lib/insforge'
import SignalChart from '../components/SignalChart'
import ImuChart from '../components/ImuChart'
import ProgressStrip from '../components/ProgressStrip'
import type { WindowRow, SignalPoint, ImuPoint, HumanLabel } from '../types'

function parseSignalCsv(text: string): SignalPoint[] {
  const lines = text.trim().split('\n')
  const header = lines[0].split(',')
  const ti = header.indexOf('t_sec')
  const ji = header.indexOf('jaw_open')
  const mi = header.indexOf('mar')
  return lines.slice(1)
    .filter((_, i) => i % 3 === 0)
    .map(line => {
      const cols = line.split(',')
      return { t: parseFloat(cols[ti]), jaw_open: parseFloat(cols[ji]), mar: parseFloat(cols[mi]) }
    })
    .filter(p => !isNaN(p.t))
}

function parseImuCsv(text: string): ImuPoint[] {
  const lines = text.trim().split('\n')
  const header = lines[0].split(',')
  const ti   = header.indexOf('t_rel_sec')
  const rxi  = header.indexOf('rotation_x')
  const ryi  = header.indexOf('rotation_y')
  const rzi  = header.indexOf('rotation_z')
  const axi  = header.indexOf('user_accel_x')
  const ayi  = header.indexOf('user_accel_y')
  const azi  = header.indexOf('user_accel_z')
  return lines.slice(1)
    .filter((_, i) => i % 2 === 0)   // downsample 2x → ~25 Hz
    .map(line => {
      const c = line.split(',')
      return {
        t:       parseFloat(c[ti]),
        rot_x:   parseFloat(c[rxi]),
        rot_y:   parseFloat(c[ryi]),
        rot_z:   parseFloat(c[rzi]),
        accel_x: parseFloat(c[axi]),
        accel_y: parseFloat(c[ayi]),
        accel_z: parseFloat(c[azi]),
      }
    })
    .filter(p => !isNaN(p.t))
}

function labelColor(label: string | null) {
  if (label === 'chewing')  return 'var(--chewing)'
  if (label === 'rest')     return 'var(--rest)'
  if (label === 'bad_face') return 'var(--bad)'
  return 'var(--muted)'
}

export default function Annotate() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const [searchParams] = useSearchParams()
  const nav = useNavigate()
  const mode = (searchParams.get('mode') ?? 'disagree') as 'disagree' | 'all'

  const [windows, setWindows] = useState<WindowRow[]>([])
  const [signals, setSignals] = useState<SignalPoint[]>([])
  const [imu, setImu] = useState<ImuPoint[]>([])
  const [imuPreds, setImuPreds] = useState<Record<number, number>>({})
  const [idx, setIdx] = useState(0)
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const syncFromVideoRef = useRef(false)   // true when idx was updated by timeupdate, not user
  const windowsRef = useRef<WindowRow[]>([])
  const idxRef = useRef(0)
  useEffect(() => { windowsRef.current = windows }, [windows])
  useEffect(() => { idxRef.current = idx }, [idx])

  useEffect(() => {
    if (!sessionId) return
    const windowParam = searchParams.get('window')
    const tParam = searchParams.get('t')
    fetchWindows(sessionId).then(rows => {
      // auto-fill human_label for windows where both models agree
      const toFill = rows.filter(w =>
        !w.human_label &&
        w.jaw_open_label === w.composite_label &&
        (w.jaw_open_label === 'chewing' || w.jaw_open_label === 'rest')
      )
      let updatedRows = rows
      if (toFill.length > 0) {
        const chewIds = toFill.filter(w => w.jaw_open_label === 'chewing').map(w => w.id)
        const restIds = toFill.filter(w => w.jaw_open_label === 'rest').map(w => w.id)
        const updates: Array<{ ids: number[]; label: HumanLabel }> = [
          ...(chewIds.length ? [{ ids: chewIds, label: 'chewing' as HumanLabel }] : []),
          ...(restIds.length ? [{ ids: restIds, label: 'rest' as HumanLabel }] : []),
        ]
        batchSaveLabels(updates).catch(console.error)
        const fillSet = new Set(toFill.map(w => w.id))
        updatedRows = rows.map(w =>
          fillSet.has(w.id) ? { ...w, human_label: w.jaw_open_label as HumanLabel } : w
        )
      }

      const filtered = mode === 'disagree'
        ? updatedRows.filter(w => w.jaw_open_label !== w.composite_label)
        : updatedRows
      setWindows(filtered)
      if (windowParam) {
        const targetId = Number(windowParam)
        const ti = filtered.findIndex(w => w.id === targetId)
        setIdx(ti >= 0 ? ti : 0)
      } else if (tParam) {
        const tVal = parseFloat(tParam)
        const ti = filtered.reduce((best, w, i) =>
          Math.abs(w.t_start - tVal) < Math.abs(filtered[best].t_start - tVal) ? i : best, 0)
        setIdx(ti)
      } else {
        const firstUnlabeled = filtered.findIndex(w => !w.human_label)
        setIdx(firstUnlabeled >= 0 ? firstUnlabeled : 0)
      }
    })
    fetch(signalsUrl(sessionId))
      .then(r => r.text())
      .then(text => setSignals(parseSignalCsv(text)))
      .catch(() => setSignals([]))
    fetch(imuUrl(sessionId))
      .then(r => r.text())
      .then(text => setImu(parseImuCsv(text)))
      .catch(() => setImu([]))
    fetchLatestImuPreds(sessionId).then(setImuPreds).catch(() => {})
  }, [sessionId, mode])

  // Seek video when window changes (skip if idx was driven by video playback)
  useEffect(() => {
    const win = windows[idx]
    if (win && videoRef.current && !syncFromVideoRef.current) {
      videoRef.current.currentTime = win.t_start
    }
    syncFromVideoRef.current = false
  }, [idx, windows])

  // Update idx from video playback position
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handler = () => {
      const t = video.currentTime
      const ws = windowsRef.current
      const newIdx = ws.findIndex(w => t >= w.t_start && t < w.t_end)
      if (newIdx >= 0 && newIdx !== idxRef.current) {
        syncFromVideoRef.current = true
        setIdx(newIdx)
      }
    }
    video.addEventListener('timeupdate', handler)
    return () => video.removeEventListener('timeupdate', handler)
  }, [])

  const applyLabel = useCallback(async (label: HumanLabel) => {
    const win = windows[idx]
    if (!win || saving) return
    setSaving(true)
    try {
      await saveLabel(win.id, label)
      setWindows(prev => prev.map((w, i) =>
        i === idx ? { ...w, human_label: label } : w
      ))
      setStatus(`Saved: ${label}`)
      setTimeout(() => setStatus(''), 1500)
      setIdx(i => Math.min(i + 1, windows.length - 1))
    } catch {
      setStatus('Save failed')
    } finally {
      setSaving(false)
    }
  }, [windows, idx, saving])

  const nextUnlabeled = useCallback(() => {
    const ni = windows.findIndex((w, i) => i > idx && !w.human_label)
    if (ni >= 0) setIdx(ni)
  }, [windows, idx])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      e.preventDefault()
      if (e.key === 'c') applyLabel('chewing')
      else if (e.key === 'r') applyLabel('rest')
      else if (e.key === 'b') applyLabel('bad_face')
      else if (e.key === 'a' || e.key === 'ArrowLeft')  setIdx(i => Math.max(i - 1, 0))
      else if (e.key === 'd' || e.key === 'ArrowRight') setIdx(i => Math.min(i + 1, windows.length - 1))
      else if (e.key === 'Tab') nextUnlabeled()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [applyLabel, nextUnlabeled, windows.length])

  const win = windows[idx]
  const labeled = windows.filter(w => w.human_label).length

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>
      {/* Navbar */}
      <nav className="flex items-center gap-3 px-4 py-2 shrink-0 border-b" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <button onClick={() => nav('/')} className="text-sm" style={{ color: 'var(--blue)' }}>← Sessions</button>
        <span className="font-mono text-sm font-medium" style={{ color: 'var(--text)' }}>{sessionId}</span>
        <span className="ml-auto text-xs" style={{ color: 'var(--muted)' }}>
          {labeled}/{windows.length} labeled
          {status && <span className="ml-3" style={{ color: 'var(--chewing)' }}>{status}</span>}
        </span>
        <div className="flex gap-1 ml-2">
          {(['disagree', 'all'] as const).map(m => (
            <button key={m}
              className="px-2 py-1 rounded text-xs"
              style={{ background: mode === m ? 'var(--blue)' : 'var(--surface2)', color: mode === m ? '#fff' : 'var(--muted)' }}
              onClick={() => nav(`/annotate/${sessionId}?mode=${m}`)}
            >{m}</button>
          ))}
        </div>
      </nav>

      {/* Main split */}
      <div className="flex flex-1 min-h-0">
        {/* Left: video */}
        <div className="flex flex-col w-3/5 min-w-0 border-r" style={{ borderColor: 'var(--border)' }}>
          {/* Top info bar */}
          <div className="flex items-center gap-4 px-4 py-2 shrink-0 border-b" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
            <span className="text-sm font-mono" style={{ color: 'var(--text)' }}>
              Window {idx + 1} / {windows.length}
            </span>
            {win && (
              <span className="text-sm" style={{ color: 'var(--muted)' }}>
                t = {win.t_start.toFixed(1)} – {win.t_end.toFixed(1)} s
              </span>
            )}
          </div>

          {/* Video */}
          <div className="flex-1 bg-black relative flex items-center justify-center min-h-0">
            {sessionId && (
              <video
                ref={videoRef}
                src={videoUrl(sessionId)}
                className="max-h-full max-w-full object-contain"
                controls
                muted
                playsInline
              />
            )}
            {win && (
              <div className="absolute left-0 top-0 bottom-0 w-1.5"
                style={{ background: labelColor(win.human_label) }} />
            )}
          </div>

          {/* Label bar */}
          {win && (
            <div className="flex items-center gap-6 px-4 py-3 shrink-0 border-t" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
              <span className="text-sm" style={{ color: 'var(--muted)' }}>Label:</span>
              <span className="text-xl font-semibold" style={{ color: labelColor(win.human_label) }}>
                {win.human_label ?? '—'}
              </span>
              <div className="flex gap-2 ml-auto">
                {(['chewing', 'rest', 'bad_face'] as HumanLabel[]).map(l => (
                  <button key={l}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{
                      background: win.human_label === l ? labelColor(l) : 'var(--surface2)',
                      color: win.human_label === l ? '#fff' : labelColor(l),
                      border: `1px solid ${labelColor(l)}`,
                      opacity: saving ? 0.5 : 1,
                    }}
                    onClick={() => applyLabel(l)}
                    disabled={saving}
                  >
                    {l === 'chewing' ? '[c] chewing' : l === 'rest' ? '[r] rest' : '[b] bad'}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: signal + IMU panel */}
        <div className="flex flex-col w-2/5 min-w-0">
          {/* Signal header */}
          <div className="flex items-center gap-3 px-4 py-2 shrink-0 border-b" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
            <span className="text-sm font-medium" style={{ color: 'var(--text)' }}>Signal Trace</span>
            <div className="flex items-center gap-2 ml-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgba(74,158,237,0.4)' }} />
              <span className="text-xs" style={{ color: 'var(--muted)' }}>chew</span>
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgba(34,197,94,0.35)' }} />
              <span className="text-xs" style={{ color: 'var(--muted)' }}>rest</span>
              <span className="inline-block w-3 h-1 rounded-sm" style={{ background: 'rgba(239,68,68,0.7)' }} />
              <span className="text-xs" style={{ color: 'var(--muted)' }}>disagree</span>
            </div>
            {win && (
              <span className="text-xs ml-auto" style={{ color: 'var(--muted)' }}>
                jaw {win.jaw_open_mean?.toFixed(3) ?? '—'}  ·  mar {win.mar_mean?.toFixed(3) ?? '—'}
              </span>
            )}
          </div>

          {/* Charts container — vision + IMU stacked */}
          <div className="flex-1 min-h-0 flex flex-col">
            {/* Vision signals */}
            <div className="flex-[3] min-h-0 p-3 pb-1">
              {signals.length > 0 ? (
                <SignalChart
                  signals={signals}
                  windows={windows}
                  currentIdx={idx}
                  onWindowClick={setIdx}
                />
              ) : (
                <div className="h-full flex items-center justify-center text-sm" style={{ color: 'var(--muted)' }}>
                  No signal data
                </div>
              )}
            </div>

            {/* IMU divider */}
            <div className="shrink-0 flex items-center gap-3 px-4 py-1 border-t" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
              <span className="text-xs font-medium" style={{ color: 'var(--text)' }}>IMU Rotation</span>
              <span className="text-xs" style={{ color: 'var(--muted)' }}>
                <span style={{ color: '#f97316' }}>rot_z</span>=jaw ·{' '}
                <span style={{ color: '#a78bfa' }}>rot_x</span> ·{' '}
                <span style={{ color: '#34d399' }}>rot_y</span>
              </span>
            </div>

            {/* IMU chart */}
            <div className="flex-[2] min-h-0 p-3 pt-1">
              {imu.length > 0 ? (
                <ImuChart
                  imu={imu}
                  windows={windows}
                  currentIdx={idx}
                  onWindowClick={setIdx}
                />
              ) : (
                <div className="h-full flex items-center justify-center text-sm" style={{ color: 'var(--muted)' }}>
                  No IMU data
                </div>
              )}
            </div>
          </div>

          {/* Engine labels + shortcuts */}
          <div className="px-4 py-3 shrink-0 border-t space-y-2" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
            {win && (
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                <p className="text-xs" style={{ color: 'var(--muted)' }}>
                  <span style={{ color: 'var(--blue)' }}>jaw_open</span>: {win.jaw_open_label ?? '—'}
                  &nbsp;·&nbsp;
                  <span style={{ color: 'var(--chewing)' }}>composite</span>: {win.composite_label ?? '—'}
                </p>
                {win && (() => {
                  const pred = matchImuPred(imuPreds, win.t_start)
                  if (pred === null) {
                    return Object.keys(imuPreds).length > 0
                      ? <p className="text-xs" style={{ color: 'var(--border)' }}>IMU(RF): 예측 없음</p>
                      : null
                  }
                  const label = pred === 1 ? 'chewing' : 'rest'
                  const match = win.jaw_open_label === label
                  return (
                    <p className="text-xs">
                      <span style={{ color: '#a78bfa' }}>IMU(RF)</span>:{' '}
                      <span style={{ color: pred === 1 ? 'var(--chewing)' : 'var(--rest)', fontWeight: 600 }}>
                        {label}
                      </span>
                      {!match && (
                        <span style={{ color: '#f87171', marginLeft: 4 }}>← Vision과 불일치</span>
                      )}
                    </p>
                  )
                })()}
              </div>
            )}
            <div className="rounded-lg p-2 space-y-1" style={{ background: 'var(--surface2)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--text)' }}>라벨링 기준</p>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                · 윈도우: {win ? `${(win.t_end - win.t_start).toFixed(0)}초` : '1초'} 슬라이스
              </p>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                · <span style={{ color: 'var(--blue)' }}>jaw_open</span>: 턱 개방 신호만 사용한 모델
              </p>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                · <span style={{ color: 'var(--chewing)' }}>composite</span>: jaw_open 70% + MAR 30% 가중 합산
              </p>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                · <b style={{ color: 'var(--text)' }}>disagree</b> 모드: 두 모델 의견 불일치 윈도우만 표시
              </p>
            </div>
            <p className="text-xs" style={{ color: 'var(--border)' }}>
              [a/d] prev/next · [Tab] next unlabeled
            </p>
          </div>
        </div>
      </div>

      {/* Progress strip */}
      <div className="shrink-0 border-t" style={{ borderColor: 'var(--border)' }}>
        <ProgressStrip windows={windows} currentIdx={idx} onSelect={setIdx} />
      </div>
    </div>
  )
}
