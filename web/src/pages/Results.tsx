import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { insforge, fetchPredictions, fetchAllWindowStats } from '../lib/insforge'
import type { LosoRun, LosoResult, LosoPrediction, SessionWindowStats } from '../types'

function Bar({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs w-16 shrink-0" style={{ color: 'var(--muted)' }}>{label}</span>
      <div className="flex-1 h-5 rounded overflow-hidden relative" style={{ background: 'var(--surface2)' }}>
        <div className="h-full rounded transition-all" style={{ width: `${pct}%`, background: color }} />
        <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white mix-blend-difference">
          {pct}%
        </span>
      </div>
    </div>
  )
}

const LABEL = (v: number) => v === 1 ? 'chewing' : 'rest'

function MisclassPanel({ runId, result, onNavigate }: {
  runId: number
  result: LosoResult
  onNavigate: (sessionId: string, windowId: number | null, tStart: number) => void
}) {
  const [preds, setPreds] = useState<LosoPrediction[] | null>(null)
  const [open, setOpen] = useState(false)

  async function toggle() {
    if (!open && preds === null) {
      const all = await fetchPredictions(runId, result.session_id)
      setPreds(all.filter(p => p.y_true !== p.y_pred))
    }
    setOpen(o => !o)
  }

  const miscount = preds?.length ?? null

  return (
    <div className="mt-3 border-t pt-3" style={{ borderColor: 'var(--border)' }}>
      <button
        className="text-xs px-2 py-1 rounded"
        style={{ background: 'var(--surface2)', color: 'var(--blue)' }}
        onClick={toggle}
      >
        {open ? '▲ 닫기' : `▼ 틀린 윈도우 보기${miscount !== null ? ` (${miscount})` : ''}`}
      </button>

      {open && preds !== null && (
        <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
          {preds.length === 0 ? (
            <p className="text-xs" style={{ color: 'var(--muted)' }}>틀린 윈도우 없음</p>
          ) : preds.map(p => (
            <div key={p.id} className="flex items-center gap-2 px-2 py-1 rounded text-xs" style={{ background: 'var(--surface2)' }}>
              <span style={{ color: 'var(--muted)', fontVariantNumeric: 'tabular-nums' }}>
                t={p.t_start.toFixed(1)}s
              </span>
              <span style={{ color: 'var(--rest)' }}>GT:{LABEL(p.y_true)}</span>
              <span style={{ color: 'var(--blue)' }}>→</span>
              <span style={{ color: 'var(--chewing)' }}>예측:{LABEL(p.y_pred)}</span>
              <button
                className="ml-auto px-2 py-0.5 rounded text-xs"
                style={{ background: 'var(--blue)', color: '#fff' }}
                onClick={() => onNavigate(p.session_id, p.window_id, p.t_start)}
              >
                라벨
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Results() {
  const nav = useNavigate()
  const [runs, setRuns] = useState<LosoRun[]>([])
  const [results, setResults] = useState<LosoResult[]>([])
  const [selectedRun, setSelectedRun] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [windowStats, setWindowStats] = useState<Record<string, SessionWindowStats>>({})

  useEffect(() => {
    async function load() {
      setLoading(true)
      setError('')
      try {
        const { data: runData, error: runErr } = await insforge.database
          .from('loso_runs').select('*').order('run_at', { ascending: false })
        if (runErr) throw runErr
        const allRuns = runData ?? []
        setRuns(allRuns)
        if (allRuns.length > 0) {
          const latestId = allRuns[0].id
          setSelectedRun(latestId)
          const { data: resData, error: resErr } = await insforge.database
            .from('loso_results').select('*').eq('run_id', latestId).order('session_id')
          if (resErr) throw resErr
          setResults(resData ?? [])
        }
        const stats = await fetchAllWindowStats()
        setWindowStats(stats)
      } catch (e) {
        setError(String(e))
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  async function switchRun(runId: number) {
    setSelectedRun(runId)
    const { data } = await insforge.database
      .from('loso_results').select('*').eq('run_id', runId).order('session_id')
    setResults(data ?? [])
  }

  function goAnnotate(sessionId: string, windowId: number | null, tStart: number) {
    const params = new URLSearchParams({ mode: 'all' })
    if (windowId != null) params.set('window', String(windowId))
    else params.set('t', String(tStart))
    nav(`/annotate/${sessionId}?${params}`)
  }

  const run = runs.find(r => r.id === selectedRun)

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <nav className="flex items-center gap-4 px-6 py-3 border-b" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <button onClick={() => nav('/')} className="text-sm" style={{ color: 'var(--blue)' }}>← Sessions</button>
        <span className="font-semibold text-lg" style={{ color: 'var(--text)' }}>LOSO Results</span>
        {runs.length > 1 && (
          <select
            className="ml-auto text-sm rounded px-2 py-1"
            style={{ background: 'var(--surface2)', color: 'var(--text)', border: '1px solid var(--border)' }}
            value={selectedRun ?? ''}
            onChange={e => switchRun(Number(e.target.value))}
          >
            {runs.map(r => (
              <option key={r.id} value={r.id}>
                {new Date(r.run_at).toLocaleString('ko-KR')} {r.notes ? `— ${r.notes}` : ''}
              </option>
            ))}
          </select>
        )}
      </nav>

      <main className="p-6 max-w-3xl mx-auto space-y-6">
        {loading ? (
          <p style={{ color: 'var(--muted)' }}>Loading...</p>
        ) : error ? (
          <pre className="text-xs p-3 rounded" style={{ color: '#f87171', background: 'var(--surface)', whiteSpace: 'pre-wrap' }}>
            {error}
          </pre>
        ) : runs.length === 0 ? (
          <div className="rounded-xl p-6 border text-center" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
            <p className="text-sm mb-2" style={{ color: 'var(--muted)' }}>아직 LOSO 결과가 없습니다.</p>
            <code className="text-xs" style={{ color: 'var(--blue)' }}>
              .venv/bin/python ml/save_loso_results.py
            </code>
            <p className="text-xs mt-1" style={{ color: 'var(--muted)' }}>실행 후 새로고침하세요.</p>
          </div>
        ) : (
          <>
            {run && (
              <div className="rounded-xl p-5 border" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-semibold" style={{ color: 'var(--text)' }}>전체 (Pooled)</h2>
                  <span className="text-xs" style={{ color: 'var(--muted)' }}>
                    {run.n_sessions}개 세션 · {new Date(run.run_at).toLocaleString('ko-KR')}
                    {run.notes && <span> · {run.notes}</span>}
                  </span>
                </div>
                <div className="space-y-2">
                  <Bar value={run.pooled_accuracy}   color="var(--blue)"    label="Accuracy" />
                  <Bar value={run.pooled_f1_chewing} color="var(--chewing)" label="F1-chew" />
                  <Bar value={run.pooled_f1_rest}    color="var(--muted)"   label="F1-rest" />
                </div>
                <div className="mt-3 flex gap-6 text-xs" style={{ color: 'var(--muted)' }}>
                  <span>Acc <b style={{ color: 'var(--text)' }}>{(run.pooled_accuracy * 100).toFixed(1)}%</b></span>
                  <span>F1-chew <b style={{ color: 'var(--chewing)' }}>{run.pooled_f1_chewing.toFixed(3)}</b></span>
                  <span>F1-rest <b style={{ color: 'var(--text)' }}>{run.pooled_f1_rest.toFixed(3)}</b></span>
                </div>
              </div>
            )}

            <h2 className="font-semibold text-sm" style={{ color: 'var(--muted)' }}>세션별 LOSO Fold</h2>
            <div className="space-y-3">
              {results.map(r => (
                <div key={r.id} className="rounded-xl p-4 border" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
                  <div className="flex items-center justify-between mb-3">
                    <p className="font-mono text-sm font-medium" style={{ color: 'var(--text)' }}>{r.session_id}</p>
                    <div className="text-xs flex gap-3" style={{ color: 'var(--muted)' }}>
                      <span>train {r.n_train}w</span>
                      <span>test {r.n_test}w</span>
                      <span>est. chews {r.estimated_chews}</span>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <Bar value={r.accuracy}   color="var(--blue)"    label="Accuracy" />
                    <Bar value={r.f1_chewing} color="var(--chewing)" label="F1-chew" />
                    <Bar value={r.f1_rest}    color="var(--muted)"   label="F1-rest" />
                  </div>
                  <div className="mt-2 flex gap-4 text-xs" style={{ color: 'var(--muted)' }}>
                    <span>test chew {(r.test_chew_ratio * 100).toFixed(0)}%</span>
                    <span>train chew {(r.train_chew_ratio * 100).toFixed(0)}%</span>
                  </div>
                  {selectedRun && (
                    <MisclassPanel runId={selectedRun} result={r} onNavigate={goAnnotate} />
                  )}
                </div>
              ))}
            </div>

            {(() => {
              const readyCount = Object.values(windowStats).filter(s => s.total > 0 && s.unlabeled_disagree === 0).length
              const pendingCount = Object.values(windowStats).filter(s => s.unlabeled_disagree > 0).length
              if (readyCount === 0 && pendingCount === 0) return null
              return (
                <div className="rounded-xl p-4 border" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
                  <p className="text-sm font-semibold mb-3" style={{ color: 'var(--text)' }}>파이프라인 상태</p>
                  <div className="flex gap-4 text-xs mb-3">
                    <span><span style={{ color: '#22c55e' }}>{readyCount}개</span> LOSO 준비 완료</span>
                    {pendingCount > 0 && <span><span style={{ color: '#f59e0b' }}>{pendingCount}개</span> 라벨 필요</span>}
                  </div>
                  {readyCount > 0 && (
                    <div className="rounded-lg p-2 text-xs" style={{ background: 'var(--surface2)', color: 'var(--muted)' }}>
                      <span style={{ color: 'var(--text)' }}>실행:</span>{' '}
                      <code style={{ color: 'var(--blue)' }}>.venv/bin/python ml/save_loso_results.py --notes "설명"</code>
                    </div>
                  )}
                  {pendingCount > 0 && (
                    <p className="text-xs mt-2" style={{ color: 'var(--muted)' }}>
                      라벨 미완료 세션은 Sessions 페이지에서 확인하세요.
                    </p>
                  )}
                </div>
              )
            })()}

            <div className="rounded-xl p-4 border text-xs space-y-1" style={{ background: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--muted)' }}>
              <p className="font-medium" style={{ color: 'var(--text)' }}>LOSO CV란?</p>
              <p>· Leave-One-Session-Out: 한 세션을 test로, 나머지를 train으로 반복 평가</p>
              <p>· 모델 일반화 성능 측정 — 새 사람에게도 잘 동작하는지 확인</p>
              <p>· Random Forest, n_estimators=100, composite signal (jaw_open 70% + MAR 30%)</p>
              <p className="mt-2 pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
                결과 업데이트: <code style={{ color: 'var(--blue)' }}>.venv/bin/python ml/save_loso_results.py --notes "설명"</code>
              </p>
              <p className="mt-1">
                CoreML export (첫 실행 시 <code style={{ color: 'var(--blue)' }}>bash ml/setup_cml_env.sh</code> 먼저):
              </p>
              <p>
                <code style={{ color: 'var(--blue)' }}>cml_env/bin/python ml/coreml_convert.py --notes "설명"</code>
                <span> → ml/models/chewing_v*.mlmodel</span>
              </p>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
