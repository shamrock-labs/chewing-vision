import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchSessions, fetchAllWindowStats } from '../lib/insforge'
import type { Session, SessionWindowStats } from '../types'

type PipelineStatus = 'no_windows' | 'needs_label' | 'loso_ready'

function getPipelineStatus(stats: SessionWindowStats | undefined): PipelineStatus {
  if (!stats || stats.total === 0) return 'no_windows'
  if (stats.unlabeled_disagree > 0) return 'needs_label'
  return 'loso_ready'
}

function agreeRate(stats: SessionWindowStats): number | null {
  if (stats.labeled === 0) return null
  return stats.agree / stats.labeled
}

function AgreeRateBadge({ stats }: { stats: SessionWindowStats }) {
  const rate = agreeRate(stats)
  if (rate === null) return null
  const pct = Math.round(rate * 100)
  const color = pct >= 70 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#f87171'
  return (
    <span className="text-xs px-2 py-0.5 rounded-full font-mono" style={{ background: `${color}22`, color }}>
      agree {pct}%{pct >= 70 ? ' ✓' : ''}
    </span>
  )
}

function StatusBadge({ status, stats }: { status: PipelineStatus; stats?: SessionWindowStats }) {
  if (status === 'no_windows')
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(239,68,68,0.15)', color: '#f87171' }}>Vision 필요</span>
  if (status === 'needs_label')
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(245,158,11,0.15)', color: '#f59e0b' }}>라벨 {stats?.unlabeled_disagree}개 필요</span>
  return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e' }}>LOSO 준비</span>
}

export default function SessionList() {
  const nav = useNavigate()
  const [sessions, setSessions] = useState<Session[]>([])
  const [stats, setStats] = useState<Record<string, SessionWindowStats>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    async function load() {
      setLoading(true)
      setError('')
      try {
        const [raw, allStats] = await Promise.all([fetchSessions(), fetchAllWindowStats()])
        setSessions(raw)
        setStats(allStats)
      } catch (e) {
        setError(String(e))
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const losoReadyCount = sessions.filter(s => getPipelineStatus(stats[s.id]) === 'loso_ready').length
  const mlReadyCount = sessions.filter(s => {
    const st = stats[s.id]
    if (!st || st.labeled === 0) return false
    return st.agree / st.labeled >= 0.7
  }).length

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <nav className="flex items-center gap-4 px-6 py-3 border-b" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <span className="font-semibold text-lg" style={{ color: 'var(--blue)' }}>ChewingVision</span>
        <button onClick={() => nav('/results')} className="text-sm px-3 py-1 rounded font-medium" style={{ background: 'var(--surface2)', color: 'var(--text)' }}>
          📊 Results
        </button>
      </nav>

      <main className="p-6 max-w-2xl mx-auto">
        <div className="flex items-end justify-between mb-6">
          <h1 className="text-2xl font-semibold" style={{ color: 'var(--text)' }}>Sessions</h1>
          <div className="text-sm text-right space-y-0.5">
            {losoReadyCount > 0 && (
              <p style={{ color: 'var(--muted)' }}>
                <span style={{ color: '#22c55e' }}>{losoReadyCount}개</span> LOSO 준비 →{' '}
                <code className="text-xs" style={{ color: 'var(--blue)' }}>.venv/bin/python ml/save_loso_results.py</code>
              </p>
            )}
            {mlReadyCount > 0 && (
              <p style={{ color: 'var(--muted)' }}>
                <span style={{ color: '#22c55e' }}>{mlReadyCount}개</span> agree ≥70% → ML 학습 가능
              </p>
            )}
          </div>
        </div>

        {loading ? (
          <p style={{ color: 'var(--muted)' }}>Loading...</p>
        ) : error ? (
          <pre className="text-xs p-3 rounded" style={{ color: '#f87171', background: 'var(--surface)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {error}
          </pre>
        ) : sessions.length === 0 ? (
          <p style={{ color: 'var(--muted)' }}>
            No sessions in DB. Run <code>python ml/import_to_db.py</code> first.
          </p>
        ) : (
          <div className="flex flex-col gap-4">
            {sessions.map(s => {
              const st = stats[s.id]
              const status = getPipelineStatus(st)
              const disagreeLabeled = st ? st.disagree - st.unlabeled_disagree : 0
              const pct = st && st.disagree > 0 ? Math.round((disagreeLabeled / st.disagree) * 100) : (status === 'loso_ready' ? 100 : 0)

              return (
                <div key={s.id} className="rounded-xl p-5 border" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="font-mono font-medium truncate" style={{ color: 'var(--text)' }}>{s.id}</p>
                        <StatusBadge status={status} stats={st} />
                        {st && <AgreeRateBadge stats={st} />}
                      </div>
                      <p className="text-xs mt-1" style={{ color: 'var(--muted)' }}>
                        {st ? (
                          <>
                            전체 {st.total}개 · 불일치 {st.disagree}개
                            {st.disagree > 0 && ` · 라벨 ${disagreeLabeled}/${st.disagree}`}
                          </>
                        ) : '윈도우 없음'}
                      </p>
                      {st && st.disagree > 0 && (
                        <>
                          <div className="mt-3 h-2 rounded-full overflow-hidden" style={{ background: 'var(--surface2)' }}>
                            <div className="h-full rounded-full transition-all"
                              style={{ width: `${pct}%`, background: pct === 100 ? '#22c55e' : 'var(--blue)' }} />
                          </div>
                          <p className="text-xs mt-1" style={{ color: pct === 100 ? '#22c55e' : 'var(--muted)' }}>
                            {pct === 100 ? '✓ 불일치 라벨 완료' : `${pct}%`}
                          </p>
                        </>
                      )}
                      {status === 'no_windows' && (
                        <p className="text-xs mt-2 font-mono" style={{ color: 'var(--muted)' }}>
                          → <code style={{ color: 'var(--blue)' }}>.venv/bin/python -m chewing.cli annotate sessions/{s.id.split('_')[1]}</code>
                        </p>
                      )}
                    </div>
                    <div className="flex flex-col gap-2 items-end">
                      {status !== 'no_windows' && (
                        <button
                          className="px-4 py-2 rounded-lg text-sm font-medium shrink-0"
                          style={{ background: 'var(--blue)', color: '#fff' }}
                          onClick={() => nav(`/annotate/${s.id}?mode=review`)}
                        >
                          리뷰 라벨링 →
                        </button>
                      )}
                      <button
                        className="px-3 py-1 rounded text-xs"
                        style={{ background: 'var(--surface2)', color: 'var(--muted)' }}
                        onClick={() => nav(`/annotate/${s.id}?mode=all`)}
                      >
                        전체 보기
                      </button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* Pipeline guide */}
        <div className="mt-8 rounded-xl p-4 border text-xs space-y-1" style={{ background: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--muted)' }}>
          <p className="font-medium mb-2" style={{ color: 'var(--text)' }}>파이프라인 흐름</p>
          <p><span style={{ color: '#f87171' }}>●</span> Vision 필요 → <code style={{ color: 'var(--blue)' }}>python ml/import_to_db.py</code> 실행</p>
          <p><span style={{ color: '#f59e0b' }}>●</span> 라벨 필요 → 리뷰 라벨링 버튼으로 vision/IMU 불일치 후보 검토</p>
          <p><span style={{ color: '#22c55e' }}>●</span> LOSO 준비 → <code style={{ color: 'var(--blue)' }}>.venv/bin/python ml/save_loso_results.py --notes "메모"</code></p>
        </div>
      </main>
    </div>
  )
}
