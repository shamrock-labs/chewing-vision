import { createClient } from '@insforge/sdk'
import type { Session, WindowRow, HumanLabel, LosoPrediction, SessionWindowStats } from '../types'

const BASE = import.meta.env.VITE_INSFORGE_URL as string

export const insforge = createClient({
  baseUrl: BASE,
  anonKey: import.meta.env.VITE_INSFORGE_ANON_KEY,
})

export async function fetchSessions(): Promise<Session[]> {
  const { data, error } = await insforge.database
    .from('sessions')
    .select('*')
    .order('id', { ascending: false })
  if (error) throw error
  return data ?? []
}

export async function fetchWindows(sessionId: string): Promise<WindowRow[]> {
  const { data, error } = await insforge.database
    .from('windows')
    .select('*')
    .eq('session_id', sessionId)
    .order('t_start')
  if (error) throw error
  return data ?? []
}

export async function saveLabel(windowId: number, label: HumanLabel): Promise<void> {
  const { error } = await insforge.database
    .from('windows')
    .update({ human_label: label, labeled_at: new Date().toISOString() })
    .eq('id', windowId)
  if (error) throw error
}

export async function batchSaveLabels(updates: Array<{ ids: number[]; label: HumanLabel }>): Promise<void> {
  const now = new Date().toISOString()
  await Promise.all(
    updates.map(({ ids, label }) =>
      insforge.database
        .from('windows')
        .update({ human_label: label, labeled_at: now })
        .in('id', ids)
    )
  )
}

// Returns {t_start_rounded: y_pred} — use matchImuPred() to look up by window t_start
export async function fetchLatestImuPreds(sessionId: string): Promise<Record<number, number>> {
  const { data: runs, error: e1 } = await insforge.database
    .from('loso_runs').select('id').order('run_at', { ascending: false }).limit(1)
  if (e1 || !runs?.length) return {}
  const runId = runs[0].id
  const { data, error: e2 } = await insforge.database
    .from('loso_predictions')
    .select('t_start, y_pred')
    .eq('run_id', runId)
    .eq('session_id', sessionId)
  if (e2 || !data) return {}
  // Key by t_start * 10 (0.1s granularity) to allow float key lookup
  const map: Record<number, number> = {}
  for (const p of data) map[Math.round(p.t_start * 10)] = p.y_pred
  return map
}

export function matchImuPred(preds: Record<number, number>, tStart: number): number | null {
  // Try exact and ±0.5s range (LOSO t_start has ~0.334s offset from DB t_start)
  for (let delta = 0; delta <= 6; delta++) {
    for (const sign of [1, -1]) {
      const key = Math.round(tStart * 10) + delta * sign
      if (preds[key] != null) return preds[key]
    }
  }
  return null
}

export async function fetchPredictions(runId: number, sessionId: string): Promise<LosoPrediction[]> {
  const { data, error } = await insforge.database
    .from('loso_predictions')
    .select('*')
    .eq('run_id', runId)
    .eq('session_id', sessionId)
    .order('t_start')
  if (error) throw error
  return data ?? []
}

function storageUrl(key: string): string {
  return `${BASE}/api/storage/buckets/sessions/objects/${encodeURIComponent(key)}`
}

export function videoUrl(sessionId: string): string {
  return storageUrl(`${sessionId}/video.mp4`)
}

export function signalsUrl(sessionId: string): string {
  return storageUrl(`${sessionId}/signals.csv`)
}

export function imuUrl(sessionId: string): string {
  return storageUrl(`${sessionId}/imu.csv`)
}

export async function fetchAllWindowStats(): Promise<Record<string, SessionWindowStats>> {
  const { data, error } = await insforge.database
    .from('windows')
    .select('session_id, jaw_open_label, composite_label, human_label')
  if (error) throw error

  const stats: Record<string, SessionWindowStats> = {}
  for (const w of (data ?? [])) {
    if (!stats[w.session_id]) stats[w.session_id] = { total: 0, disagree: 0, unlabeled_disagree: 0, labeled: 0, agree: 0 }
    const s = stats[w.session_id]
    s.total++
    if (w.jaw_open_label !== w.composite_label) {
      s.disagree++
      if (!w.human_label) s.unlabeled_disagree++
    }
    if (w.human_label) {
      s.labeled++
      if (w.jaw_open_label === w.human_label) s.agree++
    }
  }
  return stats
}

export async function createSession(id: string): Promise<void> {
  const { error } = await insforge.database
    .from('sessions')
    .upsert({ id, created_at: new Date().toISOString() })
  if (error) throw error
}
