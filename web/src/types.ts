export interface Session {
  id: string
  created_at: string
}

export interface WindowRow {
  id: number
  session_id: string
  t_start: number
  t_end: number
  jaw_open_label: string | null
  composite_label: string | null
  jaw_open_mean: number | null
  mar_mean: number | null
  human_label: string | null
  labeled_at: string | null
  labeled_by: string | null
}

export interface SignalPoint {
  t: number
  jaw_open: number
  mar: number
}

export interface ImuPoint {
  t: number       // t_rel_sec (approximately aligned with vision t_sec)
  rot_x: number
  rot_y: number
  rot_z: number
  accel_x: number
  accel_y: number
  accel_z: number
}

export interface SessionWindowStats {
  total: number
  disagree: number
  unlabeled_disagree: number
  labeled: number   // windows with human_label
  agree: number     // windows where jaw_open_label == human_label (and human_label not null)
}

export type HumanLabel = 'chewing' | 'rest' | 'bad_face'

export interface LosoRun {
  id: number
  run_at: string
  n_sessions: number
  pooled_accuracy: number
  pooled_f1_chewing: number
  pooled_f1_rest: number
  notes: string | null
}

export interface LosoResult {
  id: number
  run_id: number
  session_id: string
  accuracy: number
  f1_chewing: number
  f1_rest: number
  n_train: number
  n_test: number
  train_chew_ratio: number
  test_chew_ratio: number
  estimated_chews: number
}

export interface LosoPrediction {
  id: number
  run_id: number
  window_id: number | null
  session_id: string
  t_start: number
  y_true: number
  y_pred: number
}
