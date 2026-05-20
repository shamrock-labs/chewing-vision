-- loso_runs: one row per LOSO evaluation run
CREATE TABLE loso_runs (
  id              BIGSERIAL PRIMARY KEY,
  run_at          TIMESTAMPTZ DEFAULT NOW(),
  n_sessions      INT,
  pooled_accuracy FLOAT,
  pooled_f1_chewing FLOAT,
  pooled_f1_rest  FLOAT,
  notes           TEXT
);

-- loso_results: per-session fold metrics within a run
CREATE TABLE loso_results (
  id               BIGSERIAL PRIMARY KEY,
  run_id           BIGINT NOT NULL REFERENCES loso_runs(id) ON DELETE CASCADE,
  session_id       TEXT NOT NULL,
  accuracy         FLOAT,
  f1_chewing       FLOAT,
  f1_rest          FLOAT,
  n_train          INT,
  n_test           INT,
  train_chew_ratio FLOAT,
  test_chew_ratio  FLOAT,
  estimated_chews  INT
);

CREATE INDEX loso_results_run_id_idx ON loso_results(run_id);

ALTER TABLE loso_runs    ENABLE ROW LEVEL SECURITY;
ALTER TABLE loso_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "read_loso_runs"    ON loso_runs    FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "insert_loso_runs"  ON loso_runs    FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "read_loso_results" ON loso_results FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "insert_loso_results" ON loso_results FOR INSERT TO anon, authenticated WITH CHECK (true);
