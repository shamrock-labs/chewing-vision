CREATE TABLE loso_predictions (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT REFERENCES loso_runs(id) ON DELETE CASCADE,
  window_id BIGINT REFERENCES windows(id),
  session_id TEXT NOT NULL,
  t_start DOUBLE PRECISION NOT NULL,
  y_true INT NOT NULL,
  y_pred INT NOT NULL
);

ALTER TABLE loso_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_read" ON loso_predictions FOR SELECT TO anon USING (true);
CREATE POLICY "auth_all" ON loso_predictions FOR ALL TO authenticated USING (true) WITH CHECK (true);
