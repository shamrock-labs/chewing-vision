-- sessions: one row per recording session
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- windows: one row per annotation window
CREATE TABLE windows (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  t_start FLOAT NOT NULL,
  t_end FLOAT NOT NULL,
  jaw_open_label TEXT,
  composite_label TEXT,
  jaw_open_mean FLOAT,
  mar_mean FLOAT,
  human_label TEXT,
  labeled_at TIMESTAMPTZ,
  labeled_by UUID REFERENCES auth.users(id)
);

CREATE INDEX windows_session_id_idx ON windows(session_id);

ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE windows  ENABLE ROW LEVEL SECURITY;

CREATE POLICY "read_sessions"   ON sessions FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "insert_sessions" ON sessions FOR INSERT TO anon, authenticated WITH CHECK (true);

CREATE POLICY "read_windows"    ON windows FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "insert_windows"  ON windows FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "update_windows"  ON windows FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);
