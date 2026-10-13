import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer, BarChart, Bar } from "recharts";

const EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];
const fmt = (s) => `${Number(s).toFixed(1)}s`;

export default function Results({ data }) {
  const { summary, result } = data;
  const timeline = result.timeline;

  const rows = timeline.map((t) => ({
    label: `${fmt(t.start_time)}–${fmt(t.end_time)}`,
    ...t.hr_scores,
  }));
  const meanProbs = EMOTIONS.map((e) => ({
    emotion: e,
    probability: +(timeline.reduce((a, t) => a + (t.fusion.probabilities[e] || 0), 0) / timeline.length * 100).toFixed(1),
  }));

  return (
    <>
      <section className="card">
        <h2>2. Summary — {result.video}</h2>
        {result.warnings?.length > 0 && (
          <div className="warn">{result.warnings.map((w, i) => <p key={i}>⚠ {w}</p>)}</div>
        )}
        <div className="metrics">
          <div><span>Most frequent window label</span><b>{summary.dominant_emotion}</b></div>
          <div><span>Windows analysed</span><b>{summary.windows}</b></div>
          <div><span>Mean composure*</span><b>{summary.mean_scores.composure}%</b></div>
          <div><span>Mean stress*</span><b>{summary.mean_scores.stress}%</b></div>
          <div><span>Mean engagement*</span><b>{summary.mean_scores.engagement}%</b></div>
        </div>
      </section>

      <section className="card">
        <h2>3. Emotion timeline (fused model, per 3.6 s window)</h2>
        <div className="scroll">
          <table>
            <thead><tr><th>Time</th><th>Emotion</th><th>Probability</th><th>Faces found</th></tr></thead>
            <tbody>
              {timeline.map((t) => (
                <tr key={t.window}>
                  <td>{fmt(t.start_time)}–{fmt(t.end_time)}</td>
                  <td>{t.fusion.emotion}</td>
                  <td>{(t.fusion.probabilities[t.fusion.emotion] * 100).toFixed(1)}%</td>
                  <td>{t.faces_detected ?? "–"}/15</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>4. Derived indicators over time*</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" /><YAxis domain={[0, 100]} /><Tooltip /><Legend />
            <Line type="monotone" dataKey="composure" stroke="#2f5d50" dot={false} />
            <Line type="monotone" dataKey="stress" stroke="#c00000" dot={false} />
            <Line type="monotone" dataKey="engagement" stroke="#3b6fa0" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="card">
        <h2>5. Mean emotion probabilities (%)</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={meanProbs}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="emotion" /><YAxis /><Tooltip />
            <Bar dataKey="probability" fill="#2f5d50" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <p className="disclaimer">* {result.disclaimer}</p>
    </>
  );
}
