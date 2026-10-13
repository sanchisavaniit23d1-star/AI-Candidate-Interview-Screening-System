import { useEffect, useRef, useState } from "react";
import { uploadVideo, getJob, getResult, listJobs, deleteJob } from "./api.js";
import Results from "./Results.jsx";

export default function App() {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);      // current job status
  const [data, setData] = useState(null);    // finished result
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const timer = useRef(null);

  const refreshHistory = () => listJobs().then(setHistory).catch(() => {});
  useEffect(() => { refreshHistory(); return () => clearTimeout(timer.current); }, []);

  const open = async (id) => {
    setError(""); setData(null);
    try {
      const status = await getJob(id);
      setJob(status);
      if (status.status === "done") setData(await getResult(id));
      else if (status.status === "failed") setError(status.error || "Analysis failed.");
      else poll(id);
    } catch (e) { setError(e.message); }
  };

  const poll = (id) => {
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const status = await getJob(id);
        setJob(status);
        if (status.status === "done") { setData(await getResult(id)); refreshHistory(); }
        else if (status.status === "failed") { setError(status.error || "Analysis failed."); refreshHistory(); }
        else poll(id);
      } catch (e) { setError(e.message); }
    }, 1500);
  };

  const submit = async () => {
    if (!file) return;
    setError(""); setData(null); setJob(null);
    try {
      const res = await uploadVideo(file);
      await open(res.id);
      refreshHistory();
    } catch (e) { setError(e.message); }
  };

  const running = job && (job.status === "queued" || job.status === "running");

  return (
    <div className="page">
      <header>
        <h1>AI Interview Analysis</h1>
        <p className="muted">
          Audiovisual emotion recognition with attention fusion. Experimental model output, not a hiring tool.
        </p>
      </header>

      <section className="card">
        <h2>1. Upload an interview video</h2>
        <input type="file" accept=".mp4,.avi,.mov,.mkv" onChange={(e) => setFile(e.target.files[0] || null)} />
        <button onClick={submit} disabled={!file || running}>Analyze</button>
        {running && (
          <div className="progress">
            <div className="bar"><div style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} /></div>
            <span>{Math.round((job.progress || 0) * 100)}% · {job.message}</span>
          </div>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      {data && <Results data={data} />}

      <section className="card">
        <h2>Previous analyses</h2>
        {history.length === 0 && <p className="muted">No analyses yet.</p>}
        <ul className="history">
          {history.map((j) => (
            <li key={j.id}>
              <button className="link" onClick={() => open(j.id)}>{j.filename}</button>
              <span className={`badge ${j.status}`}>{j.status}</span>
              {j.status !== "running" && j.status !== "queued" && (
                <button className="link danger" onClick={async () => { await deleteJob(j.id); refreshHistory(); }}>delete</button>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
