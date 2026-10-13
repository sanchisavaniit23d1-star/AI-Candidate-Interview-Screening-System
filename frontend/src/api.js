const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export const uploadVideo = (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/api/videos", { method: "POST", body: form });
};
export const getJob = (id) => request(`/api/jobs/${id}`);
export const getResult = (id) => request(`/api/results/${id}`);
export const listJobs = () => request("/api/jobs");
export const deleteJob = (id) => request(`/api/jobs/${id}`, { method: "DELETE" });
