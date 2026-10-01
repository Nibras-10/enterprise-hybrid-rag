"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type DocumentRecord = {
  id: string;
  filename: string;
  title: string | null;
  document_type: string | null;
  status: string;
  created_at: string;
  versions: { id: string; version: number; page_count: number | null; status: string }[];
};

type Citation = {
  document_id: string;
  document_name: string;
  page_number: number | null;
  section_title: string | null;
  chunk_id: string;
};

type QueryResult = {
  query_id: string;
  answer: string;
  sources: Citation[];
  metadata: { retrieval_count: number; reranked_count: number; model: string; total_latency_ms: number };
};

type ApiError = { detail?: string };
type EvaluationRun = { id: string; dataset_name: string; status: string; started_at: string | null; completed_at: string | null };
type EvaluationReport = { run_id: string; dataset: { name: string; size: number }; strategies: Record<string, { metrics: Record<string, number | null>; results: { question: string; metrics: Record<string, number | null> }[] }> };
type SavedEvaluation = { dataset_name: string; results: { question: string; ground_truth: string | null; answer: string | null; metrics: Record<string, number | null> }[] };

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export default function HomePage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<QueryResult | null>(null);
  const [activeView, setActiveView] = useState<"documents" | "chat" | "evaluations">("documents");
  const [apiToken, setApiToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const headers = useMemo(() => ({
    ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
  }), [apiToken]);

  const loadDocuments = useCallback(async () => {
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/documents?limit=100`, { headers });
      if (!response.ok) throw new Error(await responseMessage(response));
      setDocuments(await response.json() as DocumentRecord[]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Documents could not be loaded.");
    }
  }, [headers]);

  useEffect(() => { void loadDocuments(); }, [loadDocuments]);
  useEffect(() => {
    if (!documents.some((document) => ["UPLOADED", "PROCESSING", "PARSED", "CHUNKED", "EMBEDDED", "INDEXED"].includes(document.status))) return;
    const timer = window.setInterval(() => { void loadDocuments(); }, 4000);
    return () => window.clearInterval(timer);
  }, [documents, loadDocuments]);

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const file = new FormData(form).get("file");
    if (!(file instanceof File) || file.size === 0) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`${API_URL}/api/documents`, {
        method: "POST",
        headers,
        body: new FormData(form),
      });
      if (!response.ok) throw new Error(await responseMessage(response));
      setNotice(`${file.name} was uploaded. Processing status will update as indexing finishes.`);
      form.reset();
      await loadDocuments();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The document could not be uploaded.");
    } finally {
      setLoading(false);
    }
  }

  async function removeDocument(document: DocumentRecord) {
    if (!window.confirm(`Delete ${document.filename} and its indexed content?`)) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/documents/${document.id}`, { method: "DELETE", headers });
      if (!response.ok) throw new Error(await responseMessage(response));
      setDocuments((current) => current.filter((item) => item.id !== document.id));
      setNotice(`${document.filename} was deleted.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The document could not be deleted.");
    } finally {
      setLoading(false);
    }
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setAnswer(null);
    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) throw new Error(await responseMessage(response));
      setAnswer(await response.json() as QueryResult);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The question could not be answered.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" onClick={() => setActiveView("documents")}>
          <span className="brand-mark">E</span>
          <span><strong>Enterprise RAG</strong><small>DOCUMENT INTELLIGENCE</small></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          <button className={activeView === "documents" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("documents")}>
            <span className="nav-icon">▤</span> Documents <span className="nav-count">{documents.length}</span>
          </button>
          <button className={activeView === "chat" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("chat")}>
            <span className="nav-icon">✳</span> Ask your library
          </button>
          <button className={activeView === "evaluations" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("evaluations")}>
            <span className="nav-icon">◫</span> Evaluations
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="connection-card"><span className="connection-dot" /> API connection <span>LOCAL</span></div>
          <label className="token-label" htmlFor="api-token">Access token <span>optional</span></label>
          <input id="api-token" type="password" autoComplete="off" placeholder="For secured deployments" value={apiToken} onChange={(event) => setApiToken(event.target.value)} />
          <p className="privacy-note">Kept only in this tab. Never sent to the browser build.</p>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div><span className="breadcrumb">Workspace</span><span className="breadcrumb-separator">/</span><strong>{viewTitle(activeView)}</strong></div>
          <div className="user-chip"><span className="avatar">E</span><span>Local workspace</span><span className="chevron">⌄</span></div>
        </header>

        <div className="content-wrap">
          {error && <div className="alert error" role="alert"><span>!</span>{error}<button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}
          {notice && <div className="alert success" role="status"><span>✓</span>{notice}<button onClick={() => setNotice("")} aria-label="Dismiss notice">×</button></div>}

          {activeView === "documents" && <DocumentsView documents={documents} loading={loading} onUpload={uploadDocument} onDelete={removeDocument} onRefresh={loadDocuments} />}
          {activeView === "chat" && <ChatView question={question} setQuestion={setQuestion} answer={answer} loading={loading} onSubmit={submitQuestion} documentsCount={documents.length} />}
          {activeView === "evaluations" && <EvaluationsView headers={headers} />}
        </div>
      </section>
    </main>
  );
}

function DocumentsView({ documents, loading, onUpload, onDelete, onRefresh }: {
  documents: DocumentRecord[];
  loading: boolean;
  onUpload: (event: FormEvent<HTMLFormElement>) => void;
  onDelete: (document: DocumentRecord) => void;
  onRefresh: () => void;
}) {
  return (
    <>
      <div className="page-heading"><div><div className="eyebrow">KNOWLEDGE BASE</div><h1>Your documents</h1><p>Upload and organize the source material your team relies on.</p></div><button className="button secondary" onClick={onRefresh} disabled={loading}>↻ <span>Refresh</span></button></div>
      <form className="upload-card" onSubmit={onUpload}>
        <div className="upload-symbol">↑</div>
        <div className="upload-copy"><strong>Add documents to your workspace</strong><span>PDF, DOCX, or TXT · up to 20 MB each</span></div>
        <label className="button primary file-button">Choose file<input type="file" name="file" accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain" required /></label>
        <button className="button upload-submit" type="submit" disabled={loading}>{loading ? "Uploading…" : "Upload"}</button>
      </form>

      <div className="section-heading"><div><h2>Library <span className="subtle-count">{documents.length}</span></h2><p>Files and their current processing status</p></div><div className="list-tools"><button className="tool-button" aria-label="Filter documents">☷</button><button className="tool-button" aria-label="Search documents">⌕</button></div></div>
      {documents.length === 0 ? <div className="empty-state"><div className="empty-icon">▤</div><strong>Your library is ready for its first document</strong><p>Upload a contract, report, or policy to begin asking grounded questions.</p></div> : (
        <div className="document-table-wrap"><table className="document-table"><thead><tr><th>DOCUMENT</th><th>TYPE</th><th>STATUS</th><th>ADDED</th><th aria-label="Actions" /></tr></thead><tbody>
          {documents.map((document) => <tr key={document.id}>
            <td><div className="file-cell"><span className={`file-icon ${document.document_type || ""}`}>{document.document_type?.toUpperCase() === "PDF" ? "PDF" : document.document_type?.toUpperCase() === "DOCX" ? "DOC" : "TXT"}</span><span><strong>{document.title || document.filename}</strong><small>{document.filename}{document.versions[0]?.page_count ? ` · ${document.versions[0].page_count} pages` : ""}</small></span></div></td>
            <td><span className="type-label">{document.document_type?.toUpperCase() || "FILE"}</span></td>
            <td><StatusPill status={document.status} /></td>
            <td className="date-cell">{new Date(document.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</td>
            <td><button className="row-action" disabled={loading} onClick={() => onDelete(document)} aria-label={`Delete ${document.filename}`}>•••</button></td>
          </tr>)}
        </tbody></table></div>
      )}
    </>
  );
}

function ChatView({ question, setQuestion, answer, loading, onSubmit, documentsCount }: {
  question: string;
  setQuestion: (value: string) => void;
  answer: QueryResult | null;
  loading: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  documentsCount: number;
}) {
  return (
    <div className="chat-page">
      <div className="page-heading"><div><div className="eyebrow">GROUNDED ANSWERS</div><h1>Ask your library</h1><p>Answers are generated from your indexed documents and include supporting sources.</p></div><div className="library-badge"><span className="connection-dot" /> {documentsCount} documents</div></div>
      <div className="chat-canvas">
        {!answer && !loading && <div className="chat-welcome"><div className="sparkle">✳</div><h2>What would you like to know?</h2><p>Ask about contract terms, policies, figures, or anything in your workspace.</p><div className="suggestion-grid"><button onClick={() => setQuestion("What are the key terms in the latest agreement?")}>What are the key terms in the latest agreement?<span>↗</span></button><button onClick={() => setQuestion("Which documents mention termination notice periods?")}>Which documents mention termination notice periods?<span>↗</span></button><button onClick={() => setQuestion("Summarize the payment obligations and due dates.")}>Summarize payment obligations and due dates.<span>↗</span></button></div></div>}
        {loading && <div className="loading-answer"><span className="spinner" /><strong>Searching your documents</strong><p>Finding relevant passages and preparing a cited answer…</p></div>}
        {answer && <div className="answer-card"><div className="answer-label"><span className="answer-icon">✳</span> ANSWER <span className="answer-meta">{answer.metadata.reranked_count} sources reviewed · {answer.metadata.total_latency_ms} ms</span></div><div className="answer-text">{answer.answer}</div>{answer.sources.length > 0 && <div className="sources-section"><div className="sources-heading"><strong>Sources</strong><span>{answer.sources.length} citations</span></div><div className="source-list">{answer.sources.map((source) => <div className="source-card" key={source.chunk_id}><span className="source-page">{source.page_number ? `P. ${source.page_number}` : "DOC"}</span><span className="source-details"><strong>{source.document_name}</strong><small>{source.section_title || "Document passage"}</small></span><span className="source-arrow">↗</span></div>)}</div></div>}</div>}
      </div>
      <form className="question-composer" onSubmit={onSubmit}><textarea aria-label="Your question" placeholder="Ask a question about your documents…" value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={4000} required rows={2} /><div className="composer-footer"><span>Answers are grounded in retrieved document passages.</span><button className="button primary ask-button" disabled={loading || !question.trim()}>{loading ? "Searching…" : "Ask question"}<span>↑</span></button></div></form>
    </div>
  );
}

function EvaluationsView({ headers }: { headers: Record<string, string> }) {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [savedEvaluation, setSavedEvaluation] = useState<SavedEvaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const [includeRagas, setIncludeRagas] = useState(false);
  const [message, setMessage] = useState("");

  const refreshRuns = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/evaluations`, { headers });
      if (!response.ok) throw new Error(await responseMessage(response));
      setRuns(await response.json() as EvaluationRun[]);
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "Evaluation history could not be loaded.");
    }
  }, [headers]);
  useEffect(() => { void refreshRuns(); }, [refreshRuns]);

  async function submitDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const file = new FormData(form).get("dataset");
    if (!(file instanceof File) || file.size === 0) return;
    setBusy(true);
    setMessage("");
    try {
      const contents = JSON.parse(await file.text()) as Record<string, unknown>;
      const response = await fetch(`${API_URL}/api/evaluations/run`, {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          name: contents.name || file.name.replace(/\.json$/i, ""),
          description: contents.description || "",
          questions: contents.questions,
          include_ragas: includeRagas,
        }),
      });
      if (!response.ok) throw new Error(await responseMessage(response));
      setReport(await response.json() as EvaluationReport);
      form.reset();
      await refreshRuns();
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "The evaluation could not be run.");
    } finally {
      setBusy(false);
    }
  }

  async function loadRun(id: string) {
    setBusy(true);
    setMessage("Saved runs can be opened from the run list.");
    try {
      const response = await fetch(`${API_URL}/api/evaluations/${id}`, { headers });
      if (!response.ok) throw new Error(await responseMessage(response));
      const result = await response.json() as SavedEvaluation;
      setSavedEvaluation(result);
      setMessage("");
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "The evaluation run could not be opened.");
    } finally {
      setBusy(false);
    }
  }

  return <>
    <div className="page-heading"><div><div className="eyebrow">QUALITY & PERFORMANCE</div><h1>Evaluations</h1><p>Compare dense, BM25, hybrid, and reranked retrieval against your labeled questions.</p></div><button className="button secondary" onClick={() => void refreshRuns()} disabled={busy}>↻ <span>Refresh</span></button></div>
    <form className="evaluation-upload" onSubmit={submitDataset}><div><strong>Run a dataset</strong><p>Upload JSON with questions, ground truth, and optional relevant chunk IDs.</p><label className="ragas-toggle"><input type="checkbox" checked={includeRagas} onChange={(event) => setIncludeRagas(event.target.checked)} /> Include Gemini-backed Ragas metrics</label></div><label className="button secondary evaluation-file">Choose JSON<input type="file" name="dataset" accept=".json,application/json" required /></label><button className="button primary" disabled={busy}>{busy ? "Running…" : "Compare strategies"}</button></form>
    {message && <div className="eval-message" role="status">{message}</div>}
    {report && <section className="report-panel"><div className="section-heading"><div><h2>Latest comparison</h2><p>{report.dataset.name} · {report.dataset.size} questions</p></div></div><div className="strategy-grid">{Object.entries(report.strategies).map(([name, value]) => <article className="strategy-card" key={name}><span>{name.replaceAll("_", " + ")}</span><strong>{formatMetric(value.metrics.recall_at_k)}</strong><small>Recall@K</small><div>Precision@K <b>{formatMetric(value.metrics.precision_at_k)}</b></div><div>MRR <b>{formatMetric(value.metrics.reciprocal_rank)}</b></div>{Object.entries(value.metrics).filter(([metric]) => !["recall_at_k", "precision_at_k", "reciprocal_rank"].includes(metric)).map(([metric, value]) => <div key={metric}>{metric.replaceAll("_", " ")} <b>{formatMetric(value)}</b></div>)}</article>)}</div></section>}
    {savedEvaluation && <section className="report-panel"><div className="section-heading"><div><h2>{savedEvaluation.dataset_name}</h2><p>Saved question-level results</p></div></div><div className="saved-results">{savedEvaluation.results.map((item, index) => <article key={`${item.question}-${index}`}><strong>{item.question}</strong><p>{item.answer || "No answer recorded."}</p><div className="saved-metrics">{Object.entries(item.metrics).map(([name, value]) => <span key={name}>{name.replaceAll("_", " ")} <b>{formatMetric(value)}</b></span>)}</div></article>)}</div></section>}
    <div className="section-heading eval-run-heading"><div><h2>Recent runs <span className="subtle-count">{runs.length}</span></h2><p>Each retrieval strategy is stored as a separate run.</p></div></div>
    {runs.length === 0 ? <div className="empty-state evaluation-empty"><div className="empty-icon">◫</div><strong>No evaluation runs yet</strong><p>Upload a labeled JSON dataset to compare retrieval performance.</p></div> : <div className="run-list">{runs.map((run) => <button className="run-row" key={run.id} onClick={() => void loadRun(run.id)} disabled={busy}><span className="run-icon">◫</span><span className="run-name"><strong>{run.dataset_name}</strong><small>{run.started_at ? new Date(run.started_at).toLocaleString() : "Pending"}</small></span><StatusPill status={run.status} /><span className="source-arrow">↗</span></button>)}</div>}
  </>;
}

function formatMetric(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  return <span className={`status-pill status-${normalized}`}><span />{status.replaceAll("_", " ").toLowerCase()}</span>;
}

function viewTitle(view: "documents" | "chat" | "evaluations") {
  return view === "documents" ? "Documents" : view === "chat" ? "Ask your library" : "Evaluations";
}

async function responseMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json() as ApiError;
    return payload.detail || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}
