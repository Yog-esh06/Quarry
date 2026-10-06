"use client";

import { useEffect, useState } from "react";

type Graph = { nodes: string[]; edges: string[][]; kind: string };
type Summary = { rows: number; default_rate: number; features: number; synthetic: boolean };

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [graph, setGraph] = useState<Graph | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/v1/causal/graph`).then((r) => r.json()),
      fetch(`${API}/api/v1/datasets/demo/summary`).then((r) => r.json()),
    ]).then(([g, s]) => { setGraph(g); setSummary(s); })
      .catch(() => setError("Backend unavailable. Start the API to load live analysis."));
  }, []);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="mark">Q</span><span>QUARRY</span></div>
        <p className="eyebrow">CAUSAL DECISION INTELLIGENCE</p>
        <nav>
          <a className="active" href="#overview">Overview</a>
          <a href="#graph">Causal graph</a>
          <a href="#policy">Policy lab</a>
          <a href="#monitoring">Monitoring</a>
        </nav>
        <div className="assumption">
          <span className="dot" /> Assumption-aware
          <p>Recommendations are conditional on the configured causal graph and overlap assumptions.</p>
        </div>
      </aside>
      <section className="content">
        <header className="topbar"><div><p className="eyebrow">WORKSPACE / LENDING DEMO</p><h1>Decision surface</h1></div><span className="status"><span className="dot" /> LIVE DATA</span></header>
        {error && <div className="notice">{error}</div>}
        <section id="overview" className="hero">
          <div><p className="eyebrow orange">INTERVENTION ANALYSIS</p><h2>What changes if we intervene?</h2><p className="muted">Estimate treatment effects, inspect counterfactuals, and optimize constrained policies without confusing correlation for causation.</p></div>
          <button onClick={() => window.location.reload()}>Refresh analysis ↗</button>
        </section>
        <div className="metrics">
          <Metric label="Causal dataset" value={summary ? `${summary.rows.toLocaleString()} rows` : "—"} note={summary?.synthetic ? "Controlled SCM" : "Loading"} />
          <Metric label="Observed default rate" value={summary ? `${(summary.default_rate * 100).toFixed(1)}%` : "—"} note="Reference period" />
          <Metric label="Graph variables" value={graph ? String(graph.nodes.length) : "—"} note={graph ? `${graph.edges.length} directed edges` : "Loading"} />
          <Metric label="Model state" value={error ? "Offline" : "Ready"} note="Assumptions visible" />
        </div>
        <div className="grid">
          <section id="graph" className="panel graph-panel"><PanelTitle title="Configured causal structure" tag={graph?.kind ?? "loading"} /><p className="muted small">The graph is an explicit assumption, not proof of causality.</p><div className="graph">{graph?.edges.slice(0, 8).map(([a, b]) => <div className="edge" key={`${a}-${b}`}><span>{a}</span><b>→</b><span>{b}</span></div>) ?? <div className="empty">Connect the API to inspect the DAG.</div>}</div></section>
          <section id="policy" className="panel"><PanelTitle title="Policy lab" tag="READY" /><div className="policy-card"><div><p className="eyebrow orange">RECOMMENDED ACTION</p><h3>Estimate uplift before allocation</h3><p className="muted small">Run the policy optimizer to compare expected benefit, intervention cost, overlap, and risk under a budget constraint.</p></div><span className="arrow">↗</span></div><div className="split"><div><strong>3</strong><span>treatment arms</span></div><div><strong>DR</strong><span>estimator</span></div><div><strong>95%</strong><span>uncertainty target</span></div></div></section>
        </div>
        <section id="monitoring" className="panel monitoring"><PanelTitle title="Monitoring posture" tag={summary ? "BASELINE LOADED" : "WAITING"} /><div className="monitor-note"><span className="dot" /> Live drift and policy diagnostics appear after an evaluation run.</div></section>
        <footer>QUARRY / causal decisions with visible assumptions</footer>
      </section>
    </main>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="metric"><p className="eyebrow">{label}</p><strong>{value}</strong><span>{note}</span></div>;
}
function PanelTitle({ title, tag }: { title: string; tag: string }) { return <div className="panel-title"><h3>{title}</h3><span className="tag">{tag}</span></div>; }
