import { useEffect, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Database,
  Gauge,
  Search,
  Server,
  Sparkles,
} from 'lucide-react';
import { getHealth, searchDocuments } from './api/search';
import './App.css';

const SUGGESTIONS = [
  'wall street bears',
  'oil prices and the economy',
  'commercial aerospace investment',
];

export default function App() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [efSearch, setEfSearch] = useState(200);
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState('checking');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let retryTimer;

    const checkHealth = async () => {
      try {
        await getHealth();
        if (!cancelled) setStatus('online');
      } catch {
        if (!cancelled) {
          setStatus('offline');
          retryTimer = window.setTimeout(checkHealth, 5000);
        }
      }
    };

    checkHealth();
    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
    };
  }, []);

  async function runSearch(event, value = query) {
    event?.preventDefault();
    if (!value.trim()) {
      setError('Enter a search phrase to find related documents.');
      setResults(null);
      return;
    }
    setLoading(true);
    setError('');
    try {
      const data = await searchDocuments(value.trim(), topK, efSearch);
      setResults(data);
      setStatus('online');
    } catch (requestError) {
      setError(requestError.message);
      setStatus('offline');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="Vector Database home">
          <span className="wordmark-mark">
            <Database size={17} />
          </span>
          <span>VECTOR / DB</span>
        </a>
        <div className="status-pill">
          <span className={`status-dot ${status}`} />
          {status === 'online'
            ? 'INDEX ONLINE'
            : status === 'offline'
              ? 'SERVER OFFLINE'
              : 'CONNECTING'}
        </div>
      </header>

      <main>
        <section className="hero-section">
          <div className="eyebrow">
            <Sparkles size={14} /> SEMANTIC RETRIEVAL LAB
          </div>
          <h1>
            Search the corpus
            <br />
            <em>by meaning.</em>
          </h1>
          <p className="hero-copy">
            A handwritten HNSW index over 119,921 AG News embeddings. Ask a
            question, not a keyword.
          </p>
          <form className="search-panel" onSubmit={runSearch}>
            <Search size={19} className="search-icon" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Try: how are oil prices affecting markets?"
              aria-label="Search documents"
            />
            <button className="search-button" type="submit" disabled={loading}>
              {loading ? 'Searching' : 'Search'} <ArrowRight size={17} />
            </button>
          </form>
          <div className="suggestions">
            <span>Try a query</span>
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => {
                  setQuery(suggestion);
                  runSearch(null, suggestion);
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
          {error && (
            <div className="error-banner" role="alert">
              {error}{' '}
              <button type="button" onClick={() => runSearch(null)}>
                Retry
              </button>
            </div>
          )}
        </section>

        <section className="stats-strip" aria-label="Index statistics">
          <div>
            <span className="stat-label">Corpus</span>
            <strong>119,921</strong>
            <small>documents indexed</small>
          </div>
          <div>
            <span className="stat-label">Dimensions</span>
            <strong>384</strong>
            <small>normalized values</small>
          </div>
          <div>
            <span className="stat-label">Index</span>
            <strong>HNSW</strong>
            <small>M12 / ef 200</small>
          </div>
          <div>
            <span className="stat-label">Benchmark</span>
            <strong>84.4%</strong>
            <small>Recall@10</small>
          </div>
          <div>
            <span className="stat-label">Speedup</span>
            <strong>3.27×</strong>
            <small>vs exact search</small>
          </div>
        </section>

        <section className="workspace">
          <div className="results-column">
            <div className="section-heading">
              <div>
                <span className="kicker">RETRIEVAL OUTPUT</span>
                <h2>
                  {results
                    ? `Results for “${results.query}”`
                    : 'Waiting for a query'}
                </h2>
              </div>
              {results && (
                <span className="result-count">
                  {results.count} matches <span>· {results.latency_ms} ms</span>
                </span>
              )}
            </div>
            {results?.results?.length ? (
              results.results.map((result) => (
                <article
                  className="result-card"
                  key={`${result.id}-${result.rank}`}
                >
                  <div className="result-meta">
                    <span className="rank">
                      {String(result.rank).padStart(2, '0')}
                    </span>
                    <span>ID {result.id}</span>
                    <span className="score">
                      <Gauge size={14} /> {result.score.toFixed(4)}
                    </span>
                  </div>
                  <p>{result.text}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">
                <Search size={28} />
                <p>Semantic results will appear here.</p>
                <span>
                  The API returns real documents from the indexed corpus.
                </span>
              </div>
            )}
          </div>
          <aside className="control-column">
            <div className="control-card">
              <div className="kicker">QUERY CONTROLS</div>
              <h3>Search depth</h3>
              <p>Increase the beam to explore more graph candidates.</p>
              <label htmlFor="ef-search">
                ef_search <output>{efSearch}</output>
              </label>
              <input
                id="ef-search"
                type="range"
                min="10"
                max="200"
                step="10"
                value={efSearch}
                onChange={(event) => setEfSearch(Number(event.target.value))}
              />
              <div className="range-ends">
                <span>fast</span>
                <span>thorough</span>
              </div>
              <label htmlFor="top-k">
                Top results <output>{topK}</output>
              </label>
              <input
                id="top-k"
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </div>
            <div className="note-card">
              <Server size={17} />
              <div>
                <strong>Live index</strong>
                <p>
                  Results are served by the FastAPI backend. No demo data or
                  fallback results.
                </p>
              </div>
            </div>
          </aside>
        </section>
      </main>
      <footer>
        <span>
          <Activity size={14} /> VECTOR DATABASE PROJECT
        </span>
        <span>AG NEWS · COSINE SIMILARITY</span>
      </footer>
    </div>
  );
}
