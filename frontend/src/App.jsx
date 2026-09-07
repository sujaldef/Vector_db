import { useEffect, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Database,
  Gauge,
  Search,
  Sparkles,
} from 'lucide-react';
import { getHealth, searchDocuments, searchExact } from './api/search';

const SUGGESTIONS = [
  'wall street bears',
  'oil prices and the economy',
  'commercial aerospace investment',
];

const formatMs = (value) => `${Number(value || 0).toFixed(2)} ms`;

export default function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [exactResults, setExactResults] = useState(null);
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState('checking');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let retryTimer;

    const checkHealth = async () => {
      try {
        const data = await getHealth();
        if (!cancelled) {
          setHealth(data);
          setStatus('online');
        }
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
    const cleanQuery = value.trim();
    if (!cleanQuery) {
      setError('Enter a search phrase to find related documents.');
      setResults(null);
      setExactResults(null);
      return;
    }

    setLoading(true);
    setError('');
    setNotice('');
    const responses = await Promise.allSettled([
      searchDocuments(cleanQuery, 10, 200),
      searchExact(cleanQuery, 10),
    ]);
    const [hnswResponse, exactResponse] = responses;

    if (hnswResponse.status === 'fulfilled') {
      setResults(hnswResponse.value);
      setStatus('online');
    } else {
      setError(hnswResponse.reason?.message || 'HNSW search failed.');
      setStatus('offline');
    }
    if (exactResponse.status === 'fulfilled') {
      setExactResults(exactResponse.value);
    } else {
      setNotice('Exact comparison was unavailable for this query.');
    }
    setLoading(false);
  }

  const exactIds = new Set(exactResults?.results?.map((item) => item.id) || []);
  const recall =
    results && exactResults
      ? results.results.filter((item) => exactIds.has(item.id)).length /
        Math.max(exactResults.count, 1)
      : null;
  const speedup =
    results && exactResults && results.latency_ms > 0
      ? exactResults.latency_ms / results.latency_ms
      : null;

  return (
    <div className="min-h-screen bg-[#e8eee8] text-[#17231f] selection:bg-[#f6cf8f]">
      <header className="mx-auto flex max-w-[1240px] items-center justify-between border-b border-[#b9c6bc] px-5 py-5 sm:px-10">
        <a
          className="flex items-center gap-2.5 font-mono text-sm font-semibold tracking-wide"
          href="/"
          aria-label="Vector Database home"
        >
          <span className="grid h-8 w-8 place-items-center rounded-full bg-[#183d32] text-[#f6cf8f]">
            <Database size={17} />
          </span>
          VECTOR / DB
        </a>
        <div className="flex items-center gap-2 font-mono text-[10px] tracking-widest">
          <span
            className={`h-2 w-2 rounded-full ${status === 'online' ? 'bg-[#287d59]' : status === 'offline' ? 'bg-[#ba4c3c]' : 'bg-[#b5aaa0]'}`}
          />
          {status === 'online'
            ? 'INDEX ONLINE'
            : status === 'offline'
              ? 'SERVER OFFLINE'
              : 'CONNECTING'}
        </div>
      </header>

      <main className="mx-auto max-w-[1240px] px-5 pb-24 pt-14 sm:px-10 sm:pt-20">
        <section className="max-w-3xl">
          <div className="flex items-center gap-2 font-mono text-[11px] tracking-[0.12em] text-[#5c7167]">
            <Sparkles size={14} /> SEMANTIC RETRIEVAL LAB
          </div>
      <h1 className="mt-5 text-6xl !text-[#c15f45] font-semibold leading-[0.94] tracking-[-0.06em] sm:text-8xl">
  Search the corpus
  <br />
  <em className="not-italic !text-[#c15f45]">
    by meaning.
  </em>
</h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-[#637168]">
            A handwritten HNSW index over{' '}
            {health?.vectors?.toLocaleString() || '119,921'} AG News embeddings.
            Ask a question, not a keyword.
          </p>
          <form
            className="mt-9 flex items-center gap-3 border border-[#b9c6bc] bg-[#fffdf8] p-2 pl-4 shadow-[8px_8px_0_#c9d5ca] max-sm:shadow-none"
            onSubmit={runSearch}
          >
            <Search size={19} className="shrink-0 text-[#65766d]" />
            <input
              className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-[#99a59d]"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Try: how are oil prices affecting markets?"
              aria-label="Search documents"
            />
            <button
              className="flex items-center gap-3 bg-[#183d32] px-4 py-3 text-sm font-semibold text-[#fffdf8] disabled:cursor-wait disabled:opacity-60"
              type="submit"
              disabled={loading}
            >
              {loading ? 'Searching' : 'Search'} <ArrowRight size={17} />
            </button>
          </form>
          <div className="mt-5 flex flex-wrap items-center gap-2 font-mono text-[11px] text-[#76847b]">
            <span>Try a query</span>
            {SUGGESTIONS.map((suggestion) => (
              <button
                className="rounded-full border border-[#bdc9bf] px-2.5 py-1 text-[#496057] hover:bg-[#fffdf8]"
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
            <div
              className="mt-6 flex justify-between gap-3 border border-[#df9b89] bg-[#f9ddd4] p-3 text-sm text-[#853f34]"
              role="alert"
            >
              <span>{error}</span>
              <button
                className="underline"
                type="button"
                onClick={() => runSearch(null)}
              >
                Retry
              </button>
            </div>
          )}
          {notice && <p className="mt-4 text-sm text-[#8a6741]">{notice}</p>}
        </section>

        <section
          className="mt-24 grid grid-cols-2 border-y border-[#b9c6bc] sm:grid-cols-5"
          aria-label="Index statistics"
        >
          <div className="border-b border-r border-[#b9c6bc] p-4 sm:border-b-0">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#76847b]">
              Corpus
            </span>
            <strong className="mt-2 block text-2xl">
              {health?.vectors?.toLocaleString() || '119,921'}
            </strong>
            <small className="font-mono text-[10px] text-[#829087]">
              documents indexed
            </small>
          </div>
          <div className="border-b border-[#b9c6bc] p-4 sm:border-b-0 sm:border-r">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#76847b]">
              Dimensions
            </span>
            <strong className="mt-2 block text-2xl">
              {health?.dimension || 384}
            </strong>
            <small className="font-mono text-[10px] text-[#829087]">
              normalized values
            </small>
          </div>
          <div className="border-r border-[#b9c6bc] p-4">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#76847b]">
              Index
            </span>
            <strong className="mt-2 block text-2xl">HNSW</strong>
            <small className="font-mono text-[10px] text-[#829087]">
              M12 / ef 200
            </small>
          </div>
          <div className="border-r border-[#b9c6bc] p-4">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#76847b]">
              Live recall
            </span>
            <strong className="mt-2 block text-2xl text-[#287d59]">
              {recall == null ? '—' : `${(recall * 100).toFixed(1)}%`}
            </strong>
            <small className="font-mono text-[10px] text-[#829087]">
              HNSW vs exact
            </small>
          </div>
          <div className="col-span-2 p-4 sm:col-span-1">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#76847b]">
              Live speedup
            </span>
            <strong className="mt-2 block text-2xl text-[#c15f45]">
              {speedup == null ? '—' : `${speedup.toFixed(2)}×`}
            </strong>
            <small className="font-mono text-[10px] text-[#829087]">
              exact / HNSW
            </small>
          </div>
        </section>

<section className="mt-16 grid gap-12 lg:grid-cols-[minmax(0,1fr)_280px]">
  <div>
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <span className="font-mono text-[11px] tracking-[0.12em] text-[#5c7167]">
          RETRIEVAL OUTPUT
        </span>

        <h2 className="mt-2 text-2xl font-semibold tracking-tight !text-[#304038]">
          {results
            ? `Results for “${results.query}”`
            : 'Waiting for a query'}
        </h2>
      </div>

      {results && (
        <span className="font-mono text-[11px] !text-[#65766d]">
          {results.count} matches · HNSW {formatMs(results.latency_ms)}{' '}
          · Exact {exactResults ? formatMs(exactResults.latency_ms) : '—'}
        </span>
      )}
    </div>

    {results?.results?.length ? (
      results.results.map((result) => {
        const confirmed = exactIds.has(result.id);

        return (
          <article
            className="mb-3 border border-[#b9c6bc] bg-[#fffdf8]/70 p-5"
            key={`${result.id}-${result.rank}`}
          >
            <div className="flex items-center gap-3 font-mono text-[11px] !text-[#79887e]">
              <span className="text-base !text-[#c15f45]">
                {String(result.rank).padStart(2, '0')}
              </span>

              <span>ID {result.id}</span>

              <span className="ml-auto flex items-center gap-1.5 !text-[#287d59]">
                <Gauge size={14} />
                {result.score.toFixed(4)}
              </span>
            </div>

            <p className="mt-4 text-sm leading-relaxed !text-[#304038]">
              {result.text}
            </p>

            {exactResults && !confirmed && (
              <span className="mt-3 inline-block border border-[#dfb675] bg-[#f6cf8f]/40 px-2 py-1 font-mono text-[10px] !text-[#765a35]">
                not in exact top 10
              </span>
            )}
          </article>
        );
      })
    ) : (
      <div className="grid min-h-64 place-items-center border border-dashed border-[#b9c6bc] text-center !text-[#78867d]">
        <div>
          <Search size={28} className="mx-auto" />

          <p className="mt-3 !text-[#405349]">
            Semantic results will appear here.
          </p>

          <span className="font-mono text-[11px] !text-[#78867d]">
            HNSW and exact search run together for comparison.
          </span>
        </div>
      </div>
    )}
  </div>

  <aside className="h-fit border border-[#b9c6bc] bg-[#dce6dd] p-5 !text-[#304038]">
    <div className="font-mono text-[11px] tracking-[0.12em] !text-[#5c7167]">
      COMPARISON
    </div>

    <h3 className="mt-3 text-xl font-semibold !text-[#304038]">
      Approximate vs exact
    </h3>

    <p className="mt-2 text-sm leading-relaxed !text-[#687970]">
      Every query runs through both engines. The exact scan is the
      ground truth; live metrics show what HNSW trades for speed.
    </p>

    {results && exactResults && (
      <dl className="mt-6 space-y-3 border-t border-[#b9c6bc] pt-4 font-mono text-xs !text-[#304038]">
        <div className="flex justify-between">
          <dt>Recall@10</dt>
          <dd className="!text-[#287d59]">
            {(recall * 100).toFixed(1)}%
          </dd>
        </div>

        <div className="flex justify-between">
          <dt>Speedup</dt>
          <dd className="!text-[#c15f45]">
            {speedup.toFixed(2)}×
          </dd>
        </div>

        <div className="flex justify-between">
          <dt>HNSW latency</dt>
          <dd>{formatMs(results.latency_ms)}</dd>
        </div>

        <div className="flex justify-between">
          <dt>Exact latency</dt>
          <dd>{formatMs(exactResults.latency_ms)}</dd>
        </div>
      </dl>
    )}
  </aside>
</section>
      </main>
      <footer className="mx-auto flex max-w-[1240px] justify-between border-t border-[#b9c6bc] px-5 py-5 font-mono text-[10px] tracking-widest text-[#7a887f] sm:px-10">
        <span className="flex items-center gap-2">
          <Activity size={14} /> VECTOR DATABASE PROJECT
        </span>
        <span>AG NEWS · COSINE SIMILARITY</span>
      </footer>
    </div>
  );
}
