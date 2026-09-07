function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <h1 className="text-4xl font-bold tracking-tight">
          Vector Database
        </h1>

        <p className="mt-3 text-zinc-400">
          Exact Brute-Force vs. HNSW
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
            <p className="text-sm text-zinc-500">Vectors</p>
            <p className="mt-2 text-3xl font-semibold">—</p>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
            <p className="text-sm text-zinc-500">Index</p>
            <p className="mt-2 text-3xl font-semibold">HNSW</p>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
            <p className="text-sm text-zinc-500">Status</p>
            <p className="mt-2 text-3xl font-semibold text-green-400">
              Ready
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App