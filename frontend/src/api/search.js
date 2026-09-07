const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        body?.detail || body?.message || 'The search server returned an error.',
      );
    }
    return body;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('The search request took too long. Please try again.');
    }
    if (error instanceof TypeError) {
      throw new Error(
        'Unable to connect to the vector search server. Make sure the backend is running.',
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function searchDocuments(query, top_k = 10, ef_search = 200) {
  return request('/api/search', {
    method: 'POST',
    body: JSON.stringify({ query, top_k, ef_search }),
  });
}

export function getHealth() {
  return request('/api/health');
}
