# ScholarRAG Frontend

React + Vite chat UI for [ScholarRAG](../README.md).

```bash
npm install
npm run dev      # http://localhost:5173 (proxies API to backend)
npm run build    # output to dist/ (served by FastAPI in production)
```

SSE events from `/api/chat` are parsed in `src/api.js`; see root README for API details.
