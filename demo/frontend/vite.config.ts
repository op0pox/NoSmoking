import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발·미리보기 서버도 이 PC(127.0.0.1)에서만 열리게 한다. /api 요청은 FastAPI 백엔드로 넘긴다
// (기본 8000번. 다른 백엔드를 쓰려면 NOSMOKING_BACKEND_URL로 바꾼다).
const backend = process.env.NOSMOKING_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": backend },
  },
  preview: {
    host: "127.0.0.1",
    proxy: { "/api": backend },
  },
});
