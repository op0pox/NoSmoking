import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발 서버도 이 PC(127.0.0.1)에서만 열리게 한다. /api 요청은 FastAPI 백엔드(8000번)로 넘긴다.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
