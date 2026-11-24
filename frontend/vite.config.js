import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // ws: true forwards the live gaze WebSocket (/api/gaze/stream).
      "/api": { target: "http://127.0.0.1:8001", ws: true },
      "/Capstone.yolov8/train/images": "http://127.0.0.1:8001",
    },
  },
});
