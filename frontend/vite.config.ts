import { defineConfig, type ProxyOptions } from "vite";

const apiOrigin = "http://127.0.0.1:8741";
const devHosts = new Set(["127.0.0.1:5173", "localhost:5173"]);
const devOrigins = new Set(["http://127.0.0.1:5173", "http://localhost:5173"]);

function localApiProxy(): ProxyOptions {
  return {
    target: apiOrigin,
    changeOrigin: true,
    // Do not turn an unexpected incoming Host into a trusted API Host.
    bypass(request) {
      if (!devHosts.has((request.headers.host ?? "").toLowerCase()))
        return false;
    },
    configure(proxy) {
      proxy.on("proxyReq", (outgoing, incoming) => {
        // Only this explicit local development UI gets the API origin.
        // Foreign origins and Sec-Fetch-Site pass through for API rejection.
        const origin = incoming.headers.origin;
        if (typeof origin === "string" && devOrigins.has(origin))
          outgoing.setHeader("origin", apiOrigin);
      });
    },
  };
}

export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    host: "127.0.0.1",
    allowedHosts: ["localhost", "127.0.0.1"],
    cors: false,
    proxy: {
      "/api": localApiProxy(),
      "/docs": localApiProxy(),
      "/openapi.json": localApiProxy(),
    },
  },
  build: { target: "es2022" },
});
