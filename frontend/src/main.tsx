import { createRoot } from "react-dom/client";
import App from "./App";
import { applyTheme, readTheme } from "./theme";
import "./style.css";
// Apply a stored Appearance choice before the first render; "system" needs no attribute.
applyTheme(readTheme());
createRoot(document.getElementById("root")!).render(<App />);
