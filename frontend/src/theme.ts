export type Theme = "system" | "light" | "dark";
export const THEME_KEY = "vektri.theme";
/** id, label, icon. "system" follows prefers-color-scheme through the stylesheet's light-dark() tokens. */
export const THEMES: readonly [Theme, string, string][] = [
  ["system", "System", "monitor"],
  ["light", "Light", "sun"],
  ["dark", "Dark", "moon"],
];
export function readTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    return "system";
  }
}
export function nextTheme(theme: Theme): Theme {
  return theme === "system" ? "light" : theme === "light" ? "dark" : "system";
}
/**
 * Writes the choice to <html data-theme>, which switches color-scheme in CSS;
 * every token resolves through light-dark(), so nothing else needs to know.
 */
export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Storage unavailable (private mode): the choice lasts for this page only.
  }
  // Keep the browser chrome colour in step with the resolved page background.
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta && document.body)
    meta.setAttribute(
      "content",
      getComputedStyle(document.body).backgroundColor,
    );
}
