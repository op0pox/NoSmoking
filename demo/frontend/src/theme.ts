// 라이트/다크 선택. 기본은 이 PC의 설정을 따르고, 사이드바의 버튼(또는 주소의 ?theme=light|dark)으로 고르면
// 이 브라우저에 기억한다. 저장소가 막혀 있으면(사생활 보호 모드 등) 기억만 하지 않고 동작은 그대로 한다.
export type Theme = "light" | "dark";
const KEY = "nosmoking-theme";

function readStored(): Theme | null {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function currentTheme(): Theme {
  const set = document.documentElement.dataset.theme;
  if (set === "light" || set === "dark") return set;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // 기억하지 못해도 이번 화면에는 적용된다
  }
}

/** 시작할 때 한 번: 주소의 ?theme= 가 있으면 그것을, 없으면 기억해 둔 값을 적용한다. */
export function initTheme() {
  const fromUrl = new URLSearchParams(window.location.search).get("theme");
  if (fromUrl === "light" || fromUrl === "dark") applyTheme(fromUrl);
  else {
    const stored = readStored();
    if (stored) document.documentElement.dataset.theme = stored;
  }
}
