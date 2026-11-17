const themeButton = document.querySelector('.theme-toggle');

function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const label = dark ? 'Switch to light theme' : 'Switch to dark theme';
  themeButton.setAttribute('aria-label', label);
  themeButton.setAttribute('title', label);
  themeButton.setAttribute('aria-pressed', String(dark));
}

themeButton.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  if (dark) document.documentElement.dataset.theme = 'dark';
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem('notes-theme', dark ? 'dark' : 'light'); } catch {}
  updateThemeButton();
});
updateThemeButton();
