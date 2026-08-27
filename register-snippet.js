// Add this <script> block right before your closing </body> tag in index.html,
// alongside your existing <script> that has loadDashboard() etc.

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./service-worker.js')
      .then((reg) => console.log('Service worker registered:', reg.scope))
      .catch((err) => console.error('Service worker registration failed:', err));
  });
}
