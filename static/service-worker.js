// service-worker.js
// Place inside your Flask "static" folder, alongside index.html

const CACHE_NAME = 'usdinr-forecast-cache-v4';

// Everything needed to render the page even with no internet:
// your own files + the external CDN scripts the page depends on.
const APP_SHELL = [
  '/',
  '/static/index.html',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',

  'https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js',
  'https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js',
  'https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js'
];

// Requests that rarely change once fetched (CDN libraries) —
// safe to serve cache-first for speed.
const CACHE_FIRST_HOSTS = [
  'cdn.jsdelivr.net'
];

// ---- INSTALL: cache the app shell (including CDN scripts) ----
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      // Cache each file individually so one failed CDN fetch
      // doesn't block the whole install.
      return Promise.all(
        APP_SHELL.map((url) =>
          cache.add(url).catch((err) => {
            console.warn('Failed to cache during install:', url, err);
          })
        )
      );
    })
  );
  self.skipWaiting();
});

// ---- ACTIVATE: clean up old caches ----
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ---- FETCH ----
self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // API calls: network-first, cache fallback (always try to get the
  // freshest forecast; fall back to the last successful one offline)
  if (url.includes('/api/predict') || url.includes('/api/history')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // CDN libraries (Chart.js, Hammer.js, zoom plugin): these rarely
  // change once cached, so cache-first is fine and faster.
  const isCacheFirstHost = CACHE_FIRST_HOSTS.some((host) => url.includes(host));
  if (isCacheFirstHost) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
    return;
  }

  // Everything else (the page itself, index.html, manifest.json):
  // network-first, cache fallback. This means devices that ARE
  // online always get the freshest page instead of getting stuck
  // on whatever was cached the first time — and devices that go
  // offline still fall back to the last successful copy.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});