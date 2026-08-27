// service-worker.js
// Place this in the SAME folder Flask serves index.html from
// (e.g. your Flask "static" or "templates" root, whichever the browser loads it from)

const CACHE_NAME = 'usdinr-forecast-cache-v1';

// The page itself (adjust filename if Flask serves it under a different route)
const APP_SHELL = [
  './',
  './index.html',
  './manifest.json'
];

// ---- INSTALL: cache the app shell ----
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
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

  // API calls (/api/predict, /api/history): network-first, cache fallback.
  // This is what makes "offline mode" actually work for this app --
  // last successful API response gets served when there's no connection.
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

  // Everything else (the HTML page itself, CDN scripts): cache-first, network fallback
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
