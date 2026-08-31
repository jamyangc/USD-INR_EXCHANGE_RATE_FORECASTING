// ============================================================
// USD/INR FORECAST DASHBOARD - SERVICE WORKER
// ============================================================

const CACHE_NAME = "usdinr-forecast-cache-v5";

// ============================================================
// APP SHELL
// ============================================================

const APP_SHELL = [
    "/",
    "/static/index.html",
    "/static/manifest.json",
    "/static/icon-192.png",
    "/static/icon-512.png",

    // Chart.js
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js",

    // Hammer.js
    "https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js",

    // Chart.js Zoom Plugin
    "https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js"
];


// ============================================================
// CDN HOSTS
// ============================================================

const CACHE_FIRST_HOSTS = [
    "cdn.jsdelivr.net"
];


// ============================================================
// INSTALL
// ============================================================

self.addEventListener("install", (event) => {

    event.waitUntil(

        caches.open(CACHE_NAME).then((cache) => {

            return Promise.all(

                APP_SHELL.map((url) => {

                    return cache.add(url).catch((error) => {

                        console.warn(
                            "Could not cache:",
                            url,
                            error
                        );

                    });

                })

            );

        })

    );

    self.skipWaiting();

});


// ============================================================
// ACTIVATE
// ============================================================

self.addEventListener("activate", (event) => {

    event.waitUntil(

        caches.keys().then((cacheNames) => {

            return Promise.all(

                cacheNames
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))

            );

        })

    );

    self.clients.claim();

});


// ============================================================
// FETCH HANDLER
// ============================================================

self.addEventListener("fetch", (event) => {

    const request = event.request;
    const url = request.url;


    // ========================================================
    // API REQUESTS
    // ========================================================
    // Network first:
    // Try to get the newest forecast.
    // If offline, use the previously cached response.
    // ========================================================

    if (
        url.includes("/api/predict") ||
        url.includes("/api/history")
    ) {

        event.respondWith(

            fetch(request)

                .then((response) => {

                    // Only cache successful responses
                    if (response.ok) {

                        const responseClone =
                            response.clone();

                        caches.open(CACHE_NAME)
                            .then((cache) => {

                                cache.put(
                                    request,
                                    responseClone
                                );

                            });

                    }

                    return response;

                })

                .catch(() => {

                    return caches.match(request);

                })

        );

        return;

    }


    // ========================================================
    // CDN FILES
    // ========================================================
    // Cache first:
    // Chart.js, Hammer.js and the zoom plugin rarely change.
    // ========================================================

    const isCacheFirstHost =
        CACHE_FIRST_HOSTS.some(
            (host) => url.includes(host)
        );


    if (isCacheFirstHost) {

        event.respondWith(

            caches.match(request)

                .then((cachedResponse) => {

                    if (cachedResponse) {

                        return cachedResponse;

                    }

                    return fetch(request);

                })

        );

        return;

    }


    // ========================================================
    // OTHER REQUESTS
    // ========================================================
    // Network first:
    // This keeps the dashboard updated when online.
    // Cached version is used when offline.
    // ========================================================

    event.respondWith(

        fetch(request)

            .then((response) => {

                // Cache only successful GET responses
                if (
                    request.method === "GET" &&
                    response.ok
                ) {

                    const responseClone =
                        response.clone();

                    caches.open(CACHE_NAME)
                        .then((cache) => {

                            cache.put(
                                request,
                                responseClone
                            );

                        });

                }

                return response;

            })

            .catch(() => {

                return caches.match(request);

            })

    );

});