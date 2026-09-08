// ============================================================
// USD/INR FORECAST DASHBOARD - SERVICE WORKER
// ============================================================

const CACHE_NAME = "fx-dashboard-v6";


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

    // Activate the new service worker immediately
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

    // Take control of open pages immediately
    self.clients.claim();

});


// ============================================================
// FETCH HANDLER
// ============================================================

self.addEventListener("fetch", (event) => {

    const request = event.request;
    const url = new URL(request.url);


    // ========================================================
    // API REQUESTS
    // ========================================================
    // Network first:
    //
    // 1. Try the live API
    // 2. Save successful response
    // 3. If offline, return cached API response
    //
    // This applies to each currency pair separately because
    // the query string is part of the cache key.
    // ========================================================

    if (
        url.pathname === "/api/predict" ||
        url.pathname === "/api/history"
    ) {

        event.respondWith(

            fetch(request)

                .then((response) => {

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
    // PAGE NAVIGATION
    // ========================================================
    // Network first when online.
    //
    // If internet is unavailable:
    // return the cached dashboard.
    // ========================================================

    if (request.mode === "navigate") {

        event.respondWith(

            fetch(request)

                .then((response) => {

                    if (response.ok) {

                        const responseClone =
                            response.clone();

                        caches.open(CACHE_NAME)
                            .then((cache) => {

                                cache.put(
                                    "/",
                                    responseClone
                                );

                            });

                    }

                    return response;

                })

                .catch(() => {

                    return caches.match("/")
                        .then((cachedPage) => {

                            if (cachedPage) {
                                return cachedPage;
                            }

                            return caches.match(
                                "/static/index.html"
                            );

                        });

                })

        );

        return;

    }


    // ========================================================
    // CDN FILES
    // ========================================================
    // Cache first:
    //
    // Chart.js
    // Hammer.js
    // chartjs-plugin-zoom
    //
    // These are already included in APP_SHELL.
    // ========================================================

    if (url.hostname === "cdn.jsdelivr.net") {

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
    // Network first.
    //
    // If online:
    //     get the newest resource
    //
    // If offline:
    //     use cached resource
    // ========================================================

    event.respondWith(

        fetch(request)

            .then((response) => {

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