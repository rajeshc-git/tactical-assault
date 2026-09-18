const CACHE_NAME = 'tactical-assault-v1';

// Install event - force activation immediately
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

// Activate event - claim all open clients immediately & clear old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log('[ServiceWorker] Removing old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
    ])
  );
});

// Cache-First with Network Update Strategy (caches both local and raw GitHub game models)
self.addEventListener('fetch', (event) => {
  // Only handle HTTP/HTTPS GET requests
  if (event.request.method !== 'GET' || !event.request.url.startsWith('http')) {
    return;
  }

  const url = event.request.url;

  // Exclude third-party CDN scripts/fonts (like Google Fonts) from Service Worker interception
  // only intercept local project assets & target raw GitHub model GLBs.
  const isLocal = url.startsWith(self.location.origin);
  const isGithubModel = url.includes('raw.githubusercontent.com') && url.endsWith('.glb');

  if (!isLocal && !isGithubModel) {
    return;
  }

  event.respondWith(
    caches.open(CACHE_NAME).then((cache) => {
      // Query the specific cache instance directly to prevent global CacheStorage matching TypeErrors
      return cache.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          // Return cached file immediately for instant load
          return cachedResponse;
        }

        return fetch(event.request).then((networkResponse) => {
          // Validate response before caching (opaque responses have status 0, which is normal for cross-origin assets)
          if (!networkResponse || (networkResponse.status !== 200 && networkResponse.status !== 0)) {
            return networkResponse;
          }

          // Cache the response clone for future loads
          const responseToCache = networkResponse.clone();
          cache.put(event.request, responseToCache);

          return networkResponse;
        }).catch(() => {
          // Offline fallback if fetch fails
          return new Response('Network error occurred.', { status: 408, statusText: 'Network Error' });
        });
      });
    })
  );
});
