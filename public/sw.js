const CACHE_NAME = 'tactical-assault-v4.2.0';

// Heavy static binary assets suitable for caching (models, sounds, textures, wasm)
const STATIC_EXTENSIONS = ['.glb', '.gltf', '.mp3', '.ogg', '.wav', '.png', '.jpg', '.jpeg', '.webp', '.wasm'];

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log('[SW] Purging outdated cache store:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
    ])
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || !event.request.url.startsWith('http')) {
    return;
  }

  const url = new URL(event.request.url);

  // Skip Vite internal dev requests & dynamic module timestamps
  if (url.pathname.includes('/@vite/') || url.pathname.includes('/@fs/') || url.search.includes('t=')) {
    return;
  }

  // Identify heavy static binary models, textures, sounds
  const isStaticBinary = STATIC_EXTENSIONS.some(ext => url.pathname.endsWith(ext)) ||
    (url.hostname === 'raw.githubusercontent.com' && url.pathname.endsWith('.glb'));

  if (isStaticBinary) {
    // Cache-First for heavy static media files (3D models, audio, textures)
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        const cached = await cache.match(event.request);
        if (cached) return cached;
        try {
          const networkRes = await fetch(event.request);
          if (networkRes && (networkRes.status === 200 || networkRes.status === 0)) {
            cache.put(event.request, networkRes.clone());
          }
          return networkRes;
        } catch (e) {
          return cached || new Response('Asset not available offline.', { status: 404 });
        }
      })
    );
  } else {
    // STRICT NETWORK-FIRST for HTML, JavaScript, CSS, and modules
    // Always fetch live code from server with no-cache validation so soft refreshes get new code instantly
    event.respondWith(
      fetch(event.request, { cache: 'no-cache' })
        .then((networkRes) => {
          if (networkRes && networkRes.status === 200) {
            const resClone = networkRes.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, resClone));
          }
          return networkRes;
        })
        .catch(async () => {
          const cached = await caches.match(event.request);
          if (cached) return cached;
          return new Response('Network error occurred.', { status: 408, statusText: 'Network Error' });
        })
    );
  }
});
