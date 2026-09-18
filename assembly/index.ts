const p = new Uint8Array(512);

export function init(seed: i32): void {
  const base = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    base[i] = i;
  }
  
  let s: i64 = seed;
  for (let i = 255; i > 0; i--) {
    s = ((s * 16807) + 7) % 2147483647;
    const j = i32(s % (i + 1));
    const temp = base[i];
    base[i] = base[j];
    base[j] = temp;
  }
  
  for (let i = 0; i < 512; i++) {
    p[i] = base[i & 255];
  }
}

@inline
function fade(t: f64): f64 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

@inline
function lerp(a: f64, b: f64, t: f64): f64 {
  return a + t * (b - a);
}

@inline
function grad(hash: i32, x: f64, y: f64): f64 {
  const h = hash & 3;
  return ((h & 1) === 0 ? x : -x) + ((h & 2) === 0 ? y : -y);
}

export function noise(x: f64, y: f64): f64 {
  const X = i32(Math.floor(x)) & 255;
  const Y = i32(Math.floor(y)) & 255;
  
  const xf = x - Math.floor(x);
  const yf = y - Math.floor(y);
  
  const u = fade(xf);
  const v = fade(yf);
  
  const A = i32(p[X]) + Y;
  const B = i32(p[X + 1]) + Y;
  
  const val = lerp(
    lerp(grad(p[A], xf, yf), grad(p[B], xf - 1.0, yf), u),
    lerp(grad(p[A + 1], xf, yf - 1.0), grad(p[B + 1], xf - 1.0, yf - 1.0), u),
    v
  );
  return val;
}

export function fbm(x: f64, y: f64, octaves: i32, lac: f64, gain: f64): f64 {
  let sum = 0.0;
  let amp = 1.0;
  let freq = 1.0;
  let max = 0.0;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x * freq, y * freq) * amp;
    max += amp;
    amp *= gain;
    freq *= lac;
  }
  return sum / max;
}

export function ridged(x: f64, y: f64, octaves: i32, lac: f64, gain: f64): f64 {
  let sum = 0.0;
  let amp = 1.0;
  let freq = 1.0;
  let prev = 1.0;
  for (let i = 0; i < octaves; i++) {
    let n = noise(x * freq, y * freq);
    n = 1.0 - Math.abs(n);
    n = n * n * prev;
    prev = n;
    sum += n * amp;
    freq *= lac;
    amp *= gain;
  }
  return sum;
}

@inline
function smoothstep(edge0: f64, edge1: f64, x: f64): f64 {
  const t = Math.max(0.0, Math.min(1.0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3.0 - 2.0 * t);
}

export function generateHeight(x: f64, z: f64, maxHeight: f64): f64 {
  const nx = x / 2000.0;
  const nz = z / 2000.0;

  const valleyNoise = (noise(nx * 0.5, nz * 0.5) + 1.0) * 0.5;
  const envelope = smoothstep(0.25, 0.75, valleyNoise);

  let h = (fbm(nx * 3.5 + 10.0, nz * 3.5 + 10.0, 5, 2.0, 0.5) + 1.0) * 0.5;
  const ridge = ridged(nx * 3.0 + 5.0, nz * 3.0 + 5.0, 5, 2.2, 0.52);

  h = h * 0.3 + ridge * 0.7;
  h *= envelope;

  h += (fbm(nx * 12.0, nz * 12.0, 4, 2.0, 0.45) + 1.0) * 0.04;
  h += (fbm(nx * 28.0, nz * 28.0, 3, 2.0, 0.4) + 1.0) * 0.012;

  const finalHeight = h * maxHeight;
  return finalHeight < 15.0 ? 15.0 : finalHeight;
}
