// ================================================================
//  SNOW MOUNTAIN EXPLORER — 3D Game Engine
//  Built with Three.js — Open World Action Explorer
// ================================================================
import * as THREE from 'three';
import './style.css';
import createEngineModule from './engine.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// --- IndexedDB Binary Asset Cache (Bypasses Chrome HTTP Disk Cache single-file size limits) ---
const ASSET_DB_NAME = 'TacticalAssaultAssetDB';
const ASSET_DB_VERSION = 1;
const ASSET_STORE_NAME = 'glb_models';

function openAssetDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(ASSET_DB_NAME, ASSET_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(ASSET_STORE_NAME)) {
        db.createObjectStore(ASSET_STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getCachedAssetBuffer(key: string): Promise<ArrayBuffer | null> {
  try {
    const db = await openAssetDB();
    return new Promise((resolve) => {
      const tx = db.transaction(ASSET_STORE_NAME, 'readonly');
      const store = tx.objectStore(ASSET_STORE_NAME);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch (err) {
    return null;
  }
}

async function saveAssetBuffer(key: string, buffer: ArrayBuffer): Promise<void> {
  try {
    const db = await openAssetDB();
    const tx = db.transaction(ASSET_STORE_NAME, 'readwrite');
    const store = tx.objectStore(ASSET_STORE_NAME);
    store.put(buffer, key);
  } catch (err) {
    console.warn('Failed to save asset to IndexedDB:', err);
  }
}

// ================================================================
//  PERLIN NOISE GENERATOR
// ================================================================
class PerlinNoise {
  [key: string]: any;
  constructor(seed = 42) {
    this.p = new Uint8Array(512);
    const base = new Uint8Array(256);
    for (let i = 0; i < 256; i++) base[i] = i;
    let s = seed;
    for (let i = 255; i > 0; i--) {
      s = ((s * 16807) + 7) % 2147483647;
      const j = s % (i + 1);
      [base[i], base[j]] = [base[j], base[i]];
    }
    for (let i = 0; i < 512; i++) this.p[i] = base[i & 255];
  }
  fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
  lerp(a, b, t) { return a + t * (b - a); }
  grad(hash, x, y) {
    const h = hash & 3;
    return ((h & 1) === 0 ? x : -x) + ((h & 2) === 0 ? y : -y);
  }
  noise(x, y) {
    const X = Math.floor(x) & 255, Y = Math.floor(y) & 255;
    x -= Math.floor(x); y -= Math.floor(y);
    const u = this.fade(x), v = this.fade(y), p = this.p;
    const A = p[X] + Y, B = p[X + 1] + Y;
    return this.lerp(
      this.lerp(this.grad(p[A], x, y), this.grad(p[B], x - 1, y), u),
      this.lerp(this.grad(p[A + 1], x, y - 1), this.grad(p[B + 1], x - 1, y - 1), u), v
    );
  }
  fbm(x, y, octaves = 5, lac = 2.0, gain = 0.5) {
    let sum = 0, amp = 1, freq = 1, max = 0;
    for (let i = 0; i < octaves; i++) { sum += this.noise(x * freq, y * freq) * amp; max += amp; amp *= gain; freq *= lac; }
    return sum / max;
  }
  ridged(x, y, octaves = 5, lac = 2.2, gain = 0.5) {
    let sum = 0, amp = 1, freq = 1, max = 0;
    for (let i = 0; i < octaves; i++) { let n = this.noise(x * freq, y * freq); n = 1.0 - Math.abs(n); n *= n; sum += n * amp; max += amp; amp *= gain; freq *= lac; }
    return sum / max;
  }
}

// ================================================================
//  INFINITE TERRAIN CHUNK CLASS WITH LOD
// ================================================================
class TerrainChunk {
  [key: string]: any;
  constructor(scene, cx, cz, size, segments, lod, game) {
    this.scene = scene;
    this.cx = cx;
    this.cz = cz;
    this.size = size;
    this.segments = segments;
    this.lod = lod;
    this.game = game;
    this.mesh = null;
    this.treeMeshes = null;

    this.create();
  }

  create() {
    const S = this.segments;
    const size = this.size;
    const half = size / 2;
    const startX = this.cx * size - half;
    const startZ = this.cz * size - half;

    const geo = new THREE.PlaneGeometry(size, size, S, S);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    const count = pos.count;

    for (let i = 0; i < count; i++) {
      const localX = pos.getX(i);
      const localZ = pos.getZ(i);
      const worldX = startX + localX + half;
      const worldZ = startZ + localZ + half;
      const y = this.game.generateHeight(worldX, worldZ);
      pos.setY(i, y);
    }
    geo.computeVertexNormals();

    const colors = new Float32Array(count * 3);
    const normals = geo.attributes.normal;
    for (let i = 0; i < count; i++) {
      const y = pos.getY(i);
      const ny = normals.getY(i);
      const slope = 1 - ny;
      const hr = y / this.game.MAX_HEIGHT;

      const localX = pos.getX(i);
      const localZ = pos.getZ(i);
      const worldX = startX + localX + half;
      const worldZ = startZ + localZ + half;
      const cv = this.game.wasm.noise(worldX * 0.03, worldZ * 0.03) * 0.04;

      const { r, g, b } = this.game.mapConfig.terrainColors(hr, slope, cv);

      colors[i * 3] = THREE.MathUtils.clamp(r, 0, 1);
      colors[i * 3 + 1] = THREE.MathUtils.clamp(g, 0, 1);
      colors[i * 3 + 2] = THREE.MathUtils.clamp(b, 0, 1);
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.0
    });

    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.position.set(this.cx * size, 0, this.cz * size);
    this.mesh.receiveShadow = true;
    this.mesh.castShadow = true;
    this.scene.add(this.mesh);

    if (this.lod < 2) {
      this.createTacticalCover();
      this.createGroundItems();
    }
  }

  createTrees() {
    const size = this.size;
    const startX = this.cx * size - size / 2;
    const startZ = this.cz * size - size / 2;

    const tc = this.game.mapConfig.treeColors;
    if (!tc) return; // some biomes may have no trees

    const treeRange = this.game.mapConfig.treeRange || [0.1, 0.58];
    const treePositions = [];
    const step = this.game.treeStep || 50;

    for (let x = 20; x < size; x += step) {
      for (let z = 20; z < size; z += step) {
        const worldX = startX + x + this.game.wasm.noise(startX + x, startZ + z) * 15;
        const worldZ = startZ + z + this.game.wasm.noise(startZ + z, startX + x) * 15;
        const y = this.game.getHeightAt(worldX, worldZ);
        const hr = y / this.game.MAX_HEIGHT;

        if (hr > treeRange[0] && hr < treeRange[1]) {
          const dx = this.game.getHeightAt(worldX + 2, worldZ) - this.game.getHeightAt(worldX - 2, worldZ);
          const dz = this.game.getHeightAt(worldX, worldZ + 2) - this.game.getHeightAt(worldX, worldZ - 2);
          const slope = Math.sqrt(dx * dx + dz * dz) / 4;

          if (slope < 0.95) {
            treePositions.push({ x: worldX, y, z: worldZ });
          }
        }
      }
    }

    if (treePositions.length === 0) return;

    const N = treePositions.length;
    const trunkGeo = new THREE.CylinderGeometry(0.3, 0.55, 5, 6);
    const lowerGeo = new THREE.ConeGeometry(4.2, 7, 7);
    const upperGeo = new THREE.ConeGeometry(2.8, 5.5, 7);
    const snowCapGeo = new THREE.ConeGeometry(3.0, 1.8, 7);

    const trunkMat = new THREE.MeshStandardMaterial({ color: tc.trunk, roughness: 0.95 });
    const foliageMat = new THREE.MeshStandardMaterial({ color: tc.foliage, roughness: 0.88 });
    const snowMat = new THREE.MeshStandardMaterial({ color: tc.snowColor || 0xe4ecf0, roughness: 0.75 });

    const trunkIM = new THREE.InstancedMesh(trunkGeo, trunkMat, N);
    const lowerIM = new THREE.InstancedMesh(lowerGeo, foliageMat, N);
    const upperIM = new THREE.InstancedMesh(upperGeo, foliageMat, N);

    trunkIM.castShadow = true;
    lowerIM.castShadow = true;
    lowerIM.receiveShadow = true;
    upperIM.castShadow = true;

    let snowCapIM: any = null;
    if (tc.snowCap) {
      snowCapIM = new THREE.InstancedMesh(snowCapGeo, snowMat, N);
    }

    const dummy = new THREE.Object3D();
    treePositions.forEach((pos, idx) => {
      const sc = 0.65 + Math.abs(this.game.wasm.noise(pos.x, pos.z)) * 0.8;
      const rot = Math.abs(this.game.wasm.noise(pos.z, pos.x)) * Math.PI * 2;

      dummy.position.set(pos.x - this.cx * size, pos.y + 2.5 * sc, pos.z - this.cz * size);
      dummy.rotation.set(0, rot, 0);
      dummy.scale.setScalar(sc);
      dummy.updateMatrix();
      trunkIM.setMatrixAt(idx, dummy.matrix);

      dummy.position.y = pos.y + 6.5 * sc;
      dummy.updateMatrix();
      lowerIM.setMatrixAt(idx, dummy.matrix);

      dummy.position.y = pos.y + 10 * sc;
      dummy.scale.setScalar(sc * 0.85);
      dummy.updateMatrix();
      upperIM.setMatrixAt(idx, dummy.matrix);

      if (snowCapIM) {
        dummy.position.y = pos.y + 13 * sc;
        dummy.scale.set(sc * 0.82, sc * 0.5, sc * 0.82);
        dummy.updateMatrix();
        snowCapIM.setMatrixAt(idx, dummy.matrix);
      }
    });

    trunkIM.instanceMatrix.needsUpdate = true;
    lowerIM.instanceMatrix.needsUpdate = true;
    upperIM.instanceMatrix.needsUpdate = true;

    this.mesh.add(trunkIM);
    this.mesh.add(lowerIM);
    this.mesh.add(upperIM);

    const meshes = [trunkIM, lowerIM, upperIM];

    if (snowCapIM) {
      snowCapIM.instanceMatrix.needsUpdate = true;
      this.mesh.add(snowCapIM);
      meshes.push(snowCapIM);
    }

    this.treeMeshes = meshes;
  }

  createTacticalCover() {
    const size = this.size;
    const startX = this.cx * size - size / 2;
    const startZ = this.cz * size - size / 2;
    const mapId = this.game.mapId || 'arctic';

    const coverGroup = new THREE.Group();
    this.coverObstacles = [];

    // Dynamic cover obstacle placement based on graphics quality
    const step = this.game.coverStep || (this.game.graphicsQuality === 'low' ? 240 : (this.game.graphicsQuality === 'medium' ? 140 : 90));
    const threshold = this.game.graphicsQuality === 'low' ? 0.35 : (this.game.graphicsQuality === 'medium' ? 0.18 : 0.06);
    for (let x = 30; x < size; x += step) {
      for (let z = 30; z < size; z += step) {
        const nVal = this.game.wasm.noise(startX + x * 0.05, startZ + z * 0.05);
        if (nVal > threshold) {
          const worldX = startX + x + (this.game.wasm.noise(startX + x * 3, startZ + z * 3) * 20);
          const worldZ = startZ + z + (this.game.wasm.noise(startZ + z * 3, startX + x * 3) * 20);
          const y = this.game.getHeightAt(worldX, worldZ);
          const hr = y / this.game.MAX_HEIGHT;

          if (hr > 0.08 && hr < 0.75) {
            // Generate full-range [-1, 1] noise value for index selection and rotation to fix the 2-model mixing bug
            const seedNoise = this.game.wasm.noise(worldX * 0.12, worldZ * 0.12);

            const coverObj = this.buildCoverModel(mapId, seedNoise, this.lod);
            coverObj.position.set(worldX, y, worldZ);

            // Compute the terrain surface normal at the spawn point using finite differences
            const eps = 2.0;
            const hL = this.game.getHeightAt(worldX - eps, worldZ);
            const hR = this.game.getHeightAt(worldX + eps, worldZ);
            const hB = this.game.getHeightAt(worldX, worldZ - eps);
            const hF = this.game.getHeightAt(worldX, worldZ + eps);

            const dh_dx = (hR - hL) / (2 * eps);
            const dh_dz = (hF - hB) / (2 * eps);

            const normal = new THREE.Vector3(-dh_dx, 1.0, -dh_dz).normalize();

            // Apply both a random rotation around the up-axis and tilt alignment matching the terrain slope
            const randomAngle = seedNoise * Math.PI * 2;
            const rotQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), randomAngle);
            const alignQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);

            coverObj.quaternion.copy(alignQ).multiply(rotQ);
            coverGroup.add(coverObj);

            // Retrieve selected GLTF model metadata to scale collision properties
            const normalizedNoise = (seedNoise + 1) / 2;
            const modelIndex = Math.floor(normalizedNoise * this.game.obstacleGltfs.length) % this.game.obstacleGltfs.length;
            const selectedGltf = this.game.obstacleGltfs[modelIndex];
            const isObst1 = selectedGltf && (selectedGltf as any).isObsticle1;
            const isObst2 = selectedGltf && (selectedGltf as any).isObsticle2;

            let targetScale = 12.0 + normalizedNoise * 4.0;
            if (isObst1) {
              targetScale *= 3.0;
            } else if (isObst2) {
              targetScale *= 2.0;
            }

            if (selectedGltf) {
              if (isObst1) {
                // Giant solid boulder: single wide central collision circle
                this.coverObstacles.push({ x: worldX, z: worldZ, radius: targetScale * 0.38 });
              } else if (isObst2) {
                // Arch/Gate: two separate collision circles on the left & right pillars
                // Compute local pillar offset along local X-axis (width X:1.90), rotated by the gate's world orientation
                const localOffsetVec = new THREE.Vector3(targetScale * 0.78, 0, 0);
                localOffsetVec.applyQuaternion(coverObj.quaternion);

                const leftX = worldX - localOffsetVec.x;
                const leftZ = worldZ - localOffsetVec.z;
                const rightX = worldX + localOffsetVec.x;
                const rightZ = worldZ + localOffsetVec.z;

                // Push Left Pillar/Guardhouse (thicker/wider side) and Right Pillar collisions
                this.coverObstacles.push({ x: leftX, z: leftZ, radius: targetScale * 0.40 });
                this.coverObstacles.push({ x: rightX, z: rightZ, radius: targetScale * 0.30 });
              } else {
                const obsRadius = mapId === 'arctic' ? 7.5 : (mapId === 'forest' ? 7.0 : 8.0);
                this.coverObstacles.push({ x: worldX, z: worldZ, radius: obsRadius });
              }
            } else {
              const obsRadius = mapId === 'arctic' ? 7.5 : (mapId === 'forest' ? 7.0 : 8.0);
              this.coverObstacles.push({ x: worldX, z: worldZ, radius: obsRadius });
            }
          }
        }
      }
    }

    if (coverGroup.children.length > 0) {
      this.scene.add(coverGroup);
      this.coverGroup = coverGroup;
    }
  }

  buildCoverModel(mapId: string, noiseVal: number = 0, lod: number = 0) {
    if (this.game.obstacleGltfs && this.game.obstacleGltfs.length > 0) {
      const group = new THREE.Group();

      // Determine index based on noise value (scale from [-1, 1] to [0, length-1])
      const normalizedNoise = (noiseVal + 1) / 2;
      const index = Math.floor(normalizedNoise * this.game.obstacleGltfs.length) % this.game.obstacleGltfs.length;
      const selectedGltf = this.game.obstacleGltfs[index];

      // Scale is also deterministic based on the noise value (12 to 16, or 36 to 48 for obsticle1)
      let targetScale = 12.0 + normalizedNoise * 4.0;
      const isObst1 = (selectedGltf as any).isObsticle1;
      const isObst2 = (selectedGltf as any).isObsticle2;

      if (isObst1) {
        targetScale *= 3.0; // Thrice the size for obsticle1
      } else if (isObst2) {
        targetScale *= 2.0; // Double the size for obsticle2
      }

      // LOD: If the chunk is distant (LOD > 0), use a simplified "flat-colored" thumbnail proxy of the actual model geometry
      if (lod > 0) {
        const model = selectedGltf.scene.clone();
        model.scale.setScalar(targetScale);

        // Ground alignment
        const minY = (selectedGltf as any).minY || 0;
        model.position.y = -minY * targetScale;

        model.traverse((child: any) => {
          if (child.isMesh) {
            child.castShadow = false;
            child.receiveShadow = false;
            child.layers.set(0);

            // Convert expensive MeshStandardMaterial to cheap MeshBasicMaterial while retaining textures & colors
            if (child.material) {
              const convertMat = (oldMat: any) => {
                return new THREE.MeshBasicMaterial({
                  color: oldMat.color,
                  map: oldMat.map,
                  side: THREE.DoubleSide,
                  transparent: oldMat.transparent,
                  opacity: oldMat.opacity
                });
              };

              if (Array.isArray(child.material)) {
                child.material = child.material.map(m => convertMat(m));
              } else {
                child.material = convertMat(child.material);
              }
            }
          }
        });

        group.add(model);
        return group;
      }

      const model = selectedGltf.scene.clone();
      model.scale.setScalar(targetScale);

      // Fast O(1) ground alignment using pre-calculated bounding box minY (no vertex traversal overhead!)
      const minY = (selectedGltf as any).minY || 0;
      model.position.y = -minY * targetScale;

      group.add(model);
      return group;
    }

    const group = new THREE.Group();

    if (mapId === 'arctic') {
      // 🏔️ Natural Glacier Formations & Frost Granite Boulders (Realistic Matte Ice & Stone)
      const iceMat = new THREE.MeshStandardMaterial({
        color: 0xc4e6f8,
        roughness: 0.35,
        metalness: 0.1
      });
      const graniteMat = new THREE.MeshStandardMaterial({
        color: 0x485668,
        roughness: 0.9,
        metalness: 0.05
      });

      // Natural Glacier Ice Monolith
      const glacierGeo = new THREE.DodecahedronGeometry(6.0, 1);
      const glacierMesh = new THREE.Mesh(glacierGeo, iceMat);
      glacierMesh.scale.set(1.2, 2.2, 1.4);
      glacierMesh.position.y = 7.0;
      glacierMesh.castShadow = true;
      glacierMesh.receiveShadow = true;
      group.add(glacierMesh);

      // Frost Granite Boulder for crouching cover
      const b1 = new THREE.Mesh(new THREE.DodecahedronGeometry(4.2, 0), graniteMat);
      b1.position.set(-5.0, 2.8, 2.0);
      b1.rotation.set(0.3, 0.5, 0.2);
      b1.castShadow = true;
      b1.receiveShadow = true;
      group.add(b1);

      const b2 = new THREE.Mesh(new THREE.DodecahedronGeometry(3.2, 0), iceMat);
      b2.position.set(4.2, 2.2, -2.5);
      b2.castShadow = true;
      group.add(b2);

    } else if (mapId === 'forest') {
      // 🌲 Natural Forest Mossy Boulders & Fallen Timber Log
      const rockMat = new THREE.MeshStandardMaterial({ color: 0x404a40, roughness: 0.92, metalness: 0.0 });
      const woodMat = new THREE.MeshStandardMaterial({ color: 0x3d2c1d, roughness: 0.95, metalness: 0.0 });

      // Weathered Forest Boulder
      const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(5.2, 1), rockMat);
      rock.scale.set(1.4, 1.0, 1.2);
      rock.position.y = 3.2;
      rock.castShadow = true;
      rock.receiveShadow = true;
      group.add(rock);

      // Fallen Timber Log Barricade
      const log = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 1.5, 11.0, 8), woodMat);
      log.rotation.z = Math.PI / 2;
      log.position.set(0, 1.3, 4.5);
      log.castShadow = true;
      log.receiveShadow = true;
      group.add(log);

    } else if (mapId === 'autumn') {
      // 🍂 Natural Autumn Slate Rock Formations
      const slateMat = new THREE.MeshStandardMaterial({ color: 0x56483c, roughness: 0.92, metalness: 0.0 });
      const darkRockMat = new THREE.MeshStandardMaterial({ color: 0x3e342b, roughness: 0.95, metalness: 0.0 });

      // Slate Rock Ridge Wall
      const ridge = new THREE.Mesh(new THREE.DodecahedronGeometry(5.8, 1), slateMat);
      ridge.scale.set(2.0, 0.9, 1.1);
      ridge.position.y = 3.2;
      ridge.castShadow = true;
      ridge.receiveShadow = true;
      group.add(ridge);

      // Flanking Autumn Boulder
      const b1 = new THREE.Mesh(new THREE.DodecahedronGeometry(3.8, 0), darkRockMat);
      b1.position.set(-5.5, 2.5, 2.0);
      b1.castShadow = true;
      group.add(b1);

    } else {
      // 🏜️ Natural Eroded Desert Sandstone Canyon Boulders
      const sandMat = new THREE.MeshStandardMaterial({ color: 0xbd8650, roughness: 0.96, metalness: 0.0 });
      const canyonMat = new THREE.MeshStandardMaterial({ color: 0x8c5b33, roughness: 0.94, metalness: 0.0 });

      // Eroded Sandstone Spire Monolith
      const spire = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 4.5, 12.0, 7), sandMat);
      spire.position.y = 6.0;
      spire.castShadow = true;
      spire.receiveShadow = true;
      group.add(spire);

      // Desert Canyon Rock Ridge for ducking
      const ridge = new THREE.Mesh(new THREE.DodecahedronGeometry(4.8, 0), canyonMat);
      ridge.scale.set(1.6, 0.8, 1.0);
      ridge.position.set(4.2, 2.2, 2.8);
      ridge.castShadow = true;
      group.add(ridge);
    }

    return group;
  }

  createGroundItems() {
    const size = this.size;
    const startX = this.cx * size - size / 2;
    const startZ = this.cz * size - size / 2;

    const types = ['ammo_pistol', 'ammo_smg', 'ammo_railgun', 'ammo_ak47', 'ammo_rocket', 'nanokit', 'powercell'];
    this.chunkItems = [];

    // Ensure 2-3 pick-ups per chunk, uniformly picking from all 7 types across the map
    const count = 2;
    for (let i = 0; i < count; i++) {
      const seed = Math.abs(Math.floor((startX * 37 + startZ * 17 + i * 91.3) % types.length));
      const itemType = types[seed];

      const offsetNoiseX = Math.abs(this.game.wasm.noise(startX + i * 41.5, startZ + i * 23.7));
      const offsetNoiseZ = Math.abs(this.game.wasm.noise(startZ + i * 83.1, startX + i * 19.4));

      const rx = startX + 30 + (offsetNoiseX * (size - 60));
      const rz = startZ + 30 + (offsetNoiseZ * (size - 60));

      const item = new GroundItem(this.scene, itemType, rx, rz, this.game);
      this.game.groundItems.push(item);
      this.chunkItems.push(item);
    }
  }

  destroy() {
    if (this.treeMeshes) {
      this.treeMeshes.forEach(im => {
        im.geometry.dispose();
        im.material.dispose();
      });
      this.treeMeshes = null;
    }
    if (this.coverGroup) {
      this.scene.remove(this.coverGroup);
      this.coverGroup = null;
    }
    if (this.chunkItems) {
      this.chunkItems.forEach(item => {
        if (!item.isDead) item.destroy();
      });
      this.chunkItems = null;
    }
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      if (Array.isArray(this.mesh.material)) {
        this.mesh.material.forEach(m => m.dispose());
      } else {
        this.mesh.material.dispose();
      }
      this.mesh = null;
    }
  }
}

// ================================================================
//  PLAYER ROCKET CLASS (Secondary Weapon Explosion)
// ================================================================
class PlayerRocket {
  [key: string]: any;
  static sharedBodyGeo: THREE.CylinderGeometry | null = null;
  static sharedTipGeo: THREE.ConeGeometry | null = null;
  static sharedGlowRingGeo: THREE.TorusGeometry | null = null;
  static sharedBodyMat: THREE.MeshStandardMaterial | null = null;
  static sharedTipMat: THREE.MeshBasicMaterial | null = null;
  static sharedGlowRingMat: THREE.MeshBasicMaterial | null = null;

  constructor(scene: any, startPos: THREE.Vector3, targetPos: THREE.Vector3, game: any) {
    this.scene = scene;
    this.game = game;
    this.mesh = new THREE.Group();

    if (!PlayerRocket.sharedBodyGeo) {
      PlayerRocket.sharedBodyGeo = new THREE.CylinderGeometry(0.32, 0.32, 2.2, 8);
      PlayerRocket.sharedTipGeo = new THREE.ConeGeometry(0.34, 0.8, 8);
      PlayerRocket.sharedGlowRingGeo = new THREE.TorusGeometry(0.35, 0.05, 8, 14);
      PlayerRocket.sharedBodyMat = new THREE.MeshStandardMaterial({ color: 0x1f2421, metalness: 0.9, roughness: 0.2 });
      PlayerRocket.sharedTipMat = new THREE.MeshBasicMaterial({ color: 0xff4400 });
      PlayerRocket.sharedGlowRingMat = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
    }

    // High tech rocket body mesh using pooled assets
    const body = new THREE.Mesh(PlayerRocket.sharedBodyGeo, PlayerRocket.sharedBodyMat);
    body.rotation.x = Math.PI / 2;
    this.mesh.add(body);

    const tip = new THREE.Mesh(PlayerRocket.sharedTipGeo, PlayerRocket.sharedTipMat);
    tip.rotation.x = Math.PI / 2;
    tip.position.z = 1.3;
    this.mesh.add(tip);

    const glowRing = new THREE.Mesh(PlayerRocket.sharedGlowRingGeo, PlayerRocket.sharedGlowRingMat);
    glowRing.position.z = 0.5;
    this.mesh.add(glowRing);

    this.mesh.position.copy(startPos);

    // Vector pointing directly from gun muzzle towards exact crosshair target point
    this.dir = new THREE.Vector3().subVectors(targetPos, startPos).normalize();
    this.speed = 210; // High speed rocket velocity
    this.velocity = this.dir.clone().multiplyScalar(this.speed);

    this.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this.dir);

    this.isDead = false;
    this.age = 0;
    this.targetPos = targetPos.clone();

    this.scene.add(this.mesh);
  }

  update(dt: number) {
    if (this.isDead) return;

    this.age += dt;

    // Slight realistic ballistic gravity drop over distance (after initial rocket thrust burst)
    if (this.age > 0.15) {
      this.velocity.y -= 12.0 * dt;
    }

    // Move rocket along velocity vector
    this.mesh.position.addScaledVector(this.velocity, dt);

    // Update rocket rotation so nose points in direction of flight
    if (this.velocity.lengthSq() > 0.001) {
      const flyDir = this.velocity.clone().normalize();
      this.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), flyDir);
    }

    // Trail particle sparks
    this.game.createSparks(this.mesh.position);

    // ---- CONTINUOUS COLLISION DETECTION (CCD) & ENEMY DIRECT HITS ----
    const currPos = this.mesh.position;
    const terrainHeight = this.game.getHeightAt(currPos.x, currPos.z);

    // Direct hit check on all active demons (radius 4.2m)
    let directHitDemon: any = null;
    this.game.demons.forEach((d: any) => {
      if (!d.isDead && d.mesh && !directHitDemon) {
        const dist = d.mesh.position.distanceTo(currPos);
        if (dist < 4.2) {
          directHitDemon = d;
        }
      }
    });

    const distToTarget = currPos.distanceTo(this.targetPos);

    if (directHitDemon || currPos.y <= terrainHeight + 0.3 || distToTarget < 3.5 || this.age > 3.0) {
      this.explode(directHitDemon);
    }
  }

  explode(directDemon?: any) {
    if (this.isDead) return;
    this.isDead = true;
    const pos = this.mesh.position.clone();

    this.game.triggerScreenShake(0.75);
    this.game.playRocketExplosionSound();

    this.game.createExplosionParticles(pos);
    this.game.createTerrainImpactDust(pos);

    // Massive AOE damage boosted by game level - Balanced 5% scaling per level
    const baseAoe = 160 * (1 + (this.game.gameLevel - 1) * 0.05);

    // Extra direct hit damage bonus if hit an enemy directly
    if (directDemon && !directDemon.isDead) {
      directDemon.takeDamage(baseAoe * 1.5, pos);
      this.game.triggerHitMarker(directDemon.isDead || directDemon.health <= 0);
    }

    this.game.demons.forEach((d: any) => {
      if (!d.isDead && d !== directDemon) {
        const dDist = d.mesh.position.distanceTo(pos);
        if (dDist < 52) {
          const aoeDamage = baseAoe * (1 - dDist / 52);
          d.takeDamage(aoeDamage, pos);
          this.game.triggerHitMarker(d.isDead || d.health <= 0);
        }
      }
    });

    this.scene.remove(this.mesh);
    this.mesh.traverse((child: any) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
  }
}

// ================================================================
//  ENEMY PROJECTILE CLASS (Alien plasma energy)
// ================================================================
class EnemyProjectile {
  [key: string]: any;
  static sharedGeo: THREE.SphereGeometry | null = null;
  static sharedMat: THREE.MeshBasicMaterial | null = null;

  constructor(scene, startPos, targetPos, game) {
    this.scene = scene;
    this.game = game;
    this.mesh = new THREE.Group();

    if (!EnemyProjectile.sharedGeo) {
      EnemyProjectile.sharedGeo = new THREE.SphereGeometry(0.85, 8, 8);
      EnemyProjectile.sharedMat = new THREE.MeshBasicMaterial({ color: 0xff2200 });
    }

    const orb = new THREE.Mesh(EnemyProjectile.sharedGeo, EnemyProjectile.sharedMat);
    this.mesh.add(orb);

    this.mesh.position.copy(startPos);
    this.dir = new THREE.Vector3().subVectors(targetPos, startPos).normalize();
    this.speed = Math.min(200.0, 115 + (this.game.gameLevel - 1) * 4.5); // smoothly scales, capped at 200
    this.isDead = false;
    this.age = 0;

    this.scene.add(this.mesh);
  }

  update(dt) {
    if (this.isDead) return;

    this.age += dt;
    this.mesh.position.addScaledVector(this.dir, this.speed * dt);

    const rp = this.game.robotGroup.position;
    const distToPlayer = this.mesh.position.distanceTo(new THREE.Vector3(rp.x, rp.y + 5, rp.z));

    if (distToPlayer < 4.8) {
      const lvl = this.game.gameLevel;
      let demonDamage;
      if (lvl <= 10) {
        demonDamage = Math.round(5.0 + (lvl - 1) * 0.7);
      } else {
        demonDamage = Math.round(12.0 + (lvl - 11) * 0.35);
      }
      this.game.damagePlayer(demonDamage);
      this.game.createSparks(this.mesh.position);
      this.game.triggerScreenShake(0.3);
      this.destroy();
      return;
    }

    if (this.age > 3.5) {
      this.destroy();
    }
  }

  destroy() {
    this.isDead = true;
    this.scene.remove(this.mesh);
    this.mesh.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
  }
}

// ================================================================
//  DEMON CLASS (Glowing enemies with plasma weapons)
// ================================================================
class Demon {
  [key: string]: any;
  constructor(scene, x, z, game) {
    this.scene = scene;
    this.game = game;
    this.mesh = new THREE.Group();

    // Dynamic enemy level assignment with tactical variety: scouts, standard, and elites
    const baseLvl = this.game.gameLevel;
    const rand = Math.random();
    let lvl = baseLvl;
    if (rand < 0.20 && baseLvl > 1) {
      lvl = baseLvl - 1; // Scout Minion
    } else if (rand > 0.85) {
      lvl = baseLvl + 1; // Elite Brute
    }
    this.level = lvl;

    // Smooth level-based enemy size progression (grows per level, capped at max titan size)
    const baseEnemyScale = 14.0;
    const scalePerLevel = 0.8;
    const maxEnemyScale = 25.0; // Giant titan cap (approx ~1.8x height of Level 1)
    this.enemyScale = Math.min(maxEnemyScale, baseEnemyScale + (lvl - 1) * scalePerLevel);

    // Uniform 3D scalar scaling preserves exact proportions without stretching
    this.mesh.scale.setScalar(this.enemyScale / 3.0);

    const y = this.game.getHeightAt(x, z);
    this.mesh.position.set(x, y, z); // Stands directly on the ground

    const lvlBonus = (lvl - 1);
    const sizeFactor = this.enemyScale / 3.0;

    // Health scales with mass/volume (size^2) - Balanced so Level 1-10 enemies die in 3-5 pistol shots
    this.maxHealth = Math.round((75 + lvlBonus * 25 + Math.random() * 15) * (this.enemyScale / 16.0));
    this.health = this.maxHealth;

    // Balanced enemy speed: smooth gradual curve capped at max 48.0 (fair tactical pacing below player run speed 60.0)
    const baseSpeed = 18.0;
    const speedPerLevel = 0.85;
    const maxEnemySpeed = 48.0;
    this.speed = Math.min(maxEnemySpeed, baseSpeed + (lvl - 1) * speedPerLevel);

    this.isDead = false;
    this.lastAttackTime = 0;

    this.createModel();
    this.createFloatingHeader();
    this.scene.add(this.mesh);
  }

  createFloatingHeader() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.headerCanvas = canvas;
    this.headerCtx = ctx;

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    this.headerTexture = texture;

    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: true,
      depthWrite: false
    });
    this.headerSpriteMat = spriteMat;

    const sprite = new THREE.Sprite(spriteMat);
    // Position floating directly above enemy head (scales with group size)
    sprite.position.set(0, 7.3, 0);
    sprite.scale.set(3.8, 1.425, 1.0);
    this.mesh.add(sprite);
    this.headerSprite = sprite;

    this.updateFloatingHeader();
  }

  updateFloatingHeader() {
    if (!this.headerCtx || !this.headerCanvas || !this.headerTexture) return;
    const ctx = this.headerCtx;
    const w = this.headerCanvas.width;
    const h = this.headerCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Color theme based on level tier
    let levelColor = '#00f0ff'; // Neon Cyan
    let titleText = `LVL ${this.level} DEMON`;
    if (this.level >= 10) {
      levelColor = '#ff0055'; // Crimson Titan
      titleText = `💀 LVL ${this.level} TITAN`;
    } else if (this.level >= 7) {
      levelColor = '#ff4d00'; // Molten Orange
      titleText = `🔥 LVL ${this.level} BRUTE`;
    } else if (this.level >= 4) {
      levelColor = '#ffb700'; // Gold/Amber
      titleText = `⚔️ LVL ${this.level} STALKER`;
    }

    // 1. Futuristic rounded badge container
    ctx.fillStyle = 'rgba(6, 12, 24, 0.82)';
    ctx.strokeStyle = levelColor;
    ctx.lineWidth = 3;

    // Draw rounded rect
    const r = 12;
    if (typeof (ctx as any).roundRect === 'function') {
      (ctx as any).roundRect(4, 4, w - 8, h - 8, r);
    } else {
      ctx.rect(4, 4, w - 8, h - 8);
    }
    ctx.fill();
    ctx.stroke();

    // 2. Level Badge Tag / Text
    ctx.font = 'bold 22px "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = levelColor;
    ctx.shadowColor = levelColor;
    ctx.shadowBlur = 8;
    ctx.fillText(titleText, w / 2, 28);
    ctx.shadowBlur = 0; // reset shadow

    // 3. Health Bar Background
    const barX = 20;
    const barY = 50;
    const barW = w - 40;
    const barH = 14;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.fillRect(barX, barY, barW, barH);

    // 4. Health Bar Fill
    const healthRatio = Math.max(0, Math.min(1, this.health / (this.maxHealth || 100)));
    const fillW = barW * healthRatio;

    const hpColor = healthRatio > 0.6 ? '#00ff66' : (healthRatio > 0.25 ? '#ffcc00' : '#ff3333');
    ctx.fillStyle = hpColor;
    ctx.shadowColor = hpColor;
    ctx.shadowBlur = 4;
    ctx.fillRect(barX, barY, fillW, barH);
    ctx.shadowBlur = 0;

    // 5. Border around Health Bar
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(barX, barY, barW, barH);

    // 6. Numeric HP text
    ctx.font = 'bold 12px "Segoe UI", Roboto, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`${Math.ceil(Math.max(0, this.health))} / ${Math.ceil(this.maxHealth || 100)} HP`, w / 2, 78);

    this.headerTexture.needsUpdate = true;
  }

  createModel() {
    const gltfSource = this.game.demonGltf || this.game.robotGltf;
    if (gltfSource) {
      const model = SkeletonUtils.clone(gltfSource.scene);
      this.mesh.add(model);
      this.model = model;

      // Adjust model base orientation/scale offset to fit the same physical demon collision capsule size perfectly
      // Scale based on which model is loaded: Xbot is scaled by 3.5, RobotExpressive is scaled by 1.0
      let baseScale = 1.0;
      if (gltfSource && (gltfSource as any).isXbot) {
        baseScale = 3.5;
      }
      model.scale.setScalar(baseScale);

      // Keep original GLB colors and materials but disable shadows and shininess for max FPS
      model.traverse(child => {
        if ((child as any).isMesh) {
          const mesh = child as any;
          mesh.castShadow = false;
          mesh.receiveShadow = false;
          if (mesh.material) {
            const disableShiny = (mat: any) => {
              if (mat.roughness !== undefined) mat.roughness = 1.0;
              if (mat.metalness !== undefined) mat.metalness = 0.0;
            };
            if (Array.isArray(mesh.material)) {
              mesh.material.forEach(m => disableShiny(m));
            } else {
              disableShiny(mesh.material);
            }
          }
        }
      });

      // Find bones and attach invisible colliders to them (head, spine, legs)
      this.boneHitBoxes = [];
      model.traverse(child => {
        if ((child as any).isBone) {
          const bone = child as THREE.Bone;
          const boneName = bone.name.toLowerCase();

          let geo = null;
          let offset = new THREE.Vector3();
          let boneType = '';

          if (boneName.includes('head') && !boneName.includes('end')) {
            // Calibrated Head hitbox: tight sphere aligned with head bone
            geo = new THREE.SphereGeometry(0.35, 8, 8);
            offset.set(0, 0.2, 0);
            boneType = 'head';
          } else if (boneName.includes('spine') || boneName.includes('chest')) {
            // Calibrated Torso/Spine hitbox: cylinder
            geo = new THREE.CylinderGeometry(0.35, 0.35, 1.2, 6);
            offset.set(0, 0.5, 0);
            boneType = 'torso';
          } else if (boneName.includes('leg') || boneName.includes('thigh') || boneName.includes('upleg')) {
            // Calibrated Leg hitbox: cylinder
            geo = new THREE.CylinderGeometry(0.2, 0.2, 1.0, 5);
            offset.set(0, -0.4, 0);
            boneType = 'leg';
          }

          if (geo) {
            const hitMat = new THREE.MeshBasicMaterial({ visible: false });
            const hitMesh = new THREE.Mesh(geo, hitMat);
            hitMesh.position.copy(offset);

            // Tag it so raycaster results know what bone was hit
            hitMesh.userData = {
              isDemonHitBox: true,
              parentDemon: this,
              boneType: boneType
            };

            bone.add(hitMesh);
            this.boneHitBoxes.push(hitMesh);
          }
        }
      });

      // Calibrated main body catch-all cylinder hitbox matching character mesh height and slender width
      const hitBoxGeo = new THREE.CylinderGeometry(0.35, 0.35, 4.5, 8);
      const hitBoxMat = new THREE.MeshBasicMaterial({ visible: false });
      this.hitBox = new THREE.Mesh(hitBoxGeo, hitBoxMat);
      this.hitBox.position.y = 2.25;
      this.hitBox.userData = { isDemonHitBox: true, parentDemon: this, boneType: 'torso' };
      this.mesh.add(this.hitBox);

      // Setup animation mixer for the enemy
      this.mixer = new THREE.AnimationMixer(model);
      this.animations = {};
      gltfSource.animations.forEach(clip => {
        this.animations[clip.name] = this.mixer.clipAction(clip);
      });

      // Dynamically detect walking/running animations case-insensitively
      let walkAction = null;
      Object.keys(this.animations).forEach(name => {
        const lower = name.toLowerCase();
        if (lower.includes('walk') || lower.includes('run')) {
          walkAction = this.animations[name];
        }
      });

      // Play Idle by default
      if (this.animations['Idle']) {
        this.currentAction = this.animations['Idle'];
        this.currentAction.play();
      } else if (walkAction) {
        this.currentAction = walkAction;
        this.currentAction.play();
      }

      // Match step rate to physical scale speed
      this.mixer.timeScale = 1.0;
    } else {
      // Fallback: procedural giant scrap-metal layout if GLTF hasn't loaded yet
      const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xa84c28,
        roughness: 0.75,
        metalness: 0.55,
        emissive: 0x3d170a,
        emissiveIntensity: 0.4
      });
      const glowMat = new THREE.MeshStandardMaterial({
        color: 0xff4400,
        emissive: 0xff3300,
        emissiveIntensity: 7.5
      });

      const torso = new THREE.Mesh(new THREE.BoxGeometry(4.0, 5.0, 2.8), bodyMat);
      torso.position.y = 2.5;
      torso.castShadow = true;
      this.mesh.add(torso);

      const head = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2.5, 2.5), bodyMat);
      head.position.y = 6.25;
      head.castShadow = true;
      this.mesh.add(head);

      const eye = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.5, 0.3), glowMat);
      eye.position.set(0, 6.3, 1.25);
      this.mesh.add(eye);
      this._glowMatInstance = glowMat;
    }
  }

  fadeToAction(name, duration = 0.25) {
    if (!this.animations || !this.animations[name]) return;
    const nextAction = this.animations[name];
    const prevAction = this.currentAction;

    if (prevAction === nextAction) return;

    this.currentAction = nextAction;

    if (prevAction) {
      prevAction.fadeOut(duration);
    }

    nextAction
      .reset()
      .setEffectiveTimeScale(1.0)
      .setEffectiveWeight(1.0)
      .fadeIn(duration)
      .play();
  }

  takeDamage(amount, hitPoint) {
    if (this.isDead) return;
    this.health -= amount;

    // Flash the cloned glow material instance
    if (this._glowMatInstance) {
      this._glowMatInstance.emissiveIntensity = 12.0;
      setTimeout(() => {
        if (this._glowMatInstance && !this.isDead) this._glowMatInstance.emissiveIntensity = 6.0;
      }, 100);
    }

    if (this.health <= 0) {
      this.die(hitPoint);
    } else {
      this.updateFloatingHeader();
    }
  }

  die(hitPoint) {
    this.isDead = true;
    this.deathTimer = 0;
    this.deathComplete = false;

    // Hide floating level badge on death
    if (this.headerSprite) {
      this.headerSprite.visible = false;
    }

    this.game.createExplosionParticles(this.mesh.position);
    this.game.playEnemyDeathSound();
    this.game.playExplosionSound();

    // Play Sitting/Death animation clip if available
    const deathAction = this.animations ? (this.animations['Death'] || this.animations['Sitting']) : null;
    if (deathAction) {
      deathAction.reset().setEffectiveWeight(1.0).setLoop(THREE.LoopOnce, 1);
      deathAction.clampWhenFinished = true;
      this.fadeToAction(deathAction.getClip().name, 0.15);
    }

    // Clone all mesh materials to isolate the opacity fade-out
    this.mesh.traverse((child: any) => {
      if (child.isMesh && child.material) {
        if (Array.isArray(child.material)) {
          child.material = child.material.map((m: any) => m.clone());
        } else {
          child.material = child.material.clone();
        }
      }
    });

    this.game.points += Math.round(100 * this.game.gameLevel * (this.enemyScale / 3.0));
    this.game.kills += 1;
    this.game.updateWeaponSystem();

    // Spawn random ammo/health loot drop on demon death
    const lootTypes = ['ammo_pistol', 'ammo_smg', 'ammo_railgun', 'ammo_ak47', 'ammo_rocket', 'nanokit'];
    const randomLoot = lootTypes[Math.floor(Math.random() * lootTypes.length)];
    const dropItem = new GroundItem(this.scene, randomLoot, this.mesh.position.x, this.mesh.position.z, this.game);
    this.game.groundItems.push(dropItem);

    // Check Level Up condition (every 5 kills)
    if (this.game.kills >= this.game.gameLevel * 5) {
      this.game.levelUp();
    }
  }

  destroy() {
    this.scene.remove(this.mesh);
    if (this._glowMatInstance) {
      this._glowMatInstance.dispose();
      this._glowMatInstance = null;
    }
    if (this.headerTexture) {
      this.headerTexture.dispose();
      this.headerTexture = null;
    }
    if (this.headerSpriteMat) {
      this.headerSpriteMat.dispose();
      this.headerSpriteMat = null;
    }
  }

  update(dt, playerPos) {
    if (this.isDead) {
      if (this.deathComplete) return;

      // Update skeleton animations
      if (this.mixer) this.mixer.update(dt);

      this.deathTimer += dt;

      // Smoothly tilt 90 degrees forward to face-plant on the ground
      this.mesh.rotation.x = THREE.MathUtils.lerp(0, -Math.PI / 2, Math.min(1.0, this.deathTimer / 0.8));

      // Sink slightly so it lies flat on the terrain surface
      const targetY = this.game.getHeightAt(this.mesh.position.x, this.mesh.position.z) - (this.enemyScale * 0.18);
      this.mesh.position.y = THREE.MathUtils.lerp(this.mesh.position.y, targetY, Math.min(1.0, this.deathTimer / 0.8));

      // Fade out materials during the final phase (from 0.8s to 1.8s)
      if (this.deathTimer > 0.8) {
        const fadeFactor = 1.0 - Math.min(1.0, (this.deathTimer - 0.8) / 1.0);
        this.mesh.traverse((child: any) => {
          if (child.isMesh && child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach((m: any) => {
                m.transparent = true;
                m.opacity = fadeFactor;
              });
            } else {
              child.material.transparent = true;
              child.material.opacity = fadeFactor;
            }
          }
        });
      }

      // Cleanup and remove from scene after death animation finishes
      if (this.deathTimer >= 1.8) {
        this.deathComplete = true;
        this.destroy();
      }
      return;
    }

    // Update animated skeleton
    if (this.mixer) this.mixer.update(dt);

    // Zero-GC pure scalar distance & direction calculations (eliminates frame stutter)
    const toPlayerX = playerPos.x - this.mesh.position.x;
    const toPlayerZ = playerPos.z - this.mesh.position.z;
    const distSq = toPlayerX * toPlayerX + toPlayerZ * toPlayerZ;

    // Check if punch action is playing to avoid interrupting it
    const punchAction = this.animations ? this.animations['Punch'] : null;
    const isPunching = punchAction && punchAction.isRunning();

    // Skip if outside 280 units (280^2 = 78400)
    if (distSq < 78400) {
      const dist = Math.sqrt(distSq);
      const targetAngle = Math.atan2(toPlayerX, toPlayerZ);
      this.mesh.rotation.y = targetAngle;

      const touchDist = 4.0 + (this.enemyScale * 0.45); // touch contact boundary based on size

      if (dist > touchDist && dist > 0.001) {
        // Normalize direction towards player
        let moveDirX = toPlayerX / dist;
        let moveDirZ = toPlayerZ / dist;

        // Check nearby alive demon allies to spread out & prevent gang clustering (Zero-GC)
        let closeAlliesCount = 0;
        let repX = 0;
        let repZ = 0;

        const allDemons = this.game.demons;
        if (allDemons) {
          const myX = this.mesh.position.x;
          const myZ = this.mesh.position.z;
          for (let i = 0; i < allDemons.length; i++) {
            const ally = allDemons[i];
            if (ally === this || ally.isDead || !ally.mesh) continue;

            const dx = myX - ally.mesh.position.x;
            const dz = myZ - ally.mesh.position.z;
            const dSq = dx * dx + dz * dz;

            if (dSq < 676 && dSq > 0.01) { // 26^2 = 676
              closeAlliesCount++;
              const dAlly = Math.sqrt(dSq);
              const pushStrength = (1.0 - dAlly / 26.0);
              repX += (dx / dAlly) * pushStrength;
              repZ += (dz / dAlly) * pushStrength;
            }
          }
        }

        // If >= 2 allies are already clustered nearby (forming a 3-enemy group), fan out / flank laterally
        if (closeAlliesCount >= 2) {
          moveDirX = moveDirX * 0.55 + repX * 0.85;
          moveDirZ = moveDirZ * 0.55 + repZ * 0.85;
          const len = Math.sqrt(moveDirX * moveDirX + moveDirZ * moveDirZ);
          if (len > 0.001) {
            moveDirX /= len;
            moveDirZ /= len;
          }
        }

        this.mesh.position.x += moveDirX * this.speed * dt;
        this.mesh.position.z += moveDirZ * this.speed * dt;
        this.mesh.position.y = this.game.getHeightAt(this.mesh.position.x, this.mesh.position.z);

        if (!isPunching) {
          const lvl = this.game.gameLevel;
          if (lvl <= 10) {
            this.fadeToAction('Walking', 0.2);
            if (this.mixer) this.mixer.timeScale = 1.0;
          } else {
            this.fadeToAction('Running', 0.2);
            if (this.mixer) this.mixer.timeScale = lvl > 20 ? 1.1 : 1.0;
          }
        }
      } else {
        // Hand-to-hand combat range: attack on touch cooldown (1.2 seconds)
        const now = Date.now();
        if (now - this.lastAttackTime > 1200 && !this.game.playerDead) {
          this.lastAttackTime = now;

          // Play punch/attack action if animation is available
          if (punchAction) {
            punchAction.reset().setEffectiveWeight(1.0).setLoop(THREE.LoopOnce, 1).play();
          }

          // Damage scaled by scale factor and gameLevel brackets (lower on 1-10, slightly increase on upper levels)
          const lvl = this.game.gameLevel;
          let damage;
          if (lvl <= 10) {
            damage = Math.round((5.0 + (lvl - 1) * 1.0) * (this.enemyScale / 8.0));
          } else {
            damage = Math.round((15.0 + (lvl - 11) * 0.45) * (this.enemyScale / 8.0));
          }
          this.game.damagePlayer(damage);

          // Physical pushback: push player away based on giant's impact direction
          const pushDir = new THREE.Vector3().subVectors(playerPos, this.mesh.position);
          pushDir.y = 0;
          pushDir.normalize();
          const pushForce = 45 * (this.enemyScale / 3.0);
          this.game.velocity.x += pushDir.x * pushForce;
          this.game.velocity.z += pushDir.z * pushForce;
          this.game.velocity.y += 18 * (this.enemyScale / 3.0); // lift up
          this.game.onGround = false;

          this.game.triggerScreenShake(0.4 * (this.enemyScale / 3.0));
        } else {
          if (!isPunching) {
            this.fadeToAction('Idle', 0.25);
          }
        }
      }
    } else {
      this.mesh.rotation.y += dt * 0.45;
      this.mesh.position.y = this.game.getHeightAt(this.mesh.position.x, this.mesh.position.z);
      if (!isPunching) {
        this.fadeToAction('Idle', 0.25);
      }
    }
  }
}

// ================================================================
//  GROUND ITEM CLASS
// ================================================================
class GroundItem {
  [key: string]: any;
  constructor(scene, type, x, z, game) {
    this.scene = scene;
    this.type = type; // 'crystal', 'nanokit', 'powercell'
    this.game = game;
    this.isDead = false;

    this.group = new THREE.Group();
    this.ring1 = null;
    this.ring2 = null;

    const metalMat = new THREE.MeshStandardMaterial({
      color: 0x222b3a,
      metalness: 0.9,
      roughness: 0.2
    });

    if (type === 'crystal') {
      // 💎 Plasma Crystal Artifact: Multi-faceted glowing core + orbiting energy halo
      const coreGeo = new THREE.IcosahedronGeometry(1.6, 0);
      const coreMat = new THREE.MeshStandardMaterial({
        color: 0x00e5ff,
        emissive: 0x0099cc,
        emissiveIntensity: 0.7,
        roughness: 0.1,
        metalness: 0.95
      });
      this.mesh = new THREE.Mesh(coreGeo, coreMat);
      this.group.add(this.mesh);

      // Orbiting Hologram Energy Ring
      const ringGeo = new THREE.TorusGeometry(2.4, 0.12, 8, 32);
      const ringMat = new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x00e5ff,
        emissiveIntensity: 0.9,
        metalness: 0.8
      });
      this.ring1 = new THREE.Mesh(ringGeo, ringMat);
      this.ring1.rotation.x = Math.PI / 3;
      this.group.add(this.ring1);

      this.name = "Plasma Crystal";
      this.icon = "💎";
    } else if (type === 'nanokit') {
      // 🧪 Sci-Fi Nano Medical Pod: Metallic capsule body + 3D glowing red cross
      const bodyGeo = new THREE.CylinderGeometry(1.2, 1.2, 2.4, 16);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xeef4fc,
        metalness: 0.6,
        roughness: 0.3
      });
      this.mesh = new THREE.Mesh(bodyGeo, bodyMat);

      // Top and bottom metallic caps
      const capGeo = new THREE.CylinderGeometry(1.3, 1.3, 0.4, 16);
      const topCap = new THREE.Mesh(capGeo, metalMat);
      topCap.position.y = 1.2;
      const botCap = new THREE.Mesh(capGeo, metalMat);
      botCap.position.y = -1.2;
      this.mesh.add(topCap);
      this.mesh.add(botCap);

      // Glowing 3D Health Cross
      const crossMat = new THREE.MeshStandardMaterial({
        color: 0xff1744,
        emissive: 0xff0044,
        emissiveIntensity: 0.9,
        metalness: 0.3
      });
      const cVert = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.5, 0.5), crossMat);
      const cHoriz = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.5, 0.5), crossMat);
      cVert.position.z = 1.0;
      cHoriz.position.z = 1.0;
      this.mesh.add(cVert);
      this.mesh.add(cHoriz);

      // Orbiting Pink Energy Ring
      const ringGeo = new THREE.TorusGeometry(2.0, 0.1, 8, 24);
      const ringMat = new THREE.MeshStandardMaterial({
        color: 0xff2a6d,
        emissive: 0xff1744,
        emissiveIntensity: 0.8
      });
      this.ring1 = new THREE.Mesh(ringGeo, ringMat);
      this.ring1.rotation.x = Math.PI / 4;
      this.group.add(this.ring1);

      this.group.add(this.mesh);
      this.name = "Nano-Kit Pod";
      this.icon = "🧪";
    } else { // powercell
      // ⚡ High-Tech Power Cell: Heavy-duty canister + dual counter-rotating emerald rings
      const coreGeo = new THREE.CylinderGeometry(1.0, 1.0, 2.6, 12);
      const coreMat = new THREE.MeshStandardMaterial({
        color: 0x00ff66,
        emissive: 0x00cc44,
        emissiveIntensity: 0.85,
        metalness: 0.8,
        roughness: 0.2
      });
      this.mesh = new THREE.Mesh(coreGeo, coreMat);

      // Gold armored end-caps
      const goldMat = new THREE.MeshStandardMaterial({ color: 0xffaa00, metalness: 0.9, roughness: 0.2 });
      const topCap = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 0.5, 12), goldMat);
      topCap.position.y = 1.3;
      const botCap = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 0.5, 12), goldMat);
      botCap.position.y = -1.3;
      this.mesh.add(topCap);
      this.mesh.add(botCap);

      // Dual Counter-Rotating Emerald Rings
      const ringGeo = new THREE.TorusGeometry(2.2, 0.1, 8, 24);
      const ringMat = new THREE.MeshStandardMaterial({
        color: 0x00ff88,
        emissive: 0x00ff66,
        emissiveIntensity: 0.9
      });
      this.ring1 = new THREE.Mesh(ringGeo, ringMat);
      this.ring1.rotation.x = Math.PI / 3;

      this.ring2 = new THREE.Mesh(ringGeo, ringMat);
      this.ring2.rotation.x = -Math.PI / 3;

      this.group.add(this.mesh);
      this.group.add(this.ring1);
      this.group.add(this.ring2);

      this.name = "Overcharge Cell";
      this.icon = "⚡";
    }

    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;

    // Find height
    const y = this.game.getHeightAt(x, z) + 1.6;
    this.group.position.set(x, y, z);

    // Tag group & meshes so Raycaster knows it's interactive
    this.mesh.userData = { isGroundItem: true, parentItem: this };
    this.group.traverse(child => {
      if (child.isMesh) child.userData = { isGroundItem: true, parentItem: this };
    });

    this.scene.add(this.group);
  }

  update(dt) {
    if (!this.mesh) return;
    this.mesh.rotation.y += dt * 1.8;
    this.mesh.position.y = Math.sin(Date.now() * 0.0035) * 0.3;

    if (this.ring1) {
      this.ring1.rotation.z += dt * 2.2;
    }
    if (this.ring2) {
      this.ring2.rotation.z -= dt * 2.2;
    }
  }

  destroy() {
    this.scene.remove(this.group);
    this.isDead = true;
  }
}

// ================================================================
//  MAP / BIOME CONFIGURATION
// ================================================================
const MAP_CONFIGS: Record<string, any> = {
  arctic: {
    name: 'Arctic Tundra',
    MAX_HEIGHT: 120,
    terrainColors(hr: number, slope: number, cv: number) {
      let r: number, g: number, b: number;
      if (slope > 0.52) {
        r = 0.33 + cv; g = 0.34 + cv; b = 0.40 + cv;
      } else if (hr > 0.5) {
        r = 0.94 + cv; g = 0.96 + cv; b = 0.99;
      } else if (hr > 0.18) {
        const sn = 0.74 + (cv * 2.0);
        r = 0.83 * sn; g = 0.89 * sn; b = 0.87 * sn;
      } else {
        const br = 0.88 + cv;
        r = br; g = br + 0.02; b = br + 0.05;
      }
      return { r, g, b };
    },
    treeColors: { foliage: 0x1a3a1c, trunk: 0x3a2a1a, snowCap: true, snowColor: 0xe4ecf0 },
    treeRange: [0.1, 0.55],
    skyGradient: ['#102840', '#3a7bb8', '#6aade0', '#9ecce8', '#c8dce8'],
    weather: {
      clear: { skyColor: 0x8cb8d8, fogColor: 0xbdd0e4, fogDensity: 0.0020, sunIntensity: 1.6, ambientIntensity: 0.55 },
      rain: { skyColor: 0x3a4555, fogColor: 0x5a6a7a, fogDensity: 0.0026, sunIntensity: 0.35, ambientIntensity: 0.25 },
      snow: { skyColor: 0xc0c8d4, fogColor: 0xd0d8e4, fogDensity: 0.0024, sunIntensity: 0.7, ambientIntensity: 0.65 }
    },
    weatherOptions: ['clear', 'rain', 'snow'],
    sunColor: 0xfff3e0, sunIntensity: 1.6, ambientColor: 0x8eafc8, hemiSky: 0x87ceeb, hemiGround: 0xe8eef2,
    fogColor: 0xbdd0e4, fogDensity: 0.0022
  },
  forest: {
    name: 'Green Forest',
    MAX_HEIGHT: 200,
    terrainColors(hr: number, slope: number, cv: number) {
      let r: number, g: number, b: number;
      if (slope > 0.55) {
        r = 0.38 + cv; g = 0.35 + cv; b = 0.30 + cv;
      } else if (hr > 0.65) {
        r = 0.52 + cv; g = 0.55 + cv; b = 0.50 + cv;
      } else if (hr > 0.15) {
        r = 0.18 + cv * 0.5; g = 0.42 + cv; b = 0.12 + cv * 0.3;
      } else {
        r = 0.22 + cv; g = 0.50 + cv; b = 0.18 + cv * 0.4;
      }
      return { r, g, b };
    },
    treeColors: { foliage: 0x228b22, trunk: 0x5c3a21, snowCap: false, snowColor: 0x228b22 },
    treeRange: [0.08, 0.62],
    skyGradient: ['#0a1628', '#2a6090', '#4a90d0', '#80bfe8', '#c0e8f0'],
    weather: {
      clear: { skyColor: 0x6ab4e8, fogColor: 0x90c8e0, fogDensity: 0.0018, sunIntensity: 1.8, ambientIntensity: 0.6 },
      rain: { skyColor: 0x384850, fogColor: 0x506060, fogDensity: 0.0025, sunIntensity: 0.3, ambientIntensity: 0.2 },
      snow: { skyColor: 0xa0b0c0, fogColor: 0xb8c8d0, fogDensity: 0.0022, sunIntensity: 0.6, ambientIntensity: 0.55 }
    },
    weatherOptions: ['clear', 'rain', 'snow'],
    sunColor: 0xfff8e0, sunIntensity: 1.8, ambientColor: 0x80b880, hemiSky: 0x88ccaa, hemiGround: 0xd0e8d0,
    fogColor: 0x90c8e0, fogDensity: 0.0020
  },
  autumn: {
    name: 'Autumn Park',
    MAX_HEIGHT: 120,
    terrainColors(hr: number, slope: number, cv: number) {
      let r: number, g: number, b: number;
      if (slope > 0.5) {
        r = 0.45 + cv; g = 0.35 + cv; b = 0.22 + cv;
      } else if (hr > 0.6) {
        r = 0.62 + cv; g = 0.48 + cv; b = 0.28 + cv;
      } else if (hr > 0.12) {
        r = 0.68 + cv; g = 0.52 + cv; b = 0.25 + cv * 0.5;
      } else {
        r = 0.72 + cv; g = 0.58 + cv; b = 0.30 + cv * 0.4;
      }
      return { r, g, b };
    },
    treeColors: { foliage: 0xd46a15, trunk: 0x4a2a0a, snowCap: false, snowColor: 0xd46a15 },
    treeRange: [0.08, 0.58],
    skyGradient: ['#0e1c2e', '#285885', '#528ebb', '#94c0dd', '#d6e6f2'],
    weather: {
      clear: { skyColor: 0x4e8cb8, fogColor: 0xa8b8c8, fogDensity: 0.0016, sunIntensity: 1.4, ambientIntensity: 0.6 },
      rain: { skyColor: 0x42505e, fogColor: 0x687888, fogDensity: 0.0024, sunIntensity: 0.5, ambientIntensity: 0.35 },
      snow: { skyColor: 0x8898a8, fogColor: 0x9cb0c4, fogDensity: 0.0020, sunIntensity: 0.7, ambientIntensity: 0.5 }
    },
    weatherOptions: ['clear', 'rain'],
    sunColor: 0xffe0a0, sunIntensity: 1.4, ambientColor: 0xb09878, hemiSky: 0x78a8d0, hemiGround: 0xd8b088,
    fogColor: 0xa8b8c8, fogDensity: 0.0018,
    sunPosition: { x: 400, y: 480, z: 250 }
  },
  desert: {
    name: 'Desert Wasteland',
    MAX_HEIGHT: 100,
    terrainColors(hr: number, slope: number, cv: number) {
      let r: number, g: number, b: number;
      if (slope > 0.5) {
        r = 0.60 + cv; g = 0.45 + cv; b = 0.30 + cv;
      } else if (hr > 0.55) {
        r = 0.85 + cv; g = 0.75 + cv; b = 0.58 + cv;
      } else if (hr > 0.12) {
        r = 0.88 + cv; g = 0.78 + cv; b = 0.54 + cv;
      } else {
        r = 0.92 + cv; g = 0.82 + cv; b = 0.62 + cv;
      }
      return { r, g, b };
    },
    treeColors: { foliage: 0x3a6830, trunk: 0x2a5020, snowCap: false, snowColor: 0x3a6830 },
    treeRange: [0.08, 0.35],
    skyGradient: ['#0d2238', '#1a4a75', '#2c70a8', '#559acc', '#92c2e0'],
    weather: {
      clear: { skyColor: 0x488ec4, fogColor: 0xa8c4d8, fogDensity: 0.0015, sunIntensity: 1.3, ambientIntensity: 0.6 },
      rain: { skyColor: 0x405262, fogColor: 0x687888, fogDensity: 0.0022, sunIntensity: 0.5, ambientIntensity: 0.35 },
      snow: { skyColor: 0x8094a8, fogColor: 0x9cb2c6, fogDensity: 0.0018, sunIntensity: 0.7, ambientIntensity: 0.5 }
    },
    weatherOptions: ['clear', 'rain'],
    sunColor: 0xfff0d8, sunIntensity: 1.3, ambientColor: 0xc8b088, hemiSky: 0x75a8d4, hemiGround: 0xc0a888,
    fogColor: 0xa8c4d8, fogDensity: 0.0016,
    sunPosition: { x: 400, y: 480, z: 250 }
  }
};

// ================================================================
//  GAME CLASS
// ================================================================
class Game {
  [key: string]: any;
  constructor() {
    // Determine selected map from DOM selection (default: arctic)
    const selectedCard = document.querySelector('.map-card.selected');
    this.mapId = selectedCard ? (selectedCard as HTMLElement).dataset.map || 'arctic' : 'arctic';
    this.mapConfig = MAP_CONFIGS[this.mapId] || MAP_CONFIGS.arctic;

    this.noise = new PerlinNoise(42);
    this.wasm = this.noise; // Fallback default

    // ---- Config ----
    this.chunkSize = 250;
    this.viewRadius = 3;   // Reduced 4→3: 81→49 max active chunks
    this.MAX_HEIGHT = this.mapConfig.MAX_HEIGHT;
    this.RAIN_COUNT = 4000; // Rain particle count
    this.MOVE_SPEED = 55;
    this.SPRINT_SPEED = 110;
    this.JUMP_FORCE = 75;
    this.GRAVITY = -190;
    // Perf: Track last minimap update time for throttling
    this._lastMinimapTime = 0;
    // Perf: Frame timing for 60fps cap
    this._lastFrameTime = 0;
    // Perf: Chunk generation queue to spread load over frames
    this._chunkQueue = [];
    // Perf: Cached Raycaster — don't create a new one every frame
    this._raycaster = new THREE.Raycaster();

    // ---- AAA Spring-Arm Camera ----
    this.camYaw = 0;
    this.camPitch = 0.28;
    this.camDist = 22;
    this.camDistTarget = 40;           // target distance (lerps to this)
    this.camLookTarget = new THREE.Vector3();
    this.screenShake = 0;
    this._camRayDir = new THREE.Vector3(); // reusable for collision ray
    this._camRay = new THREE.Raycaster();
    this._camSpringVel = 0;             // spring velocity for camera distance

    // ---- ADS (Aim Down Sights) System ----
    this.isADS = false;         // true when right-click held
    this.adsBlend = 0;             // 0=hip, 1=full ADS (lerped)
    this.adsFOV = 42;            // FOV when fully aimed
    this.hipFOV = 65;            // normal FOV
    this.currentFOV = 65;            // current animated FOV
    this.adsCamDist = 12;            // camera distance in ADS
    this.adsCamOffset = new THREE.Vector3(3.5, 1.5, 0); // over-the-shoulder offset

    // ---- Procedural Recoil ----
    this.recoilPitch = 0;             // current recoil kick (radians)
    this.recoilYaw = 0;
    this.recoilVelPitch = 0;             // spring velocity
    this.recoilVelYaw = 0;
    this.recoilRecoverySpeed = 12;       // spring stiffness
    this.recoilDamping = 0.82;          // spring damping

    // ---- Landing Impact & Sprint FX ----
    this.prevOnGround = true;
    this.prevVelocityY = 0;
    this.landingDip = 0;             // downward camera offset on landing
    this.landingDipVel = 0;
    this.sprintFOVBoost = 0;             // extra FOV from sprinting (lerped)
    this.footstepPhase = 0;             // phase for footstep camera bob

    // ---- State ----
    const savedQuality = localStorage.getItem('tactical_quality');
    this.graphicsQuality = savedQuality || (this.isMobile ? 'low' : 'medium');
    this.chunks = new Map();
    this.demons = [];
    this.enemyProjectiles = [];
    this.playerRockets = [];
    this.keys = {};
    this.isLocked = false;
    this.isMobile = (('ontouchstart' in window) || (navigator.maxTouchPoints > 0)) && (/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || window.innerWidth <= 1024);
    if (this.isMobile) document.body.classList.add('is-mobile');
    this.velocity = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.onGround = true;
    this.showControls = true;
    this.fpsSamples = [];
    this.fogLayers = [];
    this.landscapeTexture = null;
    this.isContextLost = false;

    // ---- Dynamic Weather System ----
    this.weatherState = 'clear';   // 'clear' | 'rain' | 'snow'
    this.weatherTimer = 0;
    this.weatherDuration = 50;        // seconds before next transition
    this.weatherTransition = 0;         // 0-1 blend factor during transitions
    this.isTransitioning = false;
    this.nextWeather = null;
    this.transitionSpeed = 0.4;       // how fast transitions happen
    this.rainParticles = null;
    this.rainVelocities = null;
    // Weather visual targets (lerped to during transitions)
    const clearWeather = this.mapConfig.weather.clear;
    this._weatherSkyColor = new THREE.Color(clearWeather.skyColor);
    this._weatherFogColor = new THREE.Color(clearWeather.fogColor);
    this._weatherFogDensity = clearWeather.fogDensity;
    this._weatherSunIntensity = clearWeather.sunIntensity;
    this._weatherAmbientIntensity = clearWeather.ambientIntensity;

    // ---- Enemy Spawn Timer ----
    this.enemySpawnTimer = 0;
    this.enemySpawnInterval = 3.5; // seconds between spawn attempts

    // ---- Demon Geometry Pool (prevents stutter on spawn) ----
    this._demonPool = null; // initialized in init()

    // ---- Auto Quality ----
    this._lowFPSFrames = 0;  // consecutive frames below 30fps
    this._qualityLevel = 2;  // 2=high, 1=medium, 0=low

    // ---- Memory Pools (Optimization) ----
    this._vecForward = new THREE.Vector3();
    this._vecRight = new THREE.Vector3();
    this._vecLookTarget = new THREE.Vector3();
    this._vecDesiredPos = new THREE.Vector3();
    this.activeKeys = new Set();
    this._frustum = new THREE.Frustum();
    this._projScreenMat = new THREE.Matrix4();

    this.lastWheelTime = 0; // Mouse wheel weapon switch cooldown timer

    // Progression, Leveling & Ammo
    this.gameLevel = 1;
    this.health = 100;
    this.lives = 3;
    this.kills = 0;
    this.points = 0;
    this.weaponLevel = 1;
    this.ammo = 15;
    this.maxAmmo = 15;
    this.reserveAmmo = 999;
    this.isReloading = false;
    this.isSprintLocked = false;
    this.playerDead = false;
    this.gyroEnabled = true;
    this.gyroListening = false;
    this.gyroPermissionGranted = false;
    this.gyroLastTime = 0;
    this.gyroSmoothPitchRate = 0;
    this.gyroSmoothYawRate = 0;
    this.gyroPrevBeta = null;
    this.gyroPrevGamma = null;
    this.gyroPrevAlpha = null;
    this.lastSecondaryTime = 0;
    this.lastAimTime = 0;
    this.lastRailgunFireTime = 0;        // 1-second cooldown timestamp after railgun blast
    this.railgunFiredThisPress = false;  // prevents auto-fire looping while holding mouse
    this.mouseRightDown = false;         // track right mouse button state
    this.mouseLeftDown = false;         // track left mouse button state
    this.activeShootSound = null;        // active looping audio reference for autos
    this.audioBuffers = {};            // loaded AudioBuffers cache

    this.audioCtx = null;

    // Robot & Weapon
    this.robotGroup = null;
    this.robotModel = null;
    this.mixer = null;
    this.animations = {};
    this.currentAction = null;
    this.gun = null;
    this.gunGlowMat = null;
    this.handBoneRef = null;
    // ---- Virtual Cursor & Inventory State ----
    this.virtualCursorX = window.innerWidth / 2;
    this.virtualCursorY = window.innerHeight / 2;
    this.cursorState = 'default';
    this.hoveredObject = null;
    this.draggedItem = null;
    this.draggedFromSlot = -1;
    this.inventory = [
      { id: 'pistol', name: 'Pistol', image: '/assets/weapons/skin/pistol.png', type: 'weapon', equipped: true, maxClip: 15, clip: 15, reserve: 999 },
      { id: 'smg', name: 'SMG', image: '/assets/weapons/skin/smg.png', type: 'weapon', equipped: true, maxClip: 30, clip: 30, reserve: 999 },
      { id: 'railgun', name: 'Railgun', image: '/assets/weapons/skin/railgun.png', type: 'weapon', equipped: true, maxClip: 3, clip: 3, reserve: 999 },
      { id: 'ak47', name: 'AK-47', image: '/assets/weapons/skin/ak47.png', type: 'weapon', equipped: true, maxClip: 30, clip: 30, reserve: 999 },
      { id: 'rocket', name: 'Rocket Launcher', image: '/assets/weapons/skin/rocket.png', type: 'weapon', equipped: true, maxClip: 1, clip: 1, reserve: 999 }
    ];
    this.activeSlot = 0;
    this.groundItems = [];

    window.game = this;

    this.init();
  }

  // ==============================================================
  //  INITIALIZATION
  // ==============================================================
  async preloadSounds() {
    const sounds = {
      pistol: '/assets/weapons/sound/pistol.mp3',
      smg: '/assets/weapons/sound/smg.mp3',
      railgun: '/assets/weapons/sound/railgun.mp3',
      ak47: '/assets/weapons/sound/ak47.mp3',
      rocket: '/assets/weapons/sound/rocket.mp3',
      reload: '/assets/weapons/sound/reload.mp3',
      dying: '/assets/weapons/sound/dying.mp3'
    };

    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    for (const [key, url] of Object.entries(sounds)) {
      try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
        this.audioBuffers[key] = audioBuffer;
        console.log(`Preloaded sound: ${key}`);
      } catch (err) {
        console.warn(`Failed to preload sound ${key} from ${url}:`, err);
      }
    }
  }

  playSoundBuffer(name: string, loop = false, volume = 0.5) {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();

      const buffer = this.audioBuffers[name];
      if (!buffer) return null;

      const source = this.audioCtx.createBufferSource();
      source.buffer = buffer;
      source.loop = loop;

      const gainNode = this.audioCtx.createGain();
      gainNode.gain.setValueAtTime(volume, this.audioCtx.currentTime);

      source.connect(gainNode);
      gainNode.connect(this.audioCtx.destination);
      source.start(0);

      // Force pistol to stop after exactly 1.0 second to ensure clean single-shot decay
      if (name === 'pistol') {
        source.stop(this.audioCtx.currentTime + 1.0);
      }

      return { source, gainNode };
    } catch (e) {
      console.warn(`Failed to play sound: ${name}`, e);
      return null;
    }
  }

  stopAutomaticFireEffects() {
    if (this.activeShootSound) {
      try {
        if (this.activeShootSound.source) {
          this.activeShootSound.source.stop();
        }
      } catch (e) { }
      this.activeShootSound = null;
    }
    this.stopRailgunChargeSound();
    this.isChargingRailgun = false;
  }

  playRailgunChargeSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;

      this.chargeOsc = ctx.createOscillator();
      this.chargeGain = ctx.createGain();

      this.chargeOsc.type = 'sine';
      this.chargeOsc.frequency.setValueAtTime(100, ctx.currentTime);
      this.chargeOsc.frequency.linearRampToValueAtTime(1400, ctx.currentTime + 1.5);

      this.chargeGain.gain.setValueAtTime(0.01, ctx.currentTime);
      this.chargeGain.gain.linearRampToValueAtTime(0.22, ctx.currentTime + 1.5);

      this.chargeOsc.connect(this.chargeGain);
      this.chargeGain.connect(ctx.destination);
      this.chargeOsc.start(0);
    } catch (e) { }
  }

  stopRailgunChargeSound() {
    if (this.chargeOsc) {
      try { this.chargeOsc.stop(); } catch (e) { }
      this.chargeOsc = null;
    }
  }

  async init() {
    // Preload audio files
    await this.preloadSounds();

    // Load WASM terrain engine
    try {
      const instance = await createEngineModule({
        locateFile: (path: string) => {
          if (path.endsWith('.wasm')) {
            return '/engine.wasm';
          }
          return path;
        }
      });
      const cppNoise = new instance.PerlinNoise();
      cppNoise.init(42);
      this.wasm = cppNoise;
      console.log("C++ WebAssembly terrain engine loaded successfully.");
    } catch (e) {
      console.warn("Failed loading C++ WebAssembly noise module. Falling back to JS terrain generator.", e);
    }

    try {
      const useAntialias = !this.isMobile; // Disabling MSAA on mobile saves >60% GPU RAM preventing Safari WebKit crashes
      this.renderer = new THREE.WebGLRenderer({
        antialias: useAntialias,
        powerPreference: this.isMobile ? 'default' : 'high-performance',
        alpha: false,
        stencil: false
      });
    } catch (e) {
      console.warn("Failed to initialize WebGLRenderer with high-performance. Trying default power preference...", e);
      try {
        this.renderer = new THREE.WebGLRenderer({ antialias: false });
      } catch (err) {
        console.error("Failed to initialize WebGLRenderer:", err);
        const errorDiv = document.createElement('div');
        errorDiv.style.position = 'fixed';
        errorDiv.style.top = '0';
        errorDiv.style.left = '0';
        errorDiv.style.width = '100%';
        errorDiv.style.height = '100%';
        errorDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
        errorDiv.style.color = '#ff6b6b';
        errorDiv.style.display = 'flex';
        errorDiv.style.flexDirection = 'column';
        errorDiv.style.justifyContent = 'center';
        errorDiv.style.alignItems = 'center';
        errorDiv.style.fontFamily = 'sans-serif';
        errorDiv.style.padding = '20px';
        errorDiv.style.boxSizing = 'border-box';
        errorDiv.style.zIndex = '9999';
        errorDiv.innerHTML = `
          <h1 style="margin-bottom: 10px;">WebGL Not Supported</h1>
          <p style="color: #ccc; text-align: center; max-width: 500px; line-height: 1.5;">
            The game could not start because WebGL could not be initialized by your browser.<br><br>
            Please check that <strong>Hardware Acceleration</strong> is enabled in your browser settings (Settings > System > Use graphics acceleration when available) and that your graphics drivers are up to date.
          </p>
        `;
        document.body.appendChild(errorDiv);
        throw err;
      }
    }

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    // Cap pixel ratio at 1.5 — avoids rendering 4× pixels on HiDPI screens
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    // PCFShadowMap: High-speed single-tap filtering optimized for Intel Iris & integrated GPUs
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.appendChild(this.renderer.domElement);

    // Safari WebKit Context Loss & Recovery Guard (Prevents "A problem repeatedly occurred")
    const canvasEl = this.renderer.domElement;
    canvasEl.addEventListener('webglcontextlost', (e: Event) => {
      e.preventDefault(); // CRITICAL for Safari: tells WebKit not to crash/reload the page
      console.warn('[WebGL] Context lost detected. Safely pausing render frame.');
      this.isContextLost = true;
    }, false);

    canvasEl.addEventListener('webglcontextrestored', () => {
      console.log('[WebGL] Context successfully restored. Resuming render loop.');
      this.isContextLost = false;
      if (this.renderer) {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
      }
    }, false);

    // Cleanup GPU resources cleanly on soft refresh or page navigation
    window.addEventListener('pagehide', () => {
      this.stopAutomaticFireEffects();
    });

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.mapConfig.weather.clear.skyColor);
    this.scene.fog = new THREE.FogExp2(this.mapConfig.fogColor, this.mapConfig.fogDensity * 0.15);

    this.camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.5, 8000);
    this.clock = new THREE.Clock();

    this.updateLoadingProgress(15, "Loading textures & environmental assets...");
    await this.loadAssets();
    this.renderer.render(this.scene, this.camera);
    await new Promise(resolve => requestAnimationFrame(resolve));

    this.updateLoadingProgress(35, "Generating procedural 3D terrain biomes...");
    this.createLighting();
    this.createSkybox();
    this.renderer.render(this.scene, this.camera);
    await new Promise(resolve => requestAnimationFrame(resolve));

    this.updateLoadingProgress(55, "Preloading enemy geometries & particle systems...");
    this.initDemonPool(); // Pre-create shared demon geometries to prevent spawn stutter
    this.createGun();
    this.renderer.render(this.scene, this.camera);
    await new Promise(resolve => requestAnimationFrame(resolve));

    this.updateLoadingProgress(75, "Instantiating 3D character & weapon armatures...");
    this.createRobot();

    // Progressive load: only generate the immediate 3x3 chunks during loading screen,
    // and let the remaining chunks generate smoothly in the background when the game starts.
    const originalRadius = this.viewRadius;
    this.viewRadius = 1; // 3x3 grid (9 chunks instead of 49 chunks)
    this.updateChunks();
    this.viewRadius = originalRadius; // Restore full view distance for progressive loading

    // Reposition camera directly behind character for early menu preview
    const rp = this.robotGroup.position;
    this.camLookTarget.set(rp.x, rp.y + 6, rp.z);
    this.camera.position.set(rp.x, rp.y + 20, rp.z + this.camDist);
    this.camera.lookAt(this.camLookTarget);
    // Pre-warm WebGL shader compilation on the GPU to prevent mid-game micro-stutters
    this.renderer.compile(this.scene, this.camera);
    this.renderer.render(this.scene, this.camera);
    await new Promise(resolve => requestAnimationFrame(resolve));

    this.updateLoadingProgress(90, "Configuring HUD overlays & tactical systems...");
    this.createBalloons();
    this.initWeatherSystem();
    this.createFogLayers();
    this.renderer.render(this.scene, this.camera);
    await new Promise(resolve => requestAnimationFrame(resolve));



    // Spawn initial ground items
    for (let i = 0; i < 25; i++) {
      const rx = (Math.random() - 0.5) * 600;
      const rz = (Math.random() - 0.5) * 600;
      const types = ['crystal', 'nanokit', 'powercell'];
      const type = types[Math.floor(Math.random() * types.length)];
      this.groundItems.push(new GroundItem(this.scene, type, rx, rz, this));
    }

    // Initial inventory UI update
    this.updateInventoryUI();
    this.useInventoryItem(0);

    this.setupControls();
    this.setupPostProcessing();
    this.setupMinimap();
    this.setupECG();

    this.updateLoadingProgress(100, "Initialization Complete!");

    const ls = document.getElementById('loading-screen');
    if (ls) {
      ls.classList.add('fade-out');
      setTimeout(() => ls.style.display = 'none', 1400);
    }

    this.animate();
  }

  updateLoadingProgress(percent: number, tipMessage?: string) {
    const bar = document.getElementById('progress-fill');
    if (bar) {
      bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
    }
    if (tipMessage) {
      const tipEl = document.querySelector('.loader-tip');
      if (tipEl) tipEl.textContent = tipMessage;
    }
  }

  loadAssets() {
    const textureLoader = new THREE.TextureLoader();
    const gltfLoader = new GLTFLoader();
    this.obstacleGltfs = [];

    const loadTexture = () => {
      return new Promise<void>(resolve => {
        textureLoader.load('Screenshot 2026-08-03 152141.png',
          tex => {
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            this.landscapeTexture = tex;
            resolve();
          },
          undefined,
          () => {
            console.warn('Landscape image failed to load - continuing');
            resolve();
          }
        );
      });
    };

    const loadGLTF = (url: string) => {
      return new Promise<any>(async (resolve) => {
        const fetchAndCache = () => {
          fetch(url)
            .then(res => {
              if (!res.ok) throw new Error(`HTTP error ${res.status}`);
              return res.arrayBuffer();
            })
            .then(buffer => {
              saveAssetBuffer(url, buffer);
              gltfLoader.parse(buffer, '', (gltf) => {
                resolve({ url, gltf });
              }, (err) => {
                console.warn(`Failed to parse fetched GLTF: ${url}`, err);
                resolve({ url, gltf: null });
              });
            })
            .catch(err => {
              console.warn(`Failed to load GLTF model: ${url}`, err);
              resolve({ url, gltf: null });
            });
        };

        try {
          // Read binary ArrayBuffer directly from IndexedDB (instant load from local SSD!)
          const cachedBuffer = await getCachedAssetBuffer(url);
          if (cachedBuffer) {
            gltfLoader.parse(cachedBuffer, '', (gltf) => {
              resolve({ url, gltf });
            }, (err) => {
              console.warn(`IndexedDB parse error for ${url}, refetching...`, err);
              fetchAndCache();
            });
            return;
          }
        } catch (e) {
          // Fallback to fetch
        }

        fetchAndCache();
      });
    };

    return new Promise<void>(async resolve => {
      this.updateLoadingProgress(20, "Preloading environment textures...");
      await loadTexture();

      this.updateLoadingProgress(30, "Establishing connection to asset servers...");

      const urls = [
        '/assets/objects/obsticle1.glb',
        '/assets/objects/obsticle2.glb',
        '/assets/objects/Xbot.glb',
        '/assets/objects/RobotExpressive.glb'
      ];

      // Download all models concurrently in parallel (massively speeds up load time!)
      let loadedCount = 0;
      const totalModels = urls.length;

      const promises = urls.map(url => {
        return loadGLTF(url).then(res => {
          loadedCount++;
          const pct = 30 + Math.round((loadedCount / totalModels) * 35);
          this.updateLoadingProgress(pct, `Preloading game models (${loadedCount}/${totalModels})...`);
          return res;
        });
      });

      const results = await Promise.all(promises);

      // Process results
      results.forEach(res => {
        if (!res || !res.gltf) return;
        const { url, gltf } = res;

        if (url.includes('obsticle1') || url.includes('obsticle2')) {
          gltf.scene.traverse((child: any) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              child.layers.set(0);
              child.raycast = THREE.Mesh.prototype.raycast;
              if (child.geometry && !child.geometry.attributes.normal) {
                child.geometry.computeVertexNormals();
              }
              if (child.material) {
                const configurePBR = (m: any) => {
                  let mat = m;
                  // Convert basic unlit materials to MeshStandardMaterial to guarantee response to directional sunlight
                  if (m.type === 'MeshBasicMaterial' || !m.isMeshStandardMaterial) {
                    mat = new THREE.MeshStandardMaterial({
                      color: m.color || 0x909090,
                      map: m.map || null,
                      roughness: 0.45,
                      metalness: 0.15,
                      side: THREE.DoubleSide
                    });
                  } else {
                    mat.side = THREE.DoubleSide;
                    mat.roughness = 0.45;
                    mat.metalness = 0.15;
                    mat.needsUpdate = true;
                  }
                  return mat;
                };
                if (Array.isArray(child.material)) {
                  child.material = child.material.map((m: any) => configurePBR(m));
                } else {
                  child.material = configurePBR(child.material);
                }
              }
            }
          });
          const isObs1 = url.includes('obsticle1');
          const box = new THREE.Box3().setFromObject(gltf.scene);
          (gltf as any).minY = box.min.y;
          (gltf as any).isObsticle1 = isObs1;
          (gltf as any).isObsticle2 = !isObs1;
          this.obstacleGltfs.push(gltf);
        } else if (url.includes('Xbot.glb')) {
          (gltf as any).isXbot = true;
          this.robotGltf = gltf;
        } else if (url.includes('RobotExpressive.glb')) {
          (gltf as any).isRobotExpressive = true;
          this.demonGltf = gltf;
        }
      });

      this.updateLoadingProgress(65, "Assets preload complete!");
      setTimeout(resolve, 100);
    });
  }

  createLighting() {
    const mc = this.mapConfig;
    this.ambientLight = new THREE.AmbientLight(mc.ambientColor, 0.55);
    this.scene.add(this.ambientLight);
    this.scene.add(new THREE.HemisphereLight(mc.hemiSky, mc.hemiGround, 0.45));
    const sun = new THREE.DirectionalLight(mc.sunColor, (mc.sunIntensity || 1.8) * 1.8);
    const sunPos = (mc as any).sunPosition || { x: 400, y: 550, z: 250 };
    sun.position.set(sunPos.x, sunPos.y, sunPos.z);
    sun.castShadow = true;
    const s = sun.shadow;
    s.mapSize.width = s.mapSize.height = 1024; // 4× less pixel buffer fill work for Intel Iris
    s.camera.near = 5;
    s.camera.far = 800;
    s.camera.left = s.camera.bottom = -180;
    s.camera.right = s.camera.top = 180;
    s.bias = -0.0003;
    s.normalBias = 0.02;
    this.scene.add(sun);
    this.scene.add(sun.target);
    this.sun = sun;

    // --- Visible 3D Sun Orb + Volumetric Sunbeams in Skybox ---
    if (this.sunMesh) {
      this.scene.remove(this.sunMesh);
      this.sunMesh = null;
    }

    const sunGroup = new THREE.Group();

    // Core glowing solar orb (depthTest: false ensures it's always visible in skybox)
    const sunGeo = new THREE.SphereGeometry(120, 32, 32);
    const sunMat = new THREE.MeshBasicMaterial({
      color: mc.sunColor || 0xfff8e0,
      fog: false,
      depthTest: false,
      depthWrite: false
    });
    const sunOrb = new THREE.Mesh(sunGeo, sunMat);
    sunGroup.add(sunOrb);

    // Inner glowing solar corona
    const haloGeo = new THREE.SphereGeometry(180, 32, 32);
    const haloMat = new THREE.MeshBasicMaterial({
      color: mc.sunColor || 0xffe0a0,
      transparent: true,
      opacity: 0.65,
      side: THREE.BackSide,
      fog: false,
      depthTest: false,
      depthWrite: false
    });
    const haloMesh = new THREE.Mesh(haloGeo, haloMat);
    sunGroup.add(haloMesh);

    // Outer atmosphere flare aura
    const outerGeo = new THREE.SphereGeometry(260, 32, 32);
    const outerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.35,
      side: THREE.BackSide,
      fog: false,
      depthTest: false,
      depthWrite: false
    });
    const outerMesh = new THREE.Mesh(outerGeo, outerMat);
    sunGroup.add(outerMesh);

    const sunDir = new THREE.Vector3(sunPos.x, sunPos.y, sunPos.z).normalize();
    this._sunDir = sunDir;
    sunGroup.position.copy(sunDir.clone().multiplyScalar(1800));
    sunGroup.renderOrder = 10; // Renders on top of skybox
    this.scene.add(sunGroup);
    this.sunMesh = sunGroup;
  }

  createSkybox() {
    if (this.mapId === 'arctic' && this.landscapeTexture) {
      const cylGeo = new THREE.CylinderGeometry(4800, 4800, 2800, 64, 1, true);
      const cylMat = new THREE.MeshBasicMaterial({
        map: this.landscapeTexture,
        side: THREE.BackSide,
        fog: false,
        depthWrite: false,
      });
      const cyl = new THREE.Mesh(cylGeo, cylMat);
      cyl.position.y = 350;
      cyl.renderOrder = -1;
      this.scene.add(cyl);
    }

    const c = document.createElement('canvas');
    c.width = 4; c.height = 512;
    const ctx = c.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 0, 512);
    const sg = this.mapConfig.skyGradient;
    g.addColorStop(0, sg[0]);
    g.addColorStop(0.25, sg[1]);
    g.addColorStop(0.5, sg[2]);
    g.addColorStop(0.75, sg[3]);
    g.addColorStop(1, sg[4]);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 4, 512);
    const skyTex = new THREE.CanvasTexture(c);

    const domeGeo = new THREE.SphereGeometry(4900, 32, 20, 0, Math.PI * 2, 0, Math.PI * 0.5);
    const domeMat = new THREE.MeshBasicMaterial({
      map: skyTex,
      side: THREE.BackSide,
      fog: false,
      depthWrite: false,
    });
    const dome = new THREE.Mesh(domeGeo, domeMat);
    dome.position.y = 350;
    dome.renderOrder = -2;
    this.scene.add(dome);
  }

  // ==============================================================
  //  DEMON GEOMETRY POOL — Pre-create shared geometries/materials
  //  Eliminates stutter when demons spawn (no new GPU uploads)
  // ==============================================================
  initDemonPool() {
    const hornGeo = new THREE.ConeGeometry(0.7, 2.8, 6);
    hornGeo.rotateX(Math.PI / 4.5);

    this._demonPool = {
      bodyGeo: new THREE.SphereGeometry(3.6, 12, 12),
      coreGeo: new THREE.SphereGeometry(2.2, 8, 8),
      hornGeo: hornGeo,
      eyeGeo: new THREE.SphereGeometry(0.45, 8, 8),
      spikeGeo: new THREE.ConeGeometry(0.4, 1.8, 4),
      wBarrelGeo: new THREE.CylinderGeometry(0.25, 0.25, 2.4, 8),
      wGlowGeo: new THREE.SphereGeometry(0.4, 8, 8),
      bodyMat: new THREE.MeshStandardMaterial({ color: 0x181818, roughness: 0.9, metalness: 0.1 }),
      glowMat: new THREE.MeshStandardMaterial({ color: 0xff1100, emissive: 0xff0000, emissiveIntensity: 3.5 }),
      wBarrelMat: new THREE.MeshStandardMaterial({ color: 0x111111, metalness: 0.8 }),
      wGlowMat: new THREE.MeshBasicMaterial({ color: 0xff0000 })
    };
  }

  generateHeight(x, z) {
    if (this.wasm && typeof this.wasm.generateHeight === 'function') {
      return this.wasm.generateHeight(x, z, this.MAX_HEIGHT);
    }
    // JS Fallback
    const nx = x / 2000;
    const nz = z / 2000;

    const valleyNoise = (this.noise.noise(nx * 0.5, nz * 0.5) + 1) * 0.5;
    const envelope = THREE.MathUtils.smoothstep(valleyNoise, 0.25, 0.75);

    let h = (this.noise.fbm(nx * 3.5 + 10, nz * 3.5 + 10, 5, 2.0, 0.5) + 1) * 0.5;
    const ridge = this.noise.ridged(nx * 3 + 5, nz * 3 + 5, 5, 2.2, 0.52);

    h = h * 0.3 + ridge * 0.7;
    h *= envelope;

    h += (this.noise.fbm(nx * 12, nz * 12, 4, 2.0, 0.45) + 1) * 0.04;
    h += (this.noise.fbm(nx * 28, nz * 28, 3, 2.0, 0.4) + 1) * 0.012;

    return Math.max(h * this.MAX_HEIGHT, 15);
  }

  getHeightAt(x, z) {
    return this.generateHeight(x, z);
  }

  createGun() {
    this.gun = new THREE.Group();
    this.rebuildGunMesh();
  }

  rebuildGunMesh() {
    if (!this.gun) return;

    // Clear all existing parts
    while (this.gun.children.length > 0) {
      const child = this.gun.children[0];
      this.gun.remove(child);
      if ((child as any).geometry) (child as any).geometry.dispose();
      if ((child as any).material) {
        if (Array.isArray((child as any).material)) {
          (child as any).material.forEach((m: any) => m.dispose());
        } else {
          (child as any).material.dispose();
        }
      }
    }

    const level = this.weaponLevel;

    // Define standard materials
    const steelMat = new THREE.MeshStandardMaterial({ color: 0x1c1d21, metalness: 0.85, roughness: 0.25 });
    const gunmetalMat = new THREE.MeshStandardMaterial({ color: 0x2e3033, metalness: 0.9, roughness: 0.2 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x111111, metalness: 0.1, roughness: 0.85 }); // polymer/matte
    const woodMat = new THREE.MeshStandardMaterial({ color: 0x5c3a21, roughness: 0.88, metalness: 0.05 }); // walnut/dark wood
    const oliveMat = new THREE.MeshStandardMaterial({ color: 0x3e4a30, metalness: 0.35, roughness: 0.65 }); // military olive

    // Recreate the glow material to bind to the new parts
    let glowColor = 0x00aaff;
    let emissiveColor = 0x0088ff;
    if (level === 2) {
      glowColor = 0x00ff66;
      emissiveColor = 0x00ff22;
    } else if (level === 3) {
      glowColor = 0xff3300;
      emissiveColor = 0xff0000;
    } else if (level === 4) {
      glowColor = 0xffaa00;
      emissiveColor = 0x885500;
    } else if (level === 5) {
      glowColor = 0xff33cc;
      emissiveColor = 0x880055;
    }

    this.gunGlowMat = new THREE.MeshStandardMaterial({
      color: glowColor,
      emissive: emissiveColor,
      emissiveIntensity: 3.5
    });

    if (level === 1) {
      // --- CYBER PISTOL (Level 1) ---
      // 1. Frame / Polymer Lower Grip (Rough matte material)
      const frame = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.45, 0.9), darkMat);
      frame.position.set(0, -0.05, -0.05);
      frame.castShadow = true;
      this.gun.add(frame);

      // Angled grip
      const grip = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.65, 0.28), darkMat);
      grip.position.set(0, -0.45, -0.3);
      grip.rotation.x = -Math.PI / 5.5;
      grip.castShadow = true;
      this.gun.add(grip);

      // Grip checkering panel plates
      const panelL = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.45, 0.2), steelMat);
      panelL.position.set(0.1, -0.45, -0.3);
      panelL.rotation.x = -Math.PI / 5.5;
      this.gun.add(panelL);

      const panelR = panelL.clone();
      panelR.position.x = -0.1;
      this.gun.add(panelR);

      // 2. Slide Receiver (Steel material, slightly elevated)
      const slide = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.35, 1.15), gunmetalMat);
      slide.position.set(0, 0.22, 0.05);
      slide.castShadow = true;
      this.gun.add(slide);

      // Ejection port visual cut
      const port = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.12, 0.25), darkMat);
      port.position.set(0.08, 0.28, 0.05);
      this.gun.add(port);

      // 3. Glowing Energy Core & Side Rails
      const core = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.6, 8), this.gunGlowMat);
      core.rotation.x = Math.PI / 2;
      core.position.set(0, 0.22, 0.05);
      this.gun.add(core);

      const sideRailL = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.06, 0.95), this.gunGlowMat);
      sideRailL.position.set(0.14, 0.22, 0.05);
      this.gun.add(sideRailL);

      const sideRailR = sideRailL.clone();
      sideRailR.position.x = -0.14;
      this.gun.add(sideRailR);

      // Glowing Torus Rings
      const ringGeo = new THREE.TorusGeometry(0.2, 0.04, 8, 14);
      for (let i = 0; i < 2; i++) {
        const ring = new THREE.Mesh(ringGeo, this.gunGlowMat);
        ring.position.set(0, 0.22, 0.45 + i * 0.45);
        this.gun.add(ring);
      }

      // 4. Threaded Suppressor / Silencer Barrel
      const silencer = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.7, 8), darkMat);
      silencer.rotation.x = Math.PI / 2;
      silencer.position.set(0, 0.22, 0.95);
      silencer.castShadow = true;
      this.gun.add(silencer);

      const innerBarrel = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.72, 8), steelMat);
      innerBarrel.rotation.x = Math.PI / 2;
      innerBarrel.position.set(0, 0.22, 0.95);
      this.gun.add(innerBarrel);

      // 5. Tactical Under-barrel attachment (Laser Pointer)
      const tacBox = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.24, 0.5), darkMat);
      tacBox.position.set(0, -0.12, 0.3);
      this.gun.add(tacBox);

      const laserLens = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.08, 8), this.gunGlowMat);
      laserLens.rotation.x = Math.PI / 2;
      laserLens.position.set(0, -0.12, 0.55);
      this.gun.add(laserLens);

      // 6. Iron Sights
      const rearSight = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.06), darkMat);
      rearSight.position.set(0, 0.42, -0.45);
      this.gun.add(rearSight);

      const frontSight = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.1, 0.06), darkMat);
      frontSight.position.set(0, 0.43, 0.55);
      this.gun.add(frontSight);

    } else if (level === 2) {
      // --- PLASMA SMG (Level 2) ---
      // 1. Lower Receiver and Handguard
      const receiver = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.55, 1.8), steelMat);
      receiver.position.set(0, 0.05, 0.0);
      receiver.castShadow = true;
      this.gun.add(receiver);

      // Perforated Handguard
      const handguard = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.45, 0.95), darkMat);
      handguard.position.set(0, -0.05, 1.0);
      handguard.castShadow = true;
      this.gun.add(handguard);

      // Ventilation cutouts
      for (let i = 0; i < 3; i++) {
        const ventL = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.15, 0.18), steelMat);
        ventL.position.set(0.17, -0.05, 0.7 + i * 0.28);
        this.gun.add(ventL);
        const ventR = ventL.clone();
        ventR.position.x = -0.17;
        this.gun.add(ventR);
      }

      // Glowing Plasma Core Cylinder inside handguard
      const core = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 1.1, 8), this.gunGlowMat);
      core.rotation.x = Math.PI / 2;
      core.position.set(0, 0.05, 0.9);
      this.gun.add(core);

      // 3 Glowing Energy Torus Rings
      const ringGeo = new THREE.TorusGeometry(0.28, 0.05, 8, 16);
      for (let i = 0; i < 3; i++) {
        const ring = new THREE.Mesh(ringGeo, this.gunGlowMat);
        ring.position.set(0, 0.05, 0.6 + i * 0.45);
        this.gun.add(ring);
      }

      // 2. Telescopic Wire Stock
      const stockArmL = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.08, 1.15), gunmetalMat);
      stockArmL.position.set(0.13, 0.0, -1.05);
      stockArmL.castShadow = true;
      this.gun.add(stockArmL);

      const stockArmR = stockArmL.clone();
      stockArmR.position.x = -0.13;
      this.gun.add(stockArmR);

      // Buttplate
      const buttplate = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.75, 0.14), darkMat);
      buttplate.position.set(0, -0.15, -1.62);
      buttplate.castShadow = true;
      this.gun.add(buttplate);

      // 3. SMG Magazine with glowing energy strip
      const mag = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.85, 0.35), darkMat);
      mag.position.set(0, -0.65, 0.45);
      mag.rotation.x = -Math.PI / 12;
      mag.castShadow = true;
      this.gun.add(mag);

      const magGlow = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.75, 0.06), this.gunGlowMat);
      magGlow.position.set(0, -0.65, 0.63);
      magGlow.rotation.x = -Math.PI / 12;
      this.gun.add(magGlow);

      // 4. Barrel & Flash Hider
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.85, 8), gunmetalMat);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(0, 0.05, 1.6);
      barrel.castShadow = true;
      this.gun.add(barrel);

      const flashHider = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.08, 0.22, 8), steelMat);
      flashHider.rotation.x = Math.PI / 2;
      flashHider.position.set(0, 0.05, 2.05);
      this.gun.add(flashHider);

      // 5. Reflex Holographic Sight with Glowing Lens
      const scopeBase = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.18, 0.55), darkMat);
      scopeBase.position.set(0, 0.42, 0.05);
      this.gun.add(scopeBase);

      const scopeLensHolder = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.25, 0.08), darkMat);
      scopeLensHolder.position.set(0, 0.58, 0.28);
      this.gun.add(scopeLensHolder);

      const lens = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.18, 0.04), this.gunGlowMat);
      lens.position.set(0, 0.58, 0.28);
      this.gun.add(lens);

      // Angled tactical foregrip
      const foregrip = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.4, 0.22), darkMat);
      foregrip.position.set(0, -0.4, 0.85);
      foregrip.rotation.x = Math.PI / 8;
      foregrip.castShadow = true;
      this.gun.add(foregrip);

    } else if (level === 3) {
      // --- M-66 RAILGUN (Level 3) ---
      const receiver = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.9, 2.5), steelMat);
      receiver.position.set(0, 0, 0);
      receiver.castShadow = true;
      this.gun.add(receiver);

      const railTop = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.15, 4.8), gunmetalMat);
      railTop.position.set(0, 0.25, 2.2);
      railTop.castShadow = true;
      this.gun.add(railTop);

      const railBottom = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.15, 4.8), gunmetalMat);
      railBottom.position.set(0, -0.25, 2.2);
      railBottom.castShadow = true;
      this.gun.add(railBottom);

      const coreGeo = new THREE.CylinderGeometry(0.25, 0.25, 1.0, 8);
      const core = new THREE.Mesh(coreGeo, this.gunGlowMat);
      core.rotation.x = Math.PI / 2;
      core.position.set(0, 0, 0.5);
      this.gun.add(core);

      const ringGeo = new THREE.TorusGeometry(0.48, 0.08, 8, 16);
      for (let i = 0; i < 3; i++) {
        const ring = new THREE.Mesh(ringGeo, this.gunGlowMat);
        ring.position.set(0, 0, 1.5 + i * 1.0);
        this.gun.add(ring);
      }

      const scope = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.3, 0.8), darkMat);
      scope.position.set(0, 0.6, -0.4);
      scope.castShadow = true;
      this.gun.add(scope);

    } else if (level === 4) {
      // --- CYBER AK-47 ASSAULT RIFLE (Level 4) ---
      // 1. Receivers (Gunmetal stamped steel)
      const receiver = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.55, 1.95), gunmetalMat);
      receiver.position.set(0, 0.05, -0.05);
      receiver.castShadow = true;
      this.gun.add(receiver);

      const dustCover = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.25, 1.6), steelMat);
      dustCover.position.set(0, 0.38, -0.2);
      dustCover.castShadow = true;
      this.gun.add(dustCover);

      // 2. High-quality wooden stock with buttplate
      const stock = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.45, 1.25), woodMat);
      stock.position.set(0, -0.08, -1.45);
      stock.rotation.x = Math.PI / 22;
      stock.castShadow = true;
      this.gun.add(stock);

      const stockButtplate = new THREE.Mesh(new THREE.BoxGeometry(0.27, 0.48, 0.06), steelMat);
      stockButtplate.position.set(0, -0.15, -2.08);
      stockButtplate.rotation.x = Math.PI / 22;
      this.gun.add(stockButtplate);

      // 3. Wooden Handguards
      const handguardLower = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.35, 1.15), woodMat);
      handguardLower.position.set(0, -0.1, 1.1);
      handguardLower.castShadow = true;
      this.gun.add(handguardLower);

      const handguardUpper = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.25, 0.95), woodMat);
      handguardUpper.position.set(0, 0.2, 1.05);
      handguardUpper.castShadow = true;
      this.gun.add(handguardUpper);

      // Glowing Gas-Tube Core Cylinder
      const core = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 1.2, 8), this.gunGlowMat);
      core.rotation.x = Math.PI / 2;
      core.position.set(0, 0.32, 1.25);
      this.gun.add(core);

      // 4 Glowing Energy Torus Rings along barrel & gas tube
      const ringGeo = new THREE.TorusGeometry(0.26, 0.05, 8, 16);
      for (let i = 0; i < 4; i++) {
        const ring = new THREE.Mesh(ringGeo, this.gunGlowMat);
        ring.position.set(0, 0.22, 0.75 + i * 0.55);
        this.gun.add(ring);
      }

      // 4. Barrel & Cleaning Rod
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.25, 8), steelMat);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(0, 0.18, 1.95);
      barrel.castShadow = true;
      this.gun.add(barrel);

      const cleaningRod = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.6, 8), gunmetalMat);
      cleaningRod.rotation.x = Math.PI / 2;
      cleaningRod.position.set(0, -0.06, 1.6);
      this.gun.add(cleaningRod);

      // 5. Curved Magazine with Glowing Indicator Strip
      const mag = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.95, 0.45), darkMat);
      mag.position.set(0, -0.68, 0.42);
      mag.rotation.x = -Math.PI / 6.5;
      mag.castShadow = true;
      this.gun.add(mag);

      const magGlow = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.85, 0.06), this.gunGlowMat);
      magGlow.position.set(0, -0.68, 0.65);
      magGlow.rotation.x = -Math.PI / 6.5;
      this.gun.add(magGlow);

      // 6. Tangent Leaf Rear Sight & Front Post Sight
      const rearSight = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.14, 0.3), gunmetalMat);
      rearSight.position.set(0, 0.44, 0.45);
      this.gun.add(rearSight);

      const frontSightBlock = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.35, 0.12), steelMat);
      frontSightBlock.position.set(0, 0.36, 2.9);
      this.gun.add(frontSightBlock);

    } else if (level === 5) {
      // --- QUANTUM ROCKET LAUNCHER (Level 5) ---
      // 1. Heavy Launcher Tube
      const tubeFront = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 1.95, 12), oliveMat);
      tubeFront.rotation.x = Math.PI / 2;
      tubeFront.position.set(0, 0.1, 1.0);
      tubeFront.castShadow = true;
      this.gun.add(tubeFront);

      const tubeRear = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 1.95, 12), oliveMat);
      tubeRear.rotation.x = Math.PI / 2;
      tubeRear.position.set(0, 0.1, -0.95);
      tubeRear.castShadow = true;
      this.gun.add(tubeRear);

      // 2. Wood Heat Shield Wrap
      const shield = new THREE.Mesh(new THREE.CylinderGeometry(0.39, 0.39, 1.15, 12), woodMat);
      shield.rotation.x = Math.PI / 2;
      shield.position.set(0, 0.1, -0.15);
      shield.castShadow = true;
      this.gun.add(shield);

      // 2 Large Glowing Energy Torus Rings around launcher tube
      const ringGeo = new THREE.TorusGeometry(0.55, 0.09, 8, 16);
      const ring1 = new THREE.Mesh(ringGeo, this.gunGlowMat);
      ring1.position.set(0, 0.1, 0.7);
      this.gun.add(ring1);

      const ring2 = new THREE.Mesh(ringGeo, this.gunGlowMat);
      ring2.position.set(0, 0.1, 1.5);
      this.gun.add(ring2);

      // 3. Flared Exhaust Cone at rear
      const exhaustCone = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.48, 0.65, 12), gunmetalMat);
      exhaustCone.rotation.x = Math.PI / 2;
      exhaustCone.position.set(0, 0.1, -2.15);
      this.gun.add(exhaustCone);

      // 4. Optical Rangefinding Scope (RPG PGO-7 sight with glowing lens)
      const scopeBracket = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.15, 0.18), darkMat);
      scopeBracket.position.set(-0.35, 0.22, 0.28);
      this.gun.add(scopeBracket);

      const scopeBody = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.08, 0.8, 8), darkMat);
      scopeBody.rotation.x = Math.PI / 2;
      scopeBody.position.set(-0.48, 0.44, 0.28);
      scopeBody.castShadow = true;
      this.gun.add(scopeBody);

      const scopeLens = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.06, 8), this.gunGlowMat);
      scopeLens.rotation.x = Math.PI / 2;
      scopeLens.position.set(-0.48, 0.44, 0.69);
      this.gun.add(scopeLens);

      // 5. Dual Support Grips and trigger box
      const triggerGrip = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.6, 0.22), darkMat);
      triggerGrip.position.set(0, -0.42, 0.55);
      triggerGrip.rotation.x = -Math.PI / 12;
      triggerGrip.castShadow = true;
      this.gun.add(triggerGrip);

      const rearGrip = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.5, 0.18), darkMat);
      rearGrip.position.set(0, -0.38, -0.65);
      rearGrip.rotation.x = -Math.PI / 12;
      rearGrip.castShadow = true;
      this.gun.add(rearGrip);

      // 6. Projectile Warhead loaded at muzzle tip with glowing ring
      const rocketCone = new THREE.Mesh(new THREE.ConeGeometry(0.48, 0.95, 12), oliveMat);
      rocketCone.rotation.x = Math.PI / 2;
      rocketCone.position.set(0, 0.1, 2.85);
      rocketCone.castShadow = true;
      this.gun.add(rocketCone);

      const warheadRing = new THREE.Mesh(new THREE.TorusGeometry(0.38, 0.05, 8, 14), this.gunGlowMat);
      warheadRing.position.set(0, 0.1, 2.65);
      this.gun.add(warheadRing);

      const rocketCap = new THREE.Mesh(new THREE.CylinderGeometry(0.48, 0.35, 0.55, 12), steelMat);
      rocketCap.rotation.x = Math.PI / 2;
      rocketCap.position.set(0, 0.1, 2.15);
      rocketCap.castShadow = true;
      this.gun.add(rocketCap);

      const rocketFuse = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.28, 8), steelMat);
      rocketFuse.rotation.x = Math.PI / 2;
      rocketFuse.position.set(0, 0.1, 3.45);
      this.gun.add(rocketFuse);

      // Stabilizing rocket fins
      for (let i = 0; i < 4; i++) {
        const fin = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.32, 0.5), steelMat);
        const angle = (i * Math.PI) / 2;
        fin.position.set(Math.cos(angle) * 0.3, 0.1 + Math.sin(angle) * 0.3, 2.15);
        fin.rotation.z = angle;
        this.gun.add(fin);
      }
    }

    // Ensure the gun is attached to the correct parent node (right hand bone for Xbot, robotGroup for fallback)
    const rightHandBone = this.robotModel ? (this.robotModel.getObjectByName('mixamorigRightHand') || this.robotModel.getObjectByName('RightHand')) : null;
    if (rightHandBone) {
      if (this.gun.parent !== rightHandBone) {
        rightHandBone.add(this.gun);
      }
      this.gun.scale.setScalar(1.0 / 10.5);
    } else {
      if (this.gun.parent !== this.robotGroup) {
        this.robotGroup.add(this.gun);
      }
      this.gun.scale.setScalar(1.0);
    }
  }

  createRobot() {
    this.robotGroup = new THREE.Group();
    this.scene.add(this.robotGroup);

    const fallbackRobot = () => {
      console.warn("Using procedural robot layout fallback.");
      const group = new THREE.Group();
      const bodyMat = new THREE.MeshStandardMaterial({ color: 0x4a5d80, metalness: 0.7, roughness: 0.25 });
      const eyeMat = new THREE.MeshBasicMaterial({ color: 0x00ffee });

      const torso = new THREE.Mesh(new THREE.BoxGeometry(4.0, 5.0, 2.8), bodyMat);
      torso.position.y = 5.75;
      torso.castShadow = true;
      group.add(torso);

      const head = new THREE.Mesh(new THREE.BoxGeometry(2.8, 2.8, 2.8), bodyMat);
      head.position.y = 9.75;
      head.castShadow = true;
      group.add(head);

      const eye = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.6, 0.3), eyeMat);
      eye.position.set(0, 9.8, 1.4);
      group.add(eye);

      this.robotModel = group;
      this.robotGroup.add(group);

      this.robotGroup.add(this.gun);
      this.gun.position.set(0.85, 5.4, 2.2);
    };

    // Use preloaded GLTF player model if available
    const gltf = this.robotGltf;
    if (gltf) {
      this.robotModel = gltf.scene;
      this.robotModel.scale.setScalar(10.5); // Scaled up to perfectly match the size layout of the scene

      // Custom player skin: Matte flat industrial steel plates (disabling shiny reflections for FPS)
      const chromeSteelMat = new THREE.MeshStandardMaterial({
        color: 0xcccccc, // Steel grey color
        roughness: 1.0,  // Matte texture (no sheen)
        metalness: 0.0   // No metallic reflection
      });

      const darkSteelJointsMat = new THREE.MeshStandardMaterial({
        color: 0x222222, // Dark gunmetal joints
        roughness: 1.0,
        metalness: 0.0
      });

      this.robotModel.traverse(child => {
        if ((child as any).isMesh) {
          const mesh = child as any;
          mesh.castShadow = true;
          mesh.receiveShadow = true;

          const meshName = mesh.name.toLowerCase();
          if (meshName.includes('surface')) {
            mesh.material = chromeSteelMat;
          } else if (meshName.includes('joints')) {
            mesh.material = darkSteelJointsMat;
          }
        }
      });

      this.robotGroup.add(this.robotModel);

      this.mixer = new THREE.AnimationMixer(this.robotModel);
      this.animations = {};

      // Dynamic mapping to uppercase triggers
      gltf.animations.forEach(clip => {
        const name = clip.name.toLowerCase();
        const action = this.mixer.clipAction(clip);

        if (name === 'idle') {
          this.animations['Idle'] = action;
        } else if (name === 'walk') {
          this.animations['Walking'] = action;
        } else if (name === 'run') {
          this.animations['Running'] = action;
        } else if (name === 'agree' || name === 'jump') {
          this.animations['Jump'] = action;
          this.animations['ThumbsUp'] = action;
          this.animations['Sitting'] = action; // Fallback for dying pose
        }

        this.animations[clip.name] = action;
      });

      if (this.animations['Idle']) {
        this.currentAction = this.animations['Idle'];
        this.currentAction.play();
      }

      // Mount weapon directly to character's right hand bone
      const rightHandBone = this.robotModel.getObjectByName('mixamorigRightHand') || this.robotModel.getObjectByName('RightHand');
      if (rightHandBone) {
        rightHandBone.add(this.gun);
        this.gun.position.set(0, 0.15, 0.2);
        this.gun.rotation.set(-Math.PI / 2, Math.PI, 0);
        this.gun.scale.setScalar(1.0 / 10.5);
      } else {
        this.robotGroup.add(this.gun);
        this.gun.position.set(0.85, 5.4, 2.2);
        this.gun.scale.setScalar(1.0);
        this.gun.rotation.set(0, 0, 0);
      }
    } else {
      fallbackRobot();
    }

    const startY = this.getHeightAt(0, 0);
    this.robotGroup.position.set(0, startY, 0);
  }

  fadeToAction(name, duration = 0.25) {
    if (!this.animations || !this.animations[name]) return;
    const nextAction = this.animations[name];
    const prevAction = this.currentAction;

    if (prevAction === nextAction) return;

    this.currentAction = nextAction;

    if (prevAction) {
      prevAction.fadeOut(duration);
    }

    nextAction
      .reset()
      .setEffectiveTimeScale(1)
      .setEffectiveWeight(1)
      .fadeIn(duration)
      .play();
  }

  // ==============================================================
  //  LEVEL UP & RELOADING PROGRESSION
  // ==============================================================
  levelUp() {
    this.gameLevel += 1;
    this.playLevelUpSound();

    const banner = document.getElementById('level-banner');
    if (banner) {
      banner.textContent = `LEVEL ${this.gameLevel} UNLOCKED!`;
      banner.classList.add('show');
      setTimeout(() => banner.classList.remove('show'), 2500);
    }

    const lvlDisp = document.getElementById('level-display');
    if (lvlDisp) {
      lvlDisp.textContent = `Level ${this.gameLevel}`;
    }

    this.health = Math.min(100, this.health + 40);
    // Reset ammo for ALL weapons in the inventory (both clip and reserve)
    for (let i = 0; i < this.inventory.length; i++) {
      const invItem = this.inventory[i];
      if (invItem && invItem.type === 'weapon') {
        invItem.clip = invItem.maxClip || 15;
        invItem.reserve = 999;
      }
    }
    const activeItem = this.inventory[this.activeSlot];
    if (activeItem && activeItem.type === 'weapon') {
      this.maxAmmo = activeItem.maxClip || 15;
      this.ammo = this.maxAmmo;
    }
    this.reserveAmmo = 999;
    this.updateAmmoDisplay();
  }

  collectGroundItem(type: string) {
    if (type.startsWith('ammo_')) {
      const weaponId = type.replace('ammo_', '');
      const weaponItem = this.inventory.find((inv: any) => inv && inv.id === weaponId);
      const ammoAmount = weaponId === 'pistol' ? 30 : (weaponId === 'smg' ? 60 : (weaponId === 'railgun' ? 10 : (weaponId === 'ak47' ? 60 : 5)));
      const weaponName = weaponItem ? weaponItem.name : weaponId.toUpperCase();

      if (weaponItem) {
        weaponItem.reserve = (weaponItem.reserve || 0) + ammoAmount;
        this.updateAmmoDisplay();
      }
      this.points += 50;
      this.showQuickBanner(`+${ammoAmount} ${weaponName.toUpperCase()} AMMO! (+50 PTS)`);
      this.playPickupSound();

    } else if (type === 'nanokit') {
      this.health = Math.min(this.health + 35, 100);
      this.points += 30;
      const hs = document.getElementById('health-status');
      if (hs) {
        hs.textContent = this.health > 50 ? "FINE" : (this.health > 25 ? "CAUTION" : "DANGER");
        hs.className = this.health > 50 ? "fine" : (this.health > 25 ? "caution" : "danger");
      }
      this.showQuickBanner("+35 HEALTH MEDKIT!");
      this.playHealSound();

    } else if (type === 'powercell') {
      this.points += 100;
      this.gameLevel += 1;
      const lvlDisp = document.getElementById('level-display');
      if (lvlDisp) lvlDisp.textContent = `Level ${this.gameLevel}`;
      this.showQuickBanner(`POWER CELL: LEVEL ${this.gameLevel}! (+100 PTS)`);
      this.playPointSound();

    } else if (type === 'crystal') {
      this.points += 150;
      this.showQuickBanner("PLASMA CRYSTAL ARTIFACT! (+150 PTS)");
      this.playPickupSound();
    }

    const scoreDisp = document.getElementById('score-display');
    if (scoreDisp) scoreDisp.textContent = `Points: ${this.points}`;
  }

  showQuickBanner(text: string) {
    const banner = document.getElementById('level-banner');
    if (banner) {
      banner.textContent = text;
      banner.classList.add('show');
      setTimeout(() => banner.classList.remove('show'), 1500);
    }
  }

  reloadWeapon() {
    const activeItem = this.inventory[this.activeSlot];
    if (!activeItem || activeItem.type !== 'weapon') return;

    const currentReserve = activeItem.reserve || 0;
    if (this.isReloading || this.ammo >= this.maxAmmo || currentReserve <= 0) return;

    this.isReloading = true;
    this.reloadTimerStart = Date.now();
    this.playReloadSound();

    const btnReload = document.getElementById('btn-mobile-reload');
    if (btnReload) {
      btnReload.classList.add('reloading');
      setTimeout(() => btnReload.classList.remove('reloading'), 1300);
    }

    const ammoDisp = document.getElementById('ammo-display');
    if (ammoDisp) ammoDisp.textContent = "RELOADING...";

    setTimeout(() => {
      const needed = this.maxAmmo - this.ammo;
      const toLoad = Math.min(needed, activeItem.reserve || 0);
      this.ammo += toLoad;
      activeItem.clip = this.ammo;
      activeItem.reserve = (activeItem.reserve || 0) - toLoad;
      this.reserveAmmo = activeItem.reserve;
      this.isReloading = false;
      this.updateAmmoDisplay();
    }, 1300);
  }

  updateAmmoDisplay() {
    const activeItem = this.inventory[this.activeSlot];
    if (activeItem && activeItem.type === 'weapon') {
      activeItem.clip = this.ammo;
      activeItem.reserve = this.reserveAmmo;
      this.updateInventoryUI();
    }
  }

  updateWeaponSystem() {
    const item = this.inventory[this.activeSlot];
    if (item && item.type === 'weapon') {
      let newLevel = 1;
      if (item.id === 'pistol') newLevel = 1;
      else if (item.id === 'smg') newLevel = 2;
      else if (item.id === 'railgun') newLevel = 3;
      else if (item.id === 'ak47') newLevel = 4;
      else if (item.id === 'rocket') newLevel = 5;

      this.weaponLevel = newLevel;

      // Rebuild the 3D model of the gun to match the new weapon type
      this.rebuildGunMesh();

      const wDisp = document.getElementById('weapon-display');
      if (wDisp) {
        wDisp.textContent = item.name;
      }
    }
  }

  playLaserSound(level: number) {
    if (level === 1) {
      this.playSoundBuffer('pistol', false, 0.55);
    } else if (level === 2) {
      this.playSoundBuffer('smg', false, 0.5);
    } else if (level === 3) {
      this.playRailgunBlastSound();
    } else if (level === 4) {
      this.playSoundBuffer('ak47', false, 0.55);
    } else if (level === 5) {
      this.playRocketSound();
    }
  }

  playRailgunBlastSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;

      // 1. Heavy Sub-bass Impact Pulse (350Hz down to 25Hz)
      const bassOsc = ctx.createOscillator();
      const bassGain = ctx.createGain();
      bassOsc.type = 'sine';
      bassOsc.frequency.setValueAtTime(350, ctx.currentTime);
      bassOsc.frequency.exponentialRampToValueAtTime(25, ctx.currentTime + 0.45);
      bassGain.gain.setValueAtTime(0.7, ctx.currentTime);
      bassGain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
      bassOsc.connect(bassGain);
      bassGain.connect(ctx.destination);
      bassOsc.start(0);

      // 2. High-energy Plasma Burst Crackle
      const bufferSize = ctx.sampleRate * 0.35;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.25));
      }
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;
      const filter = ctx.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(2400, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(200, ctx.currentTime + 0.35);
      filter.Q.value = 3.0;

      const noiseGain = ctx.createGain();
      noiseGain.gain.setValueAtTime(0.65, ctx.currentTime);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);

      noise.connect(filter);
      filter.connect(noiseGain);
      noiseGain.connect(ctx.destination);
      noise.start(0);

      // 3. Play railgun asset audio
      this.playSoundBuffer('railgun', false, 0.65);
    } catch (e) { }
  }

  playRocketSound() {
    // Clean launcher tube discharge sound
    this.playSoundBuffer('pistol', false, 0.60);
  }

  playEnemyDeathSound() {
    this.playSoundBuffer('dying', false, 0.75);
  }

  playExplosionSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();

      const ctx = this.audioCtx;
      const bufferSize = ctx.sampleRate * 0.45;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }

      const noise = ctx.createBufferSource();
      noise.buffer = buffer;

      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(700, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(20, ctx.currentTime + 0.38);

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.4, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.38);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      noise.start();
      noise.stop(ctx.currentTime + 0.38);
    } catch (e) {
      console.warn("Audio blocked:", e);
    }
  }

  playRocketExplosionSound() {
    // Asset audio for rocket triggers on ground or target impact!
    this.playSoundBuffer('rocket', false, 0.75);

    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const bufferSize = ctx.sampleRate * 0.7;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(450, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(15, ctx.currentTime + 0.65);
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.75, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.65);
      noise.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);
      noise.start();
      noise.stop(ctx.currentTime + 0.65);
    } catch (e) {
      console.warn("Audio blocked:", e);
    }
  }

  playReloadSound() {
    this.playSoundBuffer('reload', false, 0.65);
  }

  playLevelUpSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      const freqs = [523.25, 659.25, 783.99, 1046.50];
      freqs.forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now + idx * 0.12);
        gain.gain.setValueAtTime(0.4, now + idx * 0.12);
        gain.gain.exponentialRampToValueAtTime(0.01, now + idx * 0.12 + 0.4);
        osc.connect(gain); gain.connect(ctx.destination);
        osc.start(now + idx * 0.12);
        osc.stop(now + idx * 0.12 + 0.4);
      });
    } catch (e) { }
  }

  playPickupSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(600, now);
      osc.frequency.exponentialRampToValueAtTime(1200, now + 0.15);
      gain.gain.setValueAtTime(0.35, now);
      gain.gain.linearRampToValueAtTime(0.01, now + 0.15);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(now); osc.stop(now + 0.15);
    } catch (e) { }
  }

  playDropSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(400, now);
      osc.frequency.exponentialRampToValueAtTime(150, now + 0.2);
      gain.gain.setValueAtTime(0.3, now);
      gain.gain.linearRampToValueAtTime(0.01, now + 0.2);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(now); osc.stop(now + 0.2);
    } catch (e) { }
  }

  playHealSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(300, now);
      osc.frequency.exponentialRampToValueAtTime(800, now + 0.45);
      gain.gain.setValueAtTime(0.4, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.45);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(now); osc.stop(now + 0.45);
    } catch (e) { }
  }

  playPointSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.exponentialRampToValueAtTime(1760, now + 0.25);
      gain.gain.setValueAtTime(0.35, now);
      gain.gain.linearRampToValueAtTime(0.01, now + 0.25);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(now); osc.stop(now + 0.25);
    } catch (e) { }
  }

  playActionSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(500, now);
      gain.gain.setValueAtTime(0.2, now);
      gain.gain.linearRampToValueAtTime(0.01, now + 0.08);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(now); osc.stop(now + 0.08);
    } catch (e) { }
  }

  triggerMuzzleFlash() {
    if (!this.muzzleLight) {
      this.muzzleLight = new THREE.PointLight(0x00ffff, 0, 30);
      this.scene.add(this.muzzleLight);
    }
    const muzzlePos = new THREE.Vector3();
    if (this.gun) {
      this.gun.getWorldPosition(muzzlePos);
    } else {
      muzzlePos.copy(this.robotGroup.position).y += 6;
    }
    this.muzzleLight.position.copy(muzzlePos);
    this.muzzleLight.intensity = 9.0;

    setTimeout(() => {
      if (this.muzzleLight) this.muzzleLight.intensity = 0;
    }, 60);
  }

  triggerScreenShake(intensity = 0.5) {
    this.screenShake = intensity;
  }

  // ==============================================================
  //  PRIMARY & SECONDARY SHOOTING MECHANICS
  triggerHitMarker(isKill = false) {
    const el = document.getElementById('hit-marker');
    if (!el) return;
    el.className = isKill ? 'kill' : 'active';
    if ((this as any)._hitMarkerTimeout) clearTimeout((this as any)._hitMarkerTimeout);
    (this as any)._hitMarkerTimeout = setTimeout(() => {
      if (el) el.className = '';
    }, 140);
  }

  playHitSound(isKill = false) {
    try {
      const ctx = (this as any).audioCtx || new (window.AudioContext || (window as any).webkitAudioContext)();
      (this as any).audioCtx = ctx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(isKill ? 950 : 600, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(isKill ? 200 : 260, ctx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.25, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.08);
    } catch (e) { }
  }

  playHeadshotDing() {
    try {
      const ctx = (this as any).audioCtx || new (window.AudioContext || (window as any).webkitAudioContext)();
      (this as any).audioCtx = ctx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(1750, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1300, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.35, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.12);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.12);
    } catch (e) { }
  }

  shootPrimary() {
    if (!this.isLocked || this.playerDead) return;
    if (this.isReloading) return;

    if (this.ammo <= 0) {
      this.reloadWeapon();
      return;
    }

    const item = this.inventory[this.activeSlot];
    if (!item) return;

    if (item.id === 'pistol' || item.id === 'rocket') {
      // Single-shot: fire once
      this.shootPrimaryActual();
    } else if (item.id === 'smg' || item.id === 'ak47') {
      // Automatic: first shot immediately, continuous shots/sound in animate() tick
      this.shootPrimaryActual();
      this.lastAimTime = Date.now();
    } else if (item.id === 'railgun') {
      // Railgun: click starts charge in animate() tick
    }
  }

  intersectRayCylinder(ray: THREE.Ray, base: THREE.Vector3, radius: number, height: number): number | null {
    const Ox = ray.origin.x;
    const Oz = ray.origin.z;
    const Dx = ray.direction.x;
    const Dz = ray.direction.z;
    const Cx = base.x;
    const Cz = base.z;

    const A = Dx * Dx + Dz * Dz;
    const B = 2 * (Dx * (Ox - Cx) + Dz * (Oz - Cz));
    const C = (Ox - Cx) * (Ox - Cx) + (Oz - Cz) * (Oz - Cz) - radius * radius;

    if (A === 0) return null; // Ray is parallel to the cylinder axis

    const disc = B * B - 4 * A * C;
    if (disc < 0) return null;

    const sqrtDisc = Math.sqrt(disc);
    const t1 = (-B - sqrtDisc) / (2 * A);
    const t2 = (-B + sqrtDisc) / (2 * A);

    const tCandidates = [t1, t2].filter(t => t > 0).sort((a, b) => a - b);

    for (const t of tCandidates) {
      const Py = ray.origin.y + t * ray.direction.y;
      if (Py >= base.y && Py <= base.y + height) {
        return t; // Entry point distance
      }
    }

    // Top/Bottom cap plane intersections
    if (ray.direction.y !== 0) {
      const tBottom = (base.y - ray.origin.y) / ray.direction.y;
      if (tBottom > 0) {
        const Px = Ox + tBottom * Dx;
        const Pz = Oz + tBottom * Dz;
        if ((Px - Cx) * (Px - Cx) + (Pz - Cz) * (Pz - Cz) <= radius * radius) {
          return tBottom;
        }
      }
      const tTop = (base.y + height - ray.origin.y) / ray.direction.y;
      if (tTop > 0) {
        const Px = Ox + tTop * Dx;
        const Pz = Oz + tTop * Dz;
        if ((Px - Cx) * (Px - Cx) + (Pz - Cz) * (Pz - Cz) <= radius * radius) {
          return tTop;
        }
      }
    }

    return null;
  }

  getCrosshairTargetPoint(): THREE.Vector3 {
    // 1. Find where the camera is looking in the static world (terrain + covers)
    const cameraRaycaster = new THREE.Raycaster();
    cameraRaycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    const cameraRay = cameraRaycaster.ray;

    const staticOccluders = [];
    this.chunks.forEach(chunk => {
      if (chunk.mesh) staticOccluders.push(chunk.mesh);
      if (chunk.coverGroup) staticOccluders.push(chunk.coverGroup);
    });

    const staticHits = cameraRaycaster.intersectObjects(staticOccluders, true);
    let targetWorldPoint = new THREE.Vector3();
    if (staticHits.length > 0) {
      targetWorldPoint.copy(staticHits[0].point);
    } else {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      targetWorldPoint.copy(this.camera.position).addScaledVector(dir, 400);
    }

    // 2. Cast a shooting ray from the player's pivot towards the static target point (Parallax Resolution)
    const pivot = new THREE.Vector3(
      this.robotGroup.position.x,
      this.robotGroup.position.y + 5.8,
      this.robotGroup.position.z
    );
    const shootDir = new THREE.Vector3().subVectors(targetWorldPoint, pivot).normalize();
    const shootRay = new THREE.Ray(pivot, shootDir);

    // 3. Check for enemy cylinder intersections along this shooting ray (in front of the player)
    let closestDist = Infinity;
    const finalHitPoint = targetWorldPoint.clone();

    this.demons.forEach(d => {
      if (!d.isDead) {
        const toEnemy = new THREE.Vector3().subVectors(d.mesh.position, pivot);
        if (toEnemy.dot(shootDir) < -5.0) return; // Skip enemies behind the player

        const scale = d.enemyScale / 3.0;
        const radius = 0.85 * scale;
        const height = 4.5 * scale;
        const dist = this.intersectRayCylinder(shootRay, d.mesh.position, radius, height);
        if (dist !== null && dist < closestDist) {
          closestDist = dist;
          finalHitPoint.copy(pivot).addScaledVector(shootDir, dist);
        }
      }
    });

    return finalHitPoint;
  }

  shootPrimaryActual() {
    if (this.ammo <= 0) return;
    this.ammo--;
    const activeItem = this.inventory[this.activeSlot];
    if (activeItem) activeItem.clip = this.ammo;
    this.updateAmmoDisplay();
    this.lastAimTime = Date.now();

    // Auto-reload immediately if magazine is completely spent
    if (this.ammo <= 0) {
      this.reloadWeapon();
    }

    // Procedural weapon recoil impulse
    this.recoilVelPitch += 0.045 * (1.0 - this.adsBlend * 0.4);
    this.recoilVelYaw += (Math.random() - 0.5) * 0.02;

    this.triggerMuzzleFlash();

    if (this.weaponLevel !== 2 && this.weaponLevel !== 4) {
      this.playLaserSound(this.weaponLevel);
    }

    // 1. Get the world target point (which resolves parallax and finds the closest hit)
    const hitPoint = this.getCrosshairTargetPoint();

    // 2. Re-construct shoot ray parameters to accurately verify and damage the target
    const cameraRaycaster = new THREE.Raycaster();
    cameraRaycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    const cameraRay = cameraRaycaster.ray;

    const staticOccluders = [];
    this.chunks.forEach(chunk => {
      if (chunk.mesh) staticOccluders.push(chunk.mesh);
      if (chunk.coverGroup) staticOccluders.push(chunk.coverGroup);
    });

    const staticHits = cameraRaycaster.intersectObjects(staticOccluders, true);
    let targetWorldPoint = new THREE.Vector3();
    if (staticHits.length > 0) {
      targetWorldPoint.copy(staticHits[0].point);
    } else {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      targetWorldPoint.copy(this.camera.position).addScaledVector(dir, 400);
    }

    const pivot = new THREE.Vector3(
      this.robotGroup.position.x,
      this.robotGroup.position.y + 5.8,
      this.robotGroup.position.z
    );
    const shootDir = new THREE.Vector3().subVectors(targetWorldPoint, pivot).normalize();
    const shootRay = new THREE.Ray(pivot, shootDir);

    let closestEnemy = null;
    let closestEnemyDist = Infinity;

    this.demons.forEach(d => {
      if (!d.isDead) {
        const toEnemy = new THREE.Vector3().subVectors(d.mesh.position, pivot);
        if (toEnemy.dot(shootDir) < -5.0) return; // Skip enemies behind the player

        const scale = d.enemyScale / 3.0;
        const radius = 0.85 * scale;
        const height = 4.5 * scale;
        const dist = this.intersectRayCylinder(shootRay, d.mesh.position, radius, height);
        if (dist !== null && dist < closestEnemyDist) {
          closestEnemyDist = dist;
          closestEnemy = d;
        }
      }
    });

    let hitDemon = false;
    const closestStaticDist = pivot.distanceTo(targetWorldPoint);

    // Hit registration: enemy takes precedence if closer than static obstacle along player's shooting ray
    if (closestEnemy && closestEnemyDist < closestStaticDist) {
      hitPoint.copy(pivot).addScaledVector(shootDir, closestEnemyDist);
      hitDemon = true;

      // Dynamic damage balancing: 5% weapon damage scaling per level to match health growth
      const baseDmg = (this.weaponLevel === 1 ? 22 : (this.weaponLevel === 2 ? 50 : (this.weaponLevel === 3 ? 150 : 70))) * (1 + (this.gameLevel - 1) * 0.05);
      const damage = Math.round(baseDmg);

      closestEnemy.takeDamage(damage, hitPoint);
      this.triggerHitMarker(closestEnemy.isDead || closestEnemy.health <= 0);
      this.createSparks(hitPoint);
    } else {
      this.createTerrainImpactDust(hitPoint);
    }

    const muzzleWorld = new THREE.Vector3();
    if (this.gun) {
      this.gun.getWorldPosition(muzzleWorld);
    } else {
      muzzleWorld.copy(this.robotGroup.position).y += 6;
    }

    if (this.weaponLevel === 5) {
      this.playLaserSound(5);
      const rocket = new PlayerRocket(this.scene, muzzleWorld, hitPoint, this);
      this.playerRockets.push(rocket);

      // Heavy recoil for primary rocket launch
      const adsMult = 1.0 - this.adsBlend * 0.5;
      this.recoilVelPitch += 0.12 * adsMult;
      this.recoilVelYaw += (Math.random() - 0.5) * 0.04 * adsMult;
      return;
    }

    // High quality 3D volumetric tracer beam with white-hot core & outer energy aura
    const tracerColor = this.weaponLevel === 1 ? 0x00d4ff : (this.weaponLevel === 2 ? 0x00ff66 : (this.weaponLevel === 3 ? 0xff2200 : 0xffaa00));
    const tracerRadius = this.weaponLevel === 3 ? 0.28 : (this.weaponLevel === 4 ? 0.14 : (this.weaponLevel === 2 ? 0.12 : 0.10));
    this.createHighQualityTracer(muzzleWorld, hitPoint, tracerColor, tracerRadius);

    if (hitDemon) {
      this.triggerScreenShake(0.15);
    }

    // ---- Procedural recoil impulse ----
    const recoilStrength = this.weaponLevel === 1 ? 0.025 : (this.weaponLevel === 2 ? 0.045 : (this.weaponLevel === 3 ? 0.08 : 0.065));
    const adsMult = 1.0 - this.adsBlend * 0.5;
    this.recoilVelPitch += recoilStrength * adsMult;
    this.recoilVelYaw += (Math.random() - 0.5) * recoilStrength * 0.3 * adsMult;
  }

  shootSecondary() {
    if (!this.isLocked || this.playerDead || this.isReloading) return;

    const now = Date.now();
    if (now - this.lastSecondaryTime < 1100) return;
    this.lastSecondaryTime = now;
    this.lastAimTime = now;

    // Heavy recoil for rocket launcher
    this.recoilVelPitch += 0.12;
    this.recoilVelYaw += (Math.random() - 0.5) * 0.04;

    this.triggerMuzzleFlash();
    this.playRocketSound();

    const muzzleWorld = new THREE.Vector3();
    if (this.gun) {
      this.gun.getWorldPosition(muzzleWorld);
    } else {
      muzzleWorld.copy(this.robotGroup.position).y += 6;
    }

    const targetPoint = this.getCrosshairTargetPoint();

    const rocket = new PlayerRocket(this.scene, muzzleWorld, targetPoint, this);
    this.playerRockets.push(rocket);
  }

  createHighQualityTracer(startPos: THREE.Vector3, endPos: THREE.Vector3, colorHex: number, radius = 0.12) {
    const distance = startPos.distanceTo(endPos);
    if (distance < 0.1) return;

    const group = new THREE.Group();

    // 1. Inner intense white-hot core beam
    const coreGeo = new THREE.CylinderGeometry(radius * 0.35, radius * 0.35, distance, 8);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 1.0,
      depthWrite: false
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    core.position.set(0, distance / 2, 0);
    group.add(core);

    // 2. Outer volumetric glowing energy aura cylinder
    const auraGeo = new THREE.CylinderGeometry(radius, radius, distance, 12);
    const auraMat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.8,
      depthWrite: false
    });
    const aura = new THREE.Mesh(auraGeo, auraMat);
    aura.position.set(0, distance / 2, 0);
    group.add(aura);

    // Position and orient the 3D beam from startPos towards endPos
    group.position.copy(startPos);
    const dir = new THREE.Vector3().subVectors(endPos, startPos).normalize();
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    this.scene.add(group);

    // Smooth volumetric fade out
    let opacity = 1.0;
    const fade = () => {
      opacity -= 0.12;
      if (opacity <= 0) {
        this.scene.remove(group);
        coreGeo.dispose();
        coreMat.dispose();
        auraGeo.dispose();
        auraMat.dispose();
      } else {
        coreMat.opacity = opacity;
        auraMat.opacity = opacity * 0.8;
        requestAnimationFrame(fade);
      }
    };
    fade();
  }

  createSparks(pos) {
    const N = 8;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(N * 3);
    const velocities = [];

    for (let i = 0; i < N; i++) {
      positions[i * 3] = pos.x;
      positions[i * 3 + 1] = pos.y;
      positions[i * 3 + 2] = pos.z;
      velocities.push(new THREE.Vector3(
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20 + 5,
        (Math.random() - 0.5) * 20
      ));
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: this.weaponLevel === 1 ? 0x00ccff :
        (this.weaponLevel === 2 ? 0x00ff66 :
          (this.weaponLevel === 3 ? 0xff3300 :
            (this.weaponLevel === 4 ? 0xffcc00 : 0xff33cc))),
      size: 1.5,
      transparent: true,
      opacity: 1.0
    });

    const sparks = new THREE.Points(geo, mat);
    this.scene.add(sparks);

    let age = 0;
    const update = () => {
      age += 0.016;
      if (age > 0.4) {
        this.scene.remove(sparks);
        geo.dispose();
        mat.dispose();
      } else {
        const arr = geo.attributes.position.array;
        for (let i = 0; i < N; i++) {
          const i3 = i * 3;
          arr[i3] += velocities[i].x * 0.016;
          arr[i3 + 1] += velocities[i].y * 0.016;
          arr[i3 + 2] += velocities[i].z * 0.016;
          velocities[i].y -= 9.8 * 0.016;
        }
        geo.attributes.position.needsUpdate = true;
        mat.opacity = 1.0 - (age / 0.4);
        requestAnimationFrame(update);
      }
    };
    update();
  }

  createTerrainImpactDust(pos) {
    const N = 18;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(N * 3);
    const velocities = [];

    for (let i = 0; i < N; i++) {
      positions[i * 3] = pos.x;
      positions[i * 3 + 1] = pos.y + 0.5;
      positions[i * 3 + 2] = pos.z;
      velocities.push(new THREE.Vector3(
        (Math.random() - 0.5) * 28,
        Math.random() * 18 + 4,
        (Math.random() - 0.5) * 28
      ));
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xc8dce8,
      size: 4.5,
      transparent: true,
      opacity: 0.8
    });

    const dust = new THREE.Points(geo, mat);
    this.scene.add(dust);

    let age = 0;
    const update = () => {
      age += 0.016;
      if (age > 0.5) {
        this.scene.remove(dust);
        geo.dispose();
        mat.dispose();
      } else {
        const arr = geo.attributes.position.array;
        for (let i = 0; i < N; i++) {
          const i3 = i * 3;
          arr[i3] += velocities[i].x * 0.016;
          arr[i3 + 1] += velocities[i].y * 0.016;
          arr[i3 + 2] += velocities[i].z * 0.016;
        }
        geo.attributes.position.needsUpdate = true;
        mat.opacity = 0.8 - (age / 0.5) * 0.8;
        requestAnimationFrame(update);
      }
    };
    update();
  }

  createExplosionParticles(pos) {
    const N = 25;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(N * 3);
    const velocities = [];

    for (let i = 0; i < N; i++) {
      positions[i * 3] = pos.x;
      positions[i * 3 + 1] = pos.y;
      positions[i * 3 + 2] = pos.z;
      velocities.push(new THREE.Vector3(
        (Math.random() - 0.5) * 35,
        (Math.random() - 0.5) * 35 + 8,
        (Math.random() - 0.5) * 35
      ));
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xff3300,
      size: 3.5,
      transparent: true,
      opacity: 1.0
    });

    const explosion = new THREE.Points(geo, mat);
    this.scene.add(explosion);

    let age = 0;
    const update = () => {
      age += 0.016;
      if (age > 0.6) {
        this.scene.remove(explosion);
        geo.dispose();
        mat.dispose();
      } else {
        const arr = geo.attributes.position.array;
        for (let i = 0; i < N; i++) {
          const i3 = i * 3;
          arr[i3] += velocities[i].x * 0.016;
          arr[i3 + 1] += velocities[i].y * 0.016;
          arr[i3 + 2] += velocities[i].z * 0.016;
          velocities[i].y -= 5.0 * 0.016;
        }
        geo.attributes.position.needsUpdate = true;
        mat.opacity = 1.0 - (age / 0.6);
        requestAnimationFrame(update);
      }
    };
    update();
  }

  // ==============================================================
  //  HEALTH SYSTEM (RESIDENT EVIL ECG INTEGRATION)
  // ==============================================================
  setupECG() {
    this.damagePlayer(0);
  }

  updateECG(dt) {
    // Disabled to use heart + progress bar UI instead
  }

  damagePlayer(amount) {
    if (this.playerDead) return;
    this.health = Math.max(this.health - amount, 0);

    const roundedHealth = Math.round(this.health);
    const fillEl = document.getElementById('health-bar-fill');
    const pctEl = document.getElementById('health-percentage');
    const livesEl = document.getElementById('lives-display');

    if (fillEl) fillEl.style.width = `${roundedHealth}%`;
    if (pctEl) pctEl.textContent = `${roundedHealth}%`;

    const stateColor = this.health > 60 ? '#00ff66' : (this.health > 30 ? '#ffcc00' : '#ff3333');
    if (fillEl) fillEl.style.backgroundColor = stateColor;
    if (pctEl) pctEl.style.color = stateColor;

    if (livesEl) {
      livesEl.textContent = this.lives > 0 ? "❤️".repeat(this.lives) : "NONE";
    }

    // Auto-peek docked mobile HUD on taking damage so player always sees their HP
    if (this.isMobile && this.isHudDocked && amount > 0) {
      const infoPanel = document.getElementById('info-panel');
      if (infoPanel) {
        infoPanel.classList.add('is-peeking');
        if (this.hudPeekTimeout) clearTimeout(this.hudPeekTimeout);
        this.hudPeekTimeout = setTimeout(() => {
          if (infoPanel && this.isHudDocked) {
            infoPanel.classList.remove('is-peeking');
          }
        }, 2500);
      }
    }

    if (this.health <= 0) {
      if (this.lives > 1) {
        this.lives--;
        this.health = 100;
        this.playHealSound();
        this.showQuickBanner(`RESPAWNED! LIVES LEFT: ${this.lives}`);

        // Squeezed upward spawn coordinate
        const rp = this.robotGroup.position;
        const spawnY = this.getHeightAt(rp.x, rp.z);
        rp.y = spawnY + 15.0;
        this.velocity.set(0, 0, 0);
        this.onGround = false;

        this.damagePlayer(0);
      } else {
        this.lives = 0;
        if (livesEl) livesEl.textContent = "NONE";
        this.die();
      }
    }
  }

  die() {
    this.playerDead = true;
    this.fadeToAction('Sitting', 0.5);
    this.stopAutomaticFireEffects();

    if (document.exitPointerLock) {
      try { document.exitPointerLock(); } catch (e) { }
    }
    document.body.classList.add('pointer-unlocked');
    document.body.classList.remove('in-game');

    // Update Game Over Modal Stats
    const goModal = document.getElementById('game-over-modal');
    const goLevel = document.getElementById('go-level');
    const goKills = document.getElementById('go-kills');
    const goScore = document.getElementById('go-score');

    if (goLevel) goLevel.textContent = `${this.gameLevel}`;
    if (goKills) goKills.textContent = `${this.kills}`;
    if (goScore) goScore.textContent = `${this.points}`;

    if (goModal) {
      goModal.classList.add('active');
      goModal.style.display = 'flex';
    }

    const hud = document.getElementById('hud');
    if (hud) hud.style.display = 'none';

    this.playPlayerHitSound();
  }

  restartGame() {
    const goModal = document.getElementById('game-over-modal');
    if (goModal) {
      goModal.classList.remove('active');
      goModal.style.display = 'none';
    }

    const bEl = document.getElementById('blocker');
    if (bEl) bEl.style.display = 'none';

    const hudEl = document.getElementById('hud');
    if (hudEl) hudEl.style.display = 'block';

    this.respawn();

    // Sky drop spawn
    const rp = this.robotGroup.position;
    const spawnY = this.getHeightAt(0, 0);
    rp.set(0, spawnY + 40.0, 0);
    this.velocity.set(0, -35.0, 0);
    this.onGround = false;
    this.prevOnGround = false;
    this.prevVelocityY = -35.0;

    if (this.isMobile) {
      this.isLocked = true;
      document.body.classList.remove('pointer-unlocked');
      document.body.classList.add('in-game');
    } else {
      const canvas = this.renderer.domElement;
      if (canvas.requestPointerLock) {
        try { canvas.requestPointerLock(); } catch (e) { }
      }
    }
  }

  pauseGame() {
    if (this.playerDead) return;
    this.isLocked = false;
    if (document.exitPointerLock) {
      try { document.exitPointerLock(); } catch (e) { }
    }
    document.body.classList.add('pointer-unlocked');
    document.body.classList.remove('in-game');
    this.mouseLeftDown = false;
    this.stopAutomaticFireEffects();

    const deployText = document.getElementById('deploy-text');
    if (deployText && !this.playerDead) {
      deployText.textContent = "RESUME";
    }

    const bEl = document.getElementById('blocker');
    const hudEl = document.getElementById('hud');
    if (bEl) bEl.style.display = 'flex';
    if (hudEl) hudEl.style.display = 'none';
  }

  returnToMenu() {
    const goModal = document.getElementById('game-over-modal');
    if (goModal) {
      goModal.classList.remove('active');
      goModal.style.display = 'none';
    }

    this.playerDead = false;
    this.isLocked = false;
    document.body.classList.add('pointer-unlocked');
    document.body.classList.remove('in-game');

    const deployText = document.getElementById('deploy-text');
    if (deployText) deployText.textContent = "DEPLOY";

    const bEl = document.getElementById('blocker');
    if (bEl) bEl.style.display = 'flex';

    const hudEl = document.getElementById('hud');
    if (hudEl) hudEl.style.display = 'none';
  }

  respawn() {
    this.playerDead = false;
    this.gameLevel = 1;
    this.health = 100;
    this.lives = 3;
    this.kills = 0;
    this.points = 0;
    this.weaponLevel = 1;
    this.maxAmmo = 20;
    this.ammo = 20;
    this.isReloading = false;

    // Reset Inventory to initial pistol
    this.inventory = [
      { id: 'pistol', name: 'Pistol', image: '/assets/weapons/skin/pistol.png', type: 'weapon', equipped: true, maxClip: 15, clip: 15, reserve: 999 },
      { id: 'smg', name: 'SMG', image: '/assets/weapons/skin/smg.png', type: 'weapon', equipped: true, maxClip: 30, clip: 30, reserve: 180 },
      { id: 'railgun', name: 'Railgun', image: '/assets/weapons/skin/railgun.png', type: 'weapon', equipped: true, maxClip: 3, clip: 3, reserve: 20 },
      { id: 'ak47', name: 'AK-47', image: '/assets/weapons/skin/ak47.png', type: 'weapon', equipped: true, maxClip: 30, clip: 30, reserve: 150 },
      { id: 'rocket', name: 'Rocket Launcher', image: '/assets/weapons/skin/rocket.png', type: 'weapon', equipped: true, maxClip: 1, clip: 1, reserve: 5 }
    ];
    this.activeSlot = 0;
    this.updateInventoryUI();

    // Reset AAA state
    this.isADS = false;
    this.adsBlend = 0;
    this.mouseRightDown = false;
    this.recoilPitch = 0;
    this.recoilYaw = 0;
    this.recoilVelPitch = 0;
    this.recoilVelYaw = 0;
    this.landingDip = 0;
    this.landingDipVel = 0;
    this.sprintFOVBoost = 0;
    this.currentFOV = this.hipFOV;
    this.screenShake = 0;

    this.updateWeaponSystem();
    this.updateAmmoDisplay();
    this.damagePlayer(0);

    const lvlDisp = document.getElementById('level-display');
    if (lvlDisp) lvlDisp.textContent = "Level 1";

    const scoreDisp = document.getElementById('score-display');
    if (scoreDisp) scoreDisp.textContent = "Points: 0";

    const killsDisp = document.getElementById('kills-display');
    if (killsDisp) killsDisp.textContent = "Kills: 0";

    const blocker = document.getElementById('blocker');
    const header = blocker ? blocker.querySelector('h1') : null;
    const sub = blocker ? blocker.querySelector('.subtitle') : null;
    const deployText = document.getElementById('deploy-text');
    const deployIcon = document.getElementById('deploy-icon-elem');

    if (header) header.innerHTML = 'TACTICAL<br><span class="title-accent">ASSAULT</span>';
    if (sub) sub.textContent = "SELECT YOUR BATTLEFIELD";
    if (deployText) deployText.textContent = "DEPLOY";
    if (deployIcon) deployIcon.textContent = "▶";
    blocker.style.background = "rgba(5, 10, 20, 0.35)";

    const rp = this.robotGroup.position;
    rp.set(0, this.getHeightAt(0, 0) + 40.0, 0);
    this.velocity.set(0, -35.0, 0);
    this.onGround = false;
    this.prevOnGround = false;
    this.prevVelocityY = -35.0;

    this.demons.forEach(d => {
      d.destroy();
    });
    this.demons = [];

    this.enemyProjectiles.forEach(p => p.destroy());
    this.enemyProjectiles = [];

    this.playerRockets.forEach(p => p.destroy());
    this.playerRockets = [];

    this.fadeToAction('Idle', 0.1);
  }

  updateChunks() {
    const rp = this.robotGroup.position;
    const size = this.chunkSize;

    const cx = Math.floor((rp.x + size / 2) / size);
    const cz = Math.floor((rp.z + size / 2) / size);

    this.activeKeys.clear();
    const radius = this.viewRadius;

    let spawnedThisFrame = false;
    let chunksCreated = 0;

    for (let x = -radius; x <= radius; x++) {
      for (let z = -radius; z <= radius; z++) {
        const curX = cx + x;
        const curZ = cz + z;
        const key = `${curX},${curZ}`;
        this.activeKeys.add(key);

        const dist = Math.sqrt(x * x + z * z);
        let lod = 2;
        if (dist < 1.8) lod = 0;
        else if (dist < 3.2) lod = 1;

        if (this.chunks.has(key)) {
          const chunk = this.chunks.get(key);
          if (chunk.lod !== lod && chunksCreated < 1) { // Limit recreation rate
            chunk.destroy();
            chunk.segments = this.getSegmentsForLOD(lod);
            chunk.lod = lod;
            chunk.create();
            chunksCreated++;
          }
        } else if (!spawnedThisFrame && chunksCreated < 1) {
          const segments = this.getSegmentsForLOD(lod);
          const chunk = new TerrainChunk(this.scene, curX, curZ, size, segments, lod, this);
          this.chunks.set(key, chunk);
          spawnedThisFrame = true;
          chunksCreated++;
        }
      }
    }

    for (const [key, chunk] of this.chunks.entries()) {
      if (!this.activeKeys.has(key)) {
        chunk.destroy();
        this.chunks.delete(key);
      }
    }

    this.demons = this.demons.filter(d => {
      const dcx = Math.floor((d.mesh.position.x + size / 2) / size);
      const dcz = Math.floor((d.mesh.position.z + size / 2) / size);
      const key = `${dcx},${dcz}`;
      if (!this.activeKeys.has(key)) {
        d.destroy();
        return false;
      }
      return true;
    });
  }

  getSegmentsForLOD(lod) {
    if (lod === 0) return 32;
    if (lod === 1) return 16;
    return 8;
  }

  // GUARANTEED FRONT SPAWNING: Spawns demons in front of the player (Strictly capped at max 10 alive enemies)
  spawnDemonInFront() {
    if (!this.robotGroup || this.playerDead) return;

    // Hard global limit: strictly max 10 alive enemies at any time across all levels (Zero-GC loop)
    let aliveCount = 0;
    const demonsList = this.demons;
    for (let i = 0; i < demonsList.length; i++) {
      if (!demonsList[i].isDead) aliveCount++;
    }
    const MAX_ALIVE_DEMONS = Math.min(10, this.maxDemons || 10);
    if (aliveCount >= MAX_ALIVE_DEMONS) return;

    const rp = this.robotGroup.position;
    this.camera.getWorldDirection(this._vecForward);
    this._vecForward.y = 0;
    this._vecForward.normalize();

    const baseAngle = Math.atan2(this._vecForward.x, this._vecForward.z);

    // Try candidate spawn positions to ensure no more than 3 enemies cluster together
    let bestX = null;
    let bestZ = null;
    let minNearbyCount = Infinity;

    for (let attempt = 0; attempt < 8; attempt++) {
      // Fan out between ±75° in front of the player
      const angleOffset = (Math.random() - 0.5) * 2.6;
      const spawnAngle = baseAngle + angleOffset;
      const spawnDist = 110 + Math.random() * 80;

      const candX = rp.x + Math.sin(spawnAngle) * spawnDist;
      const candZ = rp.z + Math.cos(spawnAngle) * spawnDist;

      // Count how many alive enemies are within 45 units of this candidate spot (45^2 = 2025)
      let nearbyCount = 0;
      for (let i = 0; i < demonsList.length; i++) {
        const d = demonsList[i];
        if (d.isDead || !d.mesh) continue;
        const dx = d.mesh.position.x - candX;
        const dz = d.mesh.position.z - candZ;
        if (dx * dx + dz * dz < 2025) {
          nearbyCount++;
        }
      }

      // Found an open sector with < 3 enemies in the gang
      if (nearbyCount < 3) {
        bestX = candX;
        bestZ = candZ;
        minNearbyCount = nearbyCount;
        break;
      }

      if (nearbyCount < minNearbyCount) {
        minNearbyCount = nearbyCount;
        bestX = candX;
        bestZ = candZ;
      }
    }

    if (bestX !== null && bestZ !== null && minNearbyCount < 3) {
      const demon = new Demon(this.scene, bestX, bestZ, this);
      this.demons.push(demon);
    }
  }

  // Continuous enemy spawning on a timer — with cooldown after enemy deaths
  updateEnemySpawning(dt) {
    if (this.playerDead || !this.isLocked) return;

    this.enemySpawnTimer += dt;

    // Measured respawn pacing: controlled cooldown so killed enemies are replaced after a few seconds
    const interval = Math.max(2.5, this.enemySpawnInterval || 3.5);

    if (this.enemySpawnTimer >= interval) {
      this.enemySpawnTimer = 0;
      this.spawnDemonInFront();
    }
  }

  createBalloons() {
    this.balloons = [];
  }

  createFogLayers() {
    this.fogLayers = [];
  }

  // ==============================================================
  //  DYNAMIC WEATHER SYSTEM
  // ==============================================================
  initWeatherSystem() {
    this.rainParticles = null;
    this.weatherState = 'clear';
    this.weatherTimer = 0;
    this.isTransitioning = false;
    this.nextWeather = null;
  }

  createRainParticles() {
    // Disabled to eliminate particle rendering overhead & chunkiness
  }

  getWeatherVisuals(state) {
    const wConfig = this.mapConfig.weather[state] || this.mapConfig.weather.clear;
    return {
      skyColor: new THREE.Color(wConfig.skyColor),
      fogColor: new THREE.Color(wConfig.fogColor),
      fogDensity: wConfig.fogDensity,
      sunIntensity: wConfig.sunIntensity,
      ambientIntensity: wConfig.ambientIntensity
    };
  }

  pickNextWeather() {
    return 'clear';
  }

  updateWeather(dt) {
    // Keep weather fixed at clear permanently for optimal performance
    this.weatherState = 'clear';
    this.isTransitioning = false;
    this.nextWeather = null;

    const isMenuVisible = !this.isLocked;
    const clearVisuals = this.getWeatherVisuals('clear');
    if (this.scene.background) this.scene.background.copy(clearVisuals.skyColor);
    if (this.scene.fog) {
      (this.scene.fog as THREE.FogExp2).color.copy(clearVisuals.fogColor);
      (this.scene.fog as THREE.FogExp2).density = clearVisuals.fogDensity * (isMenuVisible ? 0.3 : 1.0);
    }
    if (this.sun) {
      this.sun.intensity = clearVisuals.sunIntensity;
    }
    if (this.ambientLight) {
      this.ambientLight.intensity = clearVisuals.ambientIntensity;
    }
  }

  updateRainParticles(dt) {
    // Disabled
  }

  setGraphicsQuality(quality: string) {
    this.graphicsQuality = quality;
    try {
      localStorage.setItem('tactical_quality', quality);
    } catch (e) { }

    // Native resolution according to device (no blurry downsampling)
    const nativeRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    let shadowSize = 1024;

    if (quality === 'low') {
      shadowSize = 512;
      this.enemySpawnInterval = 4.5;
      this.treeStep = 90;
      this.coverStep = 240;
      this.maxDemons = 6;
    } else if (quality === 'medium') {
      shadowSize = 1024;
      this.enemySpawnInterval = 3.5;
      this.treeStep = 50;
      this.coverStep = 140;
      this.maxDemons = 8;
    } else {
      shadowSize = 2048;
      this.enemySpawnInterval = 3.0;
      this.treeStep = 35;
      this.coverStep = 90;
      this.maxDemons = 10;
    }

    if (this.renderer) {
      this.renderer.setPixelRatio(nativeRatio);
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      if (this.sun && this.sun.shadow) {
        this.sun.shadow.mapSize.width = shadowSize;
        this.sun.shadow.mapSize.height = shadowSize;
        if (this.sun.shadow.map) {
          this.sun.shadow.map.dispose();
          this.sun.shadow.map = null;
        }
      }
    }

    // Update UI buttons
    const qualityBtns = document.querySelectorAll('.quality-btn');
    qualityBtns.forEach(btn => {
      if ((btn as HTMLElement).dataset.quality === quality) {
        btn.classList.add('selected');
      } else {
        btn.classList.remove('selected');
      }
    });
  }

  changeMap(mapId: string) {
    if (!MAP_CONFIGS[mapId]) return;
    this.mapId = mapId;
    this.mapConfig = MAP_CONFIGS[mapId];
    this.MAX_HEIGHT = this.mapConfig.MAX_HEIGHT;

    // Remove existing lights
    if (this.ambientLight) this.scene.remove(this.ambientLight);
    if (this.sun) {
      this.scene.remove(this.sun.target);
      this.scene.remove(this.sun);
    }
    // Remove hemisphere lights
    const toRemove: THREE.Object3D[] = [];
    this.scene.children.forEach(child => {
      if (child instanceof THREE.HemisphereLight) {
        toRemove.push(child);
      }
    });
    toRemove.forEach(l => this.scene.remove(l));

    // Recreate lighting
    this.createLighting();

    // Remove old skybox dome and cylinder
    const skyboxToRemove: THREE.Object3D[] = [];
    this.scene.children.forEach(child => {
      if (child instanceof THREE.Mesh && child.geometry &&
        (child.geometry instanceof THREE.SphereGeometry || child.geometry instanceof THREE.CylinderGeometry) &&
        (child.geometry as any).parameters && ((child.geometry as any).parameters.radius === 4900 || (child.geometry as any).parameters.radiusTop === 4800)) {
        skyboxToRemove.push(child);
      }
    });
    skyboxToRemove.forEach(s => this.scene.remove(s));

    // Recreate skybox
    this.createSkybox();

    // Update fog — use reduced density when blocker (menu) is visible so character is visible
    if (this.scene.fog) {
      (this.scene.fog as THREE.FogExp2).color.setHex(this.mapConfig.fogColor);
      const blockerEl = document.getElementById('blocker');
      const isMenuVisible = blockerEl && blockerEl.style.display !== 'none';
      const densityMultiplier = isMenuVisible ? 0.4 : 1.0;
      (this.scene.fog as THREE.FogExp2).density = this.mapConfig.fogDensity * densityMultiplier;
    }

    // Clear and rebuild fog layers
    if (this.fogLayers) {
      this.fogLayers.forEach(l => this.scene.remove(l));
    }
    this.createFogLayers();

    // Clear and destroy existing chunks
    this.chunks.forEach(chunk => {
      chunk.destroy();
    });
    this.chunks.clear();
    this._chunkQueue = [];

    // Clear old demons/enemies
    this.demons.forEach(demon => {
      if (demon.mesh) this.scene.remove(demon.mesh);
    });
    this.demons = [];

    // Clear old enemy projectiles
    this.enemyProjectiles.forEach(p => {
      if (p.mesh) this.scene.remove(p.mesh);
    });
    this.enemyProjectiles = [];

    // Clear player rockets
    this.playerRockets.forEach(r => {
      if (r.mesh) this.scene.remove(r.mesh);
    });
    this.playerRockets = [];

    // Clear ground items
    this.groundItems.forEach(item => {
      item.destroy();
    });
    this.groundItems = [];

    // Reset player position in the air for a cinematic entrance sky-drop
    const spawnY = this.getHeightAt(0, 0);
    this.robotGroup.position.set(0, spawnY + 40.0, 0);
    this.velocity.set(0, -35.0, 0);
    this.onGround = false;
    this.prevOnGround = false;
    this.prevVelocityY = -35.0;
    this.camYaw = 0;
    this.camPitch = 0.28;
    this.camDist = 22;
    this.camDistTarget = 40;

    // Generate chunks gradually (1 per frame for a cool loading vibe in the background)
    this.updateChunks();

    // Reposition camera directly behind character with downward pitch
    const rp = this.robotGroup.position;
    this.camLookTarget.set(rp.x, rp.y + 6, rp.z);
    this.camera.position.set(rp.x, rp.y + 20, rp.z + this.camDist);
    this.camera.lookAt(this.camLookTarget);

    // Spawn new initial ground items for the new map
    for (let i = 0; i < 25; i++) {
      const rx = (Math.random() - 0.5) * 600;
      const rz = (Math.random() - 0.5) * 600;
      const types = ['crystal', 'nanokit', 'powercell'];
      const type = types[Math.floor(Math.random() * types.length)];
      this.groundItems.push(new GroundItem(this.scene, type, rx, rz, this));
    }

    // Recreate balloons based on the new terrain heights
    if (this.balloons) {
      this.balloons.forEach(b => this.scene.remove(b.group));
    }
    this.createBalloons();

    // Reinitialize weather targets
    const clearWeather = this.mapConfig.weather.clear;
    this._weatherSkyColor.setHex(clearWeather.skyColor);
    this._weatherFogColor.setHex(clearWeather.fogColor);
    this._weatherFogDensity = clearWeather.fogDensity;
    this._weatherSunIntensity = clearWeather.sunIntensity;
    this._weatherAmbientIntensity = clearWeather.ambientIntensity;
    this.weatherState = 'clear';
    this.weatherTimer = 0;
    this.weatherDuration = 50;
    this.isTransitioning = false;
    this.nextWeather = null;

    // Update UI
    const weatherDisp = document.getElementById('weather-display');
    if (weatherDisp) weatherDisp.textContent = "Clear";
  }

  // ==============================================================
  //  CONTROLS & MOUSE LOCK
  // ==============================================================
  setupControls() {
    const canvas = this.renderer.domElement;
    const blocker = document.getElementById('blocker');

    const enterImmersiveGame = () => {
      this.requestGyroPermission();
      const deployTextEl = document.getElementById('deploy-text');
      const isFreshDeploy = deployTextEl && (deployTextEl.textContent === 'DEPLOY' || deployTextEl.textContent === 'RESPAWN');

      if (this.playerDead) {
        this.respawn();
      }

      // Change map on deploy if selected map is different from active map
      const selectedCard = document.querySelector('.map-card.selected');
      const selectedMapId = selectedCard ? (selectedCard as HTMLElement).dataset.map || 'arctic' : 'arctic';
      if (selectedMapId !== this.mapId) {
        this.changeMap(selectedMapId);
      } else if (isFreshDeploy && this.robotGroup) {
        // If map didn't change but it is a fresh deploy, trigger the sky drop entrance
        const spawnY = this.getHeightAt(0, 0);
        this.robotGroup.position.set(0, spawnY + 40.0, 0);
        this.velocity.set(0, -35.0, 0);
        this.onGround = false;
        this.prevOnGround = false;
        this.prevVelocityY = -35.0;
      }

      // Enter gameplay state immediately
      this.isLocked = true;
      document.body.classList.remove('pointer-unlocked');
      document.body.classList.add('in-game');

      const bEl = document.getElementById('blocker');
      const hudEl = document.getElementById('hud');
      if (bEl) bEl.style.display = 'none';
      if (hudEl) hudEl.style.display = 'block';

      const goModal = document.getElementById('game-over-modal');
      if (goModal) {
        goModal.classList.remove('active');
        goModal.style.display = 'none';
      }

      if (this.isMobile) {
        document.body.classList.add('is-mobile');
        try {
          const orientation = screen.orientation as any;
          if (orientation && typeof orientation.lock === 'function') {
            orientation.lock('landscape').catch(() => { });
          }
        } catch (e) { }
      } else {
        // Desktop: Request Fullscreen and Pointer Lock
        if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
          document.documentElement.requestFullscreen().catch(() => { });
        }
        if (canvas.requestPointerLock) {
          try {
            canvas.requestPointerLock();
          } catch (e) { }
        }
      }
    };

    // Desktop Pointer Lock State Change Listeners
    document.addEventListener('pointerlockchange', () => {
      if (document.pointerLockElement === canvas) {
        this.isLocked = true;
        document.body.classList.remove('pointer-unlocked');
        document.body.classList.add('in-game');
        if (blocker) blocker.style.display = 'none';
        const hud = document.getElementById('hud');
        if (hud) hud.style.display = 'block';
      } else {
        if (!this.playerDead && !this.isMobile) {
          this.pauseGame();
        }
      }
    });

    document.addEventListener('pointerlockerror', (e) => {
      console.warn('Pointer lock error/cancelled:', e);
    });

    // ==============================================================
    //  DESKTOP KEYBOARD, MOUSE LOOK & WEAPON CONTROLS
    // ==============================================================
    window.addEventListener('keydown', (e) => {
      // Allow pause/resume toggling via P or Escape
      if (e.code === 'KeyP' || e.code === 'Escape') {
        if (this.isLocked) {
          this.pauseGame();
        }
        return;
      }

      if (!this.isLocked || this.playerDead) return;

      this.keys[e.code] = true;

      // Weapon reload
      if (e.code === 'KeyR') {
        this.reloadWeapon();
      } else if (e.code === 'KeyQ') {
        // Quick rocket hotkey
        this.useInventoryItem(4);
      } else if (e.code === 'Digit1') {
        this.useInventoryItem(0);
      } else if (e.code === 'Digit2') {
        this.useInventoryItem(1);
      } else if (e.code === 'Digit3') {
        this.useInventoryItem(2);
      } else if (e.code === 'Digit4') {
        this.useInventoryItem(3);
      } else if (e.code === 'Digit5') {
        this.useInventoryItem(4);
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keys[e.code] = false;
    });

    // Mouse Look (First-Person Camera Movement)
    window.addEventListener('mousemove', (e) => {
      if (!this.isLocked || this.playerDead) return;

      const baseSensitivity = 0.0022;
      const sensitivity = baseSensitivity * (1.0 - this.adsBlend * 0.4);

      this.camYaw -= e.movementX * sensitivity;
      this.camPitch = THREE.MathUtils.clamp(
        this.camPitch - e.movementY * sensitivity,
        -1.35,
        1.35
      );
    });

    // Mouse Fire & ADS
    window.addEventListener('mousedown', (e) => {
      if (!this.isLocked || this.playerDead) return;
      const now = performance.now();
      if (now - lastFireTouchTime < 100) return; // Prevent synthetic mouse event right after touch fire button

      if (e.button === 0) {
        // Left Click: Fire
        this.mouseLeftDown = true;
        this.shootPrimaryActual();
      } else if (e.button === 2) {
        // Right Click: ADS
        this.mouseRightDown = true;
        this.isADS = true;
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) {
        this.mouseLeftDown = false;
        this.railgunFiredThisPress = false;
        this.stopAutomaticFireEffects();
      } else if (e.button === 2) {
        this.mouseRightDown = false;
        this.isADS = false;
      }
    });

    window.addEventListener('contextmenu', (e) => {
      // Prevent browser right-click menu during ADS
      e.preventDefault();
    });

    // Mouse Wheel: Cycle Weapons
    window.addEventListener('wheel', (e) => {
      if (!this.isLocked || this.playerDead) return;
      const now = Date.now();
      if (now - this.lastWheelTime < 150) return;
      this.lastWheelTime = now;

      if (e.deltaY > 0) {
        this.switchWeapon(1);
      } else if (e.deltaY < 0) {
        this.switchWeapon(-1);
      }
    });


    const unlockAudioContext = () => {
      try {
        if (!this.audioCtx) {
          this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        if (this.audioCtx && this.audioCtx.state === 'suspended') {
          this.audioCtx.resume().catch(() => { });
        }
      } catch (e) { }
    };
    window.addEventListener('click', unlockAudioContext, { capture: true, passive: true });
    window.addEventListener('touchstart', unlockAudioContext, { capture: true, passive: true });
    window.addEventListener('pointerdown', unlockAudioContext, { capture: true, passive: true });

    const isIOSDevice = () => {
      return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
        (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    };

    const tryFullscreen = () => {
      // Scroll to hide address bar on iOS / mobile browsers
      try {
        window.scrollTo(0, 1);
      } catch (e) { }

      if (!document.fullscreenElement) {
        if (document.documentElement.requestFullscreen) {
          document.documentElement.requestFullscreen().catch(() => { });
        } else if ((document.documentElement as any).webkitRequestFullscreen) {
          try { (document.documentElement as any).webkitRequestFullscreen(); } catch (e) { }
        } else if ((document.documentElement as any).webkitEnterFullscreen) {
          try { (document.documentElement as any).webkitEnterFullscreen(); } catch (e) { }
        }

        try {
          const orientation = screen.orientation as any;
          if (orientation && typeof orientation.lock === 'function') {
            orientation.lock('landscape').catch(() => { });
          }
        } catch (e) { }
      }
    };

    const toggleFullscreen = () => {
      if (isIOSDevice() && !document.documentElement.requestFullscreen) {
        try { window.scrollTo(0, 1); } catch (e) { }
        this.showQuickBanner("iOS: Add to Home Screen (Share > Add to Home) for borderless fullscreen!");
        return;
      }

      if (!document.fullscreenElement) {
        tryFullscreen();
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen().catch(() => { });
        } else if ((document as any).webkitExitFullscreen) {
          try { (document as any).webkitExitFullscreen(); } catch (e) { }
        }
      }
    };

    // Fullscreen Toggle Button Listener
    const fsBtn = document.getElementById('btn-fullscreen-toggle');
    if (fsBtn) {
      const handleFsToggle = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        toggleFullscreen();
      };
      fsBtn.addEventListener('click', handleFsToggle);
      fsBtn.addEventListener('touchend', handleFsToggle);
    }

    // Mobile Stats Panel Dock / Collapse Toggle Listener
    const dockBtn = document.getElementById('btn-hud-dock');
    const infoPanel = document.getElementById('info-panel');
    if (dockBtn && infoPanel) {
      const handleDockToggle = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.isHudDocked = !this.isHudDocked;
        if (this.isHudDocked) {
          infoPanel.classList.add('is-docked');
          dockBtn.classList.add('hud-hidden');
        } else {
          infoPanel.classList.remove('is-docked');
          infoPanel.classList.remove('is-peeking');
          dockBtn.classList.remove('hud-hidden');
        }
      };
      dockBtn.addEventListener('click', handleDockToggle);
      dockBtn.addEventListener('touchend', handleDockToggle);
    }

    // Game Over Action Buttons
    const restartBtn = document.getElementById('btn-restart-game');
    if (restartBtn) {
      const handleRestart = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.restartGame();
      };
      restartBtn.addEventListener('click', handleRestart);
      restartBtn.addEventListener('touchend', handleRestart);
    }

    const menuBtn = document.getElementById('btn-menu-game');
    if (menuBtn) {
      const handleMenu = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.returnToMenu();
      };
      menuBtn.addEventListener('click', handleMenu);
      menuBtn.addEventListener('touchend', handleMenu);
    }

    // Deploy Button (ONLY trigger to enter gameplay)
    const deployBtn = document.getElementById('deploy-btn-content');
    if (deployBtn) {
      const handleDeployAction = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        tryFullscreen();
        enterImmersiveGame();
      };
      deployBtn.addEventListener('click', handleDeployAction);
      deployBtn.addEventListener('touchend', handleDeployAction);
    }

    // Blocker Background Taps: Triggers Fullscreen request WITHOUT starting the game
    if (blocker) {
      const onBlockerBgInteraction = (e: Event) => {
        const target = e.target as HTMLElement;
        if (target.closest('#deploy-btn-content') || target.closest('.map-card') || target.closest('.quality-btn') || target.closest('#btn-fullscreen-toggle')) {
          return;
        }
        tryFullscreen();
      };
      blocker.addEventListener('click', onBlockerBgInteraction);
      blocker.addEventListener('touchend', onBlockerBgInteraction);
    }

    let joystickTouchId: number | null = null;
    let joystickStartX = 0;
    let joystickStartY = 0;
    const joystickBase = document.getElementById('mobile-joystick-base');
    const joystickKnob = document.getElementById('mobile-joystick-knob');
    const joystickZone = document.getElementById('mobile-joystick-zone');
    const lookZone = document.getElementById('mobile-look-zone');

    const updateJoystick = (currentX: number, currentY: number) => {
      if (!joystickBase || !joystickKnob) return;
      const maxRadius = Math.max(35, joystickBase.clientWidth / 2);

      let dx = currentX - joystickStartX;
      let dy = currentY - joystickStartY;
      const dist = Math.hypot(dx, dy);

      if (dist > maxRadius) {
        dx = (dx / dist) * maxRadius;
        dy = (dy / dist) * maxRadius;
      }

      // Move knob relative to center
      joystickKnob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;

      // Movement threshold
      const normalizedX = dx / maxRadius;
      const normalizedY = dy / maxRadius;
      const threshold = 0.22;

      this.keys['KeyW'] = normalizedY < -threshold;
      this.keys['KeyS'] = normalizedY > threshold;
      this.keys['KeyA'] = normalizedX < -threshold;
      this.keys['KeyD'] = normalizedX > threshold;
    };

    if (joystickZone && lookZone) {
      joystickZone.addEventListener('touchstart', (e) => {
        if (!this.isLocked || this.playerDead) return;
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (joystickTouchId === null) {
            joystickTouchId = t.identifier;
            joystickStartX = t.clientX;
            joystickStartY = t.clientY;

            if (joystickBase && joystickKnob) {
              joystickBase.style.display = 'block';
              joystickBase.style.left = `${t.clientX}px`;
              joystickBase.style.top = `${t.clientY}px`;
              joystickBase.style.bottom = 'auto';
              joystickBase.style.transform = 'translate(-50%, -50%)';
              joystickKnob.style.transform = 'translate(-50%, -50%)';
            }
          }
        }
      }, { passive: false });

      joystickZone.addEventListener('touchmove', (e) => {
        if (!this.isLocked || this.playerDead) return;
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (t.identifier === joystickTouchId) {
            updateJoystick(t.clientX, t.clientY);
          }
        }
      }, { passive: false });

      const handleJoystickEnd = (e: TouchEvent) => {
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (t.identifier === joystickTouchId) {
            joystickTouchId = null;
            if (joystickBase && joystickKnob) {
              joystickBase.style.display = 'none';
              joystickKnob.style.transform = 'translate(-50%, -50%)';
            }
            this.keys['KeyW'] = false;
            this.keys['KeyS'] = false;
            this.keys['KeyA'] = false;
            this.keys['KeyD'] = false;
          }
        }
      };

      joystickZone.addEventListener('touchend', handleJoystickEnd);
      joystickZone.addEventListener('touchcancel', handleJoystickEnd);

      // Look Zone
      let lookTouchId: number | null = null;
      let lastLookX = 0;
      let lastLookY = 0;

      lookZone.addEventListener('touchstart', (e) => {
        if (!this.isLocked || this.playerDead) return;
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (lookTouchId === null) {
            lookTouchId = t.identifier;
            lastLookX = t.clientX;
            lastLookY = t.clientY;
          }
        }
      }, { passive: false });

      lookZone.addEventListener('touchmove', (e) => {
        if (!this.isLocked || this.playerDead) return;
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (t.identifier === lookTouchId) {
            const dx = t.clientX - lastLookX;
            const dy = t.clientY - lastLookY;
            lastLookX = t.clientX;
            lastLookY = t.clientY;

            const baseSensitivity = 0.005;
            const sensitivity = baseSensitivity * (1.0 - this.adsBlend * 0.3);
            this.camYaw -= dx * sensitivity;
            this.camPitch = THREE.MathUtils.clamp(
              this.camPitch - dy * sensitivity,
              -1.35,
              1.35
            );
          }
        }
      }, { passive: false });

      const handleLookEnd = (e: TouchEvent) => {
        for (let i = 0; i < e.changedTouches.length; i++) {
          const t = e.changedTouches[i];
          if (t.identifier === lookTouchId) {
            lookTouchId = null;
          }
        }
      };

      lookZone.addEventListener('touchend', handleLookEnd);
      lookZone.addEventListener('touchcancel', handleLookEnd);
    }

    // Action Buttons - Instant Zero-Delay Capture Touch Listeners
    const btnFire = document.getElementById('btn-mobile-fire');
    const btnFireLeft = document.getElementById('btn-mobile-fire-left');

    let lastFireTouchTime = 0;
    const stopAllFiring = () => {
      this.mouseLeftDown = false;
      this.railgunFiredThisPress = false;
      if (btnFire) btnFire.classList.remove('active');
      if (btnFireLeft) btnFireLeft.classList.remove('active');
      this.stopAutomaticFireEffects();
    };

    const bindFireButton = (btn: HTMLElement | null) => {
      if (!btn) return;
      const onFireStart = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        const now = performance.now();
        if (now - lastFireTouchTime < 50) return; // Debounce dual touch events
        lastFireTouchTime = now;

        if (this.isLocked && !this.playerDead) {
          this.mouseLeftDown = true;
          btn.classList.add('active');
          this.shootPrimary(); // Fires single-shot immediately or triggers auto-fire / charging
        }
      };

      const onFireEnd = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        stopAllFiring();
      };

      btn.addEventListener('touchstart', onFireStart, { capture: true, passive: false });
      btn.addEventListener('pointerdown', onFireStart, { capture: true });
      btn.addEventListener('touchend', onFireEnd, { capture: true, passive: false });
      btn.addEventListener('touchcancel', onFireEnd, { capture: true, passive: false });
      btn.addEventListener('pointerup', onFireEnd, { capture: true });
      btn.addEventListener('pointercancel', onFireEnd, { capture: true });
    };

    bindFireButton(btnFire);
    bindFireButton(btnFireLeft);

    // Global window-level safety to ensure touch release is always caught (even if finger slides off button)
    window.addEventListener('touchend', () => {
      if (this.mouseLeftDown) stopAllFiring();
    }, { passive: true });
    window.addEventListener('touchcancel', () => {
      if (this.mouseLeftDown) stopAllFiring();
    }, { passive: true });
    window.addEventListener('pointerup', (e) => {
      if (e.pointerType === 'touch' && this.mouseLeftDown) stopAllFiring();
    });

    const btnSprint = document.getElementById('btn-mobile-sprint');
    if (btnSprint) {
      let sprintTouchStartTime = 0;
      let lastHandledTime = 0;
      let wasLockedBeforeTouch = false;

      const setSprintState = (active: boolean) => {
        this.keys['ShiftLeft'] = active;
        this.isSprintLocked = active;
        if (active) {
          btnSprint.classList.add('active');
        } else {
          btnSprint.classList.remove('active');
        }
      };

      const handleSprintStart = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        if (!this.isLocked || this.playerDead) return;
        const now = Date.now();
        if (now - lastHandledTime < 100) return; // Prevent double trigger from simultaneous touchstart + pointerdown
        lastHandledTime = now;
        sprintTouchStartTime = now;
        wasLockedBeforeTouch = this.isSprintLocked;

        // If sprint was OFF, activate sprint immediately on touch down
        if (!wasLockedBeforeTouch) {
          setSprintState(true);
        }
      };

      const handleSprintEnd = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        if (!this.isLocked || this.playerDead) return;
        const now = Date.now();
        const duration = now - sprintTouchStartTime;

        if (duration < 300) {
          // Quick Tap (< 300ms): Toggle sprint lock
          if (wasLockedBeforeTouch) {
            // Was ON -> toggle to OFF
            setSprintState(false);
          } else {
            // Was OFF -> keep sprint locked ON
            setSprintState(true);
          }
        } else {
          // Long press / Hold-to-sprint (> 300ms): Stop sprint upon release
          setSprintState(false);
        }
      };

      btnSprint.addEventListener('touchstart', handleSprintStart, { capture: true, passive: false });
      btnSprint.addEventListener('touchend', handleSprintEnd, { capture: true, passive: false });
      btnSprint.addEventListener('touchcancel', handleSprintEnd, { capture: true, passive: false });
      btnSprint.addEventListener('pointerdown', handleSprintStart, { capture: true });
      btnSprint.addEventListener('pointerup', handleSprintEnd, { capture: true });
      btnSprint.addEventListener('pointercancel', handleSprintEnd, { capture: true });
    }

    const btnAds = document.getElementById('btn-mobile-ads');
    if (btnAds) {
      const onAdsStart = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.isLocked && !this.playerDead) {
          this.mouseRightDown = true;
          this.isADS = true;
          btnAds.classList.add('active');
        }
      };
      const onAdsEnd = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.mouseRightDown = false;
        this.isADS = false;
        btnAds.classList.remove('active');
      };

      btnAds.addEventListener('touchstart', onAdsStart, { capture: true, passive: false });
      btnAds.addEventListener('pointerdown', onAdsStart, { capture: true });
      btnAds.addEventListener('touchend', onAdsEnd, { capture: true, passive: false });
      btnAds.addEventListener('touchcancel', onAdsEnd, { capture: true, passive: false });
      btnAds.addEventListener('pointerup', onAdsEnd, { capture: true });
    }

    const btnJump = document.getElementById('btn-mobile-jump');
    if (btnJump) {
      const onJumpStart = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.isLocked && !this.playerDead) {
          this.keys['Space'] = true;
          btnJump.classList.add('active');
        }
      };
      const onJumpEnd = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.keys['Space'] = false;
        btnJump.classList.remove('active');
      };

      btnJump.addEventListener('touchstart', onJumpStart, { capture: true, passive: false });
      btnJump.addEventListener('pointerdown', onJumpStart, { capture: true });
      btnJump.addEventListener('touchend', onJumpEnd, { capture: true, passive: false });
      btnJump.addEventListener('touchcancel', onJumpEnd, { capture: true, passive: false });
      btnJump.addEventListener('pointerup', onJumpEnd, { capture: true });
    }

    const btnReload = document.getElementById('btn-mobile-reload');
    if (btnReload) {
      const onReloadStart = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.isLocked && !this.playerDead) {
          btnReload.classList.add('active');
          this.reloadWeapon();
          setTimeout(() => btnReload.classList.remove('active'), 250);
        }
      };

      btnReload.addEventListener('touchstart', onReloadStart, { capture: true, passive: false });
      btnReload.addEventListener('pointerdown', onReloadStart, { capture: true });
    }

    // Settings / Pause Button
    const btnPause = document.getElementById('btn-mobile-pause');
    if (btnPause) {
      const handlePauseAction = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.pauseGame();
      };
      btnPause.addEventListener('touchstart', handlePauseAction, { capture: true, passive: false });
      btnPause.addEventListener('pointerdown', handlePauseAction, { capture: true });
      btnPause.addEventListener('click', handlePauseAction, { capture: true });
    }

    // Inventory Event Delegation (Guarantees weapon/item selection on touch & click)
    const inventoryHud = document.getElementById('inventory-hud');
    if (inventoryHud) {
      const handleInventorySelect = (e) => {
        if (!this.isLocked || this.playerDead) return;
        const target = (e.target as HTMLElement).closest('.inventory-slot') as HTMLElement;
        if (target && target.dataset && target.dataset.index !== undefined) {
          e.preventDefault();
          e.stopPropagation();
          const index = parseInt(target.dataset.index, 10);
          if (!isNaN(index)) {
            this.useInventoryItem(index);
          }
        }
      };
      inventoryHud.addEventListener('touchstart', handleInventorySelect, { passive: false });
      inventoryHud.addEventListener('pointerdown', handleInventorySelect);
      inventoryHud.addEventListener('click', handleInventorySelect);
    }

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      if (this.composer) {
        this.composer.setSize(window.innerWidth, window.innerHeight);
      }
    });

    this.setupGyroscope();
  }

  // ==============================================================
  //  GYROSCOPE AIMING (PUBG Mobile / BGMI Motion Aiming & Settings)
  // ==============================================================
  gyroEnabled: boolean = true;
  gyroMultiplier: number = 0.65;
  gyroListening: boolean = false;
  gyroPermissionGranted: boolean = false;
  gyroLastTime: number = 0;
  gyroSmoothPitchRate: number = 0;
  gyroSmoothYawRate: number = 0;
  gyroPrevBeta: number | null = null;
  gyroPrevGamma: number | null = null;
  gyroPrevAlpha: number | null = null;

  requestGyroPermission() {
    if (this.gyroPermissionGranted) return;
    try {
      if (typeof (DeviceMotionEvent as any)?.requestPermission === 'function') {
        (DeviceMotionEvent as any).requestPermission()
          .then((state: string) => {
            if (state === 'granted') {
              this.gyroPermissionGranted = true;
              this.startGyroscope();
            }
          })
          .catch(() => { });
      } else if (typeof (DeviceOrientationEvent as any)?.requestPermission === 'function') {
        (DeviceOrientationEvent as any).requestPermission()
          .then((state: string) => {
            if (state === 'granted') {
              this.gyroPermissionGranted = true;
              this.startGyroscope();
            }
          })
          .catch(() => { });
      } else {
        this.gyroPermissionGranted = true;
        this.startGyroscope();
      }
    } catch (e) { }
  }

  getScreenAngle(): number {
    if (window.screen && window.screen.orientation && typeof window.screen.orientation.angle === 'number') {
      return window.screen.orientation.angle;
    }
    if (typeof window.orientation === 'number') {
      return window.orientation;
    }
    return 90; // Standard Landscape
  }

  startGyroscope() {
    if (this.gyroListening) return;
    this.gyroListening = true;

    window.addEventListener('devicemotion', (e: DeviceMotionEvent) => {
      if (!this.gyroEnabled || !this.isLocked || this.playerDead) return;
      const r = e.rotationRate;
      if (!r || (r.alpha === null && r.beta === null && r.gamma === null)) return;

      const now = performance.now();
      const dt = this.gyroLastTime ? Math.min((now - this.gyroLastTime) / 1000, 0.05) : 0.016;
      this.gyroLastTime = now;

      let rawPitchRate = 0; // deg/sec (up/down tilt)
      let rawYawRate = 0;   // deg/sec (left/right rotation)

      const angle = this.getScreenAngle();

      if (angle === 90) {
        // Landscape Primary (Standard phone hold with top of phone to the left)
        rawPitchRate = -(r.beta || 0);
        rawYawRate = (r.gamma || 0) + (r.alpha || 0) * 0.35;
      } else if (angle === 270 || angle === -90) {
        // Landscape Reverse (Flipped 180 degrees)
        rawPitchRate = (r.beta || 0);
        rawYawRate = -(r.gamma || 0) + (r.alpha || 0) * 0.35;
      } else {
        // Portrait or other orientations
        rawPitchRate = (r.beta || 0);
        rawYawRate = (r.gamma || 0);
      }

      // Micro-jitter noise filter
      const deadzone = 0.25; // deg/sec threshold
      if (Math.abs(rawPitchRate) < deadzone) rawPitchRate = 0;
      if (Math.abs(rawYawRate) < deadzone) rawYawRate = 0;

      // Exponential Moving Average Smoothing (Snappy response without trembling)
      this.gyroSmoothPitchRate = this.gyroSmoothPitchRate * 0.35 + rawPitchRate * 0.65;
      this.gyroSmoothYawRate = this.gyroSmoothYawRate * 0.35 + rawYawRate * 0.65;

      // Sensitivities calibrated for PUBG Mobile / BGMI motion aiming
      const isADS = this.isADS;
      const gyroSensitivity = isADS ? 0.00095 : 0.00078;

      const dYaw = this.gyroSmoothYawRate * dt * gyroSensitivity * 57.2958 * this.gyroMultiplier;
      const dPitch = this.gyroSmoothPitchRate * dt * gyroSensitivity * 57.2958 * this.gyroMultiplier;

      this.camYaw += dYaw;
      this.camPitch = THREE.MathUtils.clamp(
        this.camPitch + dPitch,
        -1.35,
        1.35
      );
    }, { passive: true });

    // Fallback for devices where devicemotion.rotationRate is not exposed
    window.addEventListener('deviceorientation', (e: DeviceOrientationEvent) => {
      if (this.gyroLastTime && (performance.now() - this.gyroLastTime < 100)) return;
      if (!this.gyroEnabled || !this.isLocked || this.playerDead) return;

      const gamma = e.gamma;
      const beta = e.beta;
      const alpha = e.alpha;
      if (gamma === null || beta === null || alpha === null) return;

      if (this.gyroPrevGamma === null || this.gyroPrevBeta === null || this.gyroPrevAlpha === null) {
        this.gyroPrevGamma = gamma;
        this.gyroPrevBeta = beta;
        this.gyroPrevAlpha = alpha;
        return;
      }

      const dBeta = beta - this.gyroPrevBeta;
      const dGamma = gamma - this.gyroPrevGamma;

      this.gyroPrevGamma = gamma;
      this.gyroPrevBeta = beta;
      this.gyroPrevAlpha = alpha;

      const angle = this.getScreenAngle();
      let deltaPitch = 0;
      let deltaYaw = 0;

      if (angle === 90) {
        deltaPitch = -dBeta;
        deltaYaw = dGamma;
      } else if (angle === 270 || angle === -90) {
        deltaPitch = dBeta;
        deltaYaw = -dGamma;
      } else {
        deltaPitch = dBeta;
        deltaYaw = dGamma;
      }

      if (Math.abs(deltaPitch) > 40 || Math.abs(deltaYaw) > 40) return;

      const isADS = this.isADS;
      const scale = isADS ? 0.045 : 0.035;

      this.camYaw += (deltaYaw * (Math.PI / 180)) * scale * this.gyroMultiplier;
      this.camPitch = THREE.MathUtils.clamp(
        this.camPitch + (deltaPitch * (Math.PI / 180)) * scale * this.gyroMultiplier,
        -1.35,
        1.35
      );
    }, { passive: true });
  }

  setupGyroscope() {
    // Load saved preferences
    try {
      const savedEnabled = localStorage.getItem('tactical_gyro_enabled');
      if (savedEnabled !== null) {
        this.gyroEnabled = savedEnabled === 'true';
      }
      const savedSens = localStorage.getItem('tactical_gyro_sens');
      if (savedSens !== null) {
        const parsed = parseFloat(savedSens);
        if (!isNaN(parsed) && parsed >= 0.2 && parsed <= 3.0) {
          this.gyroMultiplier = parsed;
        }
      }
    } catch (e) { }

    // Initialize UI Elements
    const toggleBtn = document.getElementById('gyro-toggle-btn');
    const toggleStatus = document.getElementById('gyro-toggle-status');
    const slider = document.getElementById('gyro-sensitivity-slider') as HTMLInputElement | null;
    const valueBadge = document.getElementById('gyro-sensitivity-value');
    const sliderContainer = document.getElementById('gyro-slider-container');

    const updateToggleUI = () => {
      if (toggleBtn && toggleStatus) {
        if (this.gyroEnabled) {
          toggleBtn.classList.add('active');
          toggleStatus.textContent = 'ON';
          if (sliderContainer) sliderContainer.classList.remove('disabled');
        } else {
          toggleBtn.classList.remove('active');
          toggleStatus.textContent = 'OFF';
          if (sliderContainer) sliderContainer.classList.add('disabled');
        }
      }
    };

    const updateSliderUI = () => {
      const pct = Math.round(this.gyroMultiplier * 100);
      if (slider) slider.value = String(pct);
      if (valueBadge) valueBadge.textContent = `${pct}%`;
    };

    updateToggleUI();
    updateSliderUI();

    if (toggleBtn) {
      const handleToggle = (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.gyroEnabled = !this.gyroEnabled;
        updateToggleUI();
        try {
          localStorage.setItem('tactical_gyro_enabled', String(this.gyroEnabled));
        } catch (err) { }
      };
      toggleBtn.addEventListener('click', handleToggle);
      toggleBtn.addEventListener('touchend', handleToggle);
    }

    if (slider) {
      const handleSliderChange = () => {
        const val = parseInt(slider.value, 10);
        this.gyroMultiplier = Math.max(0.2, Math.min(3.0, val / 100));
        if (valueBadge) valueBadge.textContent = `${val}%`;
        try {
          localStorage.setItem('tactical_gyro_sens', String(this.gyroMultiplier));
        } catch (err) { }
      };
      slider.addEventListener('input', handleSliderChange);
      slider.addEventListener('change', handleSliderChange);
    }

    // Sensor and Device Capability Verification
    const hasGyroSensor = ('DeviceOrientationEvent' in window) || ('DeviceMotionEvent' in window);
    const isTouch = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0) || window.matchMedia('(pointer: coarse)').matches;
    const gyroSection = document.getElementById('gyro-settings-section');
    if (gyroSection) {
      if (!isTouch || !hasGyroSensor) {
        gyroSection.style.display = 'none';
      } else {
        gyroSection.style.display = 'block';
      }
    }

    this.startGyroscope();
  }

  setupPostProcessing() {
    // Disabled to prevent frame-buffer swapping, bloom, and shader passes overhead
    this.composer = null;
  }

  setupMinimap() {
    const canvas = document.getElementById('minimap') as HTMLCanvasElement;
    if (!canvas) return;
    const sz = 150;
    canvas.width = sz; canvas.height = sz;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(sz, sz);
    for (let y = 0; y < sz; y++) {
      for (let x = 0; x < sz; x++) {
        const wx = (x / sz - 0.5) * this.chunkSize * 6;
        const wz = (y / sz - 0.5) * this.chunkSize * 6;
        const h = this.getHeightAt(wx, wz) / this.MAX_HEIGHT;
        const br = Math.floor(h * 180 + 55);
        const idx = (y * sz + x) * 4;
        img.data[idx] = Math.floor(br * 0.75);
        img.data[idx + 1] = Math.floor(br * 0.82);
        img.data[idx + 2] = br;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    this.mmCtx = ctx; this.mmSize = sz; this.mmBase = ctx.getImageData(0, 0, sz, sz);
  }

  updateMinimap() {
    if (!this.mmCtx) return;
    // Throttle minimap to 2fps — no need to redraw 60 times per second
    const now = performance.now();
    if (now - this._lastMinimapTime < 500) return;
    this._lastMinimapTime = now;

    const ctx = this.mmCtx, sz = this.mmSize;
    ctx.putImageData(this.mmBase, 0, 0);

    const rp = this.robotGroup.position;
    const mapWorldSize = this.chunkSize * 6;
    const px = ((rp.x / mapWorldSize) + 0.5) * sz;
    const pz = ((rp.z / mapWorldSize) + 0.5) * sz;

    const ang = this.robotGroup.rotation.y;

    ctx.fillStyle = 'rgba(0,220,255,0.3)';
    ctx.beginPath();
    ctx.moveTo(px, pz);
    const mapAng = -(ang - Math.PI / 2);
    ctx.arc(px, pz, 18, mapAng - 0.4, mapAng + 0.4);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#00ddff';
    ctx.beginPath();
    ctx.arc(px, pz, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  updatePlayer(dt) {
    if (!this.isLocked || !this.robotGroup) return;

    if (!this.playerDead && this.health < 100) {
      this.damagePlayer(-1.5 * dt);
    }

    // ---- ADS blend (smooth 0→1 transition) ----
    const adsTarget = (this.isADS && !this.playerDead && !this.isReloading) ? 1.0 : 0.0;
    this.adsBlend = THREE.MathUtils.lerp(this.adsBlend, adsTarget, dt * 8.0);
    if (Math.abs(this.adsBlend - adsTarget) < 0.005) this.adsBlend = adsTarget;

    const sprinting = !!this.keys['ShiftLeft'] || !!this.isSprintLocked;

    // Calculate physical speed: ADS sprint is faster than ADS walk, but slower than hip normal/sprint
    const baseSpeed = sprinting ? this.SPRINT_SPEED : this.MOVE_SPEED;
    let speed = baseSpeed;
    if (this.adsBlend > 0.01) {
      const adsSpeedMult = sprinting ? 0.50 : 0.60;
      speed = THREE.MathUtils.lerp(baseSpeed, baseSpeed * adsSpeedMult, this.adsBlend);
    }

    this._vecForward.set(Math.sin(this.camYaw), 0, Math.cos(this.camYaw)).normalize();
    this._vecRight.set(-this._vecForward.z, 0, this._vecForward.x);

    this.direction.set(0, 0, 0);
    if (this.keys['KeyW']) this.direction.add(this._vecForward);
    if (this.keys['KeyS']) this.direction.sub(this._vecForward);
    if (this.keys['KeyD']) this.direction.add(this._vecRight);
    if (this.keys['KeyA']) this.direction.sub(this._vecRight);

    const isMoving = this.direction.lengthSq() > 0;
    if (isMoving) this.direction.normalize();

    this.velocity.x = this.direction.x * speed;
    this.velocity.z = this.direction.z * speed;

    // Track pre-jump velocity for landing impact
    this.prevVelocityY = this.velocity.y;

    if (this.keys['Space'] && this.onGround && !this.playerDead) {
      this.velocity.y = this.JUMP_FORCE;
      this.onGround = false;
      this.fadeToAction('Jump', 0.15);
    }

    this.velocity.y += this.GRAVITY * dt;

    const rp = this.robotGroup.position;
    rp.x += this.velocity.x * dt;
    rp.z += this.velocity.z * dt;
    rp.y += this.velocity.y * dt;

    // ---- Solid Physical Cover Collision (Glaciers, Pillars, Walls, Spires) ----
    this.chunks.forEach(chunk => {
      if (chunk.coverObstacles) {
        for (let i = 0; i < chunk.coverObstacles.length; i++) {
          const obs = chunk.coverObstacles[i];
          const dx = rp.x - obs.x;
          const dz = rp.z - obs.z;
          const distSq = dx * dx + dz * dz;
          const minDist = obs.radius + 2.0;
          if (distSq < minDist * minDist) {
            const dist = Math.sqrt(distSq);
            if (dist > 0.001) {
              const push = minDist - dist;
              rp.x += (dx / dist) * push;
              rp.z += (dz / dist) * push;
            }
          }
        }
      }
    });

    const groundY = this.getHeightAt(rp.x, rp.z);
    if (rp.y <= groundY) {
      rp.y = groundY;

      // ---- Landing impact detection ----
      if (!this.prevOnGround) {
        const fallVel = Math.abs(this.prevVelocityY);
        if (fallVel > 12) {
          this.landingBendIntensity = Math.min(fallVel / 40.0, 1.2);

          if (fallVel > 30) {
            const impactStrength = Math.min(fallVel / 150, 1.0);
            this.landingDipVel = -impactStrength * 8.0;
            this.triggerScreenShake(impactStrength * 0.3);
          }
        }
      }

      this.velocity.y = 0;
      this.onGround = true;
    }
    this.prevOnGround = this.onGround;

    // ---- AAA Smooth rotation with exponential smoothing ----
    if (!this.playerDead) {
      let targetRot;
      if (this.adsBlend > 0.3) {
        // During ADS: character faces camera direction (aim where you look)
        targetRot = this.camYaw;
      } else if (isMoving) {
        // Normal: character faces movement direction
        targetRot = Math.atan2(this.direction.x, this.direction.z);
      } else {
        targetRot = null; // idle: no rotation change
      }

      if (targetRot !== null) {
        let diff = targetRot - this.robotGroup.rotation.y;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        // Use exponential decay for smooth acceleration/deceleration
        // Faster rotation during ADS for responsive aim tracking
        const turnSpeed = this.adsBlend > 0.3 ? 0.25 : (sprinting ? 0.12 : 0.15);
        const smoothFactor = 1.0 - Math.pow(1.0 - turnSpeed, dt * 60);
        this.robotGroup.rotation.y += diff * smoothFactor;
      }
    }

    if (this.onGround && !this.playerDead) {
      const isExpressiveFallback = this.robotGltf && (this.robotGltf as any).isRobotExpressive;
      if (this.adsBlend > 0.3 && isExpressiveFallback) {
        // Fallback model uses ThumbsUp as its aim pose
        this.fadeToAction('ThumbsUp', 0.3);
      } else if (isMoving) {
        if (sprinting) {
          this.fadeToAction('Running', 0.2);
        } else {
          this.fadeToAction('Walking', 0.2);
        }
      } else {
        this.fadeToAction('Idle', 0.25);
      }

      // Adjust animation mixer rate to match physical velocity, preventing foot sliding
      if (this.mixer) {
        if (!this.jumpBlend || this.jumpBlend < 0.1) {
          const currentSpeedRatio = speed / this.MOVE_SPEED;
          this.mixer.timeScale = currentSpeedRatio * (sprinting ? 0.85 : 1.0);
        }
      }
    }

    // ---- Procedural recoil spring physics ----
    // Spring: F = -kx - cv (stiffness * displacement - damping * velocity)
    const recoilK = this.recoilRecoverySpeed;
    this.recoilVelPitch += (-recoilK * this.recoilPitch) * dt * 60;
    this.recoilVelYaw += (-recoilK * this.recoilYaw) * dt * 60;
    this.recoilVelPitch *= this.recoilDamping;
    this.recoilVelYaw *= this.recoilDamping;
    this.recoilPitch += this.recoilVelPitch * dt;
    this.recoilYaw += this.recoilVelYaw * dt;

    // ---- Landing dip spring ----
    this.landingDipVel += (-15.0 * this.landingDip) * dt * 60; // spring back
    this.landingDipVel *= 0.88; // damping
    this.landingDip += this.landingDipVel * dt;

    // ---- Sprint FOV boost (smooth ramp) ----
    const targetSprintFOV = (sprinting && isMoving) ? 7.0 : 0.0;
    this.sprintFOVBoost = THREE.MathUtils.lerp(this.sprintFOVBoost, targetSprintFOV, dt * 4.0);

    // ---- Footstep camera bob (synced to walk/run cycle) ----
    if (isMoving && this.onGround && !this.playerDead) {
      const bobFreq = sprinting ? 14.0 : 9.0;
      this.footstepPhase += dt * bobFreq;
    } else {
      // Smoothly settle back
      this.footstepPhase = THREE.MathUtils.lerp(this.footstepPhase, Math.round(this.footstepPhase / (Math.PI * 2)) * Math.PI * 2, dt * 5.0);
    }

    // ---- Weapon stance, sway, & reload tilt animation ----
    if (this.gun) {
      const now = Date.now();
      const isAiming = this.adsBlend > 0.1;

      // ADS: move gun directly to chest center for iron-sights look
      // targetX: 0.85 (hip-fire right shoulder) -> -0.05 (aim center chest)
      const targetX = THREE.MathUtils.lerp(0.85, -0.05, this.adsBlend);
      // targetY: 5.4 (hip-fire lower chest) -> 5.95 (aim eye-level)
      const targetY = THREE.MathUtils.lerp(5.4, 5.95, this.adsBlend);
      // targetZ: 2.2 (hip-fire forward) -> 3.6 (aim closer/tighter)
      const targetZ = THREE.MathUtils.lerp(2.2, 3.6, this.adsBlend);

      // Walking / Sprinting bob (reduced during ADS)
      const bobDampen = 1.0 - this.adsBlend * 0.85;
      const bobSpeed = sprinting ? 14 : 10;
      const bobScale = (sprinting ? 0.08 : 0.04) * bobDampen;
      let bobX = isMoving ? Math.sin(this.clock.elapsedTime * bobSpeed) * bobScale : 0;
      let bobY = isMoving ? Math.abs(Math.cos(this.clock.elapsedTime * bobSpeed)) * bobScale : 0;

      // Idle breathing sway (subtle when not ADS, very subtle during ADS)
      const breathScale = 0.012 * (1.0 - this.adsBlend * 0.7);
      let breathX = !isMoving ? Math.sin(this.clock.elapsedTime * 1.5) * breathScale : 0;
      let breathY = !isMoving ? Math.cos(this.clock.elapsedTime * 2.1) * breathScale * 0.6 : 0;

      let chargeJitterX = 0;
      let chargeJitterY = 0;
      let chargeJitterZ = 0;

      // ---- Railgun Charge Energy Surge & Repulsion Jitter ----
      if (this.isChargingRailgun) {
        // Freeze movement/swaying animation during energy charge
        bobX = 0;
        bobY = 0;
        breathX = 0;
        breathY = 0;

        const progress = Math.min(1.0, (this.railgunChargeTime || 0) / 1.5);
        // Energy pulse glow: surges up to 18.0 intensity with high-frequency energy flickering
        if (this.gunGlowMat) {
          const pulse = Math.sin(now * 0.08) * (2.0 + progress * 5.0);
          this.gunGlowMat.emissiveIntensity = 3.5 + progress * 14.0 + pulse;
        }

        // Repulsive energy micro-vibration (jitter) that increases as charge nears 100%
        const jitterAmount = progress * 0.045;
        chargeJitterX = (Math.random() - 0.5) * jitterAmount;
        chargeJitterY = (Math.random() - 0.5) * jitterAmount;
        chargeJitterZ = (Math.random() - 0.5) * jitterAmount;
      } else if (this.gunGlowMat) {
        // Reset glow intensity to normal when not charging
        this.gunGlowMat.emissiveIntensity = 3.5;
      }

      let reloadRotX = 0;
      let reloadRotZ = 0;
      let reloadOffsetY = 0;

      const activeItem = this.inventory[this.activeSlot];
      if (this.isReloading && activeItem?.id !== 'railgun') {
        // Multi-stage reload animation for standard weapons (Railgun stays steady without dipping down)
        const rT = (now - (this.reloadTimerStart || now)) / 1300;
        reloadRotX = Math.sin(rT * Math.PI) * 0.9; // dip down
        reloadRotZ = Math.sin(rT * Math.PI * 2) * 0.5; // twist
        reloadOffsetY = -Math.sin(rT * Math.PI) * 0.6; // move off screen slightly
      }

      // Apply recoil offset to gun position
      const recoilGunY = this.recoilPitch * 0.15;
      const recoilGunZ = -Math.abs(this.recoilPitch) * 0.08; // kick back

      // Ensure gun scale and position follow the correct parent node (hand bone or fallback chest)
      const rightHandBone = this.robotModel ? (this.robotModel.getObjectByName('mixamorigRightHand') || this.robotModel.getObjectByName('RightHand')) : null;
      if (rightHandBone && this.gun.parent === rightHandBone) {
        const scaleFactor = 1.0 / 10.5;
        this.gun.scale.setScalar(scaleFactor);

        // Keep it firmly gripped in hand palm with subtle recoil kickback & reload offsets
        this.gun.position.x = THREE.MathUtils.lerp(this.gun.position.x, chargeJitterX, 0.18);
        this.gun.position.y = THREE.MathUtils.lerp(this.gun.position.y, 0.15 + (reloadOffsetY + recoilGunY) * scaleFactor, 0.18) + chargeJitterY;
        this.gun.position.z = THREE.MathUtils.lerp(this.gun.position.z, 0.2 + recoilGunZ * scaleFactor, 0.18) + chargeJitterZ;

        this.gun.rotation.x = THREE.MathUtils.lerp(this.gun.rotation.x, -Math.PI / 2 + reloadRotX + this.recoilPitch * 0.3, 0.18);
        this.gun.rotation.y = THREE.MathUtils.lerp(this.gun.rotation.y, Math.PI, 0.18);
        this.gun.rotation.z = THREE.MathUtils.lerp(this.gun.rotation.z, reloadRotZ + this.recoilYaw * 0.2, 0.18);
      } else {
        // Fallback positioning for original expressive robot model
        this.gun.scale.setScalar(1.0);
        this.gun.position.x = THREE.MathUtils.lerp(this.gun.position.x, targetX + bobX + breathX + this.recoilYaw * 0.05, 0.18) + chargeJitterX;
        this.gun.position.y = THREE.MathUtils.lerp(this.gun.position.y, targetY + bobY + breathY + reloadOffsetY + recoilGunY, 0.18) + chargeJitterY;
        this.gun.position.z = THREE.MathUtils.lerp(this.gun.position.z, targetZ + recoilGunZ, 0.18) + chargeJitterZ;

        this.gun.rotation.x = THREE.MathUtils.lerp(this.gun.rotation.x, reloadRotX + this.recoilPitch * 0.3, 0.18);
        this.gun.rotation.y = 0;
        this.gun.rotation.z = THREE.MathUtils.lerp(this.gun.rotation.z, reloadRotZ + this.recoilYaw * 0.2, 0.18);
      }
    }

    // ---- Dynamic Right Arm Aim Stance Override ----
    if (this.robotModel && !this.playerDead) {
      const rightArm = this.robotModel.getObjectByName('mixamorigRightArm') || this.robotModel.getObjectByName('RightArm');
      const rightForeArm = this.robotModel.getObjectByName('mixamorigRightForeArm') || this.robotModel.getObjectByName('RightForeArm');

      if (rightArm) {
        // Raise arm horizontally (Z = PI/2) and point it straight forward (X = -PI/2), tracking camera pitch
        const targetRotX = -Math.PI / 2 - this.camPitch;
        const targetRotY = 0.0;
        const targetRotZ = Math.PI / 2;

        rightArm.rotation.x = THREE.MathUtils.lerp(rightArm.rotation.x, targetRotX, this.adsBlend);
        rightArm.rotation.y = THREE.MathUtils.lerp(rightArm.rotation.y, targetRotY, this.adsBlend);
        rightArm.rotation.z = THREE.MathUtils.lerp(rightArm.rotation.z, targetRotZ, this.adsBlend);

        // Force Three.js quaternion to update from the new Euler rotation
        rightArm.quaternion.setFromEuler(rightArm.rotation);
      }

      if (rightForeArm) {
        // Straighten the elbow forearm during aiming so weapon points directly forward
        rightForeArm.rotation.x = THREE.MathUtils.lerp(rightForeArm.rotation.x, 0.0, this.adsBlend);
        rightForeArm.rotation.y = THREE.MathUtils.lerp(rightForeArm.rotation.y, 0.0, this.adsBlend);
        rightForeArm.rotation.z = THREE.MathUtils.lerp(rightForeArm.rotation.z, 0.0, this.adsBlend);
        rightForeArm.quaternion.setFromEuler(rightForeArm.rotation);
      }
    }

    // ---- Dynamic Procedural Jumping & Landing Stance ----
    if (this.robotModel && !this.playerDead) {
      const targetJump = this.onGround ? 0.0 : 1.0;
      this.jumpBlend = THREE.MathUtils.lerp(this.jumpBlend || 0.0, targetJump, dt * 10.0);

      // Decay landing impact shock absorb bend smoothly
      this.landingBendIntensity = THREE.MathUtils.lerp(this.landingBendIntensity || 0.0, 0.0, dt * 8.0);
      const activeOverrideBlend = Math.max(this.jumpBlend, this.landingBendIntensity);

      // Freeze mixer updates mid-air or during deep landing absorption
      if (this.mixer) {
        this.mixer.timeScale = THREE.MathUtils.lerp(1.0, 0.02, activeOverrideBlend);
      }

      if (activeOverrideBlend > 0.01) {
        const leftUpLeg = this.robotModel.getObjectByName('mixamorigLeftUpLeg') || this.robotModel.getObjectByName('LeftUpLeg');
        const rightUpLeg = this.robotModel.getObjectByName('mixamorigRightUpLeg') || this.robotModel.getObjectByName('RightUpLeg');
        const leftLeg = this.robotModel.getObjectByName('mixamorigLeftLeg') || this.robotModel.getObjectByName('LeftLeg');
        const rightLeg = this.robotModel.getObjectByName('mixamorigRightLeg') || this.robotModel.getObjectByName('RightLeg');
        const leftArm = this.robotModel.getObjectByName('mixamorigLeftArm') || this.robotModel.getObjectByName('LeftArm');
        const rightArm = this.robotModel.getObjectByName('mixamorigRightArm') || this.robotModel.getObjectByName('RightArm');

        // 1. Calculate base jump pose targets
        let leftThighTarget = -0.7;
        let rightThighTarget = -0.7;
        let leftKneeTarget = 1.3;
        let rightKneeTarget = 1.3;

        // Differentiate: Symmetrical tuck (stationary) vs Running leap split (in motion)
        const horizSpeed = Math.sqrt(this.velocity.x * this.velocity.x + this.velocity.z * this.velocity.z);
        if (horizSpeed > 1.0) {
          // Dynamic running/walking leap split: alternate leading leg based on the current step cycle phase
          const leadingLegLeft = Math.sin(this.footstepPhase) > 0;
          if (leadingLegLeft) {
            leftThighTarget = -1.1;
            rightThighTarget = 0.45;
            leftKneeTarget = 0.95;
            rightKneeTarget = 0.5;
          } else {
            leftThighTarget = 0.45;
            rightThighTarget = -1.1;
            leftKneeTarget = 0.5;
            rightKneeTarget = 0.95;
          }
        }

        // 2. Add landing impact shock absorber bend (knees flex deep on landing)
        const impactOffset = this.landingBendIntensity || 0.0;
        leftThighTarget += -0.55 * impactOffset;
        rightThighTarget += -0.55 * impactOffset;
        leftKneeTarget += 1.1 * impactOffset;
        rightKneeTarget += 1.1 * impactOffset;

        // 3. Smoothly interpolate and apply rotations to the bone quaternions
        if (leftUpLeg) {
          leftUpLeg.rotation.x = THREE.MathUtils.lerp(leftUpLeg.rotation.x, leftThighTarget, activeOverrideBlend);
          leftUpLeg.quaternion.setFromEuler(leftUpLeg.rotation);
        }
        if (rightUpLeg) {
          rightUpLeg.rotation.x = THREE.MathUtils.lerp(rightUpLeg.rotation.x, rightThighTarget, activeOverrideBlend);
          rightUpLeg.quaternion.setFromEuler(rightUpLeg.rotation);
        }
        if (leftLeg) {
          leftLeg.rotation.x = THREE.MathUtils.lerp(leftLeg.rotation.x, leftKneeTarget, activeOverrideBlend);
          leftLeg.quaternion.setFromEuler(leftLeg.rotation);
        }
        if (rightLeg) {
          rightLeg.rotation.x = THREE.MathUtils.lerp(rightLeg.rotation.x, rightKneeTarget, activeOverrideBlend);
          rightLeg.quaternion.setFromEuler(rightLeg.rotation);
        }

        // Raise the left arm high above the head for dynamic gravity balance (positive Z goes up)
        if (leftArm) {
          leftArm.rotation.x = THREE.MathUtils.lerp(leftArm.rotation.x, -0.3, activeOverrideBlend);
          leftArm.rotation.z = THREE.MathUtils.lerp(leftArm.rotation.z, 2.0, activeOverrideBlend);
          leftArm.quaternion.setFromEuler(leftArm.rotation);
        }

        // Raise the right arm high above the head for balance ONLY if they are not actively aiming (ADS) (negative Z goes up)
        if (rightArm && this.adsBlend < 0.1) {
          rightArm.rotation.x = THREE.MathUtils.lerp(rightArm.rotation.x, -0.3, activeOverrideBlend);
          rightArm.rotation.z = THREE.MathUtils.lerp(rightArm.rotation.z, -2.0, activeOverrideBlend);
          rightArm.quaternion.setFromEuler(rightArm.rotation);
        }
      }
    }

    this.updateThirdPersonCamera(dt);

    const sunPos = (this.mapConfig as any).sunPosition || { x: 400, y: 550, z: 250 };
    this.sun.position.set(rp.x + sunPos.x, sunPos.y, rp.z + sunPos.z);
    this.sun.target.position.copy(rp);
    this.sun.target.updateMatrixWorld();
  }

  updateThirdPersonCamera(dt) {
    const rp = this.robotGroup.position;

    // ---- Dynamic Character Scale (Larger on menu screen / ESC pause) ----
    const isMenuVisible = !this.isLocked;
    if (this.robotModel) {
      const scaleFactor = (this.robotGltf && (this.robotGltf as any).isXbot) ? 3.5 : 1.0;
      // Use uniform in-game scale to prevent character height from crossing screen bounds in menu
      const targetScale = 3.0 * scaleFactor;
      this.robotModel.scale.setScalar(targetScale);
    }

    // ---- Dynamic FOV (ADS zoom + sprint boost) ----
    const targetFOV = THREE.MathUtils.lerp(this.hipFOV, this.adsFOV, this.adsBlend) + this.sprintFOVBoost;
    this.currentFOV = THREE.MathUtils.lerp(this.currentFOV, targetFOV, dt * 10.0);
    this.camera.fov = this.currentFOV;
    this.camera.updateProjectionMatrix();

    // ---- Camera distance (smooth spring between hip, menu, and ADS) ----
    const menuTargetDist = isMenuVisible ? 20 : this.camDistTarget;
    const targetDist = THREE.MathUtils.lerp(menuTargetDist, this.adsCamDist, this.adsBlend);
    const distDiff = targetDist - this.camDist;
    this._camSpringVel += distDiff * 15.0 * dt;
    this._camSpringVel *= 0.85; // damping
    this.camDist += this._camSpringVel;

    // ---- Over-the-shoulder offset (centered during menu, offset during gameplay) ----
    const baseShoulderX = isMenuVisible ? 0 : 2.2;
    const shoulderOffsetX = THREE.MathUtils.lerp(baseShoulderX, this.adsCamOffset.x, this.adsBlend);
    const shoulderOffsetY = THREE.MathUtils.lerp(0.8, this.adsCamOffset.y, this.adsBlend);

    const effectivePitch = this.camPitch + this.recoilPitch * 0.5;
    const cosPitch = Math.cos(effectivePitch);
    const sinPitch = Math.sin(effectivePitch);
    const sinYaw = Math.sin(this.camYaw);
    const cosYaw = Math.cos(this.camYaw);

    // Camera view direction vector (points out into world where crosshair is aimed)
    const forwardX = sinYaw * cosPitch;
    const forwardY = sinPitch;
    const forwardZ = cosYaw * cosPitch;

    // Perpendicular right vector
    const rightX = -cosYaw;
    const rightZ = sinYaw;

    // Pivot origin (player torso/head height)
    const pivotX = rp.x;
    const pivotY = rp.y + 5.8 + this.landingDip;
    const pivotZ = rp.z;

    // Desired camera position: offset behind player pivot along view vector + shoulder offset
    this._vecDesiredPos.set(
      pivotX - forwardX * this.camDist + rightX * shoulderOffsetX,
      pivotY - forwardY * this.camDist + shoulderOffsetY,
      pivotZ - forwardZ * this.camDist + rightZ * shoulderOffsetX
    );

    // Look target: points forward into world ahead of camera along view vector
    this._vecLookTarget.set(
      this._vecDesiredPos.x + forwardX * 100,
      this._vecDesiredPos.y + forwardY * 100,
      this._vecDesiredPos.z + forwardZ * 100
    );

    // Terrain collision height check
    const terrainHeight = this.getHeightAt(this._vecDesiredPos.x, this._vecDesiredPos.z);
    if (this._vecDesiredPos.y < terrainHeight + 2.5) {
      this._vecDesiredPos.y = terrainHeight + 2.5;
    }

    // Footstep camera bob
    const bobScale = this.adsBlend < 0.3 ? 0.12 : 0.02;
    const footBobY = Math.sin(this.footstepPhase) * bobScale;
    const footBobX = Math.cos(this.footstepPhase * 0.5) * bobScale * 0.4;
    this._vecDesiredPos.y += footBobY;
    this._vecDesiredPos.x += footBobX * rightX;
    this._vecDesiredPos.z += footBobX * rightZ;

    // Smooth camera movement lerp
    const isMoving = this.direction.lengthSq() > 0;
    let cameraLerp = isMoving ? 0.18 : 0.12;
    if (this.adsBlend > 0.3) cameraLerp = 0.35; // Snappy responsiveness during ADS
    const smoothFactor = 1.0 - Math.pow(1.0 - cameraLerp, dt * 60);

    this.camera.position.lerp(this._vecDesiredPos, smoothFactor);
    this.camLookTarget.lerp(this._vecLookTarget, smoothFactor * 1.5);
    this.camera.lookAt(this.camLookTarget);

    // Keep 3D Sun Orb anchored high in sky relative to camera position
    if (this.sunMesh && this._sunDir) {
      this.sunMesh.position.copy(this.camera.position).addScaledVector(this._sunDir, 1800);
    }
    if (this.sun && this._sunDir) {
      this.sun.position.copy(this.robotGroup.position).addScaledVector(this._sunDir, 600);
      this.sun.target.position.copy(this.robotGroup.position);
      this.sun.target.updateMatrixWorld();
    }

    // Screen shake
    if (this.screenShake > 0) {
      this.camera.position.x += (Math.random() - 0.5) * this.screenShake * 3.0;
      this.camera.position.y += (Math.random() - 0.5) * this.screenShake * 3.0;
      this.screenShake = Math.max(0, this.screenShake - dt * 3.0);
    }
  }

  updateBalloons(dt) {
    // Disabled to reduce objects in sky and improve performance
  }

  updateFogLayers() {
    // Disabled to save fill-rate overdraw overhead
  }

  updateHUD(dt) {
    this.fpsSamples.push(1 / dt);
    if (this.fpsSamples.length > 30) this.fpsSamples.shift();
    const avgFps = Math.round(this.fpsSamples.reduce((a, b) => a + b, 0) / this.fpsSamples.length);

    const scoreDisp = document.getElementById('score-display');
    const killsDisp = document.getElementById('kills-display');
    const fps = document.getElementById('fps');
    if (scoreDisp) scoreDisp.textContent = `Points: ${this.points}`;
    if (killsDisp) killsDisp.textContent = `Kills: ${this.kills}`;
    if (fps) fps.textContent = `FPS: ${avgFps}`;

    // Weather display
    const weatherDisp = document.getElementById('weather-display');
    if (weatherDisp) {
      const current = this.isTransitioning ? this.nextWeather : this.weatherState;
      const icons = { clear: '☀️ Clear', rain: '🌧️ Rain', snow: '🌨️ Snow' };
      weatherDisp.textContent = this.isTransitioning
        ? `${icons[this.weatherState] || 'Clear'} → ${icons[current] || 'Clear'}`
        : (icons[current] || '☀️ Clear');
    }
  }

  updateVirtualCursor(dt) {
    const cursorEl = document.getElementById('virtual-cursor');

    // Show crosshair only during gameplay
    if (cursorEl) {
      cursorEl.style.display = this.isLocked && !this.playerDead ? 'block' : 'none';
    }

    if (!this.isLocked || this.playerDead) return;

    // 1. Find where the camera is looking in the static world
    this._raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    const cameraRay = this._raycaster.ray;

    const staticOccluders = [];
    this.chunks.forEach(chunk => {
      if (chunk.mesh) staticOccluders.push(chunk.mesh);
      if (chunk.coverGroup) staticOccluders.push(chunk.coverGroup);
    });

    const staticHits = this._raycaster.intersectObjects(staticOccluders, true);
    let targetWorldPoint = new THREE.Vector3();
    if (staticHits.length > 0) {
      targetWorldPoint.copy(staticHits[0].point);
    } else {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      targetWorldPoint.copy(this.camera.position).addScaledVector(dir, 400);
    }

    // 2. Construct shooting ray from player pivot
    const pivot = new THREE.Vector3(
      this.robotGroup.position.x,
      this.robotGroup.position.y + 5.8,
      this.robotGroup.position.z
    );
    const shootDir = new THREE.Vector3().subVectors(targetWorldPoint, pivot).normalize();
    const shootRay = new THREE.Ray(pivot, shootDir);

    // 3. Check for enemy cylinder intersections along this shooting ray (in front of the player)
    let closestEnemy = null;
    let closestEnemyDist = Infinity;

    this.demons.forEach(d => {
      if (!d.isDead) {
        const toEnemy = new THREE.Vector3().subVectors(d.mesh.position, pivot);
        if (toEnemy.dot(shootDir) < -5.0) return; // Skip enemies behind the player

        const scale = d.enemyScale / 3.0;
        const radius = 0.85 * scale;
        const height = 4.5 * scale;
        const dist = this.intersectRayCylinder(shootRay, d.mesh.position, radius, height);
        if (dist !== null && dist < closestEnemyDist) {
          closestEnemyDist = dist;
          closestEnemy = d;
        }
      }
    });

    const closestStaticDist = pivot.distanceTo(targetWorldPoint);

    // 4. Ground item checks (from camera raycaster since they are clicked/highlighted in screen space)
    const itemTargets = [];
    const itemMap = new Map();
    this.groundItems.forEach(item => {
      if (!item.isDead && item.mesh) {
        itemTargets.push(item.mesh);
        itemMap.set(item.mesh, item);
      }
    });

    const itemHits = this._raycaster.intersectObjects(itemTargets, true);
    let closestItem = null;
    let closestItemDist = Infinity;
    if (itemHits.length > 0) {
      closestItemDist = itemHits[0].distance;
      closestItem = itemMap.get(itemHits[0].object);
    }

    this.hoveredObject = null;
    let nextState = 'default';

    // Hover logic: enemy takes precedence if closer than static block and within range
    if (closestEnemy && closestEnemyDist < closestStaticDist && closestEnemyDist < 450) {
      nextState = 'hover-enemy';
    } else if (closestItemDist < closestStaticDist && closestItemDist < 120) {
      if (closestItem && closestItem.mesh) {
        this.hoveredObject = closestItem;
        nextState = 'hover-item';
      }
    } else if (this.adsBlend > 0.2) {
      nextState = 'ads';
    } else if (this.draggedItem) {
      nextState = 'place';
    }

    if (this.cursorState !== nextState) {
      this.cursorState = nextState;
      if (cursorEl) {
        cursorEl.className = this.cursorState;
      }
    }

    // ---- Dynamic Reticle Spread Calculation ----
    const isMoving = this.direction.lengthSq() > 0;
    const sprinting = !!this.keys['ShiftLeft'] && this.adsBlend < 0.1;
    let baseSpread = 10;
    if (isMoving) baseSpread += 7;
    if (sprinting) baseSpread += 15;
    if (!this.onGround) baseSpread += 22;
    if (this.recoilPitch > 0.01) baseSpread += Math.min(this.recoilPitch * 60, 25);

    // ADS tightens reticle spread significantly
    const finalSpread = Math.max(3, baseSpread * (1.0 - this.adsBlend * 0.75));

    if (cursorEl) {
      cursorEl.style.setProperty('--spread', `${finalSpread.toFixed(1)}px`);
    }
  }

  updateInventoryUI() {
    this.inventory.forEach((item, index) => {
      const slot = document.querySelector(`.inventory-slot[data-index="${index}"]`);
      if (slot) {
        slot.className = 'inventory-slot';
        if (index === this.activeSlot) {
          slot.classList.add('active');
        }

        if (item) {
          if (item.type === 'weapon') {
            const currentClip = (index === this.activeSlot) ? this.ammo : (item.clip !== undefined ? item.clip : (item.maxClip || 15));
            const currentReserve = (index === this.activeSlot) ? this.reserveAmmo : (item.reserve !== undefined ? item.reserve : 999);
            const isRel = (index === this.activeSlot && this.isReloading);

            const imgTag = item.image
              ? `<img src="${item.image}" class="slot-gun-img" alt="${item.name}" />`
              : `<span class="slot-item-emoji">${item.icon || '🔫'}</span>`;

            const bottomHtml = isRel
              ? `<div class="slot-bottom-bar reloading"><span class="reload-tag">RELOAD</span></div>`
              : `<div class="slot-bottom-bar"><span class="clip-count">${currentClip}</span><span class="ammo-slash">/</span><span class="reserve-count">${currentReserve}</span></div>`;

            slot.innerHTML = `
              <div class="slot-top-section">
                ${imgTag}
                <span class="slot-gun-name">${item.name}</span>
              </div>
              <div class="slot-bottom-section">
                ${bottomHtml}
              </div>
            `;
          } else {
            slot.innerHTML = `
              <div class="slot-top-section">
                <span class="slot-item-emoji">${item.icon || '💎'}</span>
                <span class="slot-gun-name">${item.name || 'Item'}</span>
              </div>
              <div class="slot-bottom-section">
                <span class="slot-use-tag">USE</span>
              </div>
            `;
          }
        } else {
          slot.innerHTML = `
            <div class="slot-empty">
              <span class="empty-label">Empty</span>
            </div>
          `;
        }
      }
    });
  }

  useInventoryItem(index) {
    const item = this.inventory[index];
    if (!item) return;

    if (item.type === 'weapon') {
      // Save current weapon's ammo to its inventory object before switching
      const prevItem = this.inventory[this.activeSlot];
      if (prevItem && prevItem.type === 'weapon') {
        prevItem.clip = this.ammo;
        prevItem.maxClip = this.maxAmmo;
        prevItem.reserve = this.reserveAmmo;
      }

      // Stop any fire or charging loops of previous weapon
      this.stopAutomaticFireEffects();

      this.activeSlot = index;
      this.updateInventoryUI();
      this.playActionSound();

      // Load selected weapon's ammo
      this.maxAmmo = item.maxClip || 15;
      this.ammo = item.clip !== undefined ? item.clip : this.maxAmmo;
      this.reserveAmmo = item.reserve !== undefined ? item.reserve : 999;
      this.updateAmmoDisplay();

      if (item.id === 'pistol') {
        this.weaponLevel = 1;
      } else if (item.id === 'smg') {
        this.weaponLevel = 2;
      } else if (item.id === 'railgun') {
        this.weaponLevel = 3;
      } else if (item.id === 'ak47') {
        this.weaponLevel = 4;
      } else if (item.id === 'rocket') {
        this.weaponLevel = 5;
      }
      this.updateWeaponSystem();
    } else if (item.type === 'consumable') {
      if (item.id === 'nanokit') {
        if (this.health >= 100) return;
        this.health = Math.min(100, this.health + 40);
        this.playHealSound();
        this.inventory[index] = null;
      } else if (item.id === 'powercell') {
        if (this.ammo >= this.maxAmmo) return;
        this.ammo = Math.min(this.maxAmmo, this.ammo + 150); // Restore 150 ammo for power cell since max is 999
        this.updateAmmoDisplay();
        this.playReloadSound();
        this.inventory[index] = null;
      } else if (item.id === 'crystal') {
        this.points += 500;
        this.playPointSound();
        this.inventory[index] = null;
      }
      this.updateInventoryUI();
    }
  }

  switchWeapon(direction: number) {
    const weaponSlots: number[] = [];
    for (let i = 0; i < this.inventory.length; i++) {
      if (this.inventory[i] && this.inventory[i].type === 'weapon') {
        weaponSlots.push(i);
      }
    }
    if (weaponSlots.length <= 1) return;

    let currentIdx = weaponSlots.indexOf(this.activeSlot);
    if (currentIdx === -1) {
      currentIdx = 0;
    }

    let nextIdx = (currentIdx + direction + weaponSlots.length) % weaponSlots.length;
    const targetSlot = weaponSlots[nextIdx];

    this.useInventoryItem(targetSlot);
  }

  pickupGroundItem(item) {
    let emptySlot = -1;
    for (let i = 0; i < this.inventory.length; i++) {
      if (!this.inventory[i]) {
        emptySlot = i;
        break;
      }
    }

    if (emptySlot !== -1) {
      let icon = "💎";
      let type = "consumable";
      if (item.type === 'nanokit') icon = "❤️";
      else if (item.type === 'powercell') icon = "⚡";

      this.inventory[emptySlot] = {
        id: item.type,
        name: item.name,
        icon: icon,
        type: type
      };

      item.destroy();
      this.groundItems = this.groundItems.filter(g => g !== item);
      this.updateInventoryUI();
      this.playPickupSound();
    }
  }

  dropItemOnGround(item, slotIndex) {
    if (!item) return;
    if (item.type === 'weapon') return; // Cannot drop weapons!

    const ndcX = (this.virtualCursorX / window.innerWidth) * 2 - 1;
    const ndcY = -(this.virtualCursorY / window.innerHeight) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);

    const chunks = Array.from(this.chunks.values()).map(c => (c as any).mesh).filter(Boolean);
    const intersects = raycaster.intersectObjects(chunks);

    if (intersects.length > 0) {
      const hit = intersects[0];
      const p = hit.point;

      this.groundItems.push(new GroundItem(this.scene, item.id, p.x, p.z, this));

      this.inventory[slotIndex] = null;
      this.updateInventoryUI();
      this.playDropSound();
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const dt = Math.min(this.clock.getDelta(), 0.06);
    if (dt <= 0) return;

    // Current FPS estimate for adaptive quality
    const fps = 1 / dt;
    const isLagging = fps < 30;

    // ---- Auto-quality shadow fallback ----
    if (isLagging) {
      this._lowFPSFrames++;
      if (this._lowFPSFrames > 60 && this._qualityLevel > 0) {
        this._qualityLevel--;
        if (this.sun && this.sun.shadow) {
          const newSize = this._qualityLevel === 0 ? 512 : 1024;
          this.sun.shadow.mapSize.width = newSize;
          this.sun.shadow.mapSize.height = newSize;
          if (this.sun.shadow.map) {
            this.sun.shadow.map.dispose();
            this.sun.shadow.map = null; // forces re-creation at new size
          }
        }
        this._lowFPSFrames = 0;
      }
    } else {
      this._lowFPSFrames = Math.max(0, this._lowFPSFrames - 1);
      // Restore quality when FPS recovers
      if (fps > 50 && this._qualityLevel < 2 && this._lowFPSFrames === 0) {
        this._qualityLevel = 2;
        if (this.sun && this.sun.shadow) {
          this.sun.shadow.mapSize.width = 2048;
          this.sun.shadow.mapSize.height = 2048;
          if (this.sun.shadow.map) {
            this.sun.shadow.map.dispose();
            this.sun.shadow.map = null;
          }
        }
      }
    }

    const isPaused = !this.isLocked && !this.playerDead;

    if (!isPaused) {
      if (this.mixer) this.mixer.update(dt);

      const rp = this.robotGroup.position;
      this.demons = this.demons.filter(demon => {
        demon.update(dt, rp);
        return !demon.deathComplete;
      });

      this.demons.forEach(demon => {
        // ---- Demon Cover Collision ----
        if (demon.mesh) {
          const dp = demon.mesh.position;
          this.chunks.forEach(chunk => {
            if (chunk.coverObstacles) {
              for (let i = 0; i < chunk.coverObstacles.length; i++) {
                const obs = chunk.coverObstacles[i];
                const dx = dp.x - obs.x;
                const dz = dp.z - obs.z;
                const distSq = dx * dx + dz * dz;
                const minDist = obs.radius + 3.0;
                if (distSq < minDist * minDist) {
                  const dist = Math.sqrt(distSq);
                  if (dist > 0.001) {
                    const push = minDist - dist;
                    dp.x += (dx / dist) * push;
                    dp.z += (dz / dist) * push;
                  }
                }
              }
            }
          });
        }
      });

      this.enemyProjectiles = this.enemyProjectiles.filter(p => {
        p.update(dt);
        return !p.isDead;
      });

      this.playerRockets = this.playerRockets.filter(r => {
        r.update(dt);
        return !r.isDead;
      });

      // Update ground items & check player proximity for item pickup
      const playerPos = this.robotGroup.position;
      this.groundItems = this.groundItems.filter(item => {
        if (item.isDead) return false;
        item.update(dt);

        // Distance check from player (radius 10.0 units)
        const dx = playerPos.x - item.group.position.x;
        const dz = playerPos.z - item.group.position.z;
        const dy = Math.abs((playerPos.y + 3) - item.group.position.y);
        const dist2D = dx * dx + dz * dz;

        if (dist2D < 100.0 && dy < 15.0) { // 10 units radius
          item.destroy();
          this.collectGroundItem(item.type);
          return false;
        }
        return true;
      });

      // ---- Auto-fire and charging system ticks ----
      const activeItem = this.inventory[this.activeSlot];
      if (activeItem && activeItem.type === 'weapon') {
        if (this.mouseLeftDown && !this.isReloading) {
          if (this.ammo <= 0) {
            this.stopAutomaticFireEffects();
            this.reloadWeapon();
          } else if (activeItem.id === 'smg' || activeItem.id === 'ak47') {
            const fireInterval = (activeItem.id === 'smg') ? 85 : 130;
            if (Date.now() - this.lastAimTime >= fireInterval) {
              this.shootPrimaryActual();
            }
            if (!this.activeShootSound) {
              this.activeShootSound = this.playSoundBuffer(activeItem.id, true, 0.45);
            }
          } else if (activeItem.id === 'railgun') {
            const now = Date.now();
            const cooldownDone = (now - (this.lastRailgunFireTime || 0)) >= 1000;

            if (!this.railgunFiredThisPress && cooldownDone) {
              if (!this.isChargingRailgun) {
                this.isChargingRailgun = true;
                this.railgunChargeTime = 0;
                this.playRailgunChargeSound();
              } else {
                this.railgunChargeTime += dt;
                if (this.railgunChargeTime >= 1.5) {
                  this.stopRailgunChargeSound();
                  this.isChargingRailgun = false;
                  this.railgunChargeTime = 0;
                  this.railgunFiredThisPress = true; // Single-shot lock: require release before re-charge
                  this.lastRailgunFireTime = Date.now(); // 1.0s post-fire cooldown delay
                  this.shootPrimaryActual();
                }
              }
            }
          }
        } else {
          this.stopAutomaticFireEffects();
        }
      } else {
        this.stopAutomaticFireEffects();
      }

      this.updatePlayer(dt);
      this.updateWeather(dt);
      this.updateEnemySpawning(dt);
      this.updateBalloons(dt);
      this.updateECG(dt);
    } else {
      // Game is paused: stop fire effects and keep camera loop active for menu view
      this.stopAutomaticFireEffects();
      this.updateThirdPersonCamera(dt);
    }

    // Rebuild chunks during menu/paused state so maps load and render in real-time behind the menu overlay
    if (!isLagging) {
      this.updateChunks();
    }
    // Skip fog updates when lagging (subtle, barely visible)
    if (!isLagging) {
      this.updateFogLayers();
    }
    this.updateHUD(dt);
    this.updateMinimap(); // Already throttled to 2fps internally
    this.updateVirtualCursor(dt);

    this.renderer.render(this.scene, this.camera);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const tryFullscreen = () => {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => { });
    }
  };

  // Attempt immediate fullscreen request on DOM load
  tryFullscreen();

  // Also bind to ANY user interaction anywhere on the document (clicking cards, menu, or page)
  const onFirstInteraction = () => {
    tryFullscreen();
  };
  window.addEventListener('pointerdown', onFirstInteraction, { capture: true });
  window.addEventListener('keydown', onFirstInteraction, { capture: true });
  window.addEventListener('click', onFirstInteraction, { capture: true });

  // Map selection touch & click handlers
  const mapCards = document.querySelectorAll('.map-card');
  const blocker = document.getElementById('blocker');
  mapCards.forEach(card => {
    const handleMapSelect = (e: Event) => {
      e.preventDefault();
      e.stopPropagation(); // don't trigger deploy/blocker click
      mapCards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');

      const mapId = (card as HTMLElement).dataset.map;
      if (blocker && mapId) {
        blocker.className = blocker.className.replace(/theme-\w+/g, '');
        blocker.classList.add(`theme-${mapId}`);
      }

      tryFullscreen();

      if (mapId && (window as any).game) {
        (window as any).game.changeMap(mapId);
      }
    };

    card.addEventListener('click', handleMapSelect);
    card.addEventListener('touchend', handleMapSelect);
  });

  // Graphics / Resolution Quality touch & click handlers
  const qualityBtns = document.querySelectorAll('.quality-btn');
  qualityBtns.forEach(btn => {
    const handleQualitySelect = (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
      const quality = (btn as HTMLElement).dataset.quality;
      if (quality && (window as any).game) {
        (window as any).game.setGraphicsQuality(quality);
      }
    };

    btn.addEventListener('click', handleQualitySelect);
    btn.addEventListener('touchend', handleQualitySelect);
  });
  // Orientation Check for Touch Devices
  const checkOrientation = () => {
    const isPortrait = window.innerHeight > window.innerWidth;
    const isTouch = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0) || window.matchMedia('(pointer: coarse)').matches;

    // Ensure body.is-mobile is active on all mobile and touch devices
    if (isTouch) {
      document.body.classList.add('is-mobile');
    } else {
      document.body.classList.remove('is-mobile');
    }

    const rotateOverlay = document.getElementById('rotate-device-overlay');
    if (rotateOverlay) {
      if (isTouch && isPortrait) {
        rotateOverlay.style.display = 'flex';
      } else {
        rotateOverlay.style.display = 'none';
      }
    }
  };

  window.addEventListener('resize', checkOrientation);
  window.addEventListener('orientationchange', checkOrientation);
  checkOrientation();

  new Game();
});
