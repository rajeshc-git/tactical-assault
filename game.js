// ================================================================
//  SNOW MOUNTAIN EXPLORER — 3D Game Engine
//  Built with Three.js — Open World Action Explorer
// ================================================================
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ================================================================
//  PERLIN NOISE GENERATOR
// ================================================================
class PerlinNoise {
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

      let r, g, b;
      if (slope > 0.52) {
        r = 0.33 + cv; g = 0.34 + cv; b = 0.40 + cv;
      } else if (hr > 0.65) {
        r = 0.94 + cv; g = 0.96 + cv; b = 0.99;
      } else if (hr > 0.22 && hr < 0.52) {
        const sn = 0.74 + (cv * 2.0);
        r = 0.83 * sn; g = 0.89 * sn; b = 0.87 * sn;
      } else {
        const br = 0.88 + cv;
        r = br; g = br + 0.02; b = br + 0.05;
      }

      colors[i * 3]     = THREE.MathUtils.clamp(r, 0, 1);
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
      this.createTrees();
    }
  }

  createTrees() {
    const size = this.size;
    const startX = this.cx * size - size / 2;
    const startZ = this.cz * size - size / 2;

    const treePositions = [];
    const step = 45;

    for (let x = 20; x < size; x += step) {
      for (let z = 20; z < size; z += step) {
        const worldX = startX + x + this.game.wasm.noise(startX + x, startZ + z) * 15;
        const worldZ = startZ + z + this.game.wasm.noise(startZ + z, startX + x) * 15;
        const y = this.game.getHeightAt(worldX, worldZ);
        const hr = y / this.game.MAX_HEIGHT;

        if (hr > 0.15 && hr < 0.58) {
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

    const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2a1a, roughness: 0.95 });
    const foliageMat = new THREE.MeshStandardMaterial({ color: 0x1a3a1c, roughness: 0.88 });
    const snowMat = new THREE.MeshStandardMaterial({ color: 0xe4ecf0, roughness: 0.75 });

    const trunkIM = new THREE.InstancedMesh(trunkGeo, trunkMat, N);
    const lowerIM = new THREE.InstancedMesh(lowerGeo, foliageMat, N);
    const upperIM = new THREE.InstancedMesh(upperGeo, foliageMat, N);
    const snowCapIM = new THREE.InstancedMesh(snowCapGeo, snowMat, N);

    trunkIM.castShadow = true;
    lowerIM.castShadow = true;
    lowerIM.receiveShadow = true;
    upperIM.castShadow = true;

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

      dummy.position.y = pos.y + 13 * sc;
      dummy.scale.set(sc * 0.82, sc * 0.5, sc * 0.82);
      dummy.updateMatrix();
      snowCapIM.setMatrixAt(idx, dummy.matrix);
    });

    trunkIM.instanceMatrix.needsUpdate = true;
    lowerIM.instanceMatrix.needsUpdate = true;
    upperIM.instanceMatrix.needsUpdate = true;
    snowCapIM.instanceMatrix.needsUpdate = true;

    this.mesh.add(trunkIM);
    this.mesh.add(lowerIM);
    this.mesh.add(upperIM);
    this.mesh.add(snowCapIM);

    this.treeMeshes = [trunkIM, lowerIM, upperIM, snowCapIM];
  }

  destroy() {
    if (this.treeMeshes) {
      this.treeMeshes.forEach(im => {
        im.geometry.dispose();
        im.material.dispose();
      });
      this.treeMeshes = null;
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
  constructor(scene, startPos, targetPos, game) {
    this.scene = scene;
    this.game = game;
    this.mesh = new THREE.Group();
    
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.35, 0.35, 2.2, 8),
      new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.9 })
    );
    body.rotation.x = Math.PI / 2;
    this.mesh.add(body);

    const tip = new THREE.Mesh(
      new THREE.ConeGeometry(0.36, 0.8, 8),
      new THREE.MeshBasicMaterial({ color: 0xff4400 })
    );
    tip.rotation.x = Math.PI / 2;
    tip.position.z = 1.3;
    this.mesh.add(tip);

    const light = new THREE.PointLight(0xff6600, 4.0, 25);
    this.mesh.add(light);

    this.mesh.position.copy(startPos);
    this.dir = new THREE.Vector3().subVectors(targetPos, startPos).normalize();
    this.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this.dir);

    this.speed = 180;
    this.isDead = false;
    this.age = 0;
    this.targetPos = targetPos;

    this.scene.add(this.mesh);
  }

  update(dt) {
    if (this.isDead) return;

    this.age += dt;
    this.mesh.position.addScaledVector(this.dir, this.speed * dt);

    this.game.createSparks(this.mesh.position);

    const distToTarget = this.mesh.position.distanceTo(this.targetPos);
    const terrainHeight = this.game.getHeightAt(this.mesh.position.x, this.mesh.position.z);

    if (distToTarget < 5.0 || this.mesh.position.y <= terrainHeight || this.age > 2.5) {
      this.explode();
    }
  }

  explode() {
    this.isDead = true;
    const pos = this.mesh.position;

    this.game.triggerScreenShake(0.7);
    this.game.playRocketExplosionSound();

    this.game.createExplosionParticles(pos);
    this.game.createTerrainImpactDust(pos);

    // AOE damage boosted by game level
    const baseAoe = 140 * (1 + (this.game.gameLevel - 1) * 0.25);
    this.game.demons.forEach(d => {
      if (!d.isDead) {
        const dDist = d.mesh.position.distanceTo(pos);
        if (dDist < 48) {
          const aoeDamage = baseAoe * (1 - dDist / 48);
          d.takeDamage(aoeDamage, pos);
        }
      }
    });

    this.scene.remove(this.mesh);
    this.mesh.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
  }
}

// ================================================================
//  ENEMY PROJECTILE CLASS (Alien plasma energy)
// ================================================================
class EnemyProjectile {
  constructor(scene, startPos, targetPos, game) {
    this.scene = scene;
    this.game = game;
    this.mesh = new THREE.Group();
    
    const geo = new THREE.SphereGeometry(0.85, 8, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0xff2200 });
    const orb = new THREE.Mesh(geo, mat);
    this.mesh.add(orb);
    
    const light = new THREE.PointLight(0xff2200, 2.5, 18);
    this.mesh.add(light);
    
    this.mesh.position.copy(startPos);
    this.dir = new THREE.Vector3().subVectors(targetPos, startPos).normalize();
    this.speed = 115 + (this.game.gameLevel - 1) * 12; // speeds up with levels
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
      const demonDamage = 16 + (this.game.gameLevel - 1) * 4;
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
  constructor(scene, x, z, game) {
    this.scene = scene;
    this.game = game;
    this.mesh = new THREE.Group();
    
    const y = this.game.getHeightAt(x, z);
    this.mesh.position.set(x, y + 3.5, z);
    
    // Health & speed scale with Game Level
    const lvlBonus = (this.game.gameLevel - 1);
    this.health = 35 + lvlBonus * 18 + Math.random() * 25;
    this.speed = 18 + lvlBonus * 4 + Math.random() * 9;
    this.isDead = false;
    this.lastShootTime = Date.now() + Math.random() * 1000;
    
    this.createModel();
    this.scene.add(this.mesh);
  }

  createModel() {
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x181818,
      roughness: 0.9,
      metalness: 0.1
    });
    
    const glowMat = new THREE.MeshStandardMaterial({
      color: 0xff1100,
      emissive: 0xff0000,
      emissiveIntensity: 3.5
    });

    const body = new THREE.Mesh(new THREE.SphereGeometry(3.6, 12, 12), bodyMat);
    body.castShadow = true;
    this.mesh.add(body);

    const core = new THREE.Mesh(new THREE.SphereGeometry(2.2, 8, 8), glowMat);
    this.mesh.add(core);

    const hornGeo = new THREE.ConeGeometry(0.7, 2.8, 6);
    hornGeo.rotateX(Math.PI / 4.5);
    
    const hornL = new THREE.Mesh(hornGeo, glowMat);
    hornL.position.set(-1.6, 2.6, 0.6);
    this.mesh.add(hornL);
    
    const hornR = new THREE.Mesh(hornGeo, glowMat);
    hornR.position.set(1.6, 2.6, 0.6);
    hornR.rotation.y = Math.PI;
    this.mesh.add(hornR);

    const eyeGeo = new THREE.SphereGeometry(0.45, 8, 8);
    const eyeL = new THREE.Mesh(eyeGeo, glowMat);
    eyeL.position.set(-1.0, 0.8, 2.9);
    this.mesh.add(eyeL);
    
    const eyeR = new THREE.Mesh(eyeGeo, glowMat);
    eyeR.position.set(1.0, 0.8, 2.9);
    this.mesh.add(eyeR);

    const spikeGeo = new THREE.ConeGeometry(0.4, 1.8, 4);
    for (let i = 0; i < 6; i++) {
      const spike = new THREE.Mesh(spikeGeo, bodyMat);
      const angle = (i / 6) * Math.PI * 2;
      spike.position.set(Math.sin(angle) * 3.8, Math.cos(angle) * 3.8, 0);
      spike.rotation.z = angle - Math.PI / 2;
      this.mesh.add(spike);
    }

    const weaponGroup = new THREE.Group();
    const wBarrel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.25, 0.25, 2.4, 8),
      new THREE.MeshStandardMaterial({ color: 0x111111, metalness: 0.8 })
    );
    wBarrel.rotation.x = Math.PI / 2;
    wBarrel.position.z = 0.8;
    weaponGroup.add(wBarrel);

    const wGlow = new THREE.Mesh(
      new THREE.SphereGeometry(0.4, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0xff0000 })
    );
    wGlow.position.z = 1.6;
    weaponGroup.add(wGlow);

    weaponGroup.position.set(2.2, -0.5, 1.0);
    this.mesh.add(weaponGroup);
    this.weaponGroup = weaponGroup;
  }

  takeDamage(amount, hitPoint) {
    if (this.isDead) return;
    this.health -= amount;
    
    this.mesh.traverse(child => {
      if (child.material && child.material.emissive) {
        child.material.emissiveIntensity = 8.0;
        setTimeout(() => {
          if (child.material && !this.isDead) child.material.emissiveIntensity = 3.5;
        }, 100);
      }
    });

    if (this.health <= 0) {
      this.die(hitPoint);
    }
  }

  die(hitPoint) {
    this.isDead = true;
    this.game.createExplosionParticles(this.mesh.position);
    this.game.playExplosionSound();
    
    this.scene.remove(this.mesh);
    this.mesh.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });

    this.game.points += 100 * this.game.gameLevel;
    this.game.kills += 1;
    this.game.updateWeaponSystem();

    // Check Level Up condition (every 5 kills)
    if (this.game.kills >= this.game.gameLevel * 5) {
      this.game.levelUp();
    }
  }

  shootWeapon(playerPos) {
    const muzzle = new THREE.Vector3();
    if (this.weaponGroup) {
      this.weaponGroup.getWorldPosition(muzzle);
    } else {
      muzzle.copy(this.mesh.position);
    }

    const target = new THREE.Vector3(playerPos.x, playerPos.y + 5, playerPos.z);
    const proj = new EnemyProjectile(this.scene, muzzle, target, this.game);
    this.game.enemyProjectiles.push(proj);
  }

  update(dt, playerPos) {
    if (this.isDead) return;

    const t = Date.now() * 0.0035;
    const hoverOffset = 3.5 + Math.sin(t) * 0.75;

    const toPlayer = new THREE.Vector3().subVectors(playerPos, this.mesh.position);
    const dist = toPlayer.length();

    if (dist < 280) {
      const targetAngle = Math.atan2(toPlayer.x, toPlayer.z);
      this.mesh.rotation.y = targetAngle;

      if (dist > 30) {
        toPlayer.y = 0;
        toPlayer.normalize();
        this.mesh.position.x += toPlayer.x * this.speed * dt;
        this.mesh.position.z += toPlayer.z * this.speed * dt;
        this.mesh.position.y = this.game.getHeightAt(this.mesh.position.x, this.mesh.position.z) + hoverOffset;
      }

      const now = Date.now();
      const fireInterval = Math.max(1000, 1800 - (this.game.gameLevel - 1) * 150);
      if (dist < 200 && now - this.lastShootTime > fireInterval) {
        this.lastShootTime = now;
        this.shootWeapon(playerPos);
      }
    } else {
      this.mesh.rotation.y += dt * 0.45;
      this.mesh.position.y = this.game.getHeightAt(this.mesh.position.x, this.mesh.position.z) + hoverOffset;
    }
  }
}

// ================================================================
//  GROUND ITEM CLASS
// ================================================================
class GroundItem {
  constructor(scene, type, x, z, game) {
    this.scene = scene;
    this.type = type; // 'crystal', 'nanokit', 'powercell'
    this.game = game;
    this.isDead = false;

    this.group = new THREE.Group();

    let geo, mat;
    if (type === 'crystal') {
      geo = new THREE.OctahedronGeometry(1.6, 0);
      mat = new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x008888,
        roughness: 0.1,
        metalness: 0.9
      });
      this.name = "Glowing Crystal";
      this.icon = "💎";
    } else if (type === 'nanokit') {
      geo = new THREE.BoxGeometry(1.6, 2.0, 0.8);
      mat = new THREE.MeshStandardMaterial({
        color: 0xff3366,
        emissive: 0x550011,
        roughness: 0.4,
        metalness: 0.2
      });
      this.name = "Nano-Kit";
      this.icon = "❤️";
    } else { // powercell
      geo = new THREE.CylinderGeometry(0.8, 0.8, 2.4, 10);
      mat = new THREE.MeshStandardMaterial({
        color: 0x33ff66,
        emissive: 0x005511,
        roughness: 0.3,
        metalness: 0.7
      });
      this.name = "Power Cell";
      this.icon = "⚡";
    }

    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.group.add(this.mesh);

    // Light glow
    this.light = new THREE.PointLight(mat.color.getHex(), 2.0, 15);
    this.light.position.y = 1.5;
    this.group.add(this.light);

    // Find height
    const y = this.game.getHeightAt(x, z) + 1.2;
    this.group.position.set(x, y, z);
    
    // Tag this group so Raycaster knows it's interactive
    this.mesh.userData = { isGroundItem: true, parentItem: this };

    this.scene.add(this.group);
  }

  update(dt) {
    this.mesh.rotation.y += dt * 1.5;
    this.mesh.rotation.x += dt * 0.4;
    this.mesh.position.y = Math.sin(Date.now() * 0.003) * 0.2;
  }

  destroy() {
    this.scene.remove(this.group);
    this.isDead = true;
  }
}

// ================================================================
//  GAME CLASS
// ================================================================
class Game {
  constructor() {
    this.noise = new PerlinNoise(42);
    this.wasm = this.noise; // Fallback default

    // ---- Config ----
    this.chunkSize        = 250;
    this.viewRadius       = 4;
    this.MAX_HEIGHT       = 380;
    this.SNOW_COUNT       = 14000;
    this.MOVE_SPEED       = 55;
    this.SPRINT_SPEED     = 110;
    this.JUMP_FORCE       = 75;
    this.GRAVITY          = -190;

    // ---- Classic TPP Camera ----
    this.camYaw         = 0;
    this.camPitch       = 0.35;
    this.camDist        = 40;
    this.camLookTarget  = new THREE.Vector3();
    this.screenShake    = 0;

    // ---- State ----
    this.chunks         = new Map();
    this.demons         = [];
    this.enemyProjectiles = [];
    this.playerRockets  = [];
    this.keys           = {};
    this.isLocked       = false;
    this.velocity       = new THREE.Vector3();
    this.direction      = new THREE.Vector3();
    this.onGround       = true;
    this.showControls   = true;
    this.fpsSamples     = [];
    this.fogLayers      = [];
    this.landscapeTexture = null;

    // ---- Memory Pools (Optimization) ----
    this._vecForward    = new THREE.Vector3();
    this._vecRight      = new THREE.Vector3();
    this._vecLookTarget = new THREE.Vector3();
    this._vecDesiredPos = new THREE.Vector3();
    this.activeKeys     = new Set();
    this._frustum       = new THREE.Frustum();
    this._projScreenMat = new THREE.Matrix4();

    // Progression, Leveling & Ammo
    this.gameLevel      = 1;
    this.health         = 100;
    this.kills          = 0;
    this.points         = 0;
    this.weaponLevel    = 1;
    this.ammo           = 20;
    this.maxAmmo        = 20;
    this.isReloading    = false;
    this.playerDead     = false;
    this.lastSecondaryTime = 0;
    this.lastAimTime    = 0;

    this.audioCtx       = null;

    // Robot & Weapon
    this.robotGroup     = null;
    this.robotModel     = null;
    this.mixer          = null;
    this.animations     = {};
    this.currentAction  = null;
    this.gun            = null;
    this.gunGlowMat     = null;
    // ---- Virtual Cursor & Inventory State ----
    this.virtualCursorX = window.innerWidth / 2;
    this.virtualCursorY = window.innerHeight / 2;
    this.cursorState = 'default';
    this.hoveredObject = null;
    this.draggedItem = null;
    this.draggedFromSlot = -1;
    this.inventory = [
      { id: 'laser', name: 'Laser Blaster', icon: '🔫', type: 'weapon', equipped: true },
      null,
      null,
      null,
      null
    ];
    this.activeSlot = 0;
    this.groundItems = [];

    window.game = this;

    this.init();
  }

  // ==============================================================
  //  INITIALIZATION
  // ==============================================================
  async init() {
    // Load WASM terrain engine
    try {
      const wasmResponse = await fetch('noise.wasm');
      const bytes = await wasmResponse.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(bytes);
      this.wasm = instance.exports;
      this.wasm.init(42);
      console.log("WebAssembly terrain engine loaded successfully.");
    } catch (e) {
      console.warn("Failed loading WebAssembly noise module. Falling back to JS terrain generator.", e);
    }

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x8cb8d8);
    this.scene.fog = new THREE.FogExp2(0xbdd0e4, 0.00062);

    this.camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.5, 8000);
    this.clock = new THREE.Clock();

    await this.loadAssets();
    this.createLighting();
    this.createSkybox();
    
    this.createGun();
    this.createRobot();
    this.updateChunks();

    this.createBalloons();
    this.createSnow();
    this.createFogLayers();

    const rp = this.robotGroup.position;
    this.camLookTarget.set(rp.x, rp.y + 6, rp.z);
    this.camera.position.set(rp.x, rp.y + 20, rp.z + this.camDist);
    this.camera.lookAt(this.camLookTarget);

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

    this.setupControls();
    this.setupPostProcessing();
    this.setupMinimap();
    this.setupECG();

    const ls = document.getElementById('loading-screen');
    if (ls) {
      ls.classList.add('fade-out');
      setTimeout(() => ls.style.display = 'none', 1400);
    }

    this.animate();
  }

  loadAssets() {
    const loader = new THREE.TextureLoader();
    const bar = document.getElementById('progress-fill');
    return new Promise(resolve => {
      loader.load('Screenshot 2026-08-03 152141.png',
        tex => {
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.wrapS = THREE.RepeatWrapping;
          tex.wrapT = THREE.ClampToEdgeWrapping;
          this.landscapeTexture = tex;
          if (bar) bar.style.width = '100%';
          setTimeout(resolve, 600);
        },
        xhr => {
          if (xhr.lengthComputable && bar) {
            bar.style.width = (xhr.loaded / xhr.total * 100) + '%';
          }
        },
        () => {
          console.warn('Landscape image failed to load - continuing');
          resolve();
        }
      );
    });
  }

  createLighting() {
    this.scene.add(new THREE.AmbientLight(0x8eafc8, 0.55));
    this.scene.add(new THREE.HemisphereLight(0x87ceeb, 0xe8eef2, 0.45));
    const sun = new THREE.DirectionalLight(0xfff3e0, 1.6);
    sun.position.set(400, 550, 250);
    sun.castShadow = true;
    const s = sun.shadow;
    s.mapSize.width = s.mapSize.height = 2048;
    s.camera.near = 10;
    s.camera.far  = 2000;
    s.camera.left = s.camera.bottom = -500;
    s.camera.right = s.camera.top   =  500;
    s.bias = -0.0005;
    this.scene.add(sun);
    this.scene.add(sun.target);
    this.sun = sun;
  }

  createSkybox() {
    if (this.landscapeTexture) {
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
    g.addColorStop(0,    '#102840');
    g.addColorStop(0.25, '#3a7bb8');
    g.addColorStop(0.5,  '#6aade0');
    g.addColorStop(0.75, '#9ecce8');
    g.addColorStop(1,    '#c8dce8');
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
    const group = new THREE.Group();

    const barrel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.35, 0.35, 3.2, 12),
      new THREE.MeshStandardMaterial({ color: 0x1f1f1f, metalness: 0.8, roughness: 0.2 })
    );
    barrel.rotation.x = Math.PI / 2;
    barrel.position.z = 1.2;
    group.add(barrel);

    const ringGeo = new THREE.TorusGeometry(0.5, 0.12, 8, 24);
    this.gunGlowMat = new THREE.MeshStandardMaterial({
      color: 0x00aaff,
      emissive: 0x0088ff,
      emissiveIntensity: 2.5
    });
    const ring = new THREE.Mesh(ringGeo, this.gunGlowMat);
    ring.position.z = 1.8;
    group.add(ring);

    const base = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.8, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x3d3d3d, metalness: 0.7, roughness: 0.35 })
    );
    base.position.z = 0.0;
    group.add(base);

    this.gun = group;
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
      this.gun.position.set(2.0, 4.5, 1.5);
    };

    const loader = new GLTFLoader();
    const modelUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/r162/examples/models/gltf/RobotExpressive/RobotExpressive.glb';

    loader.load(modelUrl,
      gltf => {
        this.robotModel = gltf.scene;
        this.robotModel.scale.setScalar(3.0);
        
        this.robotModel.traverse(child => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        this.robotGroup.add(this.robotModel);

        this.mixer = new THREE.AnimationMixer(this.robotModel);
        this.animations = {};
        
        gltf.animations.forEach(clip => {
          this.animations[clip.name] = this.mixer.clipAction(clip);
        });

        if (this.animations['Idle']) {
          this.currentAction = this.animations['Idle'];
          this.currentAction.play();
        }

        let rightHand = null;
        this.robotModel.traverse(child => {
          const name = child.name.toLowerCase();
          if (child.isBone && (name.includes('hand_r') || name.includes('handr') || name.includes('wrist_r') || name.includes('right_wrist') || name.includes('right_hand'))) {
            rightHand = child;
          }
        });

        if (rightHand) {
          rightHand.add(this.gun);
          this.gun.position.set(0, -0.3, 0.4);
          this.gun.rotation.set(0, 0, 0);
          this.gun.scale.setScalar(0.35);
        } else {
          this.robotGroup.add(this.gun);
          this.gun.position.set(2.0, 4.5, 1.5);
        }
      },
      undefined,
      err => {
        console.error("Error loading GLTF robot model:", err);
        fallbackRobot();
      }
    );

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
    this.maxAmmo = 20 + (this.gameLevel - 1) * 5;
    this.ammo = this.maxAmmo;
    this.updateAmmoDisplay();
  }

  reloadWeapon() {
    if (this.isReloading || this.ammo >= this.maxAmmo) return;
    
    this.isReloading = true;
    this.reloadTimerStart = Date.now();
    this.playReloadSound();
    
    const ammoDisp = document.getElementById('ammo-display');
    if (ammoDisp) ammoDisp.textContent = "RELOADING...";

    setTimeout(() => {
      this.ammo = this.maxAmmo;
      this.isReloading = false;
      this.updateAmmoDisplay();
    }, 1300);
  }

  updateAmmoDisplay() {
    const ammoDisp = document.getElementById('ammo-display');
    if (ammoDisp && !this.isReloading) {
      ammoDisp.textContent = `Ammo: ${this.ammo} / ${this.maxAmmo}`;
    }
  }

  updateWeaponSystem() {
    let newLevel = 1;
    if (this.kills >= 10) newLevel = 3;
    else if (this.kills >= 3) newLevel = 2;

    if (this.weaponLevel !== newLevel) {
      this.weaponLevel = newLevel;
      
      if (this.gunGlowMat) {
        if (this.weaponLevel === 1) {
          this.gunGlowMat.color.setHex(0x00aaff);
          this.gunGlowMat.emissive.setHex(0x0088ff);
          document.getElementById('weapon-display').textContent = "Laser Pistol";
        } else if (this.weaponLevel === 2) {
          this.gunGlowMat.color.setHex(0x00ff66);
          this.gunGlowMat.emissive.setHex(0x00ff22);
          document.getElementById('weapon-display').textContent = "Plasma Rifle";
        } else {
          this.gunGlowMat.color.setHex(0xff3300);
          this.gunGlowMat.emissive.setHex(0xff0000);
          document.getElementById('weapon-display').textContent = "M-66 Railgun";
        }
      }
    }
  }

  playLaserSound(level) {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);

      const now = ctx.currentTime;
      if (level === 1) {
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.exponentialRampToValueAtTime(120, now + 0.14);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.linearRampToValueAtTime(0.01, now + 0.14);
        osc.start(now);
        osc.stop(now + 0.14);
      } else if (level === 2) {
        osc.frequency.setValueAtTime(580, now);
        osc.frequency.exponentialRampToValueAtTime(70, now + 0.18);
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.linearRampToValueAtTime(0.01, now + 0.18);
        osc.type = 'sawtooth';
        osc.start(now);
        osc.stop(now + 0.18);
      } else {
        osc.frequency.setValueAtTime(140, now);
        osc.frequency.linearRampToValueAtTime(1100, now + 0.08);
        osc.frequency.exponentialRampToValueAtTime(30, now + 0.42);
        gain.gain.setValueAtTime(0.5, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.42);
        osc.type = 'triangle';
        osc.start(now);
        osc.stop(now + 0.42);
      }
    } catch (e) {
      console.warn("Audio blocked:", e);
    }
  }

  playRocketSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);

      const now = ctx.currentTime;
      osc.frequency.setValueAtTime(320, now);
      osc.frequency.linearRampToValueAtTime(75, now + 0.35);
      gain.gain.setValueAtTime(0.55, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
      osc.type = 'sawtooth';
      osc.start(now);
      osc.stop(now + 0.35);
    } catch (e) {
      console.warn("Audio blocked:", e);
    }
  }

  playExplosionSound() {
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
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
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
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
    try {
      if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = 'triangle';
      osc1.frequency.setValueAtTime(400, now);
      osc1.frequency.exponentialRampToValueAtTime(100, now + 0.1);
      gain1.gain.setValueAtTime(0.3, now);
      gain1.gain.linearRampToValueAtTime(0.01, now + 0.1);
      osc1.connect(gain1); gain1.connect(ctx.destination);
      osc1.start(now); osc1.stop(now + 0.1);

      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sawtooth';
      osc2.frequency.setValueAtTime(200, now + 0.7);
      osc2.frequency.exponentialRampToValueAtTime(900, now + 0.85);
      gain2.gain.setValueAtTime(0.4, now + 0.7);
      gain2.gain.linearRampToValueAtTime(0.01, now + 0.85);
      osc2.connect(gain2); gain2.connect(ctx.destination);
      osc2.start(now + 0.7); osc2.stop(now + 0.85);
    } catch (e) {}
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
    } catch (e) {}
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
    } catch (e) {}
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
    } catch (e) {}
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
    } catch (e) {}
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
    } catch (e) {}
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
    } catch (e) {}
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
  // ==============================================================
  shootPrimary() {
    if (!this.isLocked || this.playerDead) return;
    if (this.isReloading) return;
    
    if (this.ammo <= 0) {
      this.reloadWeapon();
      return;
    }

    this.ammo--;
    this.updateAmmoDisplay();
    this.lastAimTime = Date.now();

    this.triggerMuzzleFlash();
    this.playLaserSound(this.weaponLevel);

    const laserGeo = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({
      color: this.weaponLevel === 1 ? 0x00aaff : (this.weaponLevel === 2 ? 0x00ff66 : 0xff3300),
      linewidth: 3.5,
      transparent: true,
      opacity: 1.0
    });

    const muzzleWorld = new THREE.Vector3();
    if (this.gun) {
      this.gun.getWorldPosition(muzzleWorld);
    } else {
      muzzleWorld.copy(this.robotGroup.position).y += 6;
    }

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);

    const targets = [];
    const demonMap = new Map();
    this.demons.forEach(d => {
      if (!d.isDead) {
        targets.push(d.mesh);
        d.mesh.traverse(child => {
          if (child.isMesh) demonMap.set(child, d);
        });
      }
    });

    const intersects = raycaster.intersectObjects(targets, true);
    let hitPoint = new THREE.Vector3();
    let hitDemon = false;

    if (intersects.length > 0) {
      const hit = intersects[0];
      hitPoint.copy(hit.point);
      const demon = demonMap.get(hit.object);
      if (demon) {
        hitDemon = true;
        const damage = (this.weaponLevel === 1 ? 22 : (this.weaponLevel === 2 ? 50 : 150)) * (1 + (this.gameLevel - 1) * 0.2);
        demon.takeDamage(damage, hit.point);
        this.createSparks(hit.point);
      } else {
        this.createTerrainImpactDust(hit.point);
      }
    } else {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      hitPoint.copy(muzzleWorld).addScaledVector(dir, 450);
      this.createTerrainImpactDust(hitPoint);
    }

    const points = [muzzleWorld, hitPoint];
    laserGeo.setFromPoints(points);
    const laserLine = new THREE.Line(laserGeo, material);
    this.scene.add(laserLine);

    let alpha = 1.0;
    const fade = () => {
      alpha -= 0.14;
      if (alpha <= 0) {
        this.scene.remove(laserLine);
        laserGeo.dispose();
        material.dispose();
      } else {
        material.opacity = alpha;
        requestAnimationFrame(fade);
      }
    };
    fade();

    if (hitDemon) {
      this.triggerScreenShake(0.15);
    }
  }

  shootSecondary() {
    if (!this.isLocked || this.playerDead || this.isReloading) return;

    const now = Date.now();
    if (now - this.lastSecondaryTime < 1100) return;
    this.lastSecondaryTime = now;
    this.lastAimTime = now;

    this.triggerMuzzleFlash();
    this.playRocketSound();

    const muzzleWorld = new THREE.Vector3();
    if (this.gun) {
      this.gun.getWorldPosition(muzzleWorld);
    } else {
      muzzleWorld.copy(this.robotGroup.position).y += 6;
    }

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);

    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    const targetPoint = new THREE.Vector3().copy(muzzleWorld).addScaledVector(dir, 350);

    const rocket = new PlayerRocket(this.scene, muzzleWorld, targetPoint, this);
    this.playerRockets.push(rocket);
  }

  createSparks(pos) {
    const N = 8;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(N * 3);
    const velocities = [];

    for (let i = 0; i < N; i++) {
      positions[i * 3]     = pos.x;
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
      color: this.weaponLevel === 1 ? 0x00ccff : (this.weaponLevel === 2 ? 0x00ff66 : 0xff3300),
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
          arr[i3]     += velocities[i].x * 0.016;
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
      positions[i * 3]     = pos.x;
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
          arr[i3]     += velocities[i].x * 0.016;
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
      positions[i * 3]     = pos.x;
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
          arr[i3]     += velocities[i].x * 0.016;
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
    this.ecgCanvas = document.getElementById('ecg-monitor');
    if (!this.ecgCanvas) return;
    this.ecgCtx = this.ecgCanvas.getContext('2d');
    this.ecgPoints = new Array(90).fill(10);
    this.ecgTime = 0;
  }

  updateECG(dt) {
    if (!this.ecgCtx) return;
    const ctx = this.ecgCtx;
    const w = this.ecgCanvas.width;
    const h = this.ecgCanvas.height;
    
    ctx.clearRect(0, 0, w, h);

    if (this.playerDead) {
      ctx.strokeStyle = '#ff3333';
      ctx.lineWidth = 2.0;
      ctx.shadowBlur = 4;
      ctx.shadowColor = ctx.strokeStyle;
      ctx.beginPath();
      ctx.moveTo(0, h/2);
      ctx.lineTo(w, h/2);
      ctx.stroke();
      ctx.shadowBlur = 0;
      return;
    }

    let hr = 1.8;
    if (this.health < 30) hr = 5.5;
    else if (this.health < 60) hr = 3.6;
    
    if (this.keys['ShiftLeft'] && this.keys['KeyW']) hr += 1.2;

    this.ecgTime += dt * hr * Math.PI * 2;
    
    const x = this.ecgTime % (Math.PI * 2);
    let wave = 0;
    
    if (x > 0 && x < 0.4) {
      wave = Math.sin((x / 0.4) * Math.PI) * 2;
    } else if (x >= 0.5 && x < 0.6) {
      wave = -((x - 0.5) / 0.1) * 3;
    } else if (x >= 0.6 && x < 0.75) {
      wave = 11 - ((x - 0.6) / 0.15) * 14;
    } else if (x >= 0.75 && x < 0.85) {
      wave = -3 + ((x - 0.75) / 0.1) * 3;
    } else if (x >= 1.0 && x < 1.4) {
      wave = Math.sin(((x - 1.0) / 0.4) * Math.PI) * 3.2;
    }

    if (this.health < 30) {
      wave += (Math.random() - 0.5) * 1.5;
    }

    this.ecgPoints.push(h/2 - wave);
    this.ecgPoints.shift();

    ctx.strokeStyle = this.health > 60 ? '#00ff66' : (this.health > 30 ? '#ffcc00' : '#ff3333');
    ctx.lineWidth = 1.8;
    ctx.shadowBlur = 4;
    ctx.shadowColor = ctx.strokeStyle;
    
    ctx.beginPath();
    for (let i = 0; i < this.ecgPoints.length; i++) {
      if (i === 0) ctx.moveTo(i, this.ecgPoints[i]);
      else ctx.lineTo(i, this.ecgPoints[i]);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  damagePlayer(amount) {
    if (this.playerDead) return;
    this.health = Math.max(this.health - amount, 0);
    
    const hs = document.getElementById('health-status');
    if (hs) {
      if (this.health > 60) {
        hs.textContent = "FINE";
        hs.className = "fine";
      } else if (this.health > 30) {
        hs.textContent = "CAUTION";
        hs.className = "caution";
      } else if (this.health > 0) {
        hs.textContent = "DANGER";
        hs.className = "danger";
      } else {
        hs.textContent = "DEAD";
        hs.className = "danger";
        this.die();
      }
    }
  }

  die() {
    this.playerDead = true;
    this.fadeToAction('Sitting', 0.5);
    
    const blocker = document.getElementById('blocker');
    const header = blocker.querySelector('h1');
    const sub = blocker.querySelector('.subtitle');
    const pulse = blocker.querySelector('.start-pulse');
    
    if (header) header.textContent = "SYSTEM TERMINATED";
    if (sub) sub.textContent = `You reached Level ${this.gameLevel} | Kills: ${this.kills} | Points: ${this.points}`;
    if (pulse) pulse.textContent = "▶ Click to Respawn";
    
    blocker.style.background = "rgba(40, 10, 10, 0.85)";
    document.exitPointerLock();
  }

  respawn() {
    this.playerDead = false;
    this.gameLevel = 1;
    this.health = 100;
    this.kills = 0;
    this.points = 0;
    this.weaponLevel = 1;
    this.maxAmmo = 20;
    this.ammo = 20;
    this.isReloading = false;
    
    this.updateWeaponSystem();
    this.updateAmmoDisplay();
    
    const hs = document.getElementById('health-status');
    if (hs) {
      hs.textContent = "FINE";
      hs.className = "fine";
    }

    const lvlDisp = document.getElementById('level-display');
    if (lvlDisp) lvlDisp.textContent = "Level 1";

    const blocker = document.getElementById('blocker');
    const header = blocker.querySelector('h1');
    const sub = blocker.querySelector('.subtitle');
    const pulse = blocker.querySelector('.start-pulse');
    
    if (header) header.textContent = "Snow Mountain Explorer";
    if (sub) sub.textContent = "Click anywhere to begin your alpine adventure";
    if (pulse) pulse.textContent = "▶ Click to Start";
    blocker.style.background = "rgba(5, 10, 25, 0.7)";

    const rp = this.robotGroup.position;
    rp.set(0, this.getHeightAt(0, 0), 0);
    this.velocity.set(0, 0, 0);

    this.demons.forEach(d => {
      this.scene.remove(d.mesh);
      d.mesh.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
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

        const dist = Math.sqrt(x*x + z*z);
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

          const maxDemons = 6 + (this.gameLevel - 1) * 3;
          if (lod === 0 && Math.random() > 0.4 && this.demons.length < maxDemons) {
            this.spawnDemonInChunk(curX, curZ);
          }
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
        this.scene.remove(d.mesh);
        d.mesh.traverse(child => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) child.material.dispose();
        });
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

  // GUARANTEED FRONT SPAWNING: Spawns demons strictly within the view frustum direction
  spawnDemonInChunk(cx, cz) {
    const rp = this.robotGroup.position;
    this.camera.getWorldDirection(this._vecForward);
    this._vecForward.y = 0;
    this._vecForward.normalize();

    let foundValidSpawn = false;
    let rx, rz;
    
    for (let attempts = 0; attempts < 5; attempts++) {
      const angleOffset = (Math.random() - 0.5) * 1.5;
      const spawnAngle = Math.atan2(this._vecForward.x, this._vecForward.z) + angleOffset;
      const spawnDist  = 120 + Math.random() * 80; // Spawn them further out in the fog

      rx = rp.x + Math.sin(spawnAngle) * spawnDist;
      rz = rp.z + Math.cos(spawnAngle) * spawnDist;
      
      foundValidSpawn = true;
      break;
    }

    if (foundValidSpawn) {
      const maxDemons = 6 + (this.gameLevel - 1) * 3;
      if (this.demons.length < maxDemons) {
        const demon = new Demon(this.scene, rx, rz, this);
        this.demons.push(demon);
      }
    }
  }

  createBalloons() {
    this.balloons = [];
    const spots = [
      { x: 180, z: -250, color1: 0xd03030, color2: 0xfafafa },
      { x: -500, z: 400, color1: 0x3090d0, color2: 0xfafafa },
      { x: 600, z: 800, color1: 0xd09030, color2: 0x303030 }
    ];

    spots.forEach((spot, index) => {
      const group = new THREE.Group();
      const envGeo = new THREE.SphereGeometry(14, 24, 18);
      const envColors = new Float32Array(envGeo.attributes.position.count * 3);
      const tmpV = new THREE.Vector3();
      const c1 = new THREE.Color(spot.color1);
      const c2 = new THREE.Color(spot.color2);

      for (let i = 0; i < envGeo.attributes.position.count; i++) {
        tmpV.fromBufferAttribute(envGeo.attributes.position, i);
        const angle = Math.atan2(tmpV.x, tmpV.z);
        const isPattern = Math.sin(angle * 5) > 0;
        const c = isPattern ? c1 : c2;
        envColors[i * 3]     = c.r;
        envColors[i * 3 + 1] = c.g;
        envColors[i * 3 + 2] = c.b;
      }
      envGeo.setAttribute('color', new THREE.BufferAttribute(envColors, 3));

      const envMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.0 });
      const envelope = new THREE.Mesh(envGeo, envMat);
      envelope.scale.set(1, 1.35, 1);
      envelope.position.y = 18;
      envelope.castShadow = true;
      group.add(envelope);

      const basketGeo = new THREE.BoxGeometry(5.5, 3.5, 5.5);
      const basketMat = new THREE.MeshStandardMaterial({ color: 0x8B6914, roughness: 0.92 });
      const basket = new THREE.Mesh(basketGeo, basketMat);
      basket.position.y = -3;
      basket.castShadow = true;
      group.add(basket);

      const rim = new THREE.Mesh(new THREE.BoxGeometry(6, 0.6, 6), basketMat);
      rim.position.y = -1.2;
      group.add(rim);

      const ropeMat = new THREE.LineBasicMaterial({ color: 0x5a4020 });
      [[-2.5, 2.5], [2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]].forEach(([cx, cz]) => {
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(cx, -1.5, cz), new THREE.Vector3(cx * 1.8, 6, cz * 1.8)
        ]), ropeMat));
      });

      const y = this.getHeightAt(spot.x, spot.z) + 90;
      group.position.set(spot.x, y, spot.z);
      
      this.scene.add(group);
      this.balloons.push({
        group,
        baseY: y,
        speed: 0.25 + index * 0.06
      });
    });
  }

  createFogLayers() {
    this.fogLayers = [];
    const fogGeo = new THREE.PlaneGeometry(1600, 1600);
    fogGeo.rotateX(-Math.PI / 2);
    for (let i = 0; i < 4; i++) {
      const fogMat = new THREE.MeshBasicMaterial({
        color: 0xeeeeff,
        transparent: true,
        opacity: 0.04 - (i * 0.005),
        depthWrite: false
      });
      const fogMesh = new THREE.Mesh(fogGeo, fogMat);
      fogMesh.position.y = 15 + (i * 25);
      this.scene.add(fogMesh);
      this.fogLayers.push(fogMesh);
    }
  }

  createSnow() {
    const N = this.SNOW_COUNT;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(N * 3);
    this.snowVelocities = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * 900;
      positions[i3 + 1] = Math.random() * 300;
      positions[i3 + 2] = (Math.random() - 0.5) * 900;
      this.snowVelocities[i3] = (Math.random() - 0.5) * 4;
      this.snowVelocities[i3 + 1] = -(1.8 + Math.random() * 3.5);
      this.snowVelocities[i3 + 2] = (Math.random() - 0.5) * 4;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, transparent: true, opacity: 0.75, depthWrite: false, sizeAttenuation: true, fog: true });
    this.snowParticles = new THREE.Points(geo, mat);
    this.scene.add(this.snowParticles);
  }

  createFogLayers() {
    const planeGeo = new THREE.PlaneGeometry(2200, 2200);
    planeGeo.rotateX(-Math.PI / 2);
    [70, 110, 155, 200, 240].forEach(h => {
      const mat = new THREE.MeshBasicMaterial({ color: 0xc8dce8, transparent: true, opacity: 0.06 + Math.random() * 0.06, depthWrite: false, side: THREE.DoubleSide });
      const plane = new THREE.Mesh(planeGeo.clone(), mat);
      plane.position.y = h;
      this.fogLayers.push(plane);
      this.scene.add(plane);
    });
  }

  // ==============================================================
  //  CONTROLS & MOUSE LOCK
  // ==============================================================
  setupControls() {
    const canvas = this.renderer.domElement;

    document.getElementById('blocker').addEventListener('click', () => {
      if (this.playerDead) {
        this.respawn();
      }
      canvas.requestPointerLock();
    });

    document.addEventListener('pointerlockchange', () => {
      this.isLocked = document.pointerLockElement === canvas;
      document.getElementById('blocker').style.display = this.isLocked ? 'none' : 'flex';
      document.getElementById('hud').style.display = this.isLocked ? 'block' : 'none';
      if (this.isLocked) {
        document.body.classList.remove('pointer-unlocked');
      } else {
        document.body.classList.add('pointer-unlocked');
      }
    });

    document.addEventListener('contextmenu', e => e.preventDefault());

    document.addEventListener('mousemove', e => {
      if (!this.isLocked) {
        this.virtualCursorX = e.clientX;
        this.virtualCursorY = e.clientY;
        return;
      }

      if (Math.abs(e.movementX) > 120 || Math.abs(e.movementY) > 120) return;

      this.virtualCursorX = Math.max(0, Math.min(window.innerWidth, this.virtualCursorX + e.movementX));
      this.virtualCursorY = Math.max(0, Math.min(window.innerHeight, this.virtualCursorY + e.movementY));
    });

    document.addEventListener('wheel', e => {
      this.camDist = THREE.MathUtils.clamp(this.camDist + e.deltaY * 0.04, 18, 85);
    }, { passive: true });

    document.addEventListener('keydown', e => {
      this.keys[e.code] = true;
      if (e.code === 'KeyH') {
        this.showControls = !this.showControls;
        const p = document.getElementById('controls-panel');
        if (p) p.style.opacity = this.showControls ? '1' : '0';
      }
      if (e.code === 'KeyR') {
        this.reloadWeapon();
      }
    });
    document.addEventListener('keyup', e => { this.keys[e.code] = false; });

    document.addEventListener('mousedown', e => {
      if (!this.isLocked || this.playerDead) return;

      if (e.button === 0) { // Left Click
        const el = document.elementFromPoint(this.virtualCursorX, this.virtualCursorY);
        const slot = el ? el.closest('.inventory-slot') : null;

        if (slot) {
          const index = parseInt(slot.dataset.index);
          const item = this.inventory[index];
          if (item) {
            this.draggedItem = item;
            this.draggedFromSlot = index;
            slot.classList.add('dragging');
          }
        } else {
          // If hovering over a ground item, pick it up
          if (this.hoveredObject) {
            this.pickupGroundItem(this.hoveredObject);
          } else {
            // Normal shoot
            this.shootPrimary();
          }
        }
      } else if (e.button === 2) { // Right Click
        this.shootSecondary();
      }
    });

    document.addEventListener('mouseup', e => {
      if (!this.isLocked || this.playerDead) return;

      if (e.button === 0 && this.draggedItem) {
        const el = document.elementFromPoint(this.virtualCursorX, this.virtualCursorY);
        const slot = el ? el.closest('.inventory-slot') : null;

        // Remove dragging class
        const slots = document.querySelectorAll('.inventory-slot');
        slots.forEach(s => s.classList.remove('dragging'));

        if (slot) {
          const targetIndex = parseInt(slot.dataset.index);
          if (targetIndex === this.draggedFromSlot) {
            // Click to use or equip
            this.useInventoryItem(targetIndex);
          } else {
            // Swap items
            const temp = this.inventory[targetIndex];
            this.inventory[targetIndex] = this.draggedItem;
            this.inventory[this.draggedFromSlot] = temp;
            
            if (this.draggedItem.type === 'weapon') {
              this.activeSlot = targetIndex;
            }
            this.updateInventoryUI();
          }
        } else {
          // Dragged to 3D world: Drop
          this.dropItemOnGround(this.draggedItem, this.draggedFromSlot);
        }

        this.draggedItem = null;
        this.draggedFromSlot = -1;
      }
    });

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      if (this.composer) {
        this.composer.setSize(window.innerWidth, window.innerHeight);
      }
    });
  }

  setupPostProcessing() {
    this.composer = new EffectComposer(this.renderer);
    
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.35,
      0.4, 
      0.85 
    );
    this.composer.addPass(bloomPass);

    const colorGradingShader = {
      uniforms: {
        tDiffuse: { value: null },
        exposure: { value: 1.05 },
        contrast: { value: 1.05 },
        saturation: { value: 0.9 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform float exposure;
        uniform float contrast;
        uniform float saturation;
        varying vec2 vUv;
        void main() {
          vec4 texel = texture2D(tDiffuse, vUv);
          vec3 color = texel.rgb * exposure;
          color = (color - 0.5) * contrast + 0.5;
          float luma = dot(color, vec3(0.299, 0.587, 0.114));
          color = mix(vec3(luma), color, saturation);
          gl_FragColor = vec4(color, texel.a);
        }
      `
    };
    
    const colorGradPass = new ShaderPass(colorGradingShader);
    this.composer.addPass(colorGradPass);

    const outputPass = new OutputPass();
    this.composer.addPass(outputPass);
  }

  setupMinimap() {
    const canvas = document.getElementById('minimap');
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

    const sprinting = !!this.keys['ShiftLeft'];
    const speed = sprinting ? this.SPRINT_SPEED : this.MOVE_SPEED;

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

    const groundY = this.getHeightAt(rp.x, rp.z);
    if (rp.y <= groundY) {
      rp.y = groundY;
      this.velocity.y = 0;
      this.onGround = true;
    }

    // Smooth robot rotation
    if (isMoving && !this.playerDead) {
      const targetRot = Math.atan2(this.direction.x, this.direction.z);
      let diff = targetRot - this.robotGroup.rotation.y;
      while (diff >  Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      this.robotGroup.rotation.y += diff * 0.2;
    }

    if (this.onGround && !this.playerDead) {
      if (isMoving) {
        if (sprinting) {
          this.fadeToAction('Running', 0.2);
        } else {
          this.fadeToAction('Walking', 0.2);
        }
      } else {
        this.fadeToAction('Idle', 0.25);
      }
    }

    // Weapon stance, sway, & reload tilt animation
    if (this.gun) {
      const now = Date.now();
      const isAiming = (now - this.lastAimTime < 600); // ADS duration
      
      // Hip fire to Shoulder ADS transition
      const targetX = isAiming ? -0.12 : 0.22; // Move left towards center for ADS
      const targetY = isAiming ? -0.10 : -0.25;
      const targetZ = isAiming ? 0.70 : 0.35;
      
      // Walking / Sprinting bob
      const bobSpeed = sprinting ? 14 : 10;
      const bobScale = sprinting ? 0.08 : 0.04;
      const bobX = isMoving && !isAiming ? Math.sin(this.clock.elapsedTime * bobSpeed) * bobScale : 0;
      const bobY = isMoving && !isAiming ? Math.abs(Math.cos(this.clock.elapsedTime * bobSpeed)) * bobScale : 0;

      let reloadRotX = 0;
      let reloadRotZ = 0;
      let reloadOffsetY = 0;

      if (this.isReloading) {
        // Multi-stage reload animation
        const rT = (now - (this.reloadTimerStart || now)) / 1300; 
        reloadRotX = Math.sin(rT * Math.PI) * 0.9; // dip down
        reloadRotZ = Math.sin(rT * Math.PI * 2) * 0.5; // twist
        reloadOffsetY = -Math.sin(rT * Math.PI) * 0.6; // move off screen slightly
      }

      this.gun.position.x = THREE.MathUtils.lerp(this.gun.position.x, targetX + bobX, 0.15);
      this.gun.position.y = THREE.MathUtils.lerp(this.gun.position.y, targetY + bobY + reloadOffsetY, 0.15);
      this.gun.position.z = THREE.MathUtils.lerp(this.gun.position.z, targetZ, 0.15);
      
      this.gun.rotation.x = THREE.MathUtils.lerp(this.gun.rotation.x, reloadRotX, 0.15);
      this.gun.rotation.z = THREE.MathUtils.lerp(this.gun.rotation.z, reloadRotZ, 0.15);
    }

    this.updateThirdPersonCamera(dt);

    this.sun.position.set(rp.x + 400, 550, rp.z + 250);
    this.sun.target.position.copy(rp);
    this.sun.target.updateMatrixWorld();
  }

  updateThirdPersonCamera(dt) {
    const rp = this.robotGroup.position;
    this._vecLookTarget.set(rp.x, rp.y + 6, rp.z);

    const cosPitch = Math.cos(this.camPitch);
    const sinPitch = Math.sin(this.camPitch);
    
    this._vecDesiredPos.set(
      rp.x - Math.sin(this.camYaw) * cosPitch * this.camDist,
      rp.y + sinPitch * this.camDist + 6,
      rp.z - Math.cos(this.camYaw) * cosPitch * this.camDist
    );

    const terrainHeight = this.getHeightAt(this._vecDesiredPos.x, this._vecDesiredPos.z);
    if (this._vecDesiredPos.y < terrainHeight + 3) {
      this._vecDesiredPos.y = terrainHeight + 3;
    }

    this.camera.position.lerp(this._vecDesiredPos, 0.12);
    this.camLookTarget.lerp(this._vecLookTarget, 0.12);
    this.camera.lookAt(this.camLookTarget);

    if (this.screenShake > 0) {
      this.camera.position.x += (Math.random() - 0.5) * this.screenShake * 3.0;
      this.camera.position.y += (Math.random() - 0.5) * this.screenShake * 3.0;
      this.screenShake = Math.max(0, this.screenShake - dt * 3.0);
    }
  }

  updateSnow(dt) {
    if (!this.snowParticles) return;
    const arr = this.snowParticles.geometry.attributes.position.array;
    const t = this.clock.elapsedTime;
    for (let i = 0; i < this.SNOW_COUNT; i++) {
      const i3 = i * 3;
      const wind = Math.sin(t * 0.4 + i * 0.007) * 3.5;
      arr[i3]     += (this.snowVelocities[i3] + wind) * dt;
      arr[i3 + 1] += this.snowVelocities[i3 + 1] * dt;
      arr[i3 + 2] += (this.snowVelocities[i3 + 2] + wind * 0.6) * dt;

      if (arr[i3 + 1] < -30) {
        arr[i3]     = (Math.random() - 0.5) * 900;
        arr[i3 + 1] = 200 + Math.random() * 100;
        arr[i3 + 2] = (Math.random() - 0.5) * 900;
      }
    }
    const rp = this.robotGroup.position;
    this.snowParticles.position.set(rp.x, 0, rp.z);
    this.snowParticles.geometry.attributes.position.needsUpdate = true;
  }

  updateBalloons(dt) {
    if (!this.balloons) return;
    const t = this.clock.elapsedTime;
    this.balloons.forEach(balloon => {
      balloon.group.position.y = balloon.baseY + Math.sin(t * balloon.speed) * 8;
      balloon.group.rotation.y += dt * 0.04;
    });
  }

  updateFogLayers() {
    const t = this.clock.elapsedTime;
    this.fogLayers.forEach((fog, i) => {
      fog.position.x = Math.sin(t * 0.07 + i * 1.3) * 30;
      fog.position.z = Math.cos(t * 0.05 + i * 2.1) * 30;
    });
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
  }

  updateVirtualCursor(dt) {
    // Position DOM element
    const cursorEl = document.getElementById('virtual-cursor');
    if (cursorEl) {
      cursorEl.style.left = `${this.virtualCursorX}px`;
      cursorEl.style.top = `${this.virtualCursorY}px`;
    }

    if (!this.isLocked || this.playerDead) return;

    // Convert mouse position to NDC (-1 to +1) for raycasting
    const ndcX = (this.virtualCursorX / window.innerWidth) * 2 - 1;
    const ndcY = -(this.virtualCursorY / window.innerHeight) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera({ x: ndcX, y: ndcY }, this.camera);

    const interactives = [];
    this.groundItems.forEach(item => {
      if (!item.isDead) interactives.push(item.mesh);
    });

    const intersects = raycaster.intersectObjects(interactives, true);
    
    this.hoveredObject = null;
    let nextState = 'default';

    if (this.draggedItem) {
      nextState = 'place';
    } else if (intersects.length > 0) {
      const hit = intersects[0];
      if (hit.distance < 120) { 
        if (hit.object.userData.isGroundItem) {
          this.hoveredObject = hit.object.userData.parentItem;
          nextState = 'hover';
        }
      }
    }

    // Check if virtual cursor is hovering over an inventory slot
    const el = document.elementFromPoint(this.virtualCursorX, this.virtualCursorY);
    if (el && el.closest('.inventory-slot')) {
      if (this.draggedItem) nextState = 'place';
      else nextState = 'equip';
    }

    if (this.cursorState !== nextState) {
      this.cursorState = nextState;
      if (cursorEl) {
        cursorEl.className = '';
        cursorEl.classList.add(this.cursorState);
      }
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
          slot.innerHTML = `<span class="item-icon">${item.icon}</span><span class="item-label">${item.name}</span>`;
        } else {
          slot.innerHTML = `<span class="item-label">Empty</span>`;
        }
      }
    });
  }

  useInventoryItem(index) {
    const item = this.inventory[index];
    if (!item) return;

    if (item.type === 'weapon') {
      this.activeSlot = index;
      this.updateInventoryUI();
      this.playActionSound();
      
      if (item.id === 'laser') {
        this.weaponLevel = 1;
      } else if (item.id === 'plasma') {
        this.weaponLevel = 2;
      } else if (item.id === 'railgun') {
        this.weaponLevel = 3;
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
        this.ammo = Math.min(this.maxAmmo, this.ammo + 15);
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
    const ndcX = (this.virtualCursorX / window.innerWidth) * 2 - 1;
    const ndcY = -(this.virtualCursorY / window.innerHeight) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera({ x: ndcX, y: ndcY }, this.camera);
    
    const chunks = Array.from(this.chunks.values()).map(c => c.mesh).filter(Boolean);
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

    if (this.mixer) this.mixer.update(dt);

    const rp = this.robotGroup.position;
    this.demons.forEach(demon => {
      demon.update(dt, rp);
    });

    this.enemyProjectiles = this.enemyProjectiles.filter(p => {
      p.update(dt);
      return !p.isDead;
    });

    this.playerRockets = this.playerRockets.filter(r => {
      r.update(dt);
      return !r.isDead;
    });

    // Update ground items
    this.groundItems = this.groundItems.filter(item => {
      item.update(dt);
      return !item.isDead;
    });

    this.updatePlayer(dt);
    this.updateChunks();
    this.updateSnow(dt);
    this.updateBalloons(dt);
    this.updateFogLayers();
    this.updateECG(dt);
    this.updateHUD(dt);
    this.updateMinimap();
    this.updateVirtualCursor(dt);

    this.composer.render();
  }
}

window.addEventListener('DOMContentLoaded', () => new Game());
