import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const cap = 2_000_000;

function parse(text) {
  const rows = text.split(/\r?\n/).filter((row) => row.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((row) => row.length));
  const floor = new Uint8Array(w * h);
  const goals = [];
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ch = rows[y][x] ?? '#';
      const cell = y * w + x;
      if (ch === '#') continue;
      floor[cell] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(cell);
      if (ch === '$' || ch === '*') boxes.push(cell);
      if (ch === '@' || ch === '+') player = cell;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) floor[y * w + x] = 0;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, floor, goals, boxes, player };
}

function components(level) {
  const { w, h, floor, boxes, goals, player } = level;
  if (player < 0) return null;
  const hallRow = Math.floor(player / w);
  const part = new Int32Array(w * h).fill(-1);
  const rooms = [];
  for (let start = 0; start < floor.length; start++) {
    if (!floor[start] || Math.floor(start / w) === hallRow || part[start] >= 0) continue;
    const cells = [start];
    part[start] = rooms.length;
    for (let head = 0; head < cells.length; head++) {
      const p = cells[head];
      for (const d of [-w, w, -1, 1]) {
        const q = p + d;
        if (floor[q] && Math.floor(q / w) !== hallRow && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
    }
    rooms.push({ cells, boxes: [], goals: [], doors: [] });
  }
  for (const b of boxes) {
    if (Math.floor(b / w) === hallRow || part[b] < 0) return null;
    rooms[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (Math.floor(g / w) === hallRow || part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  const hall = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) if (floor[hallRow * w + x]) hall[hallRow * w + x] = 1;
  for (const room of rooms) {
    for (const cell of room.cells) {
      for (const d of [-w, w]) {
        const q = cell + d;
        if (Math.floor(q / w) === hallRow && hall[q]) room.doors.push({ inside: cell, outside: q });
      }
    }
    room.doors = room.doors.filter((door, i, list) => list.findIndex((other) => other.inside === door.inside && other.outside === door.outside) === i);
    if (room.boxes.length !== room.goals.length || room.doors.length !== 1) return null;
    room.door = room.doors[0].inside;
    room.hallDoor = room.doors[0].outside;
    room.floor = new Uint8Array(w * h);
    for (const cell of room.cells) room.floor[cell] = 1;
  }
  const outsideDoors = new Set();
  for (const room of rooms) {
    if (outsideDoors.has(room.hallDoor)) return null;
    outsideDoors.add(room.hallDoor);
  }
  rooms.sort((a, b) => (a.hallDoor % w) - (b.hallDoor % w));
  return { hallRow, hall, rooms };
}

function reverseDistances(floor, w, goals) {
  return goals.map((goal) => {
    const dist = new Int16Array(floor.length).fill(-1);
    const queue = [goal];
    dist[goal] = 0;
    for (let head = 0; head < queue.length; head++) {
      const cell = queue[head];
      for (const d of [-w, w, -1, 1]) {
        const prior = cell - d;
        const support = prior - d;
        if (floor[prior] && floor[support] && dist[prior] < 0) {
          dist[prior] = dist[cell] + 1;
          queue.push(prior);
        }
      }
    }
    return dist;
  });
}

function matchingHeuristic(boxes, goalDistances) {
  const count = boxes.length;
  let best = Infinity;
  function assign(i, used, sum) {
    if (sum >= best) return;
    if (i === count) {
      best = sum;
      return;
    }
    for (let g = 0; g < count; g++) {
      if (used & (1 << g)) continue;
      const distance = goalDistances[g][boxes[i]];
      if (distance >= 0) assign(i + 1, used | (1 << g), sum + distance);
    }
  }
  assign(0, 0, 0);
  return best;
}

function assignmentHeuristic(boxes, goalDistances) {
  const n = boxes.length;
  if (n <= 4) return matchingHeuristic(boxes, goalDistances);
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const raw = goalDistances[j - 1][boxes[i0 - 1]];
        const cost = raw < 0 ? 1e9 : raw;
        const current = cost - u[i0] - v[j];
        if (current < minv[j]) {
          minv[j] = current;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  const result = -v[0];
  return result >= 1e9 ? Infinity : result;
}

function reachable(floor, boxes, start, w) {
  const blocked = new Uint8Array(floor.length);
  for (const b of boxes) blocked[b] = 1;
  const seen = new Uint8Array(floor.length);
  const queue = [start];
  let low = start;
  seen[start] = 1;
  for (let head = 0; head < queue.length; head++) {
    const p = queue[head];
    if (p < low) low = p;
    for (const d of [-w, w, -1, 1]) {
      const q = p + d;
      if (floor[q] && !blocked[q] && !seen[q]) {
        seen[q] = 1;
        queue.push(q);
      }
    }
  }
  return { seen, low };
}

function isCorner(floor, box, w) {
  const up = !floor[box - w];
  const down = !floor[box + w];
  const left = !floor[box - 1];
  const right = !floor[box + 1];
  return (up || down) && (left || right);
}

function frozen2x2(floor, goalMask, occupied, box, w) {
  for (const top of [box, box - 1, box - w, box - w - 1]) {
    const cells = [top, top + 1, top + w, top + w + 1];
    if (!cells.every((cell) => !floor[cell] || occupied[cell])) continue;
    if (cells.some((cell) => goalMask[cell])) continue;
    if (cells.some((cell) => occupied[cell])) return true;
  }
  return false;
}

function makeHeap(nodes, astar) {
  const heap = [];
  const less = (a, b) => {
    const na = nodes[a];
    const nb = nodes[b];
    const fa = astar ? na.g + na.h : na.h;
    const fb = astar ? nb.g + nb.h : nb.h;
    if (fa !== fb) return fa < fb;
    if (na.g !== nb.g) return na.g > nb.g;
    return na.key < nb.key;
  };
  return {
    get length() { return heap.length; },
    push(value) {
      let i = heap.length;
      heap.push(value);
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (!less(heap[i], heap[parent])) break;
        [heap[i], heap[parent]] = [heap[parent], heap[i]];
        i = parent;
      }
    },
    pop() {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let child = 2 * i + 1;
          if (child >= heap.length) break;
          if (child + 1 < heap.length && less(heap[child + 1], heap[child])) child++;
          if (!less(heap[child], heap[i])) break;
          [heap[child], heap[i]] = [heap[i], heap[child]];
          i = child;
        }
      }
      return top;
    },
  };
}

function search(floor, w, goals, startBoxes, startPlayer, exitDoor, astar, limit) {
  const goalMask = new Uint8Array(floor.length);
  for (const g of goals) goalMask[g] = 1;
  const goalDistances = reverseDistances(floor, w, goals);
  const live = new Uint8Array(floor.length);
  for (const dist of goalDistances) for (let i = 0; i < dist.length; i++) if (dist[i] >= 0) live[i] = 1;
  if (startBoxes.some((b) => !live[b])) return { moves: null, expanded: 0, capped: false };
  const hCache = new Map();
  const heuristic = (boxes) => {
    const key = boxes.join(',');
    if (!hCache.has(key)) hCache.set(key, assignmentHeuristic(boxes, goalDistances));
    return hCache.get(key);
  };
  const initialBoxes = startBoxes.slice().sort((a, b) => a - b);
  const initialReach = reachable(floor, initialBoxes, startPlayer, w);
  const initialKey = `${initialBoxes.join(',')}|${initialReach.low}`;
  const nodes = [{ boxes: initialBoxes, player: initialReach.low, g: 0, h: heuristic(initialBoxes), key: initialKey, parent: -1, from: -1, delta: 0 }];
  const best = new Map([[initialKey, 0]]);
  const heap = makeHeap(nodes, astar);
  heap.push(0);
  const occupied = new Uint8Array(floor.length);
  let expanded = 0;
  let found = -1;
  while (heap.length && expanded < limit) {
    const index = heap.pop();
    const node = nodes[index];
    if (best.get(node.key) !== node.g) continue;
    const reach = reachable(floor, node.boxes, node.player, w);
    if (node.boxes.every((b) => goalMask[b]) && (exitDoor < 0 || reach.seen[exitDoor])) {
      found = index;
      break;
    }
    expanded++;
    for (const b of node.boxes) occupied[b] = 1;
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const box = node.boxes[bi];
      for (const d of [-w, w, -1, 1]) {
        const support = box - d;
        const target = box + d;
        if (!reach.seen[support] || !floor[target] || occupied[target]) continue;
        occupied[box] = 0;
        occupied[target] = 1;
        if ((!live[target] && !goalMask[target]) || (!goalMask[target] && isCorner(floor, target, w)) || frozen2x2(floor, goalMask, occupied, target, w)) {
          occupied[target] = 0;
          occupied[box] = 1;
          continue;
        }
        const boxes = node.boxes.slice();
        boxes[bi] = target;
        boxes.sort((a, b) => a - b);
        const nextReach = reachable(floor, boxes, box, w);
        const key = `${boxes.join(',')}|${nextReach.low}`;
        const g = node.g + 1;
        const prior = best.get(key);
        if (prior === undefined || g < prior) {
          const h = heuristic(boxes);
          if (Number.isFinite(h)) {
            const child = nodes.length;
            nodes.push({ boxes, player: nextReach.low, g, h, key, parent: index, from: box, delta: d });
            best.set(key, g);
            heap.push(child);
          }
        }
        occupied[target] = 0;
        occupied[box] = 1;
      }
    }
    for (const b of node.boxes) occupied[b] = 0;
  }
  if (found < 0) return { moves: null, expanded, capped: heap.length > 0 && expanded >= limit };
  const path = [];
  for (let at = found; nodes[at].parent >= 0; at = nodes[at].parent) path.push(at);
  path.reverse();
  let moves = '';
  let boxes = initialBoxes;
  let player = startPlayer;
  for (const index of path) {
    const step = nodes[index];
    const support = step.from - step.delta;
    moves += walkPath(floor, boxes, player, support, w);
    moves += step.delta === -w ? 'u' : step.delta === w ? 'd' : step.delta === -1 ? 'l' : 'r';
    player = step.from;
    boxes = step.boxes;
  }
  return { moves, expanded, capped: false };
}

function walkPath(floor, boxes, start, target, w) {
  if (start === target) return '';
  const blocked = new Uint8Array(floor.length);
  for (const b of boxes) blocked[b] = 1;
  const prev = new Int32Array(floor.length).fill(-1);
  const how = new Array(floor.length);
  const queue = [start];
  prev[start] = start;
  const deltas = [-w, w, -1, 1];
  const letters = ['u', 'd', 'l', 'r'];
  for (let head = 0; head < queue.length && prev[target] < 0; head++) {
    const p = queue[head];
    for (let k = 0; k < 4; k++) {
      const q = p + deltas[k];
      if (!floor[q] || blocked[q] || prev[q] >= 0) continue;
      prev[q] = p;
      how[q] = letters[k];
      queue.push(q);
    }
  }
  if (prev[target] < 0) throw new Error('requested walk path is unreachable');
  let result = '';
  for (let p = target; p !== start; p = prev[p]) result = how[p] + result;
  return result;
}

function applyMoves(level, state, moves) {
  for (const letter of moves) {
    const d = letter === 'u' ? -level.w : letter === 'd' ? level.w : letter === 'l' ? -1 : 1;
    const next = state.player + d;
    if (!level.floor[next]) throw new Error('solver generated a step into a wall');
    const boxAt = state.boxes.indexOf(next);
    if (boxAt >= 0) {
      const beyond = next + d;
      if (!level.floor[beyond] || state.boxes.includes(beyond)) throw new Error('solver generated an illegal push');
      state.boxes[boxAt] = beyond;
      state.boxes.sort((a, b) => a - b);
    }
    state.player = next;
  }
}

function solveRooms(level, layout, id) {
  const state = { player: level.player, boxes: level.boxes.slice() };
  let moves = '';
  for (let i = 0; i < layout.rooms.length; i++) {
    const room = layout.rooms[i];
    const hallWalk = walkPath(layout.hall, [], state.player, room.hallDoor, level.w);
    moves += hallWalk;
    applyMoves(level, state, hallWalk);
    const enter = room.door - room.hallDoor;
    const enterMove = enter === -level.w ? 'u' : enter === level.w ? 'd' : enter === -1 ? 'l' : 'r';
    moves += enterMove;
    applyMoves(level, state, enterMove);
    const roomBoxes = state.boxes.filter((b) => room.floor[b]).sort((a, b) => a - b);
    const roomGoals = room.goals.slice().sort((a, b) => a - b);
    const optimal = search(room.floor, level.w, roomGoals, roomBoxes, room.door, room.door, true, cap);
    console.error(`[solver] ${id} room ${i + 1}: A* expanded ${optimal.expanded}/${cap}${optimal.capped ? ' (cap)' : ''}`);
    let answer = optimal;
    if (answer.moves === null) {
      answer = search(room.floor, level.w, roomGoals, roomBoxes, room.door, room.door, false, cap);
      console.error(`[solver] ${id} room ${i + 1}: greedy expanded ${answer.expanded}/${cap}${answer.capped ? ' (cap)' : ''}`);
    }
    if (answer.moves === null) return null;
    moves += answer.moves;
    applyMoves(level, state, answer.moves);
    const roomWalk = walkPath(room.floor, roomGoals, state.player, room.door, level.w);
    moves += roomWalk;
    applyMoves(level, state, roomWalk);
    const exit = room.hallDoor - room.door;
    const exitMove = exit === -level.w ? 'u' : exit === level.w ? 'd' : exit === -1 ? 'l' : 'r';
    moves += exitMove;
    applyMoves(level, state, exitMove);
  }
  if (!state.boxes.every((b) => level.goals.includes(b))) return null;
  return moves;
}

function solveWhole(level, id) {
  const answer = search(level.floor, level.w, level.goals, level.boxes, level.player, -1, true, 100_000);
  console.error(`[solver] ${id}: whole-level fallback A* expanded ${answer.expanded}/100000${answer.capped ? ' (cap)' : ''}`);
  return answer.moves;
}

function solve(text, id) {
  const level = parse(text);
  const layout = components(level);
  if (!layout) {
    console.error(`[solver] ${id}: room decomposition failed; using whole-level fallback`);
    return solveWhole(level, id) ?? '';
  }
  console.error(`[rooms] ${id}: ${layout.rooms.length} room(s): ${layout.rooms.map((room) => `boxes=${room.boxes.length},goals=${room.goals.length},door=${room.door % level.w}`).join('; ')}`);
  const moves = solveRooms(level, layout, id);
  if (moves !== null) return moves;
  console.error(`[solver] ${id}: room search failed; using whole-level fallback`);
  const fallback = solveWhole(level, id);
  if (fallback === null) console.error(`[solver] ${id}: no fallback solution found; writing empty answer`);
  return fallback ?? '';
}

for (const id of readdirSync(inputDir).sort()) {
  const moves = solve(readFileSync(join(inputDir, id, 'level.txt'), 'utf8'), id);
  mkdirSync(join(outputDir, id), { recursive: true });
  writeFileSync(join(outputDir, id, 'solution.txt'), `${moves}\n`);
}
