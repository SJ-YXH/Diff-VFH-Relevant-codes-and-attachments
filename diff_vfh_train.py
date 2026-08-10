# -*- coding: utf-8 -*-
"""
diff_vfh_train_three_stage_min30_60_90_inherit_speed.py

Revision note for this edited version:
  - Dynamic obstacles default to whole-arena random initialization and random velocity.
  - Route-crossing enforcement is disabled by default; crossing/forced modes remain available through CLI.

v14 fixes DER promotion, terminal tqdm, and forced-scene fallback:

  - DER is recorded/logged/displayed only; it is not used as a promotion gate.
  - tqdm is the default terminal display; rolling criteria prints are disabled by default.
  - forced dynamic scene generation accepts the best complete fallback scene when
    strict route-crossing validation fails but route_ratio is still acceptable.


Three-stage Diff-VFH training code for SCI experiments. Dense uniform static scene generation is enabled by default.

This revision sets stage-level minimum training windows to Stage 1=30, Stage 2=60, Stage 3=90 episodes, and Stage 3 dynamic speed inherits the previous stage end speed instead of using an independent stage-start speed.

Three-stage Diff-VFH training code: UAV speed is encouraged by a gated soft loss, but curriculum promotion does not use average flight speed as a hard gate. vmax is constrained to [2.00, 8.00] m/s.

This version adds a stronger speed-push mechanism because the previous gated speed
loss could be suppressed by occupancy/TTC gates, causing the UAV to cruise around
3.3 m/s. The new rollout applies a safety-gated cruise-speed floor and uses a
less over-conservative speed modulation in DiffVFH.forward().

v6 keeps forced-encounter dynamic obstacle generation and faster training
defaults. Since dynamic obstacles are already forced to cross the nominal UAV
route, dynamic-stage promotion no longer relies on DER as a hard threshold;
instead, it emphasizes success, true goal-reaching, collision rate, dynamic
collision rate, and dynamic safety.

v12 changes the static-obstacle generator:

  - no route/corridor/gate-first obstacle generation;
  - no pre-designed path is used before placing obstacles;
  - static obstacles are distributed over the whole rectangular arena by
    stratified Poisson-disk-like sampling;
  - minimum surface gap between obstacles is larger than the UAV body/safety
    passage width, so the UAV can physically pass between obstacles;
  - start/goal safety zones are kept free;
  - after generation, BFS traversability checks only verify that a feasible path
    exists; they do not impose a planned route;
  - every completed sub-level inside Stage 1-4 can be visualized automatically
    to inspect the scenario and trajectory.

Recommended stage setting:
  Stage 1: static 50, dynamic 0; train directly on the final uniform-static difficulty.
  Stage 2: dynamic scene starts with static reset to --dynamic_static_start; dynamic 1 -> 6 while static decreases to --dynamic_static_end.
           The dynamic-stage logic follows the previous Stage 3 curriculum, but is now the second training stage.
  Stage 4: disabled by default. Additional high-speed dynamic-obstacle training is not considered.

Run default all stages, ending after Stage 3:
  python train_diff_vfh_v5_no_stage4.py --stage all --episodes 3000 --device cuda --seed 2026 --vis_o3d

Run only Stage 1:
  python train_diff_vfh_stagewise_fast_static_curriculum.py --stage 1 --stage1_max_episodes 800 --device cuda --seed 2026 --vis_o3d

Visualize a trained checkpoint:
  python train_diff_vfh_stagewise_v12_uniform_static.py --mode visualize --stage 3 --init_checkpoint checkpoints/<run>/stage3_diff_vfh.pth --device cuda --seed 2026 --vis_episodes 3
"""

from __future__ import annotations
import argparse, math, os, random, time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ----------------------------- reproducibility -----------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def ep_rng(seed: int, ep: int, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed + 1000003 * ep + 91763 * stream)


# ----------------------------- data classes -----------------------------

@dataclass
class CircleObstacle:
    x: float; y: float; r: float; height: float = 1.25

@dataclass
class RectObstacle:
    x: float; y: float; sx: float; sy: float; yaw: float = 0.0; height: float = 1.25
    @property
    def proxy_radius(self) -> float:
        return 0.5 * math.sqrt(self.sx * self.sx + self.sy * self.sy)

@dataclass
class DynamicObstacle:
    x: float; y: float; r: float; vx: float; vy: float; crossing: bool = False; height: float = 1.0; motion_mode: str = "random"; cross_x: float = 0.0; cross_y: float = 0.0; lane_x: float = 0.0

@dataclass
class Scene:
    width: float; height: float
    start: np.ndarray; goal: np.ndarray
    static_circles: List[CircleObstacle]; static_rects: List[RectObstacle]
    dynamic_init: List[DynamicObstacle]
    dynamic_traj: np.ndarray; dynamic_vel_traj: np.ndarray
    dt: float; max_steps: int; goal_radius: float
    stage_id: int; stage_name: str
    static_target: int; dynamic_target: int; dynamic_speed: float

@dataclass
class CurState:
    stage_id: int = 1
    stage_name: str = "S1_static_15_to_30"
    static_count: int = 15
    dynamic_count: int = 0
    dynamic_speed: float = 0.0
    level_ep: int = 0
    stage_ep: int = 0


# ----------------------------- curriculum -----------------------------

class Curriculum:
    def __init__(self, cfg):
        self.cfg = cfg
        self.s = CurState(static_count=cfg.stage1_static_start)
        self.win_succ = deque(maxlen=cfg.level_window)
        self.win_col = deque(maxlen=cfg.level_window)
        self.win_den = deque(maxlen=cfg.level_window)
        self.win_dcol = deque(maxlen=cfg.level_window)
        self.win_dsafe = deque(maxlen=cfg.level_window)
        self.win_reached = deque(maxlen=cfg.level_window)
        self.win_speed = deque(maxlen=cfg.level_window)
        self.log = []

    def settings(self) -> Dict[str, float]:
        return dict(
            stage_id=self.s.stage_id, stage_name=self.s.stage_name,
            static_count=self.s.static_count, dynamic_count=self.s.dynamic_count,
            dynamic_speed=self.s.dynamic_speed, level_ep=self.s.level_ep, stage_ep=self.s.stage_ep,
            lsr=float(np.mean(self.win_succ)) if self.win_succ else 0.0,
            lcr=float(np.mean(self.win_col)) if self.win_col else 0.0,
            lder=float(np.mean(self.win_den)) if self.win_den else 0.0,
            ldcr=float(np.mean(self.win_dcol)) if self.win_dcol else 0.0,
            ldsafe=float(np.mean(self.win_dsafe)) if self.win_dsafe else 0.0,
            lrr=float(np.mean(self.win_reached)) if self.win_reached else 0.0,
            lvavg=float(np.mean(self.win_speed)) if self.win_speed else 0.0,
        )

    def _clear_windows(self):
        self.s.level_ep = 0
        self.win_succ.clear(); self.win_col.clear(); self.win_den.clear(); self.win_dsafe.clear(); self.win_reached.clear(); self.win_speed.clear()

    def _static_for_dynamic(self, dynamic_count: int) -> int:
        d0 = int(self.cfg.stage3_dynamic_start)
        d1 = int(self.cfg.stage3_dynamic_end)
        n0 = int(self.cfg.dynamic_static_start)
        n1 = int(self.cfg.dynamic_static_end)
        if d1 <= d0:
            return n0
        alpha = (int(dynamic_count) - d0) / float(d1 - d0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        # power < 1.0 decreases static density earlier, giving Nd=4 more lateral
        # space while preserving the same start/end counts.
        power = float(getattr(self.cfg, "dynamic_static_schedule_power", 1.0))
        if power > 1e-6:
            alpha = float(alpha ** power)
        return int(round(n0 + alpha * (n1 - n0)))

    def _set_stage(self, sid: int):
        if sid == 1:
            self.s = CurState(1, "S1_static_15_to_30", self.cfg.stage1_static_start, 0, 0.0)
        elif sid == 2:
            self.s = CurState(2, "S2_static_30_to_45", self.cfg.stage2_static_start, 0, 0.0)
        elif sid == 3:
            self.s = CurState(3, "S3_dynamic_1_to_6", self._static_for_dynamic(self.cfg.stage3_dynamic_start), self.cfg.stage3_dynamic_start, self.cfg.stage3_dynamic_speed)
        else:
            self.s = CurState(4, "S4_speed_1_to_5", self._static_for_dynamic(self.cfg.stage3_dynamic_end), self.cfg.stage3_dynamic_end, self.cfg.stage4_speed_start)
        self._clear_windows()

    def step(self, success: float, collision: float, dyn_enc: float, dyn_safe: float, ep: int, writer=None):
        self.s.level_ep += 1; self.s.stage_ep += 1
        self.win_succ.append(success); self.win_col.append(collision)
        self.win_den.append(dyn_enc); self.win_dsafe.append(dyn_safe)

        dynamic_stage = self.s.stage_id >= 3
        min_ep = self.cfg.dynamic_min_level_episodes if dynamic_stage else self.cfg.min_level_episodes
        sr_thr = self.cfg.dynamic_success_threshold if dynamic_stage else self.cfg.level_success_threshold
        cr_thr = self.cfg.dynamic_collision_threshold if dynamic_stage else self.cfg.level_collision_threshold
        if len(self.win_succ) < self.cfg.level_window or self.s.level_ep < min_ep:
            return False, None

        sr, cr = float(np.mean(self.win_succ)), float(np.mean(self.win_col))
        der, dsr = float(np.mean(self.win_den)), float(np.mean(self.win_dsafe))
        ok = sr >= sr_thr and cr <= cr_thr
        if dynamic_stage:
            # The previous fixed DER threshold was too strict when Nd=1.
            # A single moving obstacle cannot guarantee >0.55 encounter rate in a large arena,
            # even when it is path-crossing.  Therefore the required encounter rate is
            # increased gradually with the number of dynamic obstacles.
            der_thr = min(
                self.cfg.dynamic_encounter_threshold,
                self.cfg.dynamic_encounter_threshold_base
                + self.cfg.dynamic_encounter_threshold_slope * max(0, self.s.dynamic_count - 1),
            )
            ok = ok and dsr >= self.cfg.dynamic_safe_threshold  # DER is logged only; not a promotion gate.
        if not ok:
            return False, None

        rec = dict(ep=ep, stage_id=self.s.stage_id, stage=self.s.stage_name,
                   static=self.s.static_count, dynamic=self.s.dynamic_count,
                   dyn_speed=self.s.dynamic_speed, sr=sr, cr=cr, der=der, dsr=dsr)
        self.log.append(rec)
        if writer:
            writer.add_text("curriculum/transition", str(rec), ep)

        if self.s.stage_id == 1:
            if self.s.static_count < self.cfg.stage1_static_end:
                self.s.static_count = min(self.cfg.stage1_static_end, self.s.static_count + self.cfg.static_increment)
                msg = f"S1 difficulty increased: static -> {self.s.static_count}"
                self._clear_windows(); return False, msg
            msg = "S1 converged at 30 static obstacles. Entering S2."
            self._set_stage(2); return False, msg

        if self.s.stage_id == 2:
            if self.s.static_count < self.cfg.stage2_static_end:
                self.s.static_count = min(self.cfg.stage2_static_end, self.s.static_count + self.cfg.static_increment)
                msg = f"S2 difficulty increased: static -> {self.s.static_count}"
                self._clear_windows(); return False, msg
            msg = "S2 converged at 45 static obstacles. Entering S3."
            self._set_stage(3); return False, msg

        if self.s.stage_id == 3:
            if self.s.dynamic_count < self.cfg.stage3_dynamic_end:
                self.s.dynamic_count = min(self.cfg.stage3_dynamic_end, self.s.dynamic_count + self.cfg.dynamic_increment)
                self.s.static_count = self._static_for_dynamic(self.s.dynamic_count)
                msg = f"S3 difficulty increased: dynamic -> {self.s.dynamic_count}, static -> {self.s.static_count}"
                self._clear_windows(); return False, msg
            msg = f"S3 converged at {self.cfg.stage3_dynamic_end} dynamic obstacles and {self.s.static_count} static obstacles. Entering S4."
            self._set_stage(4); return False, msg

        if self.s.dynamic_speed < self.cfg.stage4_speed_end - 1e-6:
            self.s.dynamic_speed = min(self.cfg.stage4_speed_end, self.s.dynamic_speed + self.cfg.speed_increment)
            msg = "S4 difficulty increased: dynamic-obstacle difficulty increased."
            self._clear_windows(); return False, msg
        return self.cfg.early_stop_after_final_stage, "S4 converged. Training can stop."


# ----------------------------- geometry -----------------------------

def nrm(v): return float(np.linalg.norm(v))

def in_safe(x, y, r, start, goal, safe):
    p = np.array([x, y], dtype=np.float64)
    return nrm(p - start) < safe + r or nrm(p - goal) < safe + r

def sep_circles(x, y, r, obs, gap):
    return all(math.hypot(x - o.x, y - o.y) >= r + o.r + gap for o in obs)

def sep_rects(x, y, r, rects, gap):
    return all(math.hypot(x - o.x, y - o.y) >= r + o.proxy_radius + gap for o in rects)

def proxy(rect: RectObstacle) -> CircleObstacle:
    return CircleObstacle(rect.x, rect.y, rect.proxy_radius, rect.height)

def proxies(cirs, rects):
    return list(cirs) + [proxy(r) for r in rects]

def traversable(width, height, start, goal, obs, clearance, res):
    nx, ny = int(math.ceil(width / res)), int(math.ceil(height / res))
    xs = np.linspace(-width / 2 + res / 2, width / 2 - res / 2, nx)
    ys = np.linspace(-height / 2 + res / 2, height / 2 - res / 2, ny)
    occ = np.zeros((nx, ny), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if x < -width/2 + clearance or x > width/2 - clearance or y < -height/2 + clearance or y > height/2 - clearance:
                occ[i, j] = True; continue
            for o in obs:
                if math.hypot(x-o.x, y-o.y) < o.r + clearance:
                    occ[i, j] = True; break
    def idx(p):
        return int(np.clip(np.searchsorted(xs, float(p[0])), 0, nx-1)), int(np.clip(np.searchsorted(ys, float(p[1])), 0, ny-1))
    s, g = idx(start), idx(goal)
    if occ[s] or occ[g]: return False
    q, seen = deque([s]), np.zeros_like(occ, dtype=bool); seen[s] = True
    nb = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        i, j = q.popleft()
        if (i, j) == g: return True
        for di, dj in nb:
            ni, nj = i+di, j+dj
            if 0 <= ni < nx and 0 <= nj < ny and (not seen[ni,nj]) and (not occ[ni,nj]):
                seen[ni,nj] = True; q.append((ni,nj))
    return False


# ----------------------------- scene generation -----------------------------


def point_segment_features_np(p, a, b):
    """Return normalized segment coordinate s and distance to segment."""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den < 1e-9:
        return 0.0, float(np.linalg.norm(p - a))
    s = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    q = a + s * ab
    return s, float(np.linalg.norm(p - q))


def estimate_static_capacity(cfg):
    """Conservative estimate of a reasonable uniformly distributed obstacle count."""
    w = float(cfg.arena_width)
    h = float(cfg.arena_height)
    margin = float(cfg.static_edge_margin)
    usable_w = max(1e-6, w - 2.0 * margin)
    usable_h = max(1e-6, h - 2.0 * margin)

    avg_circle_r = 0.5 * (float(cfg.circle_radius_min) + float(cfg.circle_radius_max))
    avg_rect_r = 0.25 * math.sqrt((float(cfg.rect_width_min)+float(cfg.rect_width_max))**2 +
                                  (float(cfg.rect_length_min)+float(cfg.rect_length_max))**2)
    avg_r = (1.0 - float(cfg.rect_ratio)) * avg_circle_r + float(cfg.rect_ratio) * avg_rect_r
    center_pitch = max(1e-6, 2.0 * avg_r + static_surface_gap(cfg))
    grid_count = (usable_w / center_pitch) * (usable_h / center_pitch)
    conservative = int(max(1, math.floor(grid_count * float(cfg.static_capacity_factor))))
    return conservative, center_pitch

def auto_static_grid(w, h, n, cfg):
    if cfg.static_grid_cols > 0 and cfg.static_grid_rows > 0:
        return int(cfg.static_grid_cols), int(cfg.static_grid_rows)

    # Oversample grid cells so start/goal exclusion and rejection sampling do not
    # force obstacles to cluster. By default each cell can hold at most one static
    # obstacle, making the layout visually close to blue-noise / uniform sampling.
    aspect = max(0.25, float(w) / max(1e-6, float(h)))
    target_cells = max(1.0, float(n) * float(cfg.uniform_grid_oversample))
    cols = int(math.ceil(math.sqrt(target_cells * aspect)))
    rows = int(math.ceil(target_cells / max(1, cols)))
    cols = max(4, cols)
    rows = max(3, rows)
    return cols, rows

def static_cell_index(x, y, w, h, cols, rows):
    ix = int(np.clip(math.floor((x + w / 2.0) / (w / cols)), 0, cols - 1))
    iy = int(np.clip(math.floor((y + h / 2.0) / (h / rows)), 0, rows - 1))
    return ix, iy

def static_edge_distance(x, y, w, h):
    return min(w / 2.0 - abs(x), h / 2.0 - abs(y))

def static_surface_gap(cfg):
    """Effective minimum surface gap between static obstacles.

    The old code used --obstacle_gap directly.  This version also exposes the
    paper-friendly UAV-diameter multiplier, so the constraint can be reported as
    "at least k times the UAV diameter" while remaining backward compatible.
    """
    base_gap = float(getattr(cfg, "obstacle_gap", 1.50))
    drone_radius = float(getattr(cfg, "drone_radius", 0.32))
    mult = float(getattr(cfg, "obstacle_gap_uav_diameter_multiplier", 2.35))
    return max(base_gap, 2.0 * drone_radius * mult)

def sample_static_shape(rng, cfg):
    if rng.random() < cfg.rect_ratio:
        sx = float(rng.uniform(cfg.rect_width_min, cfg.rect_width_max))
        sy = float(rng.uniform(cfg.rect_length_min, cfg.rect_length_max))
        yaw = float(rng.uniform(-0.45 * math.pi, 0.45 * math.pi))
        pr = 0.5 * math.sqrt(sx * sx + sy * sy)
        return "rect", pr, (sx, sy, yaw)
    r = float(rng.uniform(cfg.circle_radius_min, cfg.circle_radius_max))
    return "circle", r, (r,)

def make_static_obstacle(kind, x, y, params, rng, cfg):
    if kind == "rect":
        sx, sy, yaw = params
        return RectObstacle(float(x), float(y), float(sx), float(sy), float(yaw), float(rng.uniform(cfg.static_obstacle_height_min, cfg.static_obstacle_height_max)))
    r, = params
    return CircleObstacle(float(x), float(y), float(r), float(rng.uniform(cfg.static_obstacle_height_min, cfg.static_obstacle_height_max)))

def static_candidate_ok(x, y, r, cirs, rects, start, goal, w, h, cfg, edge_count, n_total):
    margin = float(cfg.static_edge_margin) + float(r)
    if x < -w / 2 + margin or x > w / 2 - margin or y < -h / 2 + margin or y > h / 2 - margin:
        return False
    if in_safe(x, y, r, start, goal, cfg.start_goal_clearance):
        return False
    if not (sep_circles(x, y, r, cirs, static_surface_gap(cfg)) and sep_rects(x, y, r, rects, static_surface_gap(cfg))):
        return False

    # Do not allow the sampler to fill mostly boundary cells.  This is the
    # failure mode you observed visually: many obstacles accumulated near walls
    # while the nominal UAV path remained sparse.
    if static_edge_distance(x, y, w, h) < cfg.static_edge_band:
        if edge_count >= int(math.ceil(float(n_total) * float(cfg.static_edge_max_fraction))):
            return False
    return True

def propose_corridor_center(rng, w, h, start, goal, cfg, s_low=None, s_high=None):
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    ab = b - a
    L = max(1e-6, float(np.linalg.norm(ab)))
    u = ab / L
    nvec = np.array([-u[1], u[0]], dtype=np.float64)
    if s_low is None:
        s_low, s_high = cfg.static_corridor_s_min, cfg.static_corridor_s_max
    s = float(rng.uniform(s_low, s_high))
    center = a + s * ab
    sign = -1.0 if rng.random() < 0.5 else 1.0
    offset = sign * float(rng.uniform(cfg.static_corridor_band_min, cfg.static_corridor_band_max))
    along = float(rng.normal(0.0, cfg.static_corridor_along_jitter))
    lateral = float(rng.normal(0.0, cfg.static_corridor_lateral_jitter))
    p = center + (along * u) + ((offset + lateral) * nvec)
    return float(p[0]), float(p[1])

def propose_grid_center(rng, w, h, cols, rows, cell):
    ix, iy = cell
    x0 = -w / 2.0 + ix * w / cols
    x1 = -w / 2.0 + (ix + 1) * w / cols
    y0 = -h / 2.0 + iy * h / rows
    y1 = -h / 2.0 + (iy + 1) * h / rows
    # Keep candidates away from cell borders to improve spacing regularity.
    jx = float(np.clip(0.5 + rng.uniform(-0.5, 0.5) * 0.85, 0.08, 0.92))
    jy = float(np.clip(0.5 + rng.uniform(-0.5, 0.5) * 0.85, 0.08, 0.92))
    return float(x0 + jx * (x1 - x0)), float(y0 + jy * (y1 - y0))

def choose_underfilled_cell(rng, cell_counts):
    min_count = int(np.min(cell_counts))
    candidates = np.argwhere(cell_counts == min_count)
    k = int(rng.integers(0, len(candidates)))
    ix, iy = candidates[k]
    return int(ix), int(iy)



def sample_static(rng, w, h, n, start, goal, cfg):
    """Uniform static obstacle sampler.

    Obstacles are placed over the full arena by a stratified Poisson-disk-like
    process. No route corridor is planned beforehand. The generator only enforces
    physical feasibility: non-overlap, sufficient surface gap, start and goal
    clearance, edge limits, and the final BFS connectivity check in generate_scene().
    """
    cirs, rects = [], []
    if n <= 0:
        return cirs, rects

    cols, rows = auto_static_grid(w, h, n, cfg)
    cell_counts = np.zeros((cols, rows), dtype=np.int32)
    max_cell_count = max(1, int(cfg.static_max_per_cell))
    edge_count = 0

    all_cells = [(i, j) for i in range(cols) for j in range(rows)]
    rng.shuffle(all_cells)

    def cell_center_ok(cell):
        ix, iy = cell
        cx = -w / 2.0 + (ix + 0.5) * w / cols
        cy = -h / 2.0 + (iy + 0.5) * h / rows
        # only cell pre-filter; exact check still happens in static_candidate_ok
        return not in_safe(cx, cy, float(cfg.circle_radius_max), start, goal, cfg.start_goal_clearance)

    primary_cells = [c for c in all_cells if cell_center_ok(c)]
    secondary_cells = [c for c in all_cells if not cell_center_ok(c)]
    cell_order = primary_cells + secondary_cells

    def try_place_in_cell(cell, allow_second=False):
        nonlocal edge_count
        if cell_counts[cell[0], cell[1]] >= max_cell_count and not allow_second:
            return False

        for _ in range(int(cfg.static_cell_attempts)):
            kind, pr, params = sample_static_shape(rng, cfg)
            x, y = propose_grid_center(rng, w, h, cols, rows, cell)

            if not static_candidate_ok(x, y, pr, cirs, rects, start, goal, w, h, cfg, edge_count, n):
                continue

            obj = make_static_obstacle(kind, x, y, params, rng, cfg)
            if kind == "rect":
                rects.append(obj)
            else:
                cirs.append(obj)

            cell_counts[cell[0], cell[1]] += 1
            if static_edge_distance(x, y, w, h) < cfg.static_edge_band:
                edge_count += 1
            return True
        return False

    # First pass: at most one obstacle in each selected cell.
    for cell in cell_order:
        if len(cirs) + len(rects) >= n:
            break
        try_place_in_cell(cell, allow_second=False)

    # Additional balanced sweeps: revisit underfilled cells first.
    for _ in range(int(cfg.static_uniform_sweeps)):
        if len(cirs) + len(rects) >= n:
            break
        cells = list(cell_order)
        rng.shuffle(cells)
        cells.sort(key=lambda c: cell_counts[c[0], c[1]])
        placed_any = False
        for cell in cells:
            if len(cirs) + len(rects) >= n:
                break
            if try_place_in_cell(cell, allow_second=False):
                placed_any = True
        if not placed_any:
            break

    # Rare fallback: allow a second obstacle in large cells only if needed.
    # The surface-gap constraint is still mandatory, so this does not create
    # impassable or overlapping clusters.
    if len(cirs) + len(rects) < n and cfg.allow_relaxed_uniform_fallback:
        old_max = max_cell_count
        max_cell_count = max(max_cell_count, 2)
        cells = list(cell_order)
        rng.shuffle(cells)
        cells.sort(key=lambda c: cell_counts[c[0], c[1]])
        for cell in cells:
            if len(cirs) + len(rects) >= n:
                break
            try_place_in_cell(cell, allow_second=False)
        max_cell_count = old_max

    return cirs, rects



def static_distribution_metrics(scene, cfg):
    stat = proxies(scene.static_circles, scene.static_rects)
    n = len(stat)
    if n == 0:
        return dict(
            static_edge_ratio=0.0,
            static_grid_occupancy=0.0,
            static_cell_cv=0.0,
            static_quadrant_cv=0.0,
            static_min_pair_surface_gap=0.0,
            static_mean_pair_surface_gap=0.0,
            static_capacity_estimate=0.0,
        )

    cols, rows = auto_static_grid(scene.width, scene.height, n, cfg)
    counts = np.zeros((cols, rows), dtype=np.float64)
    quadrant = np.zeros((2, 2), dtype=np.float64)
    edge = 0

    for o in stat:
        x, y = float(o.x), float(o.y)
        if static_edge_distance(x, y, scene.width, scene.height) < cfg.static_edge_band:
            edge += 1
        ix, iy = static_cell_index(x, y, scene.width, scene.height, cols, rows)
        counts[ix, iy] += 1
        quadrant[0 if x < 0 else 1, 0 if y < 0 else 1] += 1

    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            oi, oj = stat[i], stat[j]
            gaps.append(math.hypot(oi.x - oj.x, oi.y - oj.y) - oi.r - oj.r)
    min_gap = float(np.min(gaps)) if gaps else 0.0
    mean_gap = float(np.mean(gaps)) if gaps else 0.0

    occupied = float(np.count_nonzero(counts) / counts.size)
    mean = float(np.mean(counts))
    cell_cv = float(np.std(counts) / (mean + 1e-6))
    qmean = float(np.mean(quadrant))
    quadrant_cv = float(np.std(quadrant) / (qmean + 1e-6))
    cap, _ = estimate_static_capacity(cfg)

    return dict(
        static_edge_ratio=float(edge / n),
        static_grid_occupancy=occupied,
        static_cell_cv=cell_cv,
        static_quadrant_cv=quadrant_cv,
        static_min_pair_surface_gap=min_gap,
        static_mean_pair_surface_gap=mean_gap,
        static_capacity_estimate=float(cap),
    )


def dyn_ok(x, y, r, w, h, stat, dyns, start, goal, cfg):
    """Check whether a dynamic obstacle can be initialized at (x, y)."""
    margin = r + max(0.35, float(cfg.dynamic_spawn_margin))
    if x < -w / 2 + margin or x > w / 2 - margin or y < -h / 2 + margin or y > h / 2 - margin:
        return False

    # Do not spawn directly on the UAV or goal.
    p = np.array([x, y], dtype=np.float64)
    if np.linalg.norm(p - np.asarray(start, dtype=np.float64)) < cfg.start_goal_clearance + r:
        return False
    if np.linalg.norm(p - np.asarray(goal, dtype=np.float64)) < cfg.goal_radius + cfg.start_goal_clearance * 0.35 + r:
        return False

    # Dynamic obstacles are physical bodies and must not overlap static obstacles.
    for o in stat:
        if math.hypot(x - o.x, y - o.y) < r + o.r + cfg.dynamic_static_gap:
            return False

    # They must not overlap each other at initialization.
    for d in dyns:
        if math.hypot(x - d.x, y - d.y) < r + d.r + cfg.dynamic_dynamic_gap:
            return False

    return True

def sample_rand_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg):
    """Whole-arena random dynamic obstacle with random position and velocity."""
    for _ in range(int(cfg.dynamic_place_attempts)):
        r = float(rng.uniform(cfg.dynamic_radius_min, cfg.dynamic_radius_max))
        x = float(rng.uniform(-w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        y = float(rng.uniform(-h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))
        if not dyn_ok(x, y, r, w, h, stat, dyns, start, goal, cfg):
            continue

        # v9: fully random velocity direction.  The obstacle is not intentionally
        # aimed at the UAV route; subsequent wall/static/dynamic bounce handling
        # creates a random but physically feasible moving path.
        ang = float(rng.uniform(-math.pi, math.pi))
        direction = np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))
        v = float(speed) * direction
        return DynamicObstacle(float(x), float(y), float(r), float(v[0]), float(v[1]), False, float(rng.uniform(cfg.dynamic_obstacle_height_min, cfg.dynamic_obstacle_height_max)), "random")
    return None


def sample_route_random_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=0, total=1):
    """Sample a dispersed dynamic obstacle near the current start-goal line.

    This mode is intentionally not a forced encounter.  It samples the dynamic
    obstacle around the task corridor, spreads multiple dynamic obstacles along
    the route coordinate, and assigns a randomized velocity direction.  Therefore
    the UAV is exposed to nearby dynamic obstacles without every obstacle forming
    a synchronized group blockade.
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    ab = b - a
    L = max(1e-6, float(np.linalg.norm(ab)))
    u = ab / L
    nvec = np.array([-u[1], u[0]], dtype=np.float64)

    s_low = float(getattr(cfg, "route_random_s_min", 0.18))
    s_high = float(getattr(cfg, "route_random_s_max", 0.90))
    if s_high <= s_low:
        s_low, s_high = 0.18, 0.90

    for _ in range(int(cfg.dynamic_place_attempts)):
        r = float(rng.uniform(cfg.dynamic_radius_min, cfg.dynamic_radius_max))

        # Stratify route coordinate to prevent all dynamic obstacles from being
        # initialized in the same small group.
        if total > 1:
            frac = (float(index) + float(rng.uniform(0.15, 0.85))) / float(total)
            s_route = s_low + (s_high - s_low) * frac
        else:
            s_route = float(rng.uniform(s_low, s_high))
        s_route += float(rng.normal(0.0, getattr(cfg, "route_random_s_jitter", 0.035)))
        s_route = float(np.clip(s_route, 0.05, 0.95))

        center = a + s_route * ab
        lateral_band = float(getattr(cfg, "route_random_lateral_band", 4.8))
        lateral = float(rng.uniform(-lateral_band, lateral_band))
        along = float(rng.normal(0.0, getattr(cfg, "route_random_along_jitter", 1.0)))
        p = center + lateral * nvec + along * u

        x = float(np.clip(p[0], -w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        y = float(np.clip(p[1], -h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))

        if not dyn_ok(x, y, r, w, h, stat, dyns, start, goal, cfg):
            continue

        # Random velocity.  By default, avoid repeatedly sampling a velocity that
        # is strongly aligned with the UAV's start-to-goal direction.
        direction = None
        avoid_same = bool(getattr(cfg, "route_random_avoid_same_direction", True))
        max_forward_dot = float(getattr(cfg, "route_random_max_forward_dot", 0.45))
        for _vtry in range(32):
            ang = float(rng.uniform(-math.pi, math.pi))
            cand = np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
            if (not avoid_same) or float(np.dot(cand, u)) <= max_forward_dot:
                direction = cand
                break
        if direction is None:
            ang = float(rng.uniform(-math.pi, math.pi))
            direction = np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))

        v = float(speed) * direction
        return DynamicObstacle(
            float(x), float(y), float(r), float(v[0]), float(v[1]),
            False, float(rng.uniform(cfg.dynamic_obstacle_height_min, cfg.dynamic_obstacle_height_max)),
            "route_random"
        )

    return None


def sample_cross_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg):
    """Sample a path-crossing dynamic obstacle.

    The obstacle is initialized on one side of the nominal start-goal corridor
    and moves toward a crossing point near the future UAV route.  It still obeys
    physical initialization constraints: no overlap with static or dynamic
    obstacles, and it starts inside the arena.
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    ab = b - a
    L = max(1e-6, float(np.linalg.norm(ab)))
    u = ab / L
    nvec = np.array([-u[1], u[0]], dtype=np.float64)

    for _ in range(int(cfg.dynamic_place_attempts)):
        r = float(rng.uniform(cfg.dynamic_radius_min, cfg.dynamic_radius_max))

        # Select a crossing point along the route; jitter avoids repeatedly using
        # the exact same intersection point.
        s = float(rng.uniform(cfg.crossing_s_min, cfg.crossing_s_max) + rng.normal(0.0, cfg.crossing_s_jitter))
        s = float(np.clip(s, 0.05, 0.95))
        cross = a + s * ab
        cross = cross + rng.normal(0.0, cfg.crossing_lateral_jitter, size=1)[0] * nvec

        # The UAV approximate arrival time at the crossing point.  Dynamic
        # obstacle starts far enough to meet the UAV near the crossing point.
        t_uav = s * L / max(0.2, float(cfg.crossing_uav_speed_guess))
        t_uav = float(np.clip(t_uav, cfg.crossing_t_min, cfg.crossing_t_max))
        spawn_dist = float(speed) * t_uav
        spawn_dist = float(np.clip(spawn_dist, cfg.crossing_spawn_distance_min, cfg.crossing_spawn_distance_max))

        side = -1.0 if rng.random() < 0.5 else 1.0
        along_offset = float(rng.uniform(-cfg.crossing_along_ratio * spawn_dist, cfg.crossing_along_ratio * spawn_dist))
        start_pos = cross + side * spawn_dist * nvec + along_offset * u

        # Pull back into the arena if the ideal crossing start lies outside.
        start_pos[0] = float(np.clip(start_pos[0], -w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        start_pos[1] = float(np.clip(start_pos[1], -h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))

        x, y = float(start_pos[0]), float(start_pos[1])
        if not dyn_ok(x, y, r, w, h, stat, dyns, start, goal, cfg):
            continue

        direction = cross - start_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm
        # Add a small tangential disturbance so trajectories are not perfectly
        # repeated but remain crossing-oriented.
        direction = direction + rng.normal(0.0, cfg.crossing_velocity_jitter, size=2)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))
        v = float(speed) * direction

        return DynamicObstacle(float(x), float(y), float(r), float(v[0]), float(v[1]), True, float(rng.uniform(cfg.dynamic_obstacle_height_min, cfg.dynamic_obstacle_height_max)), "crossing")

    return None



def sample_forced_encounter_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=0, total=1):
    """Sample a time-synchronized route-crossing dynamic obstacle.

    The obstacle crosses the nominal start-goal route from top to bottom.  Its
    initial vertical distance is selected from an estimated UAV arrival time so
    that the obstacle and UAV are likely to arrive at the crossing zone together.
    This is intended for training dynamic avoidance ability, not for passive
    random-motion evaluation.
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    ab = b - a
    L = max(1e-6, float(np.linalg.norm(ab)))

    s_low = float(cfg.forced_encounter_s_min)
    s_high = float(cfg.forced_encounter_s_max)
    if s_high <= s_low:
        s_low, s_high = 0.25, 0.70

    for _ in range(int(cfg.dynamic_place_attempts)):
        r = float(rng.uniform(cfg.dynamic_radius_min, cfg.dynamic_radius_max))

        if total > 1:
            frac = (float(index) + float(rng.uniform(0.10, 0.90))) / float(total)
            s_route = s_low + (s_high - s_low) * frac
        else:
            s_route = float(rng.uniform(s_low, s_high))
        s_route += float(rng.normal(0.0, cfg.forced_encounter_s_jitter))
        s_route = float(np.clip(s_route, 0.05, 0.95))

        cross = a + s_route * ab
        cross_x = float(cross[0] + rng.normal(0.0, cfg.forced_encounter_x_jitter))
        cross_y = float(cross[1] + rng.normal(0.0, cfg.forced_encounter_y_jitter))

        x_half = float(cfg.forced_encounter_x_halfwidth)
        cross_x = float(np.clip(cross_x, -x_half, x_half))
        cross_x = float(np.clip(cross_x, -w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        cross_y = float(np.clip(cross_y, -h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))

        # Estimate when the UAV reaches this route coordinate.  The dynamic
        # obstacle starts above the crossing point so it arrives there near that
        # time.  If the arena top clips the spawn point, the distance is reduced.
        t_uav = s_route * L / max(0.2, float(cfg.forced_encounter_uav_speed_guess))
        t_enc = t_uav + float(rng.normal(0.0, cfg.forced_encounter_time_jitter))
        t_enc = float(np.clip(t_enc, cfg.forced_encounter_t_min, cfg.forced_encounter_t_max))

        spawn_dist = float(speed) * t_enc
        spawn_dist = float(np.clip(spawn_dist, cfg.forced_encounter_spawn_distance_min, cfg.forced_encounter_spawn_distance_max))
        top_y = h / 2 - r - cfg.dynamic_spawn_margin
        y = min(cross_y + spawn_dist, top_y)
        spawn_dist = y - cross_y
        if spawn_dist < max(1.0, 2.2 * r):
            continue

        x = cross_x + float(rng.normal(0.0, cfg.forced_encounter_spawn_x_jitter))
        x = float(np.clip(x, -w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        y = float(np.clip(y, -h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))

        if not dyn_ok(x, y, r, w, h, stat, dyns, start, goal, cfg):
            continue

        start_pos = np.array([x, y], dtype=np.float64)
        target = np.array([cross_x, cross_y], dtype=np.float64)
        direction = target - start_pos
        if direction[1] >= -1e-6:
            direction = np.array([0.0, -1.0], dtype=np.float64)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))
        direction[0] += float(rng.normal(0.0, cfg.forced_encounter_velocity_x_jitter))
        direction[1] = -abs(direction[1]) * float(cfg.forced_encounter_downward_bias)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))

        if direction[1] > -0.35:
            direction[1] = -0.35
            direction = direction / max(1e-6, float(np.linalg.norm(direction)))

        v = float(speed) * direction
        return DynamicObstacle(
            float(x), float(y), float(r), float(v[0]), float(v[1]),
            True, float(rng.uniform(cfg.dynamic_obstacle_height_min, cfg.dynamic_obstacle_height_max)), "forced_encounter",
            float(cross_x), float(cross_y), float(cross_x)
        )

    return None


def sample_center_vertical_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=0, total=1):
    """Sample a top-to-bottom dynamic obstacle in the central crossing zone.

    This mode is designed for dynamic-obstacle training rather than passive random
    evaluation. Each obstacle is placed above a crossing point on the middle part
    of the start-goal route and moves downward through that region. Therefore the
    UAV is much more likely to encounter dynamic obstacles instead of succeeding
    simply because moving obstacles stayed far away.
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    ab = b - a
    L = max(1e-6, float(np.linalg.norm(ab)))

    s_low = float(cfg.center_vertical_s_min)
    s_high = float(cfg.center_vertical_s_max)
    if s_high <= s_low:
        s_low, s_high = 0.35, 0.75

    for _ in range(int(cfg.dynamic_place_attempts)):
        r = float(rng.uniform(cfg.dynamic_radius_min, cfg.dynamic_radius_max))

        # Stratify crossing points so multiple dynamic obstacles do not all pass
        # through the exact same point.  The whole zone remains in the middle
        # segment of the UAV route.
        if total > 1:
            frac = (float(index) + float(rng.uniform(0.15, 0.85))) / float(total)
            s_route = s_low + (s_high - s_low) * frac
            s_route += float(rng.normal(0.0, cfg.center_vertical_s_jitter))
        else:
            s_route = float(rng.uniform(s_low, s_high))
        s_route = float(np.clip(s_route, 0.05, 0.95))

        cross = a + s_route * ab
        cross_x = float(cross[0] + rng.normal(0.0, cfg.center_vertical_x_jitter))
        cross_y = float(cross[1] + rng.normal(0.0, cfg.center_vertical_y_jitter))

        # Keep the crossing zone central and away from the start/goal safety area.
        x_half = float(cfg.center_vertical_x_halfwidth)
        cross_x = float(np.clip(cross_x, -x_half, x_half))
        cross_x = float(np.clip(cross_x, -w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        cross_y = float(np.clip(cross_y, -h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))

        # Spawn above the crossing point.  The distance is linked to the UAV
        # estimated arrival time, so the dynamic obstacle tends to reach the
        # crossing zone when the UAV is also nearby.
        t_uav = s_route * L / max(0.2, float(cfg.center_vertical_uav_speed_guess))
        t_uav = float(np.clip(t_uav, cfg.center_vertical_t_min, cfg.center_vertical_t_max))
        spawn_dist = float(speed) * t_uav
        spawn_dist = float(np.clip(spawn_dist, cfg.center_vertical_spawn_distance_min, cfg.center_vertical_spawn_distance_max))

        x = cross_x + float(rng.normal(0.0, cfg.center_vertical_spawn_x_jitter))
        y = cross_y + spawn_dist
        x = float(np.clip(x, -w / 2 + r + cfg.dynamic_spawn_margin, w / 2 - r - cfg.dynamic_spawn_margin))
        y = float(np.clip(y, -h / 2 + r + cfg.dynamic_spawn_margin, h / 2 - r - cfg.dynamic_spawn_margin))

        # Ensure the obstacle starts above its intended crossing point.  If the
        # arena boundary clips too much, try another candidate.
        if y <= cross_y + max(1.0, 2.0 * r):
            continue

        if not dyn_ok(x, y, r, w, h, stat, dyns, start, goal, cfg):
            continue

        target = np.array([cross_x, cross_y], dtype=np.float64)
        start_pos = np.array([x, y], dtype=np.float64)
        direction = target - start_pos
        if direction[1] >= -1e-6:
            direction = np.array([0.0, -1.0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue

        direction = direction / norm
        # Keep a strong downward component; a small horizontal component allows
        # lane variation and later static-obstacle avoidance.
        direction[0] += float(rng.normal(0.0, cfg.center_vertical_velocity_x_jitter))
        direction[1] = -abs(direction[1]) * float(cfg.center_vertical_downward_bias)
        direction = direction / max(1e-6, float(np.linalg.norm(direction)))
        if direction[1] > -0.20:
            direction[1] = -0.20
            direction = direction / max(1e-6, float(np.linalg.norm(direction)))

        v = float(speed) * direction
        return DynamicObstacle(float(x), float(y), float(r), float(v[0]), float(v[1]), True, float(rng.uniform(cfg.dynamic_obstacle_height_min, cfg.dynamic_obstacle_height_max)), "center_vertical", float(cross_x), float(cross_y), float(cross_x))

    return None


def sample_dynamic(rng, w, h, n, speed, stat, start, goal, cfg):
    """Generate dynamic obstacles.

    Default random mode samples dynamic obstacles across the whole arena and
    assigns randomized velocity directions.  Route-neighborhood and forced
    crossing modes are still available through --dynamic_motion_mode, but they
    are no longer the default.
    """
    dyns = []
    if n <= 0:
        return dyns

    mode = getattr(cfg, "dynamic_motion_mode", "random").lower()
    for i in range(n):
        o = None

        if mode in ("random", "uniform_random", "scene_random", "whole_arena_random"):
            o = sample_rand_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg)
        elif mode in ("route_random", "near_route_random", "random_near_route"):
            o = sample_route_random_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=i, total=n)
        elif mode in ("forced_encounter", "forced", "must_encounter"):
            o = sample_forced_encounter_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=i, total=n)
            if o is None:
                o = sample_center_vertical_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=i, total=n)
        elif mode in ("center_vertical", "vertical", "vertical_crossing"):
            o = sample_center_vertical_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=i, total=n)
        elif mode == "crossing":
            o = sample_cross_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg)
        elif mode == "mixed":
            n_cross = min(n, max(cfg.min_crossing_dynamic, int(math.ceil(cfg.crossing_ratio * n))))
            prefer_cross = i < n_cross or rng.random() < cfg.crossing_ratio
            if prefer_cross:
                o = sample_forced_encounter_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=i, total=n)
                if o is None:
                    o = sample_center_vertical_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg, index=i, total=n)
                if o is None:
                    o = sample_cross_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg)

        if o is None:
            o = sample_rand_dyn(rng, w, h, speed, stat, dyns, start, goal, cfg)

        if o is not None:
            dyns.append(o)
    return dyns

def precompute_dyn(dyns, stat, w, h, steps, dt, rng, cfg=None):
    n = len(dyns)
    traj = np.zeros((steps, n, 2), dtype=np.float32)
    vel_traj = np.zeros((steps, n, 2), dtype=np.float32)
    if n == 0: return traj, vel_traj
    pos = np.array([[d.x, d.y] for d in dyns], dtype=np.float32)
    vel = np.array([[d.vx, d.vy] for d in dyns], dtype=np.float32)
    radii = np.array([d.r for d in dyns], dtype=np.float32)
    modes = [getattr(d, "motion_mode", "") for d in dyns]
    vertical_flow = np.array([m in ("center_vertical", "forced_encounter") for m in modes], dtype=bool)
    forced_flow = np.array([m == "forced_encounter" for m in modes], dtype=bool)
    lane_x = np.array([float(getattr(d, "lane_x", d.x)) for d in dyns], dtype=np.float32)
    cross_y = np.array([float(getattr(d, "cross_y", d.y)) for d in dyns], dtype=np.float32)
    base_speed = np.linalg.norm(vel, axis=1).astype(np.float32)
    base_speed = np.maximum(base_speed, 1e-6)

    def keep_vertical_down(i):
        """Keep vertical-flow obstacles moving downward after avoidance."""
        spd = float(base_speed[i])
        if forced_flow[i]:
            vx_limit = spd * float(getattr(cfg, "forced_encounter_avoid_lateral_scale", 0.70))
            err = float(lane_x[i] - pos[i, 0])
            lane_vx = float(np.clip(float(getattr(cfg, "forced_encounter_lane_gain", 0.65)) * err, -vx_limit, vx_limit))
            vx = 0.65 * float(vel[i, 0]) + 0.35 * lane_vx
            vx = float(np.clip(vx, -vx_limit, vx_limit))
        else:
            vx_limit = spd * float(getattr(cfg, "center_vertical_avoid_lateral_scale", 0.65))
            vx = float(np.clip(vel[i, 0], -vx_limit, vx_limit))
        vy = -math.sqrt(max(1e-6, spd * spd - vx * vx))
        vel[i, 0] = vx
        vel[i, 1] = vy

    for t in range(steps):
        traj[t] = pos; vel_traj[t] = vel
        pos2 = pos + vel * dt

        for i in range(n):
            r = float(radii[i])

            # Wall handling.  Center-vertical obstacles wrap from bottom to top,
            # so they repeatedly cross the UAV's route from above instead of
            # bouncing upward after reaching the lower boundary.
            if pos2[i,0] < -w/2+r:
                pos2[i,0] = -w/2+r
                vel[i,0] = abs(vel[i,0])
            if pos2[i,0] >  w/2-r:
                pos2[i,0] =  w/2-r
                vel[i,0] = -abs(vel[i,0])

            if vertical_flow[i]:
                if pos2[i,1] < -h/2+r:
                    pos2[i,1] = h/2 - r - float(rng.uniform(0.0, getattr(cfg, "center_vertical_respawn_y_jitter", 1.0)))
                    if forced_flow[i]:
                        pos2[i,0] = float(np.clip(lane_x[i] + rng.normal(0.0, getattr(cfg, "forced_encounter_respawn_x_jitter", 0.25)),
                                                   -w/2+r, w/2-r))
                    vel[i,1] = -abs(vel[i,1])
                    keep_vertical_down(i)
                if pos2[i,1] > h/2-r:
                    pos2[i,1] = h/2-r
                    vel[i,1] = -abs(vel[i,1])
                    keep_vertical_down(i)
            else:
                if pos2[i,1] < -h/2+r:
                    pos2[i,1] = -h/2+r
                    vel[i,1] = abs(vel[i,1])
                if pos2[i,1] >  h/2-r:
                    pos2[i,1] =  h/2-r
                    vel[i,1] = -abs(vel[i,1])

            if forced_flow[i]:
                keep_vertical_down(i)

            # Static-obstacle avoidance.  Random/crossing obstacles keep the old
            # reflection model.  Vertical-flow obstacles sidestep the obstacle
            # and then continue downward.
            for o in stat:
                c = np.array([o.x,o.y], dtype=np.float32)
                diff = pos2[i] - c
                dist = float(np.linalg.norm(diff))
                md = float(radii[i] + o.r + 0.10)
                if dist < md:
                    normal = diff / dist if dist > 1e-6 else np.array([1.0,0.0], dtype=np.float32)
                    pos2[i] = c + normal * md
                    if vertical_flow[i]:
                        side = 1.0 if float(normal[0]) >= 0.0 else -1.0
                        scale_name = "forced_encounter_avoid_lateral_scale" if forced_flow[i] else "center_vertical_avoid_lateral_scale"
                        vel[i,0] = side * float(base_speed[i]) * float(getattr(cfg, scale_name, 0.70))
                        keep_vertical_down(i)
                    else:
                        vel[i] = vel[i] - 2 * np.dot(vel[i], normal) * normal

        # Dynamic-dynamic separation.
        for i in range(n):
            for j in range(i+1, n):
                diff = pos2[i] - pos2[j]
                dist = float(np.linalg.norm(diff))
                md = float(radii[i] + radii[j] + 0.10)
                if dist < md:
                    normal = diff / dist if dist > 1e-6 else np.array([1.0,0.0], dtype=np.float32)
                    delta = 0.5 * (md - dist) * normal
                    pos2[i] += delta
                    pos2[j] -= delta
                    vi, vj = vel[i].copy(), vel[j].copy()
                    if vertical_flow[i] or vertical_flow[j]:
                        if vertical_flow[i]:
                            vel[i,0] = float(np.sign(normal[0]) if abs(normal[0]) > 1e-6 else 1.0) * float(base_speed[i]) * 0.45
                            keep_vertical_down(i)
                        else:
                            vel[i] = vi - 2*np.dot(vi, normal)*normal
                        if vertical_flow[j]:
                            vel[j,0] = -float(np.sign(normal[0]) if abs(normal[0]) > 1e-6 else 1.0) * float(base_speed[j]) * 0.45
                            keep_vertical_down(j)
                        else:
                            vel[j] = vj - 2*np.dot(vj, -normal)*(-normal)
                    else:
                        vel[i] = vi - 2*np.dot(vi, normal)*normal
                        vel[j] = vj - 2*np.dot(vj, -normal)*(-normal)
        pos = pos2
    return traj, vel_traj


def route_crossing_quality(start, goal, dyn_traj, cfg):
    """Scene-level check that dynamic obstacles cross the nominal route corridor."""
    n = dyn_traj.shape[1] if dyn_traj.ndim == 3 else 0
    if n == 0:
        return 1.0, 0, []
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    ab = b - a
    den = max(1e-9, float(np.dot(ab, ab)))
    steps = min(int(getattr(cfg, "forced_route_check_steps", dyn_traj.shape[0])), dyn_traj.shape[0])
    good = 0
    min_dists = []
    for j in range(n):
        dmin = 1e9
        good_j = False
        for t in range(steps):
            p = dyn_traj[t, j].astype(np.float64)
            s_seg = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
            q = a + s_seg * ab
            d = float(np.linalg.norm(p - q))
            dmin = min(dmin, d)
            if d <= float(cfg.forced_route_crossing_radius) and float(cfg.forced_encounter_s_min) <= s_seg <= float(cfg.forced_encounter_s_max):
                good_j = True
                break
        min_dists.append(dmin)
        if good_j:
            good += 1
    return float(good / max(1, n)), int(good), min_dists



def sample_start_goal(rng, w, h, cfg):
    """Sample the navigation task endpoints.

    The start point is fixed to avoid changing the takeoff side of the task.
    The goal is randomly sampled on the right half, preferably near the right
    side, so each scene is not constrained to one fixed diagonal route.
    """
    sx = getattr(cfg, "fixed_start_x", None)
    sy = getattr(cfg, "fixed_start_y", None)
    if sx is None:
        sx = -w / 2.0 + float(getattr(cfg, "fixed_start_left_margin", 3.2))
    if sy is None:
        sy = -h / 2.0 + float(getattr(cfg, "fixed_start_bottom_margin", 3.0))
    start = np.array([float(sx), float(sy)], dtype=np.float32)

    right_margin = float(getattr(cfg, "goal_right_margin", 3.2))
    right_band = float(getattr(cfg, "goal_right_band", 10.0))
    y_margin = float(getattr(cfg, "goal_y_margin", 3.0))
    min_dist = float(getattr(cfg, "min_start_goal_distance", 18.0))

    x_hi = w / 2.0 - right_margin
    x_lo = max(0.0, w / 2.0 - right_band)
    if x_hi <= x_lo + 1e-6:
        x_lo = max(0.0, x_hi - 2.0)

    y_lo = -h / 2.0 + y_margin
    y_hi = h / 2.0 - y_margin

    for _ in range(200):
        gx = float(rng.uniform(x_lo, x_hi))
        gy = float(rng.uniform(y_lo, y_hi))
        goal = np.array([gx, gy], dtype=np.float32)
        if float(np.linalg.norm(goal - start)) >= min_dist:
            return start, goal

    # Fallback remains on the right side.
    return start, np.array([x_hi, float(rng.uniform(y_lo, y_hi))], dtype=np.float32)



def effective_scene_max_steps(cfg, dynamic_count: int) -> int:
    """Return rollout length for the current scene.

    Static scenes keep --max_steps unchanged.  In dynamic stages, the route
    becomes longer because the UAV must slow down, detour, or wait for moving
    obstacles.  Therefore the maximum rollout steps can grow with the number of
    dynamic obstacles while remaining bounded by --dynamic_max_steps_cap.
    """
    base_steps = int(getattr(cfg, "max_steps", 420))
    if not bool(getattr(cfg, "adaptive_dynamic_steps", True)):
        return max(1, base_steps)

    nd = max(0, int(dynamic_count))
    if nd <= 0:
        return max(1, base_steps)

    start_nd = int(getattr(cfg, "dynamic_step_start_count", 1))
    if start_nd <= 0:
        start_nd = int(getattr(cfg, "stage3_dynamic_start", 1))
    extra_levels = max(0, nd - start_nd)
    step_inc = max(0, int(getattr(cfg, "dynamic_step_increment", 35)))
    steps = base_steps + extra_levels * step_inc

    cap = int(getattr(cfg, "dynamic_max_steps_cap", 0))
    if cap > 0:
        steps = min(steps, cap)
    return max(1, int(steps))


def generate_scene(cfg, ep, cur):
    st = cur.settings(); rng = ep_rng(cfg.seed, ep, st["stage_id"])
    effective_steps = effective_scene_max_steps(cfg, int(st["dynamic_count"]))
    w, h = float(cfg.arena_width), float(cfg.arena_height)
    start, goal = sample_start_goal(rng, w, h, cfg)
    cirs, rects = [], []
    # Dense-uniform scene generation.  There is deliberately NO start-to-goal
    # BFS/traversability screening here; otherwise the sampler preferentially
    # accepts scenes that already contain a visible corridor.  A scene is accepted
    # only when the requested number of obstacles can be placed while satisfying
    # boundary, start/goal exclusion, non-overlap, and UAV-size gap constraints.
    for _ in range(cfg.scene_retries):
        start, goal = sample_start_goal(rng, w, h, cfg)
        cirs, rects = sample_static(rng, w, h, st["static_count"], start, goal, cfg)
        stat = proxies(cirs, rects)
        if len(stat) == st["static_count"]:
            break
    stat = proxies(cirs, rects)

    dyns, tr, vt = [], None, None
    dyn_retries = int(getattr(cfg, "forced_dynamic_retries", 1)) if st["dynamic_count"] > 0 else 1
    dyn_retries = max(1, dyn_retries)
    target_dyn = int(st["dynamic_count"])
    mode = getattr(cfg, "dynamic_motion_mode", "random").lower()
    forced_like_mode = mode in ("forced_encounter", "forced", "must_encounter", "center_vertical", "vertical", "vertical_crossing")
    need_route_crossing_check = bool(getattr(cfg, "enforce_route_crossing", False)) and forced_like_mode
    accepted_dynamic_scene = False

    # Best complete candidate among retries.  We never accept a candidate with
    # fewer than target_dyn dynamic obstacles, because that silently weakens the
    # curriculum level.  But if the requested number of obstacles exists and only
    # strict route-ratio validation fails, fallback can accept the best candidate.
    best_dyns, best_tr, best_vt = None, None, None
    best_actual_dyn = 0
    best_route_ratio = 0.0
    best_route_good = 0
    best_route_dists = None
    last_actual_dyn = 0
    last_route_ratio = 0.0

    for _dyn_try in range(dyn_retries):
        cand_dyns = sample_dynamic(rng, w, h, target_dyn, st["dynamic_speed"], stat, start, goal, cfg)
        last_actual_dyn = int(len(cand_dyns))

        if len(cand_dyns) != target_dyn:
            best_actual_dyn = max(best_actual_dyn, last_actual_dyn)
            continue

        cand_tr, cand_vt = precompute_dyn(cand_dyns, stat, w, h, effective_steps, cfg.dt, rng, cfg)

        if target_dyn <= 0 or not need_route_crossing_check:
            dyns, tr, vt = cand_dyns, cand_tr, cand_vt
            accepted_dynamic_scene = True
            break

        route_ratio, route_good, route_dists = route_crossing_quality(start, goal, cand_tr, cfg)
        route_ratio = float(route_ratio)
        route_good = int(route_good)
        last_route_ratio = route_ratio

        if (
            best_dyns is None
            or route_ratio > best_route_ratio
            or (abs(route_ratio - best_route_ratio) < 1e-12 and route_good > best_route_good)
        ):
            best_dyns = cand_dyns
            best_tr = cand_tr
            best_vt = cand_vt
            best_actual_dyn = int(len(cand_dyns))
            best_route_ratio = route_ratio
            best_route_good = route_good
            best_route_dists = route_dists

        if route_ratio >= float(getattr(cfg, "forced_min_route_crossing_ratio", 1.0)):
            dyns, tr, vt = cand_dyns, cand_tr, cand_vt
            accepted_dynamic_scene = True
            break

    if target_dyn > 0 and not accepted_dynamic_scene:
        fallback_enabled = bool(getattr(cfg, "allow_forced_dynamic_fallback", True))
        fallback_min_ratio = float(getattr(cfg, "forced_dynamic_fallback_min_ratio", 0.50))

        if (
            fallback_enabled
            and best_dyns is not None
            and best_actual_dyn >= target_dyn
            and best_route_ratio >= fallback_min_ratio
        ):
            dyns, tr, vt = best_dyns, best_tr, best_vt
            accepted_dynamic_scene = True
            if bool(getattr(cfg, "print_forced_dynamic_fallback", False)):
                print(
                    "[warn] forced dynamic fallback accepted: "
                    f"target_dyn={target_dyn}, actual_dyn={best_actual_dyn}, "
                    f"route_ratio={best_route_ratio:.3f}, "
                    f"strict_ratio={float(getattr(cfg, 'forced_min_route_crossing_ratio', 1.0)):.3f}, "
                    f"fallback_min_ratio={fallback_min_ratio:.3f}"
                )

        if not accepted_dynamic_scene:
            raise RuntimeError(
                f"Failed to generate a valid forced dynamic scene: "
                f"target_dyn={target_dyn}, actual_dyn={last_actual_dyn}, "
                f"last_route_ratio={last_route_ratio:.3f}, "
                f"best_actual_dyn={best_actual_dyn}, best_route_ratio={best_route_ratio:.3f}. "
                f"Try increasing --forced_dynamic_retries or --dynamic_place_attempts, "
                f"or lower --forced_dynamic_fallback_min_ratio."
            )

    # Defensive fallback for static/no-dynamic scenes.
    if tr is None or vt is None:
        dyns = [] if dyns is None else dyns
        tr, vt = precompute_dyn(dyns, stat, w, h, effective_steps, cfg.dt, rng, cfg)

    return Scene(w, h, start, goal, cirs, rects, dyns, tr, vt, cfg.dt, effective_steps, cfg.goal_radius,
                 st["stage_id"], st["stage_name"], st["static_count"], st["dynamic_count"], st["dynamic_speed"])


# ----------------------------- Diff-VFH -----------------------------

def inv_range(v, lo, hi):
    v = min(max(v, lo+1e-6), hi-1e-6); p = (v-lo)/(hi-lo)
    return math.log(p/(1-p))

def range_param(raw, lo, hi):
    return lo + (hi-lo) * torch.sigmoid(raw)

def wrap_pi(a): return torch.atan2(torch.sin(a), torch.cos(a))
def circ_dist(a,b): return 1.0 - torch.cos(a-b)

def blend_angle(a,b,wb):
    wb = float(np.clip(wb, 0.0, 1.0))
    return torch.atan2((1-wb)*torch.sin(a)+wb*torch.sin(b), (1-wb)*torch.cos(a)+wb*torch.cos(b))

def abs_angle(a,b): return torch.abs(torch.atan2(torch.sin(a-b), torch.cos(a-b)))

def heading_inertia(cfg, stage):
    return [0, cfg.heading_inertia_s1, cfg.heading_inertia_s2, cfg.heading_inertia_s3, cfg.heading_inertia_s4][stage]

def pref_clear(cfg, stage):
    return [0, cfg.preferred_clearance_s1, cfg.preferred_clearance_s2, cfg.preferred_clearance_s3, cfg.preferred_clearance_s4][stage]

def escape_heading(pos, goal, obs_pos, obs_r, goal_blend):
    gv = goal - pos; ga = torch.atan2(gv[1], gv[0])
    if obs_pos.shape[0] == 0: return ga
    rel = pos[None,:] - obs_pos
    dist = torch.linalg.norm(rel, dim=1).clamp_min(1e-6)
    idx = torch.argmin(dist - obs_r)
    radial = torch.atan2(rel[idx,1], rel[idx,0])
    left, right = wrap_pi(radial + math.pi/2), wrap_pi(radial - math.pi/2)
    tangent = torch.where(abs_angle(left, ga) <= abs_angle(right, ga), left, right)
    return blend_angle(tangent, ga, goal_blend)

class DiffVFH(nn.Module):
    def __init__(self, n_sectors=72, device="cpu"):
        super().__init__()
        self.n = int(n_sectors)
        th = torch.linspace(-math.pi, math.pi, self.n+1, device=device)[:-1]
        self.register_buffer("theta", th)
        init = {
            "kappa": (18.0,12.0,32.0), "beta_occ": (12.0,8.0,24.0),
            "tau": (0.48,0.22,0.90), "vmax": (7.00,2.00,8.00),
            "w_goal": (2.8,1.3,5.5), "w_obs": (10.5,5.0,22.0),
            "w_smooth": (1.25,0.5,2.8), "w_clearance": (8.0,2.0,22.0),
            "w_ttc": (11.0,2.0,32.0), "speed_beta": (10.0,6.0,20.0),
            "occ_threshold": (0.22,0.10,0.60)
        }
        self.ranges = {}
        for k,(v,lo,hi) in init.items():
            self.ranges[k] = (lo,hi)
            self.register_parameter("raw_"+k, nn.Parameter(torch.tensor(inv_range(v,lo,hi), dtype=torch.float32)))
    def p(self,k):
        lo,hi = self.ranges[k]; return range_param(getattr(self,"raw_"+k),lo,hi)
    def readable_params(self):
        return {k: float(self.p(k).detach().cpu()) for k in self.ranges}

    def forward(self, pos, goal, obs_pos, obs_r_eff, obs_vel, dyn_mask, prev_dir,
                drone_radius, sensor_range, safe_clearance, preferred_clearance,
                ttc_horizon, ttc_clearance, ttc_beta, ttc_decay,
                arena_width, arena_height, wall_influence, wall_beta,
                w_wall, w_recovery, dynamic_recovery_gain,
                dynamic_brake_clearance, dynamic_brake_beta,
                dynamic_min_speed_scale, dynamic_dir_clearance_weight,
                enable_wait_action, dynamic_wait_clearance,
                dynamic_wait_ttc_cost, dynamic_wait_static_density,
                dynamic_wait_beta, dynamic_wait_max_gate,
                dynamic_wait_speed_scale, dynamic_wait_goal_disable_distance):
        dev = pos.device
        gv = goal - pos
        ga = torch.atan2(gv[1], gv[0])
        gd = torch.linalg.norm(gv).clamp_min(1e-6)

        if obs_pos.shape[0] == 0:
            hist = torch.zeros(self.n, device=dev)
            min_clr = torch.tensor(sensor_range, device=dev)
            clearance_cost = torch.zeros(self.n, device=dev)
        else:
            rel = obs_pos - pos[None,:]
            cd = torch.linalg.norm(rel, dim=1).clamp_min(1e-6)
            surf = cd - obs_r_eff - float(drone_radius)
            min_clr = torch.min(surf)
            oa = torch.atan2(rel[:,1], rel[:,0])
            gate = torch.sigmoid(8.0 * (float(sensor_range)-cd))
            strength = torch.exp(-0.75 * torch.clamp(surf, -1.0, float(sensor_range))) * gate * (1 + 0.35*torch.clamp(obs_r_eff,0,2.5))
            delta = wrap_pi(oa[:,None] - self.theta[None,:])
            kernel = torch.softmax(self.p("kappa") * torch.cos(delta), dim=1)
            hist = torch.sum(strength[:,None]*kernel, dim=0)
            proj = cd[:,None] * torch.cos(delta)
            lateral = torch.abs(cd[:,None] * torch.sin(delta))
            pass_clr = lateral - obs_r_eff[:,None] - float(drone_radius)
            front = torch.sigmoid(4.0 * proj)
            clearance_cost = torch.max(torch.sigmoid(10.0*(float(preferred_clearance)-pass_clr)) * front * gate[:,None], dim=0).values

        occ = torch.sigmoid(self.p("beta_occ") * (hist - self.p("occ_threshold")))

        # Local static density used by the hover/wait gate.  Waiting should be
        # preferred mainly when dynamic avoidance would otherwise force the UAV
        # into a locally dense static-obstacle region.
        static_density = torch.tensor(0.0, device=dev)
        static_min_clearance = torch.tensor(float(sensor_range), device=dev)
        if obs_pos.shape[0] > 0:
            static_mask = ~dyn_mask
            if torch.any(static_mask):
                sp_local = obs_pos[static_mask]
                sr_local = obs_r_eff[static_mask]
                rel_s = sp_local - pos[None, :]
                dist_s = torch.linalg.norm(rel_s, dim=1).clamp_min(1e-6)
                surf_s = dist_s - sr_local - float(drone_radius)
                static_min_clearance = torch.min(surf_s)
                range_gate_s = torch.sigmoid(3.0 * (float(sensor_range) - dist_s))
                density_each = torch.sigmoid(5.0 * (float(preferred_clearance) + 0.55 - surf_s)) * range_gate_s
                static_density = torch.mean(density_each)

        ttc_cost = torch.zeros(self.n, device=dev)
        min_pred_dyn_clr = torch.tensor(float(ttc_clearance)+10.0, device=dev)
        if obs_pos.shape[0] > 0 and torch.any(dyn_mask):
            dp = obs_pos[dyn_mask]
            dv = obs_vel[dyn_mask]
            dr = obs_r_eff[dyn_mask]
            relp = dp - pos[None,:]
            dirs = torch.stack([torch.cos(self.theta), torch.sin(self.theta)], dim=1)

            # Candidate-speed TTC: use several possible UAV speeds instead of a
            # single 0.8*vmax.  This is more robust when dynamic braking is active.
            cand_speed_scales = torch.tensor([0.35, 0.60, 0.85, 1.00], dtype=torch.float32, device=dev)
            cand_v = self.p("vmax") * cand_speed_scales[:, None, None] * dirs[None, :, :]  # K,N,2
            relv = dv[:, None, None, :] - cand_v[None, :, :, :]                            # M,K,N,2
            relp2 = relp[:, None, None, :]                                                  # M,1,1,2
            rv2 = torch.sum(relv * relv, dim=3).clamp_min(1e-6)
            tstar = -torch.sum(relp2 * relv, dim=3) / rv2
            tstar = torch.clamp(tstar, 0.0, float(ttc_horizon))
            closest = torch.linalg.norm(relp2 + relv * tstar[:, :, :, None], dim=3) - dr[:, None, None] - float(drone_radius)

            # Take the worst risk over dynamic obstacles and candidate speeds.
            base_risk = torch.sigmoid(float(ttc_beta) * (float(ttc_clearance) - closest)) * torch.exp(-tstar / float(ttc_decay))
            close_risk = torch.sigmoid(float(ttc_beta) * (float(dynamic_brake_clearance) - closest)) * torch.exp(-0.5 * tstar / float(ttc_decay))
            range_gate = torch.sigmoid(4.0 * (float(sensor_range) + 1.5 - torch.linalg.norm(relp, dim=1)))[:, None, None]
            risk = (base_risk + float(dynamic_dir_clearance_weight) * close_risk) * range_gate
            ttc_cost = torch.max(risk.reshape(-1, self.n), dim=0).values
            min_pred_dyn_clr = torch.min(closest)

        goal_cost = circ_dist(self.theta, ga)
        smooth_cost = circ_dist(self.theta, prev_dir)

        # Boundary-direction cost: penalize candidate directions that point toward
        # a nearby wall before the UAV reaches the wall. This removes the common
        # boundary-trap failure in dense dynamic scenes.
        dirs = torch.stack([torch.cos(self.theta), torch.sin(self.theta)], dim=1)
        d_left = pos[0] + float(arena_width) / 2.0 - float(drone_radius)
        d_right = float(arena_width) / 2.0 - pos[0] - float(drone_radius)
        d_bottom = pos[1] + float(arena_height) / 2.0 - float(drone_radius)
        d_top = float(arena_height) / 2.0 - pos[1] - float(drone_radius)
        g_left = torch.sigmoid(float(wall_beta) * (float(wall_influence) - d_left))
        g_right = torch.sigmoid(float(wall_beta) * (float(wall_influence) - d_right))
        g_bottom = torch.sigmoid(float(wall_beta) * (float(wall_influence) - d_bottom))
        g_top = torch.sigmoid(float(wall_beta) * (float(wall_influence) - d_top))
        wall_cost = (
            g_left * F.relu(-dirs[:, 0]) ** 2
            + g_right * F.relu(dirs[:, 0]) ** 2
            + g_bottom * F.relu(-dirs[:, 1]) ** 2
            + g_top * F.relu(dirs[:, 1]) ** 2
        )

        # Goal-recovery cost becomes stronger when dynamic risk is active. It
        # prevents the UAV from over-escaping sideways and then staying near walls.
        recovery_cost = goal_cost * torch.clamp(torch.mean(ttc_cost) * float(dynamic_recovery_gain), 0.0, 1.0)

        cost = (
            self.p("w_goal")*goal_cost
            + self.p("w_obs")*occ
            + self.p("w_smooth")*smooth_cost
            + self.p("w_clearance")*clearance_cost
            + self.p("w_ttc")*ttc_cost
            + float(w_wall)*wall_cost
            + float(w_recovery)*recovery_cost
        )
        prob = torch.softmax(-cost/self.p("tau"), dim=0)
        xm, ym = torch.sum(prob*torch.cos(self.theta)), torch.sum(prob*torch.sin(self.theta))
        direction = torch.atan2(ym, xm)
        conc = torch.sqrt(xm*xm + ym*ym).clamp(0,1)
        avg_occ = torch.mean(occ)
        selected_ttc = torch.sum(prob * ttc_cost)
        sb = self.p("speed_beta")

        # Dynamic braking: when the predicted dynamic clearance is small or the
        # selected TTC risk is high, slow down before steering.
        clr_brake = torch.sigmoid(float(dynamic_brake_beta) * (min_pred_dyn_clr - float(dynamic_brake_clearance)))
        risk_brake = torch.sigmoid(float(dynamic_brake_beta) * (0.55 - selected_ttc))
        dyn_brake = float(dynamic_min_speed_scale) + (1.0 - float(dynamic_min_speed_scale)) * clr_brake * risk_brake

        # High-speed modulation.  The previous constants (0.36 for occupancy,
        # 0.55 for TTC risk, and 0.25+0.75*concentration) were too conservative
        # in the enlarged scene: even safe trajectories were throttled to about
        # 3.3 m/s.  These thresholds still slow the UAV near obstacles, but do
        # not suppress speed in moderately occupied sectors.
        occ_speed_gate = torch.sigmoid(sb * (0.62 - avg_occ))
        ttc_speed_gate = torch.sigmoid(sb * (0.72 - selected_ttc))
        direction_conf_gate = 0.55 + 0.45 * conc
        clearance_gate = torch.sigmoid(sb * (min_clr - float(safe_clearance)))
        goal_gate_speed = torch.clamp(gd / 3.0, 0.18, 1.0)
        base_speed = self.p("vmax") * clearance_gate * occ_speed_gate * ttc_speed_gate * direction_conf_gate * goal_gate_speed * dyn_brake

        # Optional short wait / hover action for dense dynamic encounters.
        # It is disabled by default for comparability with earlier experiments.
        # When enabled, the UAV slows down strongly only when dynamic TTC risk is
        # high and the goal is still far enough that waiting is meaningful.
        if bool(enable_wait_action) and obs_pos.shape[0] > 0 and torch.any(dyn_mask):
            wait_risk = torch.sigmoid(float(dynamic_wait_beta) * (selected_ttc - float(dynamic_wait_ttc_cost)))
            wait_clear = torch.sigmoid(float(dynamic_wait_beta) * (float(dynamic_wait_clearance) - min_pred_dyn_clr))
            wait_density = torch.sigmoid(float(dynamic_wait_beta) * (static_density - float(dynamic_wait_static_density)))
            wait_goal_gate = torch.sigmoid(float(dynamic_wait_beta) * (gd - float(dynamic_wait_goal_disable_distance)))
            wait_gate = wait_risk * torch.maximum(wait_clear, wait_density) * wait_goal_gate
            wait_gate = torch.clamp(wait_gate, 0.0, float(dynamic_wait_max_gate))
        else:
            wait_gate = torch.tensor(0.0, dtype=torch.float32, device=dev)
        speed = base_speed * ((1.0 - wait_gate) + wait_gate * float(dynamic_wait_speed_scale))

        vel = speed * torch.stack([torch.cos(direction), torch.sin(direction)])
        return dict(direction=direction, speed=speed, vel=vel, selected_ttc_cost=selected_ttc, min_pred_dynamic_clearance=min_pred_dyn_clr,
                    dynamic_brake=dyn_brake, wait_gate=wait_gate, local_static_density=static_density,
                    static_min_clearance=static_min_clearance, min_clearance=min_clr, avg_occ=avg_occ,
                    wall_cost=torch.sum(prob*wall_cost),
                    entropy=-torch.sum(prob*torch.log(prob+1e-6)), concentration=conc)


def stage_uav_speed_cap(cfg, stage_id: int, has_dynamic: bool = False) -> float:
    """Stage-wise desired maximum UAV cruising speed.

    In aggressive_adaptive mode, the UAV starts from this speed cap in every
    stage and learns to brake only when local static/dynamic risk becomes high.
    The value is still clamped by the learnable model parameter vmax.
    """
    sid = int(stage_id)
    if sid <= 1:
        return float(getattr(cfg, "uav_max_speed_s1", getattr(cfg, "target_static_avg_speed", 8.0)))
    if sid == 2:
        return float(getattr(cfg, "uav_max_speed_s2", getattr(cfg, "target_dynamic_avg_speed", 8.0)))
    return float(getattr(cfg, "uav_max_speed_s3", getattr(cfg, "target_dynamic_avg_speed", 8.0)))



# ----------------------------- rollout -----------------------------

def static_tensors(scene, device):
    c, r = [], []
    for o in scene.static_circles: c.append([o.x,o.y]); r.append(o.r)
    for o in scene.static_rects: c.append([o.x,o.y]); r.append(o.proxy_radius)
    if not c:
        return torch.empty((0,2),device=device), torch.empty((0,),device=device)
    return torch.tensor(c,dtype=torch.float32,device=device), torch.tensor(r,dtype=torch.float32,device=device)

def wall_tensors(scene, cfg, device):
    # Virtual wall obstacles are placed slightly outside the arena. They are used
    # by the VFH direction selector, while exact wall collision is still checked
    # by wall_clr(). This gives the UAV feed-forward boundary perception.
    if not cfg.use_virtual_walls:
        return torch.empty((0,2),device=device), torch.empty((0,),device=device)
    spacing = float(cfg.wall_point_spacing)
    margin = float(cfg.wall_virtual_margin)
    xs = np.arange(-scene.width/2, scene.width/2 + 1e-6, spacing, dtype=np.float32)
    ys = np.arange(-scene.height/2, scene.height/2 + 1e-6, spacing, dtype=np.float32)
    pts = []
    for x in xs:
        pts.append([x, -scene.height/2 - margin])
        pts.append([x,  scene.height/2 + margin])
    for y in ys:
        pts.append([-scene.width/2 - margin, y])
        pts.append([ scene.width/2 + margin, y])
    centers = torch.tensor(pts, dtype=torch.float32, device=device)
    radii = torch.full((centers.shape[0],), float(cfg.wall_virtual_radius), dtype=torch.float32, device=device)
    return centers, radii

def dyn_tensors(scene, device):
    p = torch.tensor(scene.dynamic_traj,dtype=torch.float32,device=device)
    v = torch.tensor(scene.dynamic_vel_traj,dtype=torch.float32,device=device)
    r = torch.tensor([d.r for d in scene.dynamic_init],dtype=torch.float32,device=device)
    return p,v,r

def wall_clr(pos,w,h,dr):
    vals = [pos[0]+w/2-dr, w/2-pos[0]-dr, pos[1]+h/2-dr, h/2-pos[1]-dr]
    return torch.min(torch.stack(vals))

def min_clr(pos, centers, radii, dr):
    if centers.shape[0] == 0:
        return torch.tensor(1e3,dtype=torch.float32,device=pos.device)
    return torch.min(torch.linalg.norm(centers-pos[None,:], dim=1) - radii - float(dr))

def ttc_metric(pos, vel, dp, dv, dr, drone_r, horizon):
    if dp.shape[0] == 0:
        return torch.tensor(horizon,device=pos.device), torch.tensor(1e3,device=pos.device)
    relp, relv = dp-pos[None,:], dv-vel[None,:]
    rv2 = torch.sum(relv*relv, dim=1).clamp_min(1e-6)
    t = -torch.sum(relp*relv, dim=1)/rv2
    t = torch.clamp(t, 0.0, float(horizon))
    closest = torch.linalg.norm(relp+relv*t[:,None],dim=1)-dr-float(drone_r)
    idx = torch.argmin(closest)
    return t[idx], closest[idx]

def rollout(model, scene, cfg, device):
    start = torch.tensor(scene.start,dtype=torch.float32,device=device)
    goal = torch.tensor(scene.goal,dtype=torch.float32,device=device)
    pos = start.clone()
    sp, sr = static_tensors(scene,device)
    # v5: do not inject virtual wall points into the VFH histogram.
    # Wall awareness is handled separately by boundary-direction cost in DiffVFH.forward().
    dpos_tr, dvel_tr, dr = dyn_tensors(scene,device)
    drone_r, safe, pc = cfg.drone_radius, cfg.safe_clearance, pref_clear(cfg, scene.stage_id)
    init_d = torch.linalg.norm(goal-pos).clamp_min(1e-6)
    prev_dir = torch.atan2((goal-pos)[1], (goal-pos)[0])
    prev_vel = torch.zeros(2,device=device)
    positions, velocities = [pos], []
    progress_hist = deque(maxlen=cfg.stuck_window)
    clearances=[]; dyn_clearances=[]; speeds=[]; cmd_speeds=[]; occs=[]; ents=[]; concs=[]; dyn_brakes=[]; wait_gates=[]; static_densities=[]; speed_push_gates=[]; speed_target_gates=[]; speed_adapt_risks=[]
    progress_terms=[]; smooth_terms=[]; stag_terms=[]; osc_terms=[]; ttc_terms=[]; dyn_pred_terms=[]; wall_terms=[]
    speed_target_terms=[]; speed_gate_terms=[]
    reached=collided=dyn_collision=False
    dyn_enc=0.0; esc_steps=0; stuck_activated=0.0
    min_ttc_val, min_ttc_clr_val = cfg.ttc_horizon, 1e3

    for t in range(scene.max_steps):
        if dpos_tr.shape[1] > 0:
            dp, dv = dpos_tr[t], dvel_tr[t]
            dspeed = torch.linalg.norm(dv,dim=1)
            infl = torch.clamp(dspeed*cfg.reaction_time, max=cfg.max_dynamic_inflation)
            dr_eff = dr + infl
            obs_p_vfh = torch.cat([sp, dp], dim=0)
            obs_r_eff_vfh = torch.cat([sr, dr_eff], dim=0)
            obs_p = torch.cat([sp, dp], dim=0)
            obs_r_eff = torch.cat([sr, dr_eff], dim=0)
            obs_r_phys = torch.cat([sr, dr], dim=0)
            obs_v = torch.cat([torch.zeros_like(sp), dv], dim=0)
            dyn_mask = torch.cat([torch.zeros(sp.shape[0],dtype=torch.bool,device=device), torch.ones(dp.shape[0],dtype=torch.bool,device=device)])
        else:
            dp = torch.empty((0,2),device=device); dv = torch.empty((0,2),device=device)
            obs_p_vfh, obs_r_eff_vfh = sp, sr
            obs_p, obs_r_eff, obs_r_phys = sp, sr, sr
            obs_v = torch.zeros_like(obs_p_vfh)
            dyn_mask = torch.zeros(obs_p_vfh.shape[0],dtype=torch.bool,device=device)

        old_d = torch.linalg.norm(goal-pos)
        stuck=False
        if len(progress_hist) >= cfg.stuck_window and len(positions) > cfg.stuck_window:
            rp = float(np.mean(progress_hist))
            disp = float(torch.linalg.norm(pos-positions[-cfg.stuck_window]).detach().cpu())
            stuck = old_d.item() > scene.goal_radius*cfg.stuck_goal_factor and rp < cfg.stuck_avg_progress and disp < cfg.stuck_max_displacement

        out = model(pos, goal, obs_p_vfh, obs_r_eff_vfh, obs_v, dyn_mask, prev_dir, drone_r, cfg.sensor_range, safe,
                    pc*(cfg.escape_clearance_scale if stuck else 1.0), cfg.ttc_horizon, cfg.ttc_clearance, cfg.ttc_beta, cfg.ttc_decay,
                    scene.width, scene.height, cfg.wall_influence_distance, cfg.wall_beta,
                    cfg.w_wall_direction, cfg.w_goal_recovery, cfg.dynamic_recovery_gain,
                    cfg.dynamic_brake_clearance, cfg.dynamic_brake_beta,
                    cfg.dynamic_min_speed_scale, cfg.dynamic_dir_clearance_weight,
                    float(cfg.enable_wait_action), cfg.dynamic_wait_clearance,
                    cfg.dynamic_wait_ttc_cost, cfg.dynamic_wait_static_density,
                    cfg.dynamic_wait_beta, cfg.dynamic_wait_max_gate,
                    cfg.dynamic_wait_speed_scale, cfg.dynamic_wait_goal_disable_distance)

        # Wait action is disabled, so no wait-based stuck suppression is used.

        cmd_dir = blend_angle(out["direction"], prev_dir, heading_inertia(cfg, scene.stage_id))
        cmd_speed = out["speed"]

        # Target executed speed for the current scene.  Static-only training aims
        # at about 8 m/s; mixed static-dynamic training aims at about 5 m/s.
        if str(getattr(cfg, "speed_strategy", "gated")).lower() == "aggressive_adaptive":
            # In this mode, every stage starts from the user-defined maximum UAV
            # speed.  The controller learns when it has to reduce speed.
            target_speed_value = stage_uav_speed_cap(cfg, scene.stage_id, int(scene.dynamic_target) > 0)
        else:
            target_speed_value = (
                float(getattr(cfg, "target_dynamic_avg_speed", 5.0))
                if int(scene.dynamic_target) > 0
                else float(getattr(cfg, "target_static_avg_speed", 8.0))
            )
        target_speed = torch.tensor(target_speed_value, device=device, dtype=torch.float32)

        # Additional near-obstacle braking based on current dynamic clearance.
        # This term is differentiable and reduces collision caused by late reaction.
        current_dyn_clr = torch.tensor(1e3, device=device, dtype=torch.float32)
        if dp.shape[0] > 0:
            current_dyn_clr = min_clr(pos, dp, dr, drone_r)
            near_brake = cfg.dynamic_min_speed_scale + (1.0 - cfg.dynamic_min_speed_scale) * torch.sigmoid(
                cfg.dynamic_brake_beta * (current_dyn_clr - cfg.dynamic_brake_clearance)
            )
            cmd_speed = cmd_speed * near_brake
        else:
            near_brake = torch.tensor(1.0, device=device)

        # ------------------------------------------------------------------
        # Aggressive high-speed adaptive policy
        # ------------------------------------------------------------------
        # Instead of multiplying several conservative gates into speed from the
        # beginning, start from the stage-wise maximum speed and reduce it only
        # when differentiable risk indicators become high.  Collision / near-miss
        # losses then train the model to brake early enough in similar states.
        if str(getattr(cfg, "speed_strategy", "gated")).lower() == "aggressive_adaptive":
            stage_cap = torch.tensor(
                stage_uav_speed_cap(cfg, scene.stage_id, int(scene.dynamic_target) > 0),
                dtype=torch.float32, device=device,
            )
            stage_cap = torch.minimum(stage_cap, model.p("vmax"))
            beta = model.p("speed_beta") * float(getattr(cfg, "aggressive_speed_beta_scale", 1.0))

            static_trigger = float(safe) + float(getattr(cfg, "aggressive_static_clearance_margin", 0.22))
            static_risk = torch.sigmoid(beta * (static_trigger - out["min_clearance"]))

            if dp.shape[0] > 0:
                dyn_clr_risk = torch.sigmoid(
                    beta * (float(getattr(cfg, "aggressive_dynamic_clearance_trigger", 1.00)) - current_dyn_clr)
                )
                dyn_ttc_risk = torch.sigmoid(
                    beta * (out["selected_ttc_cost"] - float(getattr(cfg, "aggressive_ttc_risk_trigger", 0.60)))
                )
                dyn_risk = 1.0 - (1.0 - dyn_clr_risk) * (1.0 - dyn_ttc_risk)
                min_scale = float(getattr(cfg, "aggressive_min_speed_scale_dynamic", 0.42))
            else:
                dyn_risk = torch.tensor(0.0, device=device, dtype=torch.float32)
                min_scale = float(getattr(cfg, "aggressive_min_speed_scale_static", 0.55))

            risk = torch.clamp(1.0 - (1.0 - static_risk) * (1.0 - dyn_risk), 0.0, 1.0)
            goal_gate_adapt = torch.clamp(
                old_d / float(getattr(cfg, "aggressive_goal_slow_distance", 4.0)),
                float(getattr(cfg, "aggressive_goal_min_speed_scale", 0.18)),
                1.0,
            )
            cmd_speed = stage_cap * goal_gate_adapt * (1.0 - risk * (1.0 - min_scale))
            speed_adapt_risks.append(risk)
        else:
            speed_adapt_risks.append(torch.tensor(0.0, device=device, dtype=torch.float32))

        # ------------------------------------------------------------------
        # Safety-gated cruise-speed floor
        # ------------------------------------------------------------------
        # The learnable VFH speed gate can become overly conservative in the
        # enlarged 80 x 45 m field, especially because avg_occ and predicted TTC
        # risk are multiplied into the speed.  This differentiable floor does
        # not force high speed in truly dangerous states; it only pushes the
        # command upward when clearance/TTC/goal-distance gates indicate that a
        # cruising segment is safe.  It is part of the controller and is used in
        # both training and validation because validation imports this rollout().
        local_gate = torch.sigmoid(
            float(getattr(cfg, "speed_floor_gate_beta", 5.0))
            * (out["min_clearance"] - (float(safe) + float(getattr(cfg, "speed_floor_clearance_margin", 0.20))))
        )
        goal_speed_gate = torch.clamp(old_d / float(getattr(cfg, "speed_floor_goal_distance", 6.0)), 0.0, 1.0)
        if dp.shape[0] > 0:
            dyn_speed_gate = torch.sigmoid(
                float(getattr(cfg, "speed_floor_gate_beta", 5.0))
                * (current_dyn_clr - float(getattr(cfg, "dynamic_speed_floor_clearance", 0.75)))
            ) * torch.sigmoid(
                float(getattr(cfg, "speed_floor_gate_beta", 5.0))
                * (float(getattr(cfg, "dynamic_speed_floor_ttc_cost", 0.62)) - out["selected_ttc_cost"])
            )
            floor_ratio = float(getattr(cfg, "dynamic_speed_floor_ratio", 0.88))
        else:
            dyn_speed_gate = torch.tensor(1.0, device=device, dtype=torch.float32)
            floor_ratio = float(getattr(cfg, "static_speed_floor_ratio", 0.90))

        speed_push_gate = torch.clamp(local_gate * goal_speed_gate * dyn_speed_gate, 0.0, 1.0)
        cruise_floor = target_speed * floor_ratio
        floor_beta = float(getattr(cfg, "speed_floor_softplus_beta", 6.0))
        floor_gain = float(getattr(cfg, "speed_floor_gain", 1.0))
        floor_deficit = F.softplus(cruise_floor - cmd_speed, beta=floor_beta) / floor_beta
        cmd_speed = cmd_speed + floor_gain * speed_push_gate * floor_deficit
        cmd_speed = torch.minimum(cmd_speed, model.p("vmax"))
        speed_push_gates.append(speed_push_gate)

        if stuck:
            esc_steps += 1; stuck_activated = 1.0
            ed = escape_heading(pos, goal, obs_p, obs_r_eff, cfg.escape_goal_blend)
            cmd_dir = blend_angle(cmd_dir, ed, cfg.escape_blend)
            cmd_speed = torch.maximum(cmd_speed*(1+cfg.escape_speed_boost), torch.tensor(cfg.escape_min_speed,device=device))
        cmd_speed = torch.minimum(cmd_speed, model.p("vmax"))
        cmd_speeds.append(cmd_speed)

        # Wait action disabled: use normal velocity smoothing.
        eff_smoothing = float(cfg.velocity_smoothing)
        vel = eff_smoothing * prev_vel + (1.0 - eff_smoothing) * (
            cmd_speed * torch.stack([torch.cos(cmd_dir), torch.sin(cmd_dir)])
        )
        actual_speed = torch.linalg.norm(vel)
        pos2 = pos + scene.dt*vel
        clr = torch.minimum(min_clr(pos2, obs_p, obs_r_phys, drone_r), wall_clr(pos2, scene.width, scene.height, drone_r))

        # High-speed target loss.  The previous version used free_gate*dyn_gate
        # directly; in dynamic scenes this often became almost zero, so the model
        # received little gradient to increase speed.  Keep the goal-distance gate
        # but add a small minimum local gate, so safe cruising is consistently
        # rewarded without encouraging acceleration at the final goal.
        free_gate = torch.sigmoid(
            float(getattr(cfg, "speed_target_gate_beta", 5.0))
            * (out["min_clearance"] - (float(safe) + float(getattr(cfg, "speed_target_clearance_margin", 0.20))))
        )
        goal_gate = torch.clamp(old_d / float(getattr(cfg, "speed_target_goal_distance", 6.0)), 0.0, 1.0)
        if dp.shape[0] > 0:
            dyn_gate = torch.sigmoid(
                float(getattr(cfg, "speed_target_gate_beta", 5.0))
                * (current_dyn_clr - float(getattr(cfg, "dynamic_speed_loss_clearance", 0.70)))
            ) * torch.sigmoid(
                float(getattr(cfg, "speed_target_gate_beta", 5.0))
                * (float(getattr(cfg, "dynamic_speed_loss_ttc_cost", 0.62)) - out["selected_ttc_cost"])
            )
            min_local_gate = float(getattr(cfg, "speed_target_min_gate_dynamic", 0.22))
        else:
            dyn_gate = torch.tensor(1.0, device=device, dtype=torch.float32)
            min_local_gate = float(getattr(cfg, "speed_target_min_gate_static", 0.35))
        local_speed_gate = torch.clamp(free_gate * dyn_gate, min=min_local_gate, max=1.0)
        speed_gate = (goal_gate * local_speed_gate).detach()
        speed_target_terms.append(speed_gate * (F.relu(target_speed - actual_speed) / target_speed.clamp_min(1e-6)) ** 2)
        speed_gate_terms.append(speed_gate)
        speed_target_gates.append(speed_gate)
        if dp.shape[0] > 0:
            dclr = min_clr(pos2, dp, dr, drone_r)
            dyn_clearances.append(dclr)
            tt, tc = ttc_metric(pos, vel, dp, dv, dr, drone_r, cfg.ttc_horizon)
            min_ttc_val = min(min_ttc_val, float(tt.detach().cpu()))
            min_ttc_clr_val = min(min_ttc_clr_val, float(tc.detach().cpu()))
            if float(dclr.detach().cpu()) < cfg.dynamic_encounter_distance or (float(tc.detach().cpu()) < cfg.ttc_clearance and float(tt.detach().cpu()) < cfg.ttc_horizon):
                dyn_enc = 1.0
            if float(dclr.detach().cpu()) < 0.0:
                dyn_collision = True
        else:
            dclr = torch.tensor(1e3,device=device)

        new_d = torch.linalg.norm(goal-pos2)
        prog = old_d - new_d
        progress_hist.append(float(prog.detach().cpu()))
        progress_terms.append(F.relu(cfg.min_step_progress - prog))
        smooth_terms.append(circ_dist(cmd_dir, prev_dir) + 0.15*torch.sum((vel-prev_vel)**2))
        stag_terms.append(torch.sigmoid(4.0*(new_d-scene.goal_radius*1.5)) * F.relu(cfg.stagnation_progress_threshold-prog)**2)
        if velocities:
            pv = velocities[-1]
            cosv = torch.dot(pv, vel)/(torch.linalg.norm(pv)*torch.linalg.norm(vel)+1e-6)
            osc_terms.append(F.relu(cfg.oscillation_cos_threshold-cosv)**2)
        else:
            osc_terms.append(torch.tensor(0.0,device=device))
        ttc_terms.append(out["selected_ttc_cost"]); dyn_pred_terms.append(out["min_pred_dynamic_clearance"]); wall_terms.append(out["wall_cost"])
        dyn_brakes.append(out["dynamic_brake"] * near_brake)
        wait_gates.append(out["wait_gate"])
        static_densities.append(out["local_static_density"])
        positions.append(pos2); velocities.append(vel); speeds.append(actual_speed)
        clearances.append(clr); occs.append(out["avg_occ"]); ents.append(out["entropy"]); concs.append(out["concentration"])

        if float(clr.detach().cpu()) < 0.0:
            collided=True; break
        if float(new_d.detach().cpu()) <= scene.goal_radius:
            reached=True; break
        pos, prev_dir, prev_vel = pos2, cmd_dir, vel

    final = positions[-1]
    final_d = torch.linalg.norm(goal-final)
    min_goal_d = torch.min(torch.stack([torch.linalg.norm(goal-p) for p in positions]))
    cseq = torch.stack(clearances) if clearances else torch.tensor([0.0],device=device)
    dseq = torch.stack(dyn_clearances) if dyn_clearances else torch.tensor([1e3],device=device)
    min_c, avg_c = torch.min(cseq), torch.mean(cseq)
    min_dc, avg_dc = torch.min(dseq), torch.mean(dseq)
    goal_loss = 0.60*(final_d/init_d) + 0.40*(min_goal_d/init_d)
    safety_loss = torch.mean((F.softplus((safe-cseq)*cfg.safety_beta)/cfg.safety_beta)**2)
    comfort_loss = torch.mean((F.softplus((pc-cseq)*cfg.comfort_beta)/cfg.comfort_beta)**2)
    dyn_loss = torch.mean((F.softplus((cfg.dynamic_clearance_target-dseq)*cfg.dynamic_beta)/cfg.dynamic_beta)**2)
    dynamic_collision_loss = (F.softplus(-min_dc*35.0)/35.0)**2
    pred_dyn_seq = torch.stack(dyn_pred_terms) if dyn_pred_terms else torch.tensor([1e3], device=device)
    pred_dynamic_clearance_loss = torch.mean((F.softplus((cfg.dynamic_pred_clearance_target-pred_dyn_seq)*cfg.dynamic_beta)/cfg.dynamic_beta)**2)
    collision_loss = (F.softplus(-min_c*30)/30)**2
    if speeds:
        speed_seq = torch.stack(speeds)
        # Penalize maintaining high speed in already-dangerous states.  This is
        # the main signal that converts collision/near-miss experience into a
        # learned speed adaptation policy.
        danger_seq = torch.sigmoid(model.p("speed_beta") * (float(safe) - cseq))
        overspeed_danger_loss = torch.mean(danger_seq * (speed_seq / model.p("vmax").clamp_min(1e-6)) ** 2)
        if str(getattr(cfg, "speed_strategy", "gated")).lower() != "aggressive_adaptive":
            overspeed_danger_loss = overspeed_danger_loss * 0.0
    else:
        overspeed_danger_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    smooth_loss = torch.mean(torch.stack(smooth_terms)) if smooth_terms else torch.tensor(0.0,device=device)
    progress_loss = torch.mean(torch.stack(progress_terms)) if progress_terms else torch.tensor(0.0,device=device)
    stagnation_loss = torch.mean(torch.stack(stag_terms)) if stag_terms else torch.tensor(0.0,device=device)
    oscillation_loss = torch.mean(torch.stack(osc_terms)) if osc_terms else torch.tensor(0.0,device=device)
    ttc_loss = torch.mean(torch.stack(ttc_terms)) if ttc_terms else torch.tensor(0.0,device=device)
    wall_loss = torch.mean(torch.stack(wall_terms)) if wall_terms else torch.tensor(0.0,device=device)
    wait_seq = torch.stack(wait_gates) if wait_gates else torch.tensor([0.0], device=device)
    wait_loss = torch.mean(wait_seq**2)
    if speed_target_terms:
        speed_gate_seq = torch.stack(speed_gate_terms)
        speed_target_loss = torch.sum(torch.stack(speed_target_terms)) / (torch.sum(speed_gate_seq) + 1e-6)
    else:
        speed_target_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    # Waiting is useful only as a short safety action; this mild loss prevents
    # an always-hover policy while progress/time losses keep the UAV moving.
    time_loss = torch.tensor(len(positions)/max(1,scene.max_steps),device=device,dtype=torch.float32)
    loss = (cfg.w_goal_loss*goal_loss + cfg.w_safety_loss*safety_loss + cfg.w_comfort_loss*comfort_loss +
            cfg.w_collision_loss*collision_loss + cfg.w_smooth_loss*smooth_loss + cfg.w_progress_loss*progress_loss +
            cfg.w_stagnation_loss*stagnation_loss + cfg.w_oscillation_loss*oscillation_loss +
            cfg.w_ttc_loss*ttc_loss + cfg.w_dynamic_clearance_loss*dyn_loss +
            cfg.w_dynamic_collision_loss*dynamic_collision_loss + cfg.w_dynamic_pred_loss*pred_dynamic_clearance_loss +
            cfg.w_wait_loss*wait_loss + cfg.w_wall_loss*wall_loss + cfg.w_time_loss*time_loss +
            float(getattr(cfg, "w_speed_target_loss", 2.0))*speed_target_loss +
            float(getattr(cfg, "w_overspeed_danger_loss", 1.0))*overspeed_danger_loss)
    pts = torch.stack(positions).detach()
    path_len = float(torch.sum(torch.linalg.norm(pts[1:]-pts[:-1],dim=1)).cpu()) if pts.shape[0] > 1 else 0.0

    # ------------------------------------------------------------------
    # Success definition
    # ------------------------------------------------------------------
    # reached_success:
    #   The UAV reaches the goal without collision.
    #
    # safe_progress_success:
    #   Restores the previous safe-progress success idea, but makes it
    #   stricter.  It is no longer enough to only avoid collision and survive
    #   enough steps.  The UAV must keep moving toward the goal, must not hover
    #   for a long time, and must not obtain success by oscillating around the
    #   same area.
    # ------------------------------------------------------------------
    episode_steps = max(0, len(positions) - 1)

    reached_success = bool(
        reached
        and (not collided)
        and (not dyn_collision)
    )

    avg_speed_val = float(torch.mean(torch.stack(speeds)).detach().cpu()) if speeds else 0.0

    # Goal-distance sequence for anti-stop and anti-oscillation checks.
    goal_dist_seq = torch.stack([torch.linalg.norm(goal - p) for p in positions]).detach().cpu().numpy().astype(np.float32)
    if goal_dist_seq.size <= 0:
        goal_dist_seq = np.asarray([float(init_d.detach().cpu())], dtype=np.float32)

    init_goal_dist = float(init_d.detach().cpu())
    final_goal_dist = float(final_d.detach().cpu())
    min_goal_dist = float(min_goal_d.detach().cpu())

    # "final_progress_ratio" is used for safe-progress success.  This prevents
    # the false-success case where the UAV once approached the goal but finally
    # moved away, stopped, or oscillated.
    final_progress_ratio = float((init_goal_dist - final_goal_dist) / max(init_goal_dist, 1e-6))
    best_progress_ratio = float((init_goal_dist - min_goal_dist) / max(init_goal_dist, 1e-6))

    if len(goal_dist_seq) >= 12:
        tail_n = max(10, int(0.25 * len(goal_dist_seq)))
        first_tail = goal_dist_seq[-tail_n:-(tail_n // 2)] if tail_n // 2 > 0 else goal_dist_seq[-tail_n:]
        last_tail = goal_dist_seq[-(tail_n // 2):] if tail_n // 2 > 0 else goal_dist_seq[-tail_n:]
        tail_start_dist = float(np.mean(first_tail))
        tail_end_dist = float(np.mean(last_tail))
        tail_progress_ratio = float((tail_start_dist - tail_end_dist) / max(init_goal_dist, 1e-6))
    else:
        tail_progress_ratio = final_progress_ratio

    if len(goal_dist_seq) >= 2:
        # Positive value means the UAV moved closer to the goal in that step.
        goal_progress_steps = -np.diff(goal_dist_seq)
        backward_ratio = float(np.mean(goal_progress_steps < -0.015))
        stagnation_ratio = float(np.mean(np.abs(goal_progress_steps) < 0.003))
    else:
        backward_ratio = 1.0
        stagnation_ratio = 1.0

    # Path efficiency prevents a long looping trajectory from being counted as
    # successful just because it did not collide.
    path_efficiency = float(
        max(init_goal_dist - final_goal_dist, 0.0) / max(float(path_len), 1e-6)
    )

    safe_progress_success = bool(
        (not collided)
        and (not dyn_collision)
        and episode_steps >= int(cfg.progress_success_steps)
        and final_progress_ratio >= float(cfg.progress_success_ratio)
        and tail_progress_ratio >= float(getattr(cfg, "tail_progress_success_ratio", 0.04))
        and avg_speed_val >= float(getattr(cfg, "min_success_avg_speed", 0.22))
        and stagnation_ratio <= float(getattr(cfg, "max_success_stagnation_ratio", 0.35))
        and backward_ratio <= float(getattr(cfg, "max_success_backward_ratio", 0.30))
        and path_efficiency >= float(getattr(cfg, "min_success_path_efficiency", 0.18))
    )

    # Final success logic:
    #   goal reaching always counts;
    #   safe progress also counts, but only if the UAV continues to move toward
    #   the goal and does not rely on stopping or oscillation.
    success = bool(reached_success or safe_progress_success)

    # Keep the legacy metric name "progress_ratio", but make it represent final
    # progress rather than best-ever progress.  best_progress_ratio is logged
    # separately for diagnosis.
    progress_ratio = float(final_progress_ratio)

    dyn_safe = 1.0 if scene.dynamic_target == 0 else float((not dyn_collision) and float(min_dc.detach().cpu()) >= cfg.dynamic_safe_clearance_threshold)
    return dict(loss=loss, goal_loss=goal_loss.detach(), safety_loss=safety_loss.detach(), comfort_loss=comfort_loss.detach(),
                collision_loss=collision_loss.detach(), smooth_loss=smooth_loss.detach(), progress_loss=progress_loss.detach(),
                stagnation_loss=stagnation_loss.detach(), oscillation_loss=oscillation_loss.detach(), ttc_loss=ttc_loss.detach(),
                dynamic_clearance_loss=dyn_loss.detach(), dynamic_collision_loss=dynamic_collision_loss.detach(),
                pred_dynamic_clearance_loss=pred_dynamic_clearance_loss.detach(), wait_loss=wait_loss.detach(),
                wall_loss=wall_loss.detach(), time_loss=time_loss.detach(), speed_target_loss=speed_target_loss.detach(),
                overspeed_danger_loss=overspeed_danger_loss.detach(),
                final_distance=float(final_d.detach().cpu()), min_goal_distance=float(min_goal_d.detach().cpu()),
                min_clearance=float(min_c.detach().cpu()), avg_clearance=float(avg_c.detach().cpu()),
                min_dynamic_clearance=float(min_dc.detach().cpu()), avg_dynamic_clearance=float(avg_dc.detach().cpu()),
                min_ttc=float(min_ttc_val), min_ttc_pred_clearance=float(min_ttc_clr_val),
                dynamic_encounter=float(dyn_enc), dynamic_collision=float(dyn_collision), dynamic_safe=float(dyn_safe),
                avg_speed=float(avg_speed_val),
                avg_cmd_speed=float(torch.mean(torch.stack(cmd_speeds)).detach().cpu()) if cmd_speeds else 0.0,
                avg_speed_push_gate=float(torch.mean(torch.stack(speed_push_gates)).detach().cpu()) if speed_push_gates else 0.0,
                avg_speed_target_gate=float(torch.mean(torch.stack(speed_target_gates)).detach().cpu()) if speed_target_gates else 0.0,
                avg_speed_adapt_risk=float(torch.mean(torch.stack(speed_adapt_risks)).detach().cpu()) if speed_adapt_risks else 0.0,
                avg_dynamic_brake=float(torch.mean(torch.stack(dyn_brakes)).detach().cpu()) if dyn_brakes else 1.0,
                avg_wait_gate=float(torch.mean(wait_seq).detach().cpu()) if wait_gates else 0.0,
                wait_rate=float(torch.mean((wait_seq > cfg.wait_count_threshold).float()).detach().cpu()) if wait_gates else 0.0,
                wait_steps=float(torch.sum((wait_seq > cfg.wait_count_threshold).float()).detach().cpu()) if wait_gates else 0.0,
                avg_local_static_density=float(torch.mean(torch.stack(static_densities)).detach().cpu()) if static_densities else 0.0,
                avg_soft_occupancy=float(torch.mean(torch.stack(occs)).detach().cpu()) if occs else 0.0,
                avg_entropy=float(torch.mean(torch.stack(ents)).detach().cpu()) if ents else 0.0,
                avg_concentration=float(torch.mean(torch.stack(concs)).detach().cpu()) if concs else 0.0,
                path_length=path_len, steps=len(positions)-1, reached=float(reached), collision=float(collided),
                success=float(success), reached_success=float(reached_success),
                safe_progress_success=float(safe_progress_success), progress_ratio=float(progress_ratio),
                final_progress_ratio=float(final_progress_ratio), best_progress_ratio=float(best_progress_ratio),
                tail_progress_ratio=float(tail_progress_ratio), stagnation_ratio=float(stagnation_ratio),
                backward_ratio=float(backward_ratio), path_efficiency=float(path_efficiency),
                episode_steps=float(episode_steps), stuck_activated=float(stuck_activated), escape_steps=float(esc_steps),
                preferred_clearance_used=float(pc), trajectory=pts.cpu().numpy(), scene=scene)


# ----------------------------- Open3D -----------------------------

def try_o3d():
    try:
        import open3d as o3d
        return o3d
    except Exception as e:
        print(f"[Open3D disabled] {e}"); return None

def cyl(o3d, r, h, xy, color):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=float(r), height=float(h), resolution=32)
    m.compute_vertex_normals(); m.paint_uniform_color(color); m.translate((float(xy[0]),float(xy[1]),float(h)/2))
    return m

def sph(o3d, r, xyz, color):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=float(r), resolution=24)
    m.compute_vertex_normals(); m.paint_uniform_color(color); m.translate(xyz); return m

def box(o3d, sx, sy, sz, xy, yaw, color):
    m = o3d.geometry.TriangleMesh.create_box(width=float(sx), height=float(sy), depth=float(sz))
    m.compute_vertex_normals(); m.paint_uniform_color(color); m.translate((-sx/2,-sy/2,0))
    R = m.get_rotation_matrix_from_xyz((0,0,float(yaw))); m.rotate(R, center=(0,0,0)); m.translate((float(xy[0]),float(xy[1]),0)); return m

def line_tube(o3d, p0, p1, radius, color):
    """Create a small cylinder between two 3D points for clearer UAV-like paths."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-8:
        return None
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=length, resolution=10)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    # Cylinder default axis is z. Rotate z-axis to segment direction.
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = v / length
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-8:
        axis = axis / axis_norm
        angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
        R = m.get_rotation_matrix_from_axis_angle(axis * angle)
        m.rotate(R, center=(0, 0, 0))
    elif direction[2] < 0:
        R = m.get_rotation_matrix_from_xyz((math.pi, 0, 0))
        m.rotate(R, center=(0, 0, 0))
    m.translate(tuple((p0 + p1) / 2.0))
    return m

def show_o3d(scene, cfg, traj, title):
    o3d = try_o3d()
    if o3d is None: return
    geoms=[]; w,h=scene.width,scene.height

    uav_z = float(getattr(cfg, "uav_flight_altitude", 2.2))
    ground_thick = float(getattr(cfg, "visual_ground_thickness", 0.04))
    wall_h = float(getattr(cfg, "visual_wall_height", 0.45))
    path_r = float(getattr(cfg, "uav_path_tube_radius", 0.035))
    dyn_path_r = float(getattr(cfg, "dynamic_path_tube_radius", 0.025))

    # Ground and boundary.  Walls are intentionally kept low so buildings and
    # UAV trajectory dominate the 3D visualization.
    ground=o3d.geometry.TriangleMesh.create_box(width=w,height=h,depth=ground_thick)
    ground.paint_uniform_color((0.72,0.82,0.92))
    ground.translate((-w/2,-h/2,-ground_thick))
    geoms.append(ground)

    wc=(0.16,0.16,0.16); th=0.35
    geoms += [
        box(o3d,w+2*th,th,wall_h,(0,-h/2-th/2),0,wc),
        box(o3d,w+2*th,th,wall_h,(0,h/2+th/2),0,wc),
        box(o3d,th,h+2*th,wall_h,(-w/2-th/2,0),0,wc),
        box(o3d,th,h+2*th,wall_h,(w/2+th/2,0),0,wc)
    ]

    # Tall static obstacles: rendered as buildings rising from the ground.
    for o in scene.static_circles:
        geoms.append(cyl(o3d,o.r,o.height,(o.x,o.y),(0.42,0.42,0.42)))
    for o in scene.static_rects:
        geoms.append(box(o3d,o.sx,o.sy,o.height,(o.x,o.y),o.yaw,(0.56,0.56,0.52)))

    # Dynamic obstacles are visualized as moving airborne objects at UAV altitude.
    idx = 0 if traj is None else min(len(traj)-1, scene.dynamic_traj.shape[0]-1)
    for i,d in enumerate(scene.dynamic_init):
        x,y = scene.dynamic_traj[idx,i]
        color = (1.0,0.02,0.02) if d.crossing else (0.9,0.18,0.02)
        geoms.append(sph(o3d,d.r,(float(x),float(y),uav_z),color))

        # Show dynamic-obstacle path at the same flight layer using short tubes.
        if scene.dynamic_traj is not None and scene.dynamic_traj.shape[0] >= 2:
            step = max(1, int(scene.dynamic_traj.shape[0] // 80))
            pts_dyn = scene.dynamic_traj[::step, i, :]
            for k in range(len(pts_dyn)-1):
                p0 = (float(pts_dyn[k,0]), float(pts_dyn[k,1]), uav_z)
                p1 = (float(pts_dyn[k+1,0]), float(pts_dyn[k+1,1]), uav_z)
                seg = line_tube(o3d, p0, p1, dyn_path_r, color)
                if seg is not None:
                    geoms.append(seg)

    # Start and goal markers are lifted to UAV flight altitude.
    geoms.append(sph(o3d,cfg.drone_radius*1.25,(float(scene.start[0]),float(scene.start[1]),uav_z),(0.05,0.1,1.0)))
    geoms.append(sph(o3d,scene.goal_radius*0.34,(float(scene.goal[0]),float(scene.goal[1]),uav_z),(0.25,1.0,0.05)))

    # UAV trajectory is rendered as an elevated tube rather than a ground line.
    if traj is not None and len(traj) >= 2:
        pts=np.zeros((len(traj),3))
        pts[:,:2]=traj[:,:2]
        pts[:,2]=uav_z
        # Downsample tube segments for interactive performance.
        step = max(1, int(len(pts) // 160))
        pts_show = pts[::step]
        if not np.allclose(pts_show[-1], pts[-1]):
            pts_show = np.vstack([pts_show, pts[-1]])
        for k in range(len(pts_show)-1):
            seg = line_tube(o3d, pts_show[k], pts_show[k+1], path_r, (0.0,0.0,1.0))
            if seg is not None:
                geoms.append(seg)
        geoms.append(sph(o3d,cfg.drone_radius,(float(traj[-1,0]),float(traj[-1,1]),uav_z),(0.05,0.12,1.0)))

    # Add a faint vertical altitude reference at start and goal to make the
    # off-ground flight layer visually obvious.
    for xy, color in [((scene.start[0], scene.start[1]), (0.05,0.1,1.0)), ((scene.goal[0], scene.goal[1]), (0.25,1.0,0.05))]:
        ref = line_tube(o3d, (float(xy[0]), float(xy[1]), 0.02), (float(xy[0]), float(xy[1]), uav_z), 0.018, color)
        if ref is not None:
            geoms.append(ref)

    vis=o3d.visualization.Visualizer(); vis.create_window(window_name=title,width=1280,height=850)
    for g in geoms: vis.add_geometry(g)
    opt=vis.get_render_option(); opt.background_color=np.array([1,1,1])
    ctr=vis.get_view_control()
    ctr.set_front([0.30,-0.58,-0.76])
    ctr.set_lookat([0,0,max(1.0,uav_z*0.60)])
    ctr.set_up([0,0,1])
    ctr.set_zoom(0.62)
    vis.run(); vis.destroy_window()


# ----------------------------- train -----------------------------

def save_ckpt(path, model, opt, cfg, ep, cur, extra=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    data = dict(episode=ep, model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                config=vars(cfg), readable_params=model.readable_params(), seed=cfg.seed,
                curriculum_state=cur.settings(), transition_log=cur.log)
    if extra: data.update(extra)
    torch.save(data, path)

def log_tb(w, ep, m, model, scene, cur, cfg, lr, gn, sr100, cr100, der100, dcr100):
    if w is None:
        return
    log_every = max(1, int(getattr(cfg, "log_every", 1)))
    if ep % log_every != 0:
        return

    for k in ["loss","goal_loss","safety_loss","comfort_loss","collision_loss","smooth_loss","progress_loss","stagnation_loss","oscillation_loss","ttc_loss","dynamic_clearance_loss","dynamic_collision_loss","pred_dynamic_clearance_loss","wait_loss","wall_loss","time_loss","speed_target_loss","overspeed_danger_loss"]:
        val = float(m[k].detach().cpu()) if torch.is_tensor(m[k]) else float(m[k])
        w.add_scalar("train/"+k, val, ep)
    for k in ["success","collision","reached","reached_success","safe_progress_success","progress_ratio","final_progress_ratio","best_progress_ratio","tail_progress_ratio","stagnation_ratio","backward_ratio","path_efficiency","episode_steps","steps","min_clearance","avg_clearance","min_dynamic_clearance","avg_dynamic_clearance","min_ttc","min_ttc_pred_clearance","dynamic_encounter","dynamic_collision","dynamic_safe","avg_speed","avg_cmd_speed","avg_speed_push_gate","avg_speed_target_gate","avg_speed_adapt_risk","avg_dynamic_brake","avg_wait_gate","wait_rate","wait_steps","avg_local_static_density","avg_soft_occupancy","avg_concentration","path_length","escape_steps"]:
        w.add_scalar("train/"+k, float(m[k]), ep)

    w.add_scalar("train/success_rate_100ep", sr100, ep)
    w.add_scalar("train/collision_rate_100ep", cr100, ep)
    w.add_scalar("train/dynamic_encounter_rate_100ep", der100, ep)
    w.add_scalar("train/dynamic_collision_rate_100ep", dcr100, ep)
    w.add_scalar("train/lr", lr, ep)
    w.add_scalar("train/grad_norm", gn, ep)

    st = cur.settings()
    for k in ["stage_id","static_count","dynamic_count","dynamic_speed","level_ep","stage_ep","lsr","lcr","lder","ldcr","ldsafe","lrr","lvavg"]:
        w.add_scalar("curriculum/"+k, float(st.get(k, 0.0)), ep)

    if float(st.get("dynamic_count", 0.0)) > 0.0:
        eff_der_thr = min(
            cfg.dynamic_encounter_threshold,
            cfg.dynamic_encounter_threshold_base + cfg.dynamic_encounter_threshold_slope * max(0, st["dynamic_count"] - 1)
        )
    else:
        eff_der_thr = 0.0
    w.add_scalar("curriculum/effective_dynamic_encounter_threshold", float(eff_der_thr), ep)

    scene_every = max(1, int(getattr(cfg, "scene_metric_every", 25)))
    if ep % scene_every == 0:
        w.add_text("curriculum/stage_name", st["stage_name"], ep)
        w.add_scalar("scene/static_actual", len(scene.static_circles)+len(scene.static_rects), ep)
        w.add_scalar("scene/dynamic_actual", len(scene.dynamic_init), ep)
        w.add_scalar("scene/max_steps", int(scene.max_steps), ep)
        w.add_scalar("scene/crossing_dynamic_actual", sum(1 for d in scene.dynamic_init if d.crossing), ep)
        for mk, mv in static_distribution_metrics(scene, cfg).items():
            w.add_scalar("scene/" + mk, float(mv), ep)

    param_every = max(1, int(getattr(cfg, "param_log_every", 10)))
    if ep % param_every == 0:
        for k, v in model.readable_params().items():
            w.add_scalar("params/" + k, v, ep)


class StageLevelCurriculum:
    """Independent current-stage curriculum.

    This object is intentionally reset at the beginning of every stage.
    Therefore LSR/LCR/DER/Dsafe describe only the current stage and current level,
    not the previous easier stages.
    """
    def __init__(self, cfg, stage_id: int, inherited_dynamic_speed: Optional[float] = None):
        self.cfg = cfg
        self.stage_id = int(stage_id)
        self.inherited_dynamic_speed = None if inherited_dynamic_speed is None else float(inherited_dynamic_speed)
        self.log = []

        # Stage-specific promotion windows.  Because promotion requires both
        # the sliding window to be full and level_ep >= min_level_episodes,
        # these defaults make the earliest possible advancement exactly:
        # Stage 1: 30 episodes, Stage 2: 60 episodes, Stage 3: 90 episodes.
        if self.stage_id == 1:
            self.level_window = int(cfg.static_level_window)
            self.min_level_episodes = int(cfg.static_min_level_episodes)
        elif self.stage_id == 2:
            self.level_window = int(getattr(cfg, "stage2_level_window", cfg.dynamic_level_window))
            self.min_level_episodes = int(getattr(cfg, "stage2_min_level_episodes", cfg.dynamic_min_level_episodes))
        else:
            self.level_window = int(getattr(cfg, "stage3_level_window", cfg.dynamic_level_window))
            self.min_level_episodes = int(getattr(cfg, "stage3_min_level_episodes", cfg.dynamic_min_level_episodes))
        self.win_succ = deque(maxlen=self.level_window)
        self.win_col = deque(maxlen=self.level_window)
        self.win_den = deque(maxlen=self.level_window)
        self.win_dcol = deque(maxlen=self.level_window)
        self.win_dsafe = deque(maxlen=self.level_window)
        self.win_reached = deque(maxlen=self.level_window)
        self.win_speed = deque(maxlen=self.level_window)
        self.level_ep = 0
        self.stage_ep = 0

        def inherited_or_default(default_speed: float) -> float:
            # A previous static-only stage has dynamic_speed=0.0, so Stage 2
            # falls back to its configured dynamic-count speed.  Once a dynamic
            # stage has been completed, later stages inherit that actual end speed.
            if self.inherited_dynamic_speed is not None and self.inherited_dynamic_speed > 1e-6:
                return float(self.inherited_dynamic_speed)
            return float(default_speed)

        if self.stage_id == 1:
            self.stage_name = f"S1_uniform_static_{cfg.stage1_static_start}"
            self.static_count = cfg.stage1_static_start
            self.dynamic_count = 0
            self.dynamic_speed = 0.0
            self.target_static = cfg.stage1_static_end
            self.target_dynamic = 0
            self.target_speed = 0.0
        elif self.stage_id == 2:
            self.stage_name = (
                f"S2_dynamic_count_{cfg.stage3_dynamic_start}_to_{cfg.stage3_dynamic_end}"
                f"_static_{cfg.dynamic_static_start}_to_{cfg.dynamic_static_end}"
            )
            self.dynamic_count = cfg.stage3_dynamic_start
            self.static_count = self._static_for_dynamic(self.dynamic_count)
            self.dynamic_speed = inherited_or_default(cfg.stage3_dynamic_speed)
            self.target_dynamic = cfg.stage3_dynamic_end
            self.target_static = self._static_for_dynamic(self.target_dynamic)
            self.target_speed = self.dynamic_speed
        elif self.stage_id == 3:
            # Stage 3 is the old Stage 4 speed curriculum; the old name was a
            # numbering mistake.  It is now a normal part of --stage all.
            inherited_start_speed = inherited_or_default(cfg.stage3_dynamic_speed)
            self.stage_name = (
                "S3_dynamic_speed"
                f"_{float(inherited_start_speed):.2f}_to_{float(cfg.stage4_speed_end):.2f}"
                f"_static_{cfg.dynamic_static_end}"
            )
            self.dynamic_count = cfg.stage3_dynamic_end
            self.static_count = self._static_for_dynamic(self.dynamic_count)
            self.dynamic_speed = inherited_start_speed
            self.target_dynamic = cfg.stage3_dynamic_end
            self.target_static = self.static_count
            self.target_speed = cfg.stage4_speed_end
        elif self.stage_id == 4:
            # Legacy alias retained only for old checkpoints/commands.
            self.stage_id = 3
            inherited_start_speed = inherited_or_default(cfg.stage3_dynamic_speed)
            self.stage_name = (
                "S3_dynamic_speed"
                f"_{float(inherited_start_speed):.2f}_to_{float(cfg.stage4_speed_end):.2f}"
                f"_static_{cfg.dynamic_static_end}"
            )
            self.dynamic_count = cfg.stage3_dynamic_end
            self.static_count = self._static_for_dynamic(self.dynamic_count)
            self.dynamic_speed = inherited_start_speed
            self.target_dynamic = cfg.stage3_dynamic_end
            self.target_static = self.static_count
            self.target_speed = cfg.stage4_speed_end
        else:
            raise ValueError(f"Unsupported stage_id: {stage_id}")

    def _static_for_dynamic(self, dynamic_count: int) -> int:
        """Map dynamic-obstacle difficulty to a static-obstacle count.

        In dynamic stages, the previous 50-static-obstacle density often leaves
        too little physical avoidance space.  Therefore the dynamic curriculum
        starts by resetting static obstacles to --dynamic_static_start and then
        decreases them linearly as dynamic obstacles are added.
        """
        d0 = int(self.cfg.stage3_dynamic_start)
        d1 = int(self.cfg.stage3_dynamic_end)
        n0 = int(self.cfg.dynamic_static_start)
        n1 = int(self.cfg.dynamic_static_end)
        if d1 <= d0:
            return n0
        alpha = (int(dynamic_count) - d0) / float(d1 - d0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        # power < 1.0 decreases static density earlier, giving Nd=4 more lateral
        # space while preserving the same start/end counts.
        power = float(getattr(self.cfg, "dynamic_static_schedule_power", 1.0))
        if power > 1e-6:
            alpha = float(alpha ** power)
        return int(round(n0 + alpha * (n1 - n0)))

    def settings(self):
        return dict(
            stage_id=self.stage_id,
            stage_name=self.stage_name,
            static_count=int(self.static_count),
            dynamic_count=int(self.dynamic_count),
            dynamic_speed=float(self.dynamic_speed),
            max_steps=int(effective_scene_max_steps(self.cfg, int(self.dynamic_count))),
            level_ep=int(self.level_ep),
            stage_ep=int(self.stage_ep),
            lsr=float(np.mean(self.win_succ)) if self.win_succ else 0.0,
            lcr=float(np.mean(self.win_col)) if self.win_col else 0.0,
            lder=float(np.mean(self.win_den)) if self.win_den else 0.0,
            ldcr=float(np.mean(self.win_dcol)) if self.win_dcol else 0.0,
            ldsafe=float(np.mean(self.win_dsafe)) if self.win_dsafe else 0.0,
            lrr=float(np.mean(self.win_reached)) if self.win_reached else 0.0,
            lvavg=float(np.mean(self.win_speed)) if self.win_speed else 0.0,
        )

    def _current_thresholds(self):
        dynamic_stage = self.stage_id >= 2
        min_ep = self.min_level_episodes
        sr_thr = self.cfg.dynamic_success_threshold if dynamic_stage else self.cfg.level_success_threshold
        cr_thr = self.cfg.dynamic_collision_threshold if dynamic_stage else self.cfg.level_collision_threshold
        if dynamic_stage:
            der_thr = min(
                self.cfg.dynamic_encounter_threshold,
                self.cfg.dynamic_encounter_threshold_base
                + self.cfg.dynamic_encounter_threshold_slope * max(0, self.dynamic_count - 1),
            )
            dsafe_thr = self.cfg.dynamic_safe_threshold
        else:
            der_thr = 0.0
            dsafe_thr = 0.0
        dcr_thr = self.cfg.dynamic_collision_rate_threshold if dynamic_stage else 1.0
        rr_thr = self.cfg.dynamic_reached_success_threshold if dynamic_stage else 0.0
        return min_ep, sr_thr, cr_thr, der_thr, dsafe_thr, dcr_thr, rr_thr

    def _speed_threshold(self) -> float:
        # Historical name kept for checkpoint/log compatibility.  This is now
        # only a logged speed reference, not a promotion threshold.
        return float(
            getattr(self.cfg, "dynamic_promotion_avg_speed", 5.5)
            if self.stage_id >= 2
            else getattr(self.cfg, "static_promotion_avg_speed", 5.5)
        )

    def _reset_level(self):
        self.level_ep = 0
        self.win_succ.clear()
        self.win_col.clear()
        self.win_den.clear()
        self.win_dcol.clear()
        self.win_dsafe.clear()
        self.win_reached.clear()
        self.win_speed.clear()

    def observe(self, success, collision, dynamic_encounter, dynamic_collision, dynamic_safe, reached_success, avg_speed, global_ep, writer=None):
        self.level_ep += 1
        self.stage_ep += 1
        self.win_succ.append(float(success))
        self.win_col.append(float(collision))
        self.win_den.append(float(dynamic_encounter))
        self.win_dcol.append(float(dynamic_collision))
        self.win_dsafe.append(float(dynamic_safe))
        self.win_reached.append(float(reached_success))
        self.win_speed.append(float(avg_speed))

        min_ep, sr_thr, cr_thr, der_thr, dsafe_thr, dcr_thr, rr_thr = self._current_thresholds()
        if len(self.win_succ) < self.level_window or self.level_ep < min_ep:
            return False, None

        sr = float(np.mean(self.win_succ))
        cr = float(np.mean(self.win_col))
        der = float(np.mean(self.win_den))
        dcr = float(np.mean(self.win_dcol))
        dsafe = float(np.mean(self.win_dsafe))
        rr = float(np.mean(self.win_reached))
        vavg = float(np.mean(self.win_speed))
        speed_thr = self._speed_threshold()

        # Promotion should reflect task reliability and safety, not raw flight speed.
        # The average speed is still logged and optimized by the gated speed loss,
        # but it is NOT a hard curriculum-advancement condition.
        ok = sr >= sr_thr and cr <= cr_thr
        if self.stage_id >= 2:
            ok = ok and dcr <= dcr_thr and dsafe >= dsafe_thr and rr >= rr_thr  # DER is logged only; not a promotion gate.
        if not ok:
            return False, None

        rec = dict(
            stage_id=self.stage_id,
            stage=self.stage_name,
            stage_episode=self.stage_ep,
            level_episode=self.level_ep,
            static_count=int(self.static_count),
            dynamic_count=int(self.dynamic_count),
            dynamic_speed=float(self.dynamic_speed),
            sr=sr,
            cr=cr,
            der=der,
            dcr=dcr,
            dsafe=dsafe,
            rr=rr,
            vavg=vavg,
            speed_thr=speed_thr,
            sr_thr=sr_thr,
            cr_thr=cr_thr,
            der_thr=der_thr,
            dcr_thr=dcr_thr,
            dsafe_thr=dsafe_thr,
            rr_thr=rr_thr,
        )
        self.log.append(rec)
        if writer is not None:
            writer.add_text(f"stage{self.stage_id}/level_event", str(rec), global_ep)

        # Increase current-stage difficulty, or finish the stage if already at max difficulty.
        if self.stage_id == 1:
            if self.static_count < self.target_static:
                self.static_count = min(self.target_static, self.static_count + self.cfg.static_increment)
                msg = f"Stage 1 level passed. Static obstacles -> {self.static_count}"
                self._reset_level()
                return False, msg
            return True, f"Stage 1 converged at {self.target_static} static obstacles."

        if self.stage_id == 2:
            if self.dynamic_count < self.target_dynamic:
                self.dynamic_count = min(self.target_dynamic, self.dynamic_count + self.cfg.dynamic_increment)
                self.static_count = self._static_for_dynamic(self.dynamic_count)
                msg = (
                    f"Stage 2 level passed. Dynamic obstacles -> {self.dynamic_count}; "
                    f"static obstacles -> {self.static_count}"
                )
                self._reset_level()
                return False, msg
            return True, f"Stage 2 converged at {self.target_dynamic} dynamic obstacles and {self.static_count} static obstacles. Entering Stage 3 speed curriculum."

        if self.stage_id == 3:
            if self.dynamic_speed < self.target_speed - 1e-6:
                self.dynamic_speed = min(self.target_speed, self.dynamic_speed + self.cfg.speed_increment)
                msg = f"Stage 3 level passed. Dynamic-obstacle speed -> {self.dynamic_speed:.2f} m/s."
                self._reset_level()
                return False, msg
            return True, f"Stage 3 converged at {self.target_speed:.2f} m/s."

        if self.stage_id == 4:
            # Legacy alias, should not be reached in normal three-stage training.
            if self.dynamic_speed < self.target_speed - 1e-6:
                self.dynamic_speed = min(self.target_speed, self.dynamic_speed + self.cfg.speed_increment)
                msg = f"Stage 3 level passed. Dynamic-obstacle speed -> {self.dynamic_speed:.2f} m/s."
                self._reset_level()
                return False, msg
            return True, f"Stage 3 converged at {self.target_speed:.2f} m/s."

        return False, None


def load_model_checkpoint(model, optimizer, checkpoint_path, device, load_optimizer=False):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    return ckpt


def find_latest_checkpoint(checkpoints_root, preferred_name="final_stagewise_diff_vfh.pth"):
    """Find the newest checkpoint under checkpoints_root.

    By default this searches recursively for final_stagewise_diff_vfh.pth inside
    checkpoints/<run_name>/.  If no final checkpoint is found, it falls back to
    common stage-completion checkpoint names so a completed Stage-2/3/4 checkpoint
    can still be used for fine-tuning.
    """
    root = Path(checkpoints_root).expanduser()
    if root.is_file():
        return root

    if not root.exists():
        return None

    preferred_name = str(preferred_name).strip() or "final_stagewise_diff_vfh.pth"
    search_names = [preferred_name]
    for fallback_name in (
        "final_stagewise_diff_vfh.pth",
        "stage4_diff_vfh.pth",
        "stage3_diff_vfh.pth",
        "stage2_diff_vfh.pth",
        "stage4_best_diff_vfh.pth",
        "stage3_best_diff_vfh.pth",
        "stage2_best_diff_vfh.pth",
    ):
        if fallback_name not in search_names:
            search_names.append(fallback_name)

    candidates = []
    for name in search_names:
        candidates.extend([p for p in root.rglob(name) if p.is_file()])
        if candidates and name == preferred_name:
            break

    if not candidates:
        candidates = [p for p in root.rglob("*.pth") if p.is_file()]

    if not candidates:
        return None

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_init_checkpoint(cfg):
    """Resolve explicit --init_checkpoint or automatic latest-final resume."""
    explicit = str(getattr(cfg, "init_checkpoint", "") or "").strip()
    if explicit:
        pth = Path(explicit).expanduser()
        if not pth.exists():
            raise FileNotFoundError(f"--init_checkpoint does not exist: {pth}")
        cfg.init_checkpoint = str(pth)
        return str(pth)

    if bool(getattr(cfg, "auto_resume_final", False)):
        search_dir = str(getattr(cfg, "resume_search_dir", "") or getattr(cfg, "out_dir", "checkpoints"))
        preferred_name = str(getattr(cfg, "resume_checkpoint_name", "final_stagewise_diff_vfh.pth"))
        pth = find_latest_checkpoint(search_dir, preferred_name=preferred_name)
        if pth is None:
            raise FileNotFoundError(
                f"No checkpoint found under {search_dir!r}. "
                "Pass --init_checkpoint explicitly or check --resume_search_dir."
            )
        cfg.init_checkpoint = str(pth)
        print(f"[resume] auto-selected checkpoint: {pth}")
        return str(pth)

    return None


def maybe_use_checkpoint_seed(cfg, checkpoint_path):
    """Optionally replace --seed with the seed stored inside the checkpoint."""
    if not checkpoint_path or not bool(getattr(cfg, "use_checkpoint_seed", False)):
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "seed" in ckpt:
        old_seed = int(getattr(cfg, "seed", 2026))
        cfg.seed = int(ckpt["seed"])
        print(f"[resume] using checkpoint seed: {cfg.seed} (was --seed {old_seed})")
    else:
        print("[resume] checkpoint has no stored seed; keeping --seed.")


def reset_model_vmax(model, vmax_value: float, device) -> None:
    """Reset the learnable vmax parameter after loading an old checkpoint."""
    vmax_value = float(vmax_value)
    if vmax_value <= 0.0:
        return
    lo, hi = model.ranges["vmax"]
    with torch.no_grad():
        model.raw_vmax.copy_(
            torch.tensor(inv_range(vmax_value, lo, hi), dtype=torch.float32, device=device)
        )


def stage_ids_from_arg(stage_arg, include_stage4=False):
    """Parse requested training stages for the three-stage curriculum.

    --stage all always runs Stage 1 -> Stage 2 -> Stage 3.
    Stage 3 is the old speed-training stage that was previously named Stage 4.
    """
    arg = str(stage_arg).lower()
    if arg == "all":
        return [1, 2, 3]
    sid = int(arg)
    if sid == 4:
        print("[warn] --stage 4 is a legacy alias; running Stage 3 speed curriculum instead.")
        sid = 3
    return [sid]

def stage_episode_budget(cfg, stage_id: int) -> int:
    """Return the maximum episode budget for a specific stage.

    Stage 1/2 can be shorter because they are static-only and usually reach
    high success rates early. Stage 3/4 keep the full budget by default because
    dynamic obstacles require more samples for stable TensorBoard curves and
    reliable convergence checks.
    """
    stage_specific = {
        1: int(getattr(cfg, "stage1_max_episodes", 0)),
        2: int(getattr(cfg, "stage2_max_episodes", 0)),
        3: int(getattr(cfg, "stage3_max_episodes", 0)),
        4: int(getattr(cfg, "stage4_max_episodes", 0)),
    }.get(int(stage_id), 0)
    return max(1, stage_specific if stage_specific > 0 else int(cfg.stage_max_episodes))



def terminal_write(msg: str, pbar=None) -> None:
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg, flush=True)


def format_criteria_status(stage_id: int, ep: int, cur: StageLevelCurriculum, cfg) -> str:
    st = cur.settings()
    min_ep, sr_thr, cr_thr, der_thr, dsafe_thr, dcr_thr, rr_thr = cur._current_thresholds()
    win_fill = len(cur.win_succ)
    speed_thr = cur._speed_threshold()
    prefix = (
        f"[Stage {stage_id} ep={ep}] "
        f"level_ep={st['level_ep']}/{min_ep}, window={win_fill}/{cur.level_window}, "
        f"Nst={st['static_count']}, Nd={st['dynamic_count']}, "
        f"Vdyn={st['dynamic_speed']:.2f}, max_steps={st['max_steps']}"
    )
    if int(st["dynamic_count"]) > 0 or int(stage_id) >= 2:
        return (
            prefix
            + " | criteria: "
            + f"LSR={st['lsr']:.3f}/{sr_thr:.3f}, "
            + f"LRR={st['lrr']:.3f}/{rr_thr:.3f}, "
            + f"LCR={st['lcr']:.3f}/{cr_thr:.3f}, "
            + f"DCR={st['ldcr']:.3f}/{dcr_thr:.3f}, "
            + f"Dsafe={st['ldsafe']:.3f}/{dsafe_thr:.3f}"
            + f" | logged: DER={st['lder']:.3f}/{der_thr:.3f}, "
            + f"Lvavg={st.get('lvavg',0.0):.2f} m/s, speed_ref={speed_thr:.2f} m/s"
        )
    return (
        prefix
        + " | criteria: "
        + f"LSR={st['lsr']:.3f}/{sr_thr:.3f}, "
        + f"LCR={st['lcr']:.3f}/{cr_thr:.3f}"
        + f" | logged: Lvavg={st.get('lvavg',0.0):.2f} m/s, speed_ref={speed_thr:.2f} m/s"
    )


def train_one_stage(cfg, stage_id, model, device, out_dir, writer, init_checkpoint=None, inherited_dynamic_speed=None):
    cur = StageLevelCurriculum(cfg, stage_id, inherited_dynamic_speed=inherited_dynamic_speed)
    stage_max_ep = stage_episode_budget(cfg, stage_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, stage_max_ep),
        eta_min=cfg.lr * 0.15,
    )

    if init_checkpoint:
        load_model_checkpoint(model, optimizer=None, checkpoint_path=init_checkpoint, device=device, load_optimizer=False)
        if float(getattr(cfg, "reset_vmax", -1.0)) > 0.0:
            reset_model_vmax(model, float(cfg.reset_vmax), device)
            print(f"[Stage {stage_id}] reset_vmax={float(cfg.reset_vmax):.3f} m/s")
        print(f"[Stage {stage_id}] init_checkpoint={init_checkpoint}")

    if bool(getattr(cfg, "compact_terminal", False)) or bool(getattr(cfg, "print_stage_events", False)):
        print(f"\n========== Stage {stage_id}: {cur.stage_name} ==========")
        print(
            f"budget={stage_max_ep}, window={cur.level_window}, min_ep={cur.min_level_episodes}, "
            f"initial Nst={cur.static_count}, Nd={cur.dynamic_count}, Vdyn={cur.dynamic_speed:.2f}, max_steps={cur.settings()['max_steps']}"
        )
        print(format_criteria_status(stage_id, 0, cur, cfg))

    stage_success_window = deque(maxlen=100)
    stage_collision_window = deque(maxlen=100)
    stage_dyn_enc_window = deque(maxlen=100)
    stage_dyn_col_window = deque(maxlen=100)
    stage_reached_window = deque(maxlen=100)
    stage_safe_progress_window = deque(maxlen=100)
    stage_loss_window = deque(maxlen=100)
    best_stage_score = -1e9
    completed = False
    last_metrics = None
    last_ep = 0

    use_compact_terminal = bool(getattr(cfg, "compact_terminal", False))
    pbar = None
    episode_iter = range(1, stage_max_ep + 1)
    if not use_compact_terminal:
        pbar = tqdm(
            episode_iter,
            desc=f"Stage {stage_id}: {cur.stage_name}",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
            ascii=True,
        )
        episode_iter = pbar

    for ep in episode_iter:
        last_ep = ep
        model.train()
        global_ep_for_seed = cfg.seed_stage_offset * stage_id + ep
        scene = generate_scene(cfg, global_ep_for_seed, cur)
        metrics = rollout(model, scene, cfg, device)
        loss = metrics["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).detach().cpu())
        optimizer.step()
        scheduler.step()

        stage_success_window.append(float(metrics["success"]))
        stage_collision_window.append(float(metrics["collision"]))
        stage_dyn_enc_window.append(float(metrics["dynamic_encounter"]))
        stage_dyn_col_window.append(float(metrics["dynamic_collision"]))
        stage_reached_window.append(float(metrics["reached_success"]))
        stage_safe_progress_window.append(float(metrics["safe_progress_success"]))
        stage_loss_window.append(float(loss.detach().cpu()))

        ssr100 = float(np.mean(stage_success_window)) if stage_success_window else 0.0
        scr100 = float(np.mean(stage_collision_window)) if stage_collision_window else 0.0
        sder100 = float(np.mean(stage_dyn_enc_window)) if stage_dyn_enc_window else 0.0
        sdcr100 = float(np.mean(stage_dyn_col_window)) if stage_dyn_col_window else 0.0
        rsr100 = float(np.mean(stage_reached_window)) if stage_reached_window else 0.0
        spsr100 = float(np.mean(stage_safe_progress_window)) if stage_safe_progress_window else 0.0
        loss100 = float(np.mean(stage_loss_window)) if stage_loss_window else 0.0
        lr_now = float(optimizer.param_groups[0]["lr"])

        # Existing logger, now fed by current-stage curriculum only. Use a compact
        # continuous global TensorBoard step so shortened early stages do not leave
        # artificial gaps on train/* curves.
        global_tb_step = int(getattr(cfg, "tb_global_offset", 0)) + ep
        log_tb(writer, global_tb_step, metrics, model, scene, cur, cfg, lr_now, grad_norm, ssr100, scr100, sder100, sdcr100)

        if writer is not None and (ep % max(1, int(getattr(cfg, "log_every", 1))) == 0):
            step = ep
            writer.add_scalar(f"stage{stage_id}/success_rate_100ep", ssr100, step)
            writer.add_scalar(f"stage{stage_id}/collision_rate_100ep", scr100, step)
            writer.add_scalar(f"stage{stage_id}/dynamic_encounter_rate_100ep", sder100, step)
            writer.add_scalar(f"stage{stage_id}/dynamic_collision_rate_100ep", sdcr100, step)
            writer.add_scalar(f"stage{stage_id}/reached_success_rate_100ep", rsr100, step)
            writer.add_scalar(f"stage{stage_id}/safe_progress_success_rate_100ep", spsr100, step)
            writer.add_scalar("train/reached_success_rate_100ep", rsr100, global_tb_step)
            writer.add_scalar("train/safe_progress_success_rate_100ep", spsr100, global_tb_step)
            writer.add_scalar(f"stage{stage_id}/level_success_rate", cur.settings()["lsr"], step)
            writer.add_scalar(f"stage{stage_id}/level_collision_rate", cur.settings()["lcr"], step)
            writer.add_scalar(f"stage{stage_id}/level_dynamic_collision_rate", cur.settings()["ldcr"], step)
            writer.add_scalar(f"stage{stage_id}/level_avg_speed", cur.settings().get("lvavg", 0.0), step)
            writer.add_scalar(f"stage{stage_id}/static_count", cur.settings()["static_count"], step)
            writer.add_scalar(f"stage{stage_id}/dynamic_count", cur.settings()["dynamic_count"], step)
            writer.add_scalar(f"stage{stage_id}/dynamic_speed", cur.settings()["dynamic_speed"], step)
            writer.add_scalar(f"stage{stage_id}/loss100", loss100, step)

        score = ssr100 - 0.85 * scr100 - 0.50 * sdcr100
        if (
            ep >= 30
            and score > best_stage_score + float(getattr(cfg, "best_min_delta", 0.0))
            and ep % max(1, int(getattr(cfg, "best_save_interval", 100))) == 0
        ):
            best_stage_score = score
            save_ckpt(
                out_dir / f"stage{stage_id}_best_diff_vfh.pth",
                model,
                optimizer,
                cfg,
                ep,
                cur,
                {"stage_id": stage_id, "stage_score": score, "ssr100": ssr100, "scr100": scr100, "sder100": sder100, "sdcr100": sdcr100},
            )

        if ep % cfg.save_every == 0:
            save_ckpt(
                out_dir / f"stage{stage_id}_latest_diff_vfh.pth",
                model,
                optimizer,
                cfg,
                ep,
                cur,
                {"stage_id": stage_id, "ssr100": ssr100, "scr100": scr100, "sder100": sder100, "sdcr100": sdcr100},
            )

        # Visualization is intentionally NOT performed at fixed episode intervals.
        # For paper experiments, we visualize only after a stage is completed, so
        # the displayed trajectory corresponds to a converged policy at that stage.
        stage_done, event_msg = cur.observe(
            success=float(metrics["success"]),
            collision=float(metrics["collision"]),
            dynamic_encounter=float(metrics["dynamic_encounter"]),
            dynamic_collision=float(metrics["dynamic_collision"]),
            dynamic_safe=float(metrics["dynamic_safe"]),
            reached_success=float(metrics["reached_success"]),
            avg_speed=float(metrics.get("avg_speed", 0.0)),
            global_ep=ep,
            writer=writer,
        )
        if event_msg:
            if bool(getattr(cfg, "print_stage_events", False)) or use_compact_terminal:
                terminal_write(f"[Stage {stage_id} ep={ep}] {event_msg}", pbar)
            if pbar is not None:
                pbar.set_description_str(f"Stage {stage_id}: {cur.stage_name}")

            # v12: visualize every completed sub-level, not only the final
            # large-stage completion.  This is useful for checking obstacle
            # uniformity and whether the policy is exploiting an empty side.
            if cfg.vis_o3d and cfg.vis_each_level:
                show_o3d(
                    scene,
                    cfg,
                    metrics["trajectory"],
                    f"Dense-uniform v2 Stage {stage_id} Level Completed at Episode {ep} | Nst={scene.static_target} Nd={scene.dynamic_target}"
                )

        status_line = format_criteria_status(stage_id, ep, cur, cfg)
        terminal_log_every = max(1, int(getattr(cfg, "terminal_log_every", 50)))
        if use_compact_terminal:
            if ep == 1 or ep % terminal_log_every == 0 or event_msg or stage_done:
                terminal_write(status_line, pbar)
        elif pbar is not None:
            st_now = cur.settings()
            min_ep_now, _, _, _, _, _, _ = cur._current_thresholds()
            pbar.set_description_str(f"Stage {stage_id}: {cur.stage_name}")
            pbar.set_postfix_str(
                f"Nst={st_now['static_count']} Nd={st_now['dynamic_count']} "
                f"T={st_now['max_steps']} lvl={st_now['level_ep']}/{min_ep_now} "
                f"V={st_now.get('lvavg',0.0):.2f}/{cur._speed_threshold():.2f} "
                f"win={len(cur.win_succ)}/{cur.level_window}"
            )

        last_metrics = metrics

        if stage_done:
            completed = True
            save_ckpt(
                out_dir / f"stage{stage_id}_diff_vfh.pth",
                model,
                optimizer,
                cfg,
                ep,
                cur,
                {
                    "stage_id": stage_id,
                    "stage_completed": True,
                    "stage_episode": ep,
                    "stage_score": score,
                    "ssr100": ssr100,
                    "scr100": scr100,
                    "sder100": sder100,
                    "sdcr100": sdcr100,
                },
            )
            if bool(getattr(cfg, "print_stage_events", False)) or use_compact_terminal:
                terminal_write(f"[Stage {stage_id}] completed and saved: {out_dir / f'stage{stage_id}_diff_vfh.pth'}", pbar)

            # Stage-end visualization.  If --vis_each_level is enabled, the final
            # sub-level has already been shown above, so avoid duplicate windows.
            if cfg.vis_o3d and not cfg.vis_each_level:
                show_o3d(
                    scene,
                    cfg,
                    metrics["trajectory"],
                    f"Dense-uniform v2 Stage {stage_id} Completed at Episode {ep} | Nst={scene.static_target} Nd={scene.dynamic_target}"
                )
            break

    if pbar is not None:
        pbar.close()

    if not completed:
        save_ckpt(
            out_dir / f"stage{stage_id}_unfinished_diff_vfh.pth",
            model,
            optimizer,
            cfg,
            last_ep,
            cur,
            {"stage_id": stage_id, "stage_completed": False, "stage_episode": last_ep},
        )
        print(f"[Stage {stage_id}] did not converge within {stage_max_ep} episodes.")
        print(f"[Stage {stage_id}] saved unfinished checkpoint: {out_dir / f'stage{stage_id}_unfinished_diff_vfh.pth'}")

    cfg.tb_global_offset = int(getattr(cfg, "tb_global_offset", 0)) + int(last_ep)
    return completed, out_dir / f"stage{stage_id}_diff_vfh.pth", cur


def train(cfg):
    if cfg.stage_max_episodes <= 0:
        cfg.stage_max_episodes = int(cfg.episodes)

    init_checkpoint = resolve_init_checkpoint(cfg)
    maybe_use_checkpoint_seed(cfg, init_checkpoint)
    set_seed(cfg.seed, deterministic=not cfg.nondeterministic)
    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    run_name = cfg.run_name or time.strftime("diff_vfh_stagewise_v12_%Y%m%d-%H%M%S")
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(Path(cfg.log_dir) / run_name)) if (cfg.tensorboard and SummaryWriter is not None) else None

    model = DiffVFH(cfg.n_sectors, str(device)).to(device)
    if float(getattr(cfg, "reset_vmax", -1.0)) > 0.0 and not bool(getattr(cfg, "init_checkpoint", "")):
        reset_model_vmax(model, float(cfg.reset_vmax), device)
        print(f"reset_vmax={float(cfg.reset_vmax):.3f} m/s")

    stage_ids = stage_ids_from_arg(cfg.stage, include_stage4=getattr(cfg, "include_stage4", False))
    if bool(getattr(cfg, "auto_resume_final", False)) and str(cfg.stage).lower() == "all":
        print("[resume] Warning: --auto_resume_final with --stage all will run the normal stage list. "
              "For speed fine-tuning from the final model, use --stage 3.")

    print("\nStage-wise training mode: THREE-STAGE ADAPTIVE-SPEED NO-VGATE")
    cap_est, pitch_est = estimate_static_capacity(cfg)
    if max(cfg.stage1_static_end, cfg.dynamic_static_start, cfg.dynamic_static_end) > cap_est:
        print(f"[Warning] static target {max(cfg.stage1_static_end, cfg.dynamic_static_start, cfg.dynamic_static_end)} exceeds conservative capacity {cap_est}.")
    cfg.tb_global_offset = 0
    stage_budget_summary = {sid: stage_episode_budget(cfg, sid) for sid in stage_ids}
    print(f"device={device}, stages={stage_ids}, budgets={stage_budget_summary}")
    print(
        "criteria: "
        f"Stage1 window={cfg.static_level_window}, min_ep={cfg.static_min_level_episodes}; "
        f"Stage2 window={cfg.stage2_level_window}, min_ep={cfg.stage2_min_level_episodes}; "
        f"Stage3 window={cfg.stage3_level_window}, min_ep={cfg.stage3_min_level_episodes}; "
        f"static gates: LSR>={cfg.level_success_threshold}, LCR<={cfg.level_collision_threshold}; "
        f"dynamic gates: LSR>={cfg.dynamic_success_threshold}, LRR>={cfg.dynamic_reached_success_threshold}, "
        f"LCR<={cfg.dynamic_collision_threshold}, DCR<={cfg.dynamic_collision_rate_threshold}, "
        f"Dsafe>={cfg.dynamic_safe_threshold}; DER and Lvavg logged only"
    )
    print(
        "dynamic schedule: "
        f"static {cfg.dynamic_static_start}->{cfg.dynamic_static_end} as Nd {cfg.stage3_dynamic_start}->{cfg.stage3_dynamic_end}; "
        f"Stage2 dynamic speed fallback {cfg.stage3_dynamic_speed:.2f} m/s; "
        f"Stage3 starts from previous-stage end speed and increases to {cfg.stage4_speed_end:.2f} m/s "
        f"(increment {cfg.speed_increment:.2f}); "
        f"max_steps base={cfg.max_steps}, adaptive={getattr(cfg, 'adaptive_dynamic_steps', True)}, "
        f"increment={getattr(cfg, 'dynamic_step_increment', 35)}/dynamic, cap={getattr(cfg, 'dynamic_max_steps_cap', 0)}"
    )
    print(f"terminal: tqdm={not bool(getattr(cfg, 'compact_terminal', False))}, compact={getattr(cfg, 'compact_terminal', False)}")
    print(f"output_dir={out_dir}")

    last_cur_for_final = None
    previous_stage_dynamic_speed = None
    for idx, sid in enumerate(stage_ids):
        # For separate stage runs, the user can provide --init_checkpoint.
        # For --stage all, the same model parameters are passed forward after each
        # saved stage checkpoint, exactly equivalent to loading the previous stage.
        ckpt_to_load = init_checkpoint if idx == 0 else None
        completed, stage_ckpt, stage_cur = train_one_stage(
            cfg, sid, model, device, out_dir, writer,
            init_checkpoint=ckpt_to_load,
            inherited_dynamic_speed=previous_stage_dynamic_speed,
        )
        last_cur_for_final = stage_cur
        if stage_cur is not None:
            previous_stage_dynamic_speed = float(stage_cur.dynamic_speed)

        if not completed:
            if cfg.continue_if_stage_unfinished:
                print(f"[Stage {sid}] continuing despite unfinished stage because --continue_if_stage_unfinished was set.")
            else:
                print(f"[Stage {sid}] stopping. Later stages will not start because the current stage did not converge.")
                break

        # After a stage completes, explicitly reload its saved checkpoint before
        # the next stage to match the user's requested pth-passing protocol.
        if completed and stage_ckpt.exists() and sid != stage_ids[-1]:
            load_model_checkpoint(model, optimizer=None, checkpoint_path=stage_ckpt, device=device, load_optimizer=False)
            print(f"[Stage {sid}] checkpoint reloaded for next stage: {stage_ckpt}")

    save_ckpt(out_dir / "final_stagewise_diff_vfh.pth", model, torch.optim.AdamW(model.parameters(), lr=cfg.lr), cfg, 0, last_cur_for_final or StageLevelCurriculum(cfg, stage_ids[-1], inherited_dynamic_speed=previous_stage_dynamic_speed), {"stagewise_final": True})
    if writer:
        writer.flush()
        writer.close()

    print("\nStage-wise training finished.")
    print(f"Output directory: {out_dir}")
    print("Readable parameters:")
    for k, v in model.readable_params().items():
        print(f"  {k:>14s}: {v:.6f}")


def build_arg_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=5000, help="Fallback maximum episodes per stage when --stage_max_episodes <= 0.")
    p.add_argument("--stage_max_episodes",type=int,default=0, help="Fallback maximum episodes for stages without a stage-specific budget. 0 means use --episodes.")
    p.add_argument("--stage1_max_episodes",type=int,default=1200,
                   help="Maximum episodes for Stage 1. Static-only training is shortened by default for cleaner TensorBoard curves.")
    p.add_argument("--stage2_max_episodes",type=int,default=0,
                   help="Maximum episodes for Stage 2 dynamic curriculum. 0 means use --stage_max_episodes/--episodes.")
    p.add_argument("--stage3_max_episodes",type=int,default=0,
                   help="Maximum episodes for Stage 3. 0 means use --stage_max_episodes/--episodes.")
    p.add_argument("--stage4_max_episodes",type=int,default=0,
                   help="Maximum episodes for Stage 4. 0 means use --stage_max_episodes/--episodes.")
    p.add_argument("--stage",type=str,default="all",choices=["1","2","3","4","all"], help="Run one stage or all stages sequentially. --stage all runs Stage 1 -> Stage 2 -> Stage 3. Stage 4 is accepted only as a legacy alias of Stage 3.")
    p.add_argument("--include_stage4",action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode",type=str,default="train",choices=["train","visualize"], help="train: stagewise training; visualize: load a pth and show trajectories.")
    p.add_argument("--init_checkpoint",type=str,default="", help="Checkpoint used to initialize a single-stage run, or checkpoint to visualize.")
    p.add_argument("--auto_resume_final",action="store_true",
                   help="When --init_checkpoint is empty, automatically load the newest final_stagewise_diff_vfh.pth under --resume_search_dir/--out_dir.")
    p.add_argument("--resume_search_dir",type=str,default="",
                   help="Directory searched recursively by --auto_resume_final. Default: --out_dir, normally checkpoints/.")
    p.add_argument("--resume_checkpoint_name",type=str,default="final_stagewise_diff_vfh.pth",
                   help="Checkpoint filename preferred by --auto_resume_final.")
    p.add_argument("--use_checkpoint_seed",action="store_true",
                   help="Use the seed saved inside the checkpoint. Otherwise the explicit --seed is kept; default seed is 2026.")
    p.add_argument("--reset_vmax",type=float,default=-1.0, help="If >0, reset learnable vmax after loading an old checkpoint. Values are clamped by the learnable vmax range [2, 8].")
    p.add_argument("--continue_if_stage_unfinished",action="store_true", help="Allow the next stage to start even if the current stage does not meet convergence criteria.")
    p.add_argument("--seed_stage_offset",type=int,default=100000, help="Episode seed offset between stages for reproducibility.")
    p.add_argument("--training_mode",type=str,default="stagewise",choices=["stagewise","adaptive","staged"])
    p.add_argument("--device",type=str,default="cuda"); p.add_argument("--seed",type=int,default=2026); p.add_argument("--nondeterministic",action="store_true")
    p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--weight_decay",type=float,default=1e-4); p.add_argument("--grad_clip",type=float,default=8.0)

    p.add_argument("--level_window",type=int,default=50,
                   help="Legacy/default level window. Dynamic stages use this unless --dynamic_level_window is set.")
    p.add_argument("--static_level_window",type=int,default=30,
                   help="Sliding-window length for Stage 1 convergence checks.")
    p.add_argument("--dynamic_level_window",type=int,default=90,
                   help="Legacy dynamic-stage window. Stage 2/3 use --stage2_level_window/--stage3_level_window by default.")
    p.add_argument("--min_level_episodes",type=int,default=40,
                   help="Legacy static minimum level episodes; kept for compatibility.")
    p.add_argument("--static_min_level_episodes",type=int,default=30,
                   help="Minimum episodes at each Stage 1 difficulty level. Default 30.")
    p.add_argument("--stage2_level_window",type=int,default=60,
                   help="Sliding-window length for Stage 2 convergence checks. Default 60.")
    p.add_argument("--stage2_min_level_episodes",type=int,default=60,
                   help="Minimum episodes at each Stage 2 difficulty level. Default 60.")
    p.add_argument("--stage3_level_window",type=int,default=90,
                   help="Sliding-window length for Stage 3 convergence checks. Default 90.")
    p.add_argument("--stage3_min_level_episodes",type=int,default=90,
                   help="Minimum episodes at each Stage 3 difficulty level. Default 90.")
    p.add_argument("--dynamic_min_level_episodes",type=int,default=90)
    p.add_argument("--level_success_threshold",type=float,default=0.90); p.add_argument("--level_collision_threshold",type=float,default=0.12)
    p.add_argument("--dynamic_success_threshold",type=float,default=0.84); p.add_argument("--dynamic_collision_threshold",type=float,default=0.12)
    p.add_argument("--dynamic_reached_success_threshold",type=float,default=0.30,
                   help="Minimum reached-goal success rate required before dynamic stages can advance. In forced-encounter mode this is the main anti-safe-progress-only gate.")
    p.add_argument("--dynamic_collision_rate_threshold",type=float,default=0.10)
    p.add_argument("--dynamic_encounter_threshold",type=float,default=0.85); p.add_argument("--dynamic_encounter_threshold_base",type=float,default=0.55); p.add_argument("--dynamic_encounter_threshold_slope",type=float,default=0.07); p.add_argument("--dynamic_safe_threshold",type=float,default=0.72)
    p.add_argument("--static_promotion_avg_speed",type=float,default=5.5, help="Logged reference only. Stage promotion no longer uses average UAV speed as a hard gate.")
    p.add_argument("--dynamic_promotion_avg_speed",type=float,default=5.5, help="Logged reference only. Dynamic-stage promotion no longer uses average UAV speed as a hard gate.")
    p.add_argument("--early_stop_after_final_stage",action="store_true",default=True); p.add_argument("--no_early_stop",dest="early_stop_after_final_stage",action="store_false")

    p.add_argument("--stage1_static_start",type=int,default=70); p.add_argument("--stage1_static_end",type=int,default=70)
    p.add_argument("--stage2_static_start",type=int,default=50); p.add_argument("--stage2_static_end",type=int,default=30)
    p.add_argument("--stage3_dynamic_start",type=int,default=1); p.add_argument("--stage3_dynamic_end",type=int,default=6)
    p.add_argument("--stage3_dynamic_speed",type=float,default=0.7,
                   help="Fallback dynamic-obstacle speed for Stage 2 when there is no nonzero previous-stage speed to inherit.")
    p.add_argument("--stage4_speed_start",type=float,default=0.7,
                   help="Legacy argument retained for old commands; Stage 3 now inherits the previous stage end speed instead of using this value.")
    p.add_argument("--stage4_speed_end",type=float,default=2.0,
                   help="Final dynamic obstacle speed for Stage 4 speed fine-tuning.")
    p.add_argument("--dynamic_static_start",type=int,default=50,
                   help="Static obstacle count after entering dynamic stages. This resets density from Stage 2 before adding dynamic obstacles.")
    p.add_argument("--dynamic_static_end",type=int,default=30,
                   help="Static obstacle count at the maximum dynamic-obstacle level. Static count is linearly interpolated from dynamic_static_start to this value as dynamic_count increases.")
    p.add_argument("--dynamic_static_schedule_power",type=float,default=1.0,
                   help="Power applied to the dynamic-to-static schedule. Default 1.0 gives a linear 50 -> 30 schedule as dynamic obstacles increase from 1 -> 6.")
    p.add_argument("--static_increment",type=int,default=3); p.add_argument("--dynamic_increment",type=int,default=1)
    p.add_argument("--speed_increment",type=float,default=0.5,
                   help="Dynamic-obstacle speed increment after each passed Stage 4 level.")

    p.add_argument("--arena_width",type=float,default=80.0); p.add_argument("--arena_height",type=float,default=45.0); p.add_argument("--dt",type=float,default=0.06); p.add_argument("--max_steps",type=int,default=840,
                   help="Base maximum rollout steps per episode. Static stages use this value directly; dynamic stages can increase it with --adaptive_dynamic_steps.")
    p.add_argument("--adaptive_dynamic_steps",action="store_true",default=True,
                   help="Increase max rollout steps as the number of dynamic obstacles grows.")
    p.add_argument("--no_adaptive_dynamic_steps",dest="adaptive_dynamic_steps",action="store_false",
                   help="Disable dynamic-obstacle-based max-step scheduling and always use --max_steps.")
    p.add_argument("--dynamic_step_start_count",type=int,default=1,
                   help="Dynamic-obstacle count at which adaptive max-step scheduling starts. Counts at or below this keep --max_steps.")
    p.add_argument("--dynamic_step_increment",type=int,default=35,
                   help="Additional rollout steps for each dynamic obstacle above --dynamic_step_start_count. Default gives Nd=5 about 560 steps from a 420-step base.")
    p.add_argument("--dynamic_max_steps_cap",type=int,default=1000,
                   help="Upper bound for adaptive dynamic max steps. Use <=0 for no cap.")
    p.add_argument("--scene_retries",type=int,default=12); p.add_argument("--path_grid_res",type=float,default=0.50); p.add_argument("--path_clearance",type=float,default=0.76)
    p.add_argument("--start_goal_clearance",type=float,default=2.2)
    # Start/goal sampling.  Start is fixed; goal is sampled on the right side.
    p.add_argument("--fixed_start_x",type=float,default=None,
                   help="Fixed start x. None uses left-margin default.")
    p.add_argument("--fixed_start_y",type=float,default=None,
                   help="Fixed start y. None uses bottom-margin default.")
    p.add_argument("--fixed_start_left_margin",type=float,default=3.2,
                   help="Start x is -arena_width/2 + this margin when fixed_start_x is None.")
    p.add_argument("--fixed_start_bottom_margin",type=float,default=3.0,
                   help="Start y is -arena_height/2 + this margin when fixed_start_y is None.")
    p.add_argument("--goal_right_margin",type=float,default=3.2,
                   help="Right boundary margin for random goal sampling.")
    p.add_argument("--goal_right_band",type=float,default=10.0,
                   help="Goal x is sampled from the right-side band of this width.")
    p.add_argument("--goal_y_margin",type=float,default=3.0,
                   help="Top/bottom margin for random goal y sampling.")
    p.add_argument("--min_start_goal_distance",type=float,default=18.0,
                   help="Minimum distance between fixed start and sampled goal.")
    p.add_argument("--obstacle_gap",type=float,default=1.50,
                   help="Minimum surface gap between any two static obstacles. The effective value is max(obstacle_gap, UAV diameter * obstacle_gap_uav_diameter_multiplier).")
    p.add_argument("--obstacle_gap_uav_diameter_multiplier",type=float,default=2.35,
                   help="Minimum static-obstacle surface gap measured in UAV diameters. Effective gap=max(--obstacle_gap, 2*drone_radius*this value).")
    p.add_argument("--rect_ratio",type=float,default=0.55)
    p.add_argument("--circle_radius_min",type=float,default=0.25); p.add_argument("--circle_radius_max",type=float,default=0.55)
    p.add_argument("--rect_width_min",type=float,default=0.35); p.add_argument("--rect_width_max",type=float,default=0.75)
    p.add_argument("--rect_length_min",type=float,default=0.75); p.add_argument("--rect_length_max",type=float,default=1.55)
    # 3D visualization parameters. These do not change the 2D horizontal-plane
    # Diff-VFH training state; they make Open3D scenes look like UAV navigation
    # among building-like obstacles.
    p.add_argument("--static_obstacle_height_min",type=float,default=3.5,
                   help="Minimum rendered height of static obstacles/buildings in Open3D.")
    p.add_argument("--static_obstacle_height_max",type=float,default=8.0,
                   help="Maximum rendered height of static obstacles/buildings in Open3D.")
    p.add_argument("--dynamic_obstacle_height_min",type=float,default=1.2,
                   help="Minimum nominal rendered height of dynamic obstacles.")
    p.add_argument("--dynamic_obstacle_height_max",type=float,default=2.5,
                   help="Maximum nominal rendered height of dynamic obstacles.")
    p.add_argument("--uav_flight_altitude",type=float,default=2.2,
                   help="Rendered UAV flight altitude for Open3D start/goal/trajectory markers.")
    p.add_argument("--uav_path_tube_radius",type=float,default=0.035,
                   help="Rendered UAV trajectory tube radius in Open3D.")
    p.add_argument("--dynamic_path_tube_radius",type=float,default=0.025,
                   help="Rendered dynamic-obstacle trajectory tube radius in Open3D.")
    p.add_argument("--visual_ground_thickness",type=float,default=0.04,
                   help="Rendered ground thickness in Open3D.")
    p.add_argument("--visual_wall_height",type=float,default=0.45,
                   help="Rendered boundary wall height in Open3D.")


    # v12 whole-arena uniform static-obstacle sampler. No route corridor/gate
    # generation is used. Uniformity is controlled by grid oversampling and
    # at-most-one-obstacle-per-cell placement.
    p.add_argument("--uniform_grid_oversample",type=float,default=1.00)
    p.add_argument("--static_max_per_cell",type=int,default=1)
    p.add_argument("--static_cell_attempts",type=int,default=16)
    p.add_argument("--static_uniform_sweeps",type=int,default=3)
    p.add_argument("--allow_relaxed_uniform_fallback",action="store_true",default=True)
    p.add_argument("--no_relaxed_uniform_fallback",dest="allow_relaxed_uniform_fallback",action="store_false")
    p.add_argument("--static_capacity_factor",type=float,default=0.72)

    p.add_argument("--static_grid_cols",type=int,default=0)
    p.add_argument("--static_grid_rows",type=int,default=0)
    p.add_argument("--static_edge_margin",type=float,default=1.25)
    p.add_argument("--static_edge_band",type=float,default=2.5)
    p.add_argument("--static_edge_max_fraction",type=float,default=0.18)
    p.add_argument("--static_place_attempts",type=int,default=220)

    p.add_argument("--dynamic_radius_min",type=float,default=0.28); p.add_argument("--dynamic_radius_max",type=float,default=0.50)

    p.add_argument("--dynamic_motion_mode",type=str,default="random",
                   choices=["random","uniform_random","scene_random","whole_arena_random",
                            "route_random","near_route_random","random_near_route",
                            "forced_encounter","forced","must_encounter",
                            "center_vertical","vertical","vertical_crossing",
                            "crossing","mixed"],
                   help="Dynamic obstacle path mode. Default random samples dynamic obstacles uniformly across the whole arena with random velocity directions.")
    p.add_argument("--crossing_ratio",type=float,default=0.0); p.add_argument("--min_crossing_dynamic",type=int,default=0)
    # Route-neighborhood random dynamic obstacles. This is optional and can be
    # enabled by --dynamic_motion_mode route_random when you want more near-route encounters.
    p.add_argument("--route_random_s_min",type=float,default=0.18)
    p.add_argument("--route_random_s_max",type=float,default=0.90)
    p.add_argument("--route_random_s_jitter",type=float,default=0.035)
    p.add_argument("--route_random_lateral_band",type=float,default=4.8)
    p.add_argument("--route_random_along_jitter",type=float,default=1.0)
    p.add_argument("--route_random_avoid_same_direction",action="store_true",default=True)
    p.add_argument("--route_random_allow_same_direction",dest="route_random_avoid_same_direction",action="store_false")
    p.add_argument("--route_random_max_forward_dot",type=float,default=0.45,
                   help="When avoiding same-direction motion, reject random velocities with dot(v, start_to_goal_dir) above this value.")

    # Center-vertical dynamic obstacle zone.  Obstacles spawn above the middle
    # route segment and move downward, increasing real dynamic encounters.
    p.add_argument("--center_vertical_s_min",type=float,default=0.32)
    p.add_argument("--center_vertical_s_max",type=float,default=0.74)
    p.add_argument("--center_vertical_s_jitter",type=float,default=0.035)
    p.add_argument("--center_vertical_x_halfwidth",type=float,default=5.0)
    p.add_argument("--center_vertical_x_jitter",type=float,default=0.80)
    p.add_argument("--center_vertical_y_jitter",type=float,default=0.45)
    p.add_argument("--center_vertical_spawn_x_jitter",type=float,default=0.45)
    p.add_argument("--center_vertical_uav_speed_guess",type=float,default=5.0)
    p.add_argument("--center_vertical_spawn_distance_min",type=float,default=3.5)
    p.add_argument("--center_vertical_spawn_distance_max",type=float,default=9.5)
    p.add_argument("--center_vertical_t_min",type=float,default=1.4)
    p.add_argument("--center_vertical_t_max",type=float,default=12.0)
    p.add_argument("--center_vertical_velocity_x_jitter",type=float,default=0.08)
    p.add_argument("--center_vertical_downward_bias",type=float,default=0.95)
    p.add_argument("--center_vertical_avoid_lateral_scale",type=float,default=0.65)
    p.add_argument("--center_vertical_respawn_y_jitter",type=float,default=1.0)
    # Forced-encounter mode: recommended for training/verifying real dynamic avoidance.
    p.add_argument("--forced_encounter_s_min",type=float,default=0.22)
    p.add_argument("--forced_encounter_s_max",type=float,default=0.68)
    p.add_argument("--forced_encounter_s_jitter",type=float,default=0.025)
    p.add_argument("--forced_encounter_x_halfwidth",type=float,default=9.0)
    p.add_argument("--forced_encounter_x_jitter",type=float,default=0.25)
    p.add_argument("--forced_encounter_y_jitter",type=float,default=0.20)
    p.add_argument("--forced_encounter_spawn_x_jitter",type=float,default=0.20)
    p.add_argument("--forced_encounter_uav_speed_guess",type=float,default=5.0)
    p.add_argument("--forced_encounter_time_jitter",type=float,default=0.60)
    p.add_argument("--forced_encounter_t_min",type=float,default=2.0)
    p.add_argument("--forced_encounter_t_max",type=float,default=16.0)
    p.add_argument("--forced_encounter_spawn_distance_min",type=float,default=2.0)
    p.add_argument("--forced_encounter_spawn_distance_max",type=float,default=11.0)
    p.add_argument("--forced_encounter_velocity_x_jitter",type=float,default=0.04)
    p.add_argument("--forced_encounter_downward_bias",type=float,default=0.98)
    p.add_argument("--forced_encounter_avoid_lateral_scale",type=float,default=0.70)
    p.add_argument("--forced_encounter_lane_gain",type=float,default=0.65)
    p.add_argument("--forced_encounter_respawn_x_jitter",type=float,default=0.25)
    p.add_argument("--enforce_route_crossing",action="store_true",default=False,
                   help="Only applies to forced/vertical crossing modes. Disabled by default for random whole-arena dynamics.")
    p.add_argument("--no_enforce_route_crossing",dest="enforce_route_crossing",action="store_false")
    p.add_argument("--forced_dynamic_retries",type=int,default=6)
    p.add_argument("--allow_forced_dynamic_fallback",action="store_true",default=True,
                   help="When strict forced dynamic route-crossing generation fails, accept the best complete candidate scene if route_ratio is high enough.")
    p.add_argument("--no_allow_forced_dynamic_fallback",dest="allow_forced_dynamic_fallback",action="store_false",
                   help="Disable fallback and restore strict RuntimeError behavior.")
    p.add_argument("--forced_dynamic_fallback_min_ratio",type=float,default=0.50,
                   help="Minimum route_ratio accepted by forced-dynamic fallback. Default 0.50 accepts cases such as 1/2 route-crossing dynamic obstacles.")
    p.add_argument("--print_forced_dynamic_fallback",action="store_true",default=False,
                   help="Print a short warning when fallback dynamic scene is accepted. Disabled by default to keep tqdm clean.")
    p.add_argument("--forced_min_route_crossing_ratio",type=float,default=0.7)
    p.add_argument("--forced_route_crossing_radius",type=float,default=1.8)
    p.add_argument("--forced_route_check_steps",type=int,default=220)
    p.add_argument("--crossing_s_min",type=float,default=0.12); p.add_argument("--crossing_s_max",type=float,default=0.65)
    p.add_argument("--crossing_lateral_jitter",type=float,default=0.8); p.add_argument("--crossing_s_jitter",type=float,default=0.045); p.add_argument("--crossing_uav_speed_guess",type=float,default=5.0); p.add_argument("--crossing_spawn_distance_min",type=float,default=5.0); p.add_argument("--crossing_spawn_distance_max",type=float,default=11.0); p.add_argument("--crossing_t_min",type=float,default=1.2); p.add_argument("--crossing_t_max",type=float,default=12.0); p.add_argument("--crossing_along_ratio",type=float,default=0.18)
    p.add_argument("--crossing_velocity_jitter",type=float,default=0.08)
    p.add_argument("--dynamic_spawn_margin",type=float,default=1.0)
    p.add_argument("--dynamic_static_gap",type=float,default=0.45)
    p.add_argument("--dynamic_dynamic_gap",type=float,default=1.80)
    p.add_argument("--dynamic_place_attempts",type=int,default=360)

    p.add_argument("--drone_radius",type=float,default=0.32); p.add_argument("--goal_radius",type=float,default=1.15); p.add_argument("--safe_clearance",type=float,default=0.38)
    p.add_argument("--progress_success_steps",type=int,default=160,
                   help="Minimum collision-free rollout steps required for safe-progress success.")
    p.add_argument("--progress_success_ratio",type=float,default=0.60,
                   help="Required final goal-distance reduction ratio for safe-progress success.")
    p.add_argument("--tail_progress_success_ratio",type=float,default=0.04,
                   help="Safe-progress success requires this much progress in the final trajectory tail.")
    p.add_argument("--min_success_avg_speed",type=float,default=0.0,
                   help="Optional anti-hover speed floor for safe-progress success. Default 0 disables speed-based success gating; progress/stagnation/path-efficiency checks still prevent hover-as-success.")
    p.add_argument("--max_success_stagnation_ratio",type=float,default=0.35,
                   help="Maximum fraction of near-zero goal-progress steps for safe-progress success.")
    p.add_argument("--max_success_backward_ratio",type=float,default=0.30,
                   help="Maximum fraction of steps moving away from the goal for safe-progress success.")
    p.add_argument("--min_success_path_efficiency",type=float,default=0.18,
                   help="Minimum net-progress/path-length efficiency for safe-progress success.")
    p.add_argument("--preferred_clearance_s1",type=float,default=0.62); p.add_argument("--preferred_clearance_s2",type=float,default=0.54); p.add_argument("--preferred_clearance_s3",type=float,default=0.54); p.add_argument("--preferred_clearance_s4",type=float,default=0.50)
    p.add_argument("--sensor_range",type=float,default=18.0); p.add_argument("--reaction_time",type=float,default=0.70); p.add_argument("--max_dynamic_inflation",type=float,default=1.20); p.add_argument("--velocity_smoothing",type=float,default=0.12); p.add_argument("--n_sectors",type=int,default=72)
    p.add_argument("--target_static_avg_speed",type=float,default=5.5, help="Soft target executed speed for static scenes. It encourages fast flight but is gated by safety and is not used for promotion.")
    p.add_argument("--target_dynamic_avg_speed",type=float,default=5.5, help="Soft target executed speed for mixed static-dynamic scenes. It encourages fast flight but is gated by safety and is not used for promotion.")
    p.add_argument("--speed_strategy",type=str,default="gated",choices=["gated","aggressive_adaptive"],
                   help="gated keeps the original conservative speed gates; aggressive_adaptive starts each stage from the maximum UAV speed and learns when to brake from collision/near-miss risk.")
    p.add_argument("--uav_max_speed_s1",type=float,default=8.0, help="Stage-1 UAV speed cap used by aggressive_adaptive mode.")
    p.add_argument("--uav_max_speed_s2",type=float,default=8.0, help="Stage-2 UAV speed cap used by aggressive_adaptive mode.")
    p.add_argument("--uav_max_speed_s3",type=float,default=8.0, help="Stage-3 UAV speed cap used by aggressive_adaptive mode.")
    p.add_argument("--aggressive_static_clearance_margin",type=float,default=0.22, help="Extra static clearance margin that starts braking in aggressive_adaptive mode.")
    p.add_argument("--aggressive_dynamic_clearance_trigger",type=float,default=1.00, help="Dynamic-obstacle clearance threshold that starts braking in aggressive_adaptive mode.")
    p.add_argument("--aggressive_ttc_risk_trigger",type=float,default=0.60, help="TTC-risk threshold that starts braking in aggressive_adaptive mode.")
    p.add_argument("--aggressive_min_speed_scale_static",type=float,default=0.55, help="Minimum speed ratio under high static risk in aggressive_adaptive mode.")
    p.add_argument("--aggressive_min_speed_scale_dynamic",type=float,default=0.42, help="Minimum speed ratio under high dynamic risk in aggressive_adaptive mode.")
    p.add_argument("--aggressive_goal_slow_distance",type=float,default=4.0, help="Distance to goal where aggressive_adaptive mode starts slowing for terminal approach.")
    p.add_argument("--aggressive_goal_min_speed_scale",type=float,default=0.18, help="Minimum terminal approach speed ratio near the goal.")
    p.add_argument("--aggressive_speed_beta_scale",type=float,default=1.0, help="Multiplier applied to learnable speed_beta for aggressive_adaptive risk gates.")
    p.add_argument("--w_overspeed_danger_loss",type=float,default=1.0, help="Penalty weight for moving fast in dangerous clearance states; active mainly in aggressive_adaptive mode.")
    p.add_argument("--speed_target_clearance_margin",type=float,default=0.20, help="Extra clearance margin used to gate the speed target loss.")
    p.add_argument("--speed_target_gate_beta",type=float,default=5.0, help="Gate sharpness for speed-target loss.")
    p.add_argument("--speed_target_goal_distance",type=float,default=6.0, help="Disable speed-target pressure near the goal within this distance.")
    p.add_argument("--speed_target_min_gate_static",type=float,default=0.35, help="Minimum local speed-loss gate in static scenes.")
    p.add_argument("--speed_target_min_gate_dynamic",type=float,default=0.22, help="Minimum local speed-loss gate in dynamic scenes.")
    p.add_argument("--dynamic_speed_loss_clearance",type=float,default=0.70, help="Dynamic clearance threshold used by speed-target loss gate.")
    p.add_argument("--dynamic_speed_loss_ttc_cost",type=float,default=0.62, help="Selected TTC-cost threshold used by speed-target loss gate.")
    p.add_argument("--static_speed_floor_ratio",type=float,default=0.75, help="Safety-gated minimum command speed as a ratio of target_static_avg_speed. Lower default keeps speed adaptive to dense scenes.")
    p.add_argument("--dynamic_speed_floor_ratio",type=float,default=0.65, help="Safety-gated minimum command speed as a ratio of target_dynamic_avg_speed. Lower default keeps speed adaptive to dynamic scenes.")
    p.add_argument("--speed_floor_gain",type=float,default=0.6, help="Strength of the differentiable cruise-speed floor. Lower default encourages speed without forcing it.")
    p.add_argument("--speed_floor_clearance_margin",type=float,default=0.20, help="Extra static clearance margin for the cruise-speed floor gate.")
    p.add_argument("--dynamic_speed_floor_clearance",type=float,default=0.75, help="Dynamic clearance threshold used by the cruise-speed floor gate.")
    p.add_argument("--dynamic_speed_floor_ttc_cost",type=float,default=0.62, help="Selected TTC-cost threshold used by the cruise-speed floor gate.")
    p.add_argument("--speed_floor_goal_distance",type=float,default=6.0, help="Disable the cruise-speed floor near the goal within this distance.")
    p.add_argument("--speed_floor_gate_beta",type=float,default=5.0, help="Gate sharpness for the cruise-speed floor.")
    p.add_argument("--speed_floor_softplus_beta",type=float,default=6.0, help="Softplus beta used by the differentiable speed-floor deficit.")

    # Boundary-aware VFH.
    # v5 keeps these flags only for backward compatibility.  Virtual wall points
    # are no longer inserted into the VFH histogram because they inflate avg_occ.
    p.add_argument("--use_virtual_walls",action="store_true",default=False)
    p.add_argument("--no_virtual_walls",dest="use_virtual_walls",action="store_false")
    p.add_argument("--wall_point_spacing",type=float,default=0.85)
    p.add_argument("--wall_virtual_margin",type=float,default=0.18)
    p.add_argument("--wall_virtual_radius",type=float,default=0.34)
    p.add_argument("--wall_influence_distance",type=float,default=2.4)
    p.add_argument("--wall_beta",type=float,default=2.5)
    p.add_argument("--w_wall_direction",type=float,default=5.0)
    p.add_argument("--w_goal_recovery",type=float,default=2.0)
    p.add_argument("--dynamic_recovery_gain",type=float,default=3.4)

    p.add_argument("--ttc_horizon",type=float,default=3.4); p.add_argument("--ttc_clearance",type=float,default=0.95); p.add_argument("--ttc_beta",type=float,default=10.0); p.add_argument("--ttc_decay",type=float,default=1.6)
    p.add_argument("--dynamic_encounter_distance",type=float,default=4.2); p.add_argument("--dynamic_clearance_target",type=float,default=0.62); p.add_argument("--dynamic_pred_clearance_target",type=float,default=0.95); p.add_argument("--dynamic_safe_clearance_threshold",type=float,default=0.10); p.add_argument("--dynamic_beta",type=float,default=8.0)
    p.add_argument("--dynamic_brake_clearance",type=float,default=1.00)
    p.add_argument("--dynamic_brake_beta",type=float,default=4.0)
    p.add_argument("--dynamic_min_speed_scale",type=float,default=0.55)
    p.add_argument("--dynamic_dir_clearance_weight",type=float,default=0.45)

    # Wait action is disabled by default for comparability. It can be enabled in
    # dynamic-heavy ablation runs with --enable_wait_action.
    p.add_argument("--enable_wait_action",action="store_true",default=False)
    p.add_argument("--disable_wait_action",dest="enable_wait_action",action="store_false")
    p.add_argument("--dynamic_wait_clearance",type=float,default=1.05)
    p.add_argument("--dynamic_wait_ttc_cost",type=float,default=0.30)
    p.add_argument("--dynamic_wait_static_density",type=float,default=0.18)
    p.add_argument("--dynamic_wait_beta",type=float,default=6.0)
    p.add_argument("--dynamic_wait_max_gate",type=float,default=0.98)
    p.add_argument("--dynamic_wait_speed_scale",type=float,default=0.02)
    p.add_argument("--dynamic_wait_goal_disable_distance",type=float,default=2.2)
    p.add_argument("--wait_count_threshold",type=float,default=0.50)
    p.add_argument("--wait_stuck_suppress_gate",type=float,default=0.35)

    p.add_argument("--min_step_progress",type=float,default=0.008); p.add_argument("--stagnation_progress_threshold",type=float,default=0.018); p.add_argument("--oscillation_cos_threshold",type=float,default=0.15)
    p.add_argument("--heading_inertia_s1",type=float,default=0.42); p.add_argument("--heading_inertia_s2",type=float,default=0.24); p.add_argument("--heading_inertia_s3",type=float,default=0.24); p.add_argument("--heading_inertia_s4",type=float,default=0.20)
    p.add_argument("--stuck_window",type=int,default=14); p.add_argument("--stuck_avg_progress",type=float,default=0.006); p.add_argument("--stuck_max_displacement",type=float,default=0.85); p.add_argument("--stuck_goal_factor",type=float,default=2.2)
    p.add_argument("--escape_blend",type=float,default=0.78); p.add_argument("--escape_goal_blend",type=float,default=0.35); p.add_argument("--escape_clearance_scale",type=float,default=0.72); p.add_argument("--escape_speed_boost",type=float,default=0.20); p.add_argument("--escape_min_speed",type=float,default=0.85)

    p.add_argument("--safety_beta",type=float,default=8.0); p.add_argument("--comfort_beta",type=float,default=5.0)
    p.add_argument("--w_goal_loss",type=float,default=7.5); p.add_argument("--w_safety_loss",type=float,default=90.0); p.add_argument("--w_comfort_loss",type=float,default=7.0); p.add_argument("--w_collision_loss",type=float,default=160.0)
    p.add_argument("--w_smooth_loss",type=float,default=0.25); p.add_argument("--w_progress_loss",type=float,default=7.0); p.add_argument("--w_stagnation_loss",type=float,default=34.0); p.add_argument("--w_oscillation_loss",type=float,default=3.0)
    p.add_argument("--w_ttc_loss",type=float,default=28.0); p.add_argument("--w_dynamic_clearance_loss",type=float,default=60.0)
    p.add_argument("--w_dynamic_collision_loss",type=float,default=260.0)
    p.add_argument("--w_dynamic_pred_loss",type=float,default=40.0)
    p.add_argument("--w_wait_loss",type=float,default=0.06)
    p.add_argument("--w_wall_loss",type=float,default=2.0); p.add_argument("--w_time_loss",type=float,default=1.20)
    p.add_argument("--w_speed_target_loss",type=float,default=8.0, help="Weight of gated speed target loss for high-speed training.")

    p.add_argument("--tensorboard",action="store_true",default=True); p.add_argument("--log_dir",type=str,default="runs"); p.add_argument("--out_dir",type=str,default="checkpoints"); p.add_argument("--run_name",type=str,default=""); p.add_argument("--save_every",type=int,default=300)
    p.add_argument("--log_every",type=int,default=5, help="Write TensorBoard train/stage scalars every N episodes instead of every episode.")
    p.add_argument("--compact_terminal",action="store_true",default=False,
                   help="Disable tqdm and print compact criteria lines every --terminal_log_every episodes.")
    p.add_argument("--verbose_tqdm",dest="compact_terminal",action="store_false",
                   help="Use tqdm progress bar output. This is the default.")
    p.add_argument("--terminal_log_every",type=int,default=50,
                   help="Only used with --compact_terminal: print one criteria line every N episodes.")
    p.add_argument("--print_stage_events",action="store_true",default=False,
                   help="Print stage/level transition messages. Disabled by default to keep tqdm clean.")
    p.add_argument("--scene_metric_every",type=int,default=25, help="Write expensive scene distribution metrics every N TensorBoard steps.")
    p.add_argument("--param_log_every",type=int,default=10, help="Write model readable parameters every N TensorBoard steps.")
    p.add_argument("--best_save_interval",type=int,default=100, help="Save best checkpoint at most every N episodes.")
    p.add_argument("--best_min_delta",type=float,default=0.003, help="Minimum stage-score improvement required before best checkpoint saving is considered.")
    p.add_argument("--vis_o3d",action="store_true", help="Show Open3D visualizations.")
    p.add_argument("--vis_each_level",action="store_true",default=True,
                   help="Show an Open3D window whenever a sub-level is passed.")
    p.add_argument("--no_vis_each_level",dest="vis_each_level",action="store_false")
    p.add_argument("--vis_every",type=int,default=300, help="Deprecated in stagewise v12: kept for compatibility.")
    p.add_argument("--vis_first_episode",action="store_true", help="Deprecated in stagewise v12: kept for compatibility.")
    p.add_argument("--vis_episodes",type=int,default=3, help="Number of trajectories shown in --mode visualize.")
    p.add_argument("--vis_static_count",type=int,default=-1, help="Override static obstacle number in visualize mode; -1 uses stage default.")
    p.add_argument("--vis_dynamic_count",type=int,default=-1, help="Override dynamic obstacle number in visualize mode; -1 uses stage default.")
    p.add_argument("--vis_dynamic_speed",type=float,default=-1.0, help=argparse.SUPPRESS)
    return p

def visualize_checkpoint(cfg):
    if not cfg.init_checkpoint:
        raise ValueError("--mode visualize requires --init_checkpoint <path_to_pth>")
    set_seed(cfg.seed, deterministic=not cfg.nondeterministic)
    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # stage must be a concrete stage for visualization.
    stage_id = 3 if cfg.stage == "all" else (3 if int(cfg.stage) == 4 else int(cfg.stage))
    model = DiffVFH(cfg.n_sectors, str(device)).to(device)
    load_model_checkpoint(model, optimizer=None, checkpoint_path=cfg.init_checkpoint, device=device, load_optimizer=False)
    model.eval()

    cur = StageLevelCurriculum(cfg, stage_id)
    # By default visualize the final difficulty of the selected stage.
    if stage_id == 1:
        cur.static_count = cfg.stage1_static_end
    elif stage_id == 2:
        cur.dynamic_count = cfg.stage3_dynamic_end
        cur.static_count = cur._static_for_dynamic(cur.dynamic_count)
        cur.dynamic_speed = cfg.stage3_dynamic_speed
    elif stage_id == 3:
        cur.dynamic_count = cfg.stage3_dynamic_end
        cur.static_count = cur._static_for_dynamic(cur.dynamic_count)
        cur.dynamic_speed = cfg.stage4_speed_end

    if cfg.vis_static_count >= 0:
        cur.static_count = int(cfg.vis_static_count)
    if cfg.vis_dynamic_count >= 0:
        cur.dynamic_count = int(cfg.vis_dynamic_count)
    if cfg.vis_dynamic_speed >= 0:
        cur.dynamic_speed = float(cfg.vis_dynamic_speed)

    print(f"Visualizing checkpoint: {cfg.init_checkpoint}")
    print(f"Stage={stage_id}, static={cur.static_count}, dynamic={cur.dynamic_count}")

    for i in range(1, cfg.vis_episodes + 1):
        scene = generate_scene(cfg, cfg.seed_stage_offset * stage_id + 10000 + i, cur)
        with torch.no_grad():
            metrics = rollout(model, scene, cfg, device)
        print(
            f"[vis {i}/{cfg.vis_episodes}] success={metrics['success']:.0f}, collision={metrics['collision']:.0f}, "
            f"dyn_collision={metrics['dynamic_collision']:.0f}, reached={metrics['reached']:.0f}, "
            f"steps={metrics['steps']}, min_dyn_clearance={metrics['min_dynamic_clearance']:.3f}, "
            f"min_ttc={metrics['min_ttc']:.3f}, avg_speed={metrics['avg_speed']:.3f}, "
            f"wait_rate={metrics.get('wait_rate',0.0):.2f}"
        )
        show_o3d(scene, cfg, metrics["trajectory"], f"Diff-VFH-v12 checkpoint visualization {i}: Stage {stage_id}")

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.dynamic_level_window <= 0:
        args.dynamic_level_window = args.level_window
    if args.static_level_window <= 0:
        args.static_level_window = args.level_window
    if args.static_min_level_episodes <= 0:
        args.static_min_level_episodes = args.min_level_episodes
    if args.stage2_level_window <= 0:
        args.stage2_level_window = args.dynamic_level_window
    if args.stage2_min_level_episodes <= 0:
        args.stage2_min_level_episodes = args.dynamic_min_level_episodes
    if args.stage3_level_window <= 0:
        args.stage3_level_window = args.dynamic_level_window
    if args.stage3_min_level_episodes <= 0:
        args.stage3_min_level_episodes = args.dynamic_min_level_episodes
    if args.dynamic_static_end > args.dynamic_static_start:
        raise ValueError("--dynamic_static_end should be <= --dynamic_static_start so static obstacles decrease in dynamic stages.")
    if args.mode == "visualize":
        visualize_checkpoint(args)
    else:
        train(args)
"""*"""