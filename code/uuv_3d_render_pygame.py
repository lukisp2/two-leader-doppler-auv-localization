# -*- coding: utf-8 -*-
"""uuv_3d_render_pygame.py

Pygame renderer for the 3D (x,y,z) two-leader UUV formation/PF environment.

Why this exists
---------------
The main training/eval script should stay focused on the environment + SB3.
This module provides a *usable* human render:

  • 4 panels:
      - 3D isometric view (with depth)
      - XY top-down view
      - XZ depth profile
      - std_norm history chart
  • Velocity vectors (leaders + follower) in all panels.
  • Particle cloud (subsampled) in all panels.
  • PF covariance principal axes in 3D view.

Dependencies: pygame + numpy.

Usage (from env.render):

    from uuv_3d_render_pygame import UUV3DRenderer
    if self._renderer3d is None:
        self._renderer3d = UUV3DRenderer(self.cfg)
    self._renderer3d.draw(env=self, screen=self._screen, font=self._font)

The renderer does *not* consume the pygame event queue (so the env can keep
handling M/ESC/BACKSPACE etc.). It only reads key *state* for camera control.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None


# ------------------------
# small utilities
# ------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(v) + eps)


def _arrow(
    screen,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 2,
    head_len: int = 10,
    head_ang_deg: float = 25.0,
) -> None:
    """Draw an arrow from p0 to p1."""
    if pygame is None:
        return

    x0, y0 = p0
    x1, y1 = p1
    pygame.draw.line(screen, color, (x0, y0), (x1, y1), width)

    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux = dx / L
    uy = dy / L

    ang = math.radians(head_ang_deg)
    ca = math.cos(ang)
    sa = math.sin(ang)

    # rotate (-u) by +/- ang
    bx, by = -ux, -uy
    rx1 = bx * ca - by * sa
    ry1 = bx * sa + by * ca
    rx2 = bx * ca + by * sa
    ry2 = -bx * sa + by * ca

    pA = (int(x1 + rx1 * head_len), int(y1 + ry1 * head_len))
    pB = (int(x1 + rx2 * head_len), int(y1 + ry2 * head_len))
    pygame.draw.polygon(screen, color, [p1, pA, pB])


def _depth_shade(base: Tuple[int, int, int], z: float, z_min: float, z_max: float) -> Tuple[int, int, int]:
    """Darken color with depth (larger z => darker)."""
    if not np.isfinite(z) or z_max <= z_min + 1e-9:
        return base
    t = (float(z) - float(z_min)) / (float(z_max) - float(z_min))
    t = _clamp(t, 0.0, 1.0)
    # shallow: 1.0, deep: ~0.45
    f = 1.0 - 0.55 * t
    return (int(base[0] * f), int(base[1] * f), int(base[2] * f))


def _face_shade(base: Tuple[int, int, int], normal: np.ndarray) -> Tuple[int, int, int]:
    n = np.asarray(normal, dtype=float)
    nrm = np.linalg.norm(n)
    if nrm <= 1e-12:
        return base
    n = n / nrm
    light = np.array([0.42, -0.22, 0.88], dtype=float)
    light /= max(float(np.linalg.norm(light)), 1e-12)
    i = float(np.dot(n, light))
    g = 0.28 + 0.72 * _clamp(0.5 * (i + 1.0), 0.0, 1.0)
    return (int(base[0] * g), int(base[1] * g), int(base[2] * g))


@dataclass
class Panel:
    x: int
    y: int
    w: int
    h: int
    title: str


class FitView2D:
    """Maps 2D world coordinates to a panel rectangle."""

    def __init__(self, panel: Panel, *, flip_y: bool = True):
        self.panel = panel
        self.flip_y = bool(flip_y)
        self.center = np.zeros(2, dtype=float)
        self.scale = 1.0
        self._initialized = False

    def fit(
        self,
        pts2: np.ndarray,
        margin_px: int = 18,
        center_override: Optional[np.ndarray] = None,
        *,
        lock_scale: bool = True,
    ) -> None:
        pts2 = np.asarray(pts2, dtype=float)
        if pts2.size == 0:
            return
        if pts2.ndim != 2 or pts2.shape[1] != 2:
            pts2 = pts2.reshape(-1, 2)

        target_center = (
            np.asarray(center_override, dtype=float).reshape(2)
            if center_override is not None
            else np.mean(pts2, axis=0)
        )

        d = np.linalg.norm(pts2 - target_center[None, :], axis=1)
        if d.size > 0:
            dmax = float(np.quantile(d, 0.98))
        else:
            dmax = 1.0
        dmax = max(dmax, 1.0)

        avail_w = max(40, self.panel.w - 2 * margin_px)
        avail_h = max(40, self.panel.h - 2 * margin_px)
        target_px = 0.48 * float(min(avail_w, avail_h))
        target_scale = float(target_px / dmax)

        if not self._initialized:
            self.center = target_center
            self.scale = target_scale
            self._initialized = True
            return

        self.center = target_center
        if not lock_scale:
            self.scale = target_scale

    def to_screen(self, p2: Sequence[float]) -> Tuple[int, int]:
        px, py = float(p2[0]), float(p2[1])
        dx = px - float(self.center[0])
        dy = py - float(self.center[1])
        sx = self.panel.x + self.panel.w / 2 + dx * self.scale
        sy = self.panel.y + self.panel.h / 2 + (-dy if self.flip_y else dy) * self.scale
        return int(sx), int(sy)


class IsoView:
    """Simple isometric-ish 3D view using yaw+pitch rotation and orthographic projection."""

    def __init__(self, panel: Panel, *, yaw_deg: float = 45.0, pitch_deg: float = 35.0):
        self.panel = panel
        self.yaw = float(yaw_deg)
        self.pitch = float(pitch_deg)
        self.center = np.zeros(3, dtype=float)
        self.scale = 1.0
        self.base_scale = 1.0
        self._initialized = False

    def _R(self) -> np.ndarray:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)

        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=float)
        return Rx @ Rz

    def project_cam(self, p3: np.ndarray) -> np.ndarray:
        p3 = np.asarray(p3, dtype=float)
        R = self._R()
        return (R @ (p3 - self.center)).reshape(3)

    def fit(
        self,
        pts3: np.ndarray,
        margin_px: int = 18,
        center_override: Optional[np.ndarray] = None,
        *,
        lock_scale: bool = True,
    ) -> None:
        pts3 = np.asarray(pts3, dtype=float)
        if pts3.size == 0:
            return
        if pts3.ndim != 2 or pts3.shape[1] != 3:
            pts3 = pts3.reshape(-1, 3)

        target_center = np.asarray(center_override, dtype=float).reshape(3) if center_override is not None else np.mean(pts3, axis=0)

        if not self._initialized:
            self.center = target_center
            # rotate points to camera frame for proper fit
            R = self._R()
            pts_cam = (R @ (pts3 - self.center).T).T
            # use x,y of cam frame
            d = np.linalg.norm(pts_cam[:, :2] - np.mean(pts_cam[:, :2], axis=0)[None, :], axis=1)
            dmax = float(np.quantile(d, 0.98) if d.size > 0 else 1.0)
            avail_w = max(40, self.panel.w - 2 * margin_px)
            avail_h = max(40, self.panel.h - 2 * margin_px)
            target_px = 0.48 * float(min(avail_w, avail_h))
            self.base_scale = float(target_px / max(dmax, 1.0))
            self.scale = self.base_scale
            self._initialized = True
            return

        self.center = target_center

        # rotate points to camera frame for proper fit
        R = self._R()
        pts_cam = (R @ (pts3 - self.center).T).T
        # use x,y of cam frame
        c2 = np.mean(pts_cam[:, :2], axis=0)
        d = np.linalg.norm(pts_cam[:, :2] - c2[None, :], axis=1)
        dmax = float(np.quantile(d, 0.98) if d.size > 0 else 1.0)
        dmax = max(dmax, 1.0)

        avail_w = max(40, self.panel.w - 2 * margin_px)
        avail_h = max(40, self.panel.h - 2 * margin_px)
        target_px = 0.48 * float(min(avail_w, avail_h))
        target_scale = float(target_px / dmax)
        if not lock_scale:
            self.base_scale = target_scale
            self.scale = self.base_scale

    def to_screen(self, p3: Sequence[float]) -> Tuple[int, int]:
        cam = self.project_cam(np.asarray(p3, dtype=float))
        sx = self.panel.x + self.panel.w / 2 + cam[0] * self.scale
        sy = self.panel.y + self.panel.h / 2 - cam[1] * self.scale
        return int(sx), int(sy)

    def cam_depth(self, p3: Sequence[float]) -> float:
        """Depth used for painter sorting (larger => further away)."""
        cam = self.project_cam(np.asarray(p3, dtype=float))
        # camera z is 'into screen'
        return float(cam[2])


class UUV3DRenderer:
    """Pygame renderer for the 3D PF env."""

    def __init__(self, cfg) -> None:
        if pygame is None:
            raise RuntimeError("pygame is not installed. Install: pip install pygame")
        self.cfg = cfg

        self.iso_yaw_deg = 45.0
        self.iso_pitch_deg = 35.0
        self.iso_zoom = 1.0

        self._std_hist_t = deque(maxlen=1200)
        self._std_hist_val = deque(maxlen=1200)
        self._std_hist_err = deque(maxlen=1200)
        self._std_plot_margin = 12

        self._last_panel_layout: Optional[Tuple[int, int]] = None
        self._pan_iso: Optional[Panel] = None
        self._pan_xy: Optional[Panel] = None
        self._pan_xz: Optional[Panel] = None
        self._pan_std: Optional[Panel] = None
        self._iso_view: Optional[IsoView] = None
        self._xy_view: Optional[FitView2D] = None
        self._xz_view: Optional[FitView2D] = None
        self._last_draw_t: float = 0.0
        self._state_smooth: Dict[str, np.ndarray] = {}
        self._smooth_tau_s: float = 0.08
        self._smooth_dt_cap_s: float = 0.20

    @staticmethod
    def _nice_step(xmin: float, xmax: float, n_steps: int = 4) -> float:
        if not (np.isfinite(xmin) and np.isfinite(xmax)):
            return 1.0
        span = xmax - xmin
        if span <= 0.0:
            return 1.0
        raw = span / max(1, n_steps)
        p = 10.0 ** math.floor(math.log10(max(raw, 1e-12)))
        for s in (1.0, 2.0, 5.0, 10.0):
            if s * p >= raw:
                return s * p
        return 10.0 * p

    @staticmethod
    def _yaw_pitch_rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        yaw = math.radians(float(yaw_deg))
        pitch = math.radians(float(pitch_deg))
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)

        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        Ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], dtype=float)
        return Rz @ Ry

    @staticmethod
    def _vehicle_box(
        center: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        *,
        length: float = 12.0,
        width: float = 4.0,
        height: float = 2.8,
    ) -> np.ndarray:
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        L, W, H = float(length), float(width), float(height)
        hx, hy, hz = 0.5 * L, 0.5 * W, 0.5 * H

        local = np.array([
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, +hy, -hz],
            [-hx, +hy, -hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ], dtype=float)

        R = UUV3DRenderer._yaw_pitch_rot(yaw_deg, pitch_deg)
        return (R @ local.T).T + np.array([cx, cy, cz], dtype=float)

    def _layout(self, W: int, H: int) -> Tuple[Panel, Panel, Panel, Panel]:
        key = (int(W), int(H))
        if (
            self._last_panel_layout == key
            and self._pan_iso is not None
            and self._pan_xy is not None
            and self._pan_xz is not None
            and self._pan_std is not None
        ):
            return self._pan_iso, self._pan_xy, self._pan_xz, self._pan_std

        graph_w = max(300, int(round(0.25 * W)))
        main_w = max(1, W - graph_w)

        w_iso = int(round(0.66 * main_w))
        w_rhs = max(1, main_w - w_iso)
        h_top = H // 2
        h_bot = H - h_top

        pan_iso = Panel(0, 0, w_iso, H, "3D (isometric)")
        pan_xy = Panel(w_iso, 0, w_rhs, h_top, "XY (top-down)")
        pan_xz = Panel(w_iso, h_top, w_rhs, h_bot, "XZ (depth)")
        pan_std = Panel(main_w, 0, graph_w, H, "PF std_norm history")

        self._last_panel_layout = key
        self._pan_iso, self._pan_xy, self._pan_xz, self._pan_std = pan_iso, pan_xy, pan_xz, pan_std
        return pan_iso, pan_xy, pan_xz, pan_std

    @staticmethod
    def _draw_panel_bg(screen, panel: Panel, *, bg=(10, 10, 25), border=(90, 90, 140)) -> None:
        pygame.draw.rect(screen, bg, (panel.x, panel.y, panel.w, panel.h))
        pygame.draw.rect(screen, border, (panel.x, panel.y, panel.w, panel.h), 1)

    @staticmethod
    def _text(font, text: str, color=(235, 235, 235)):
        return font.render(text, True, color)

    def _camera_keys(self) -> None:
        """Read key state and adjust iso camera (does not consume event queue)."""
        keys = pygame.key.get_pressed()
        # camera yaw
        if keys[pygame.K_a]:
            self.iso_yaw_deg -= 1.5
        if keys[pygame.K_d]:
            self.iso_yaw_deg += 1.5
        # camera pitch
        if keys[pygame.K_r]:
            self.iso_pitch_deg = _clamp(self.iso_pitch_deg + 1.2, -80.0, 80.0)
        if keys[pygame.K_f]:
            self.iso_pitch_deg = _clamp(self.iso_pitch_deg - 1.2, -80.0, 80.0)
        # zoom
        if keys[pygame.K_z]:
            self.iso_zoom = float(_clamp(self.iso_zoom * 1.02, 0.35, 4.0))
        if keys[pygame.K_x]:
            self.iso_zoom = float(_clamp(self.iso_zoom / 1.02, 0.35, 4.0))

    def _blend_frame(self, dt: float, alpha_max: float = 1.0) -> float:
        if dt <= 0.0:
            return alpha_max
        return float(_clamp(1.0 - math.exp(-dt / max(self._smooth_tau_s, 1e-6)), 0.0, alpha_max))

    def _smooth(self, key: str, target: np.ndarray, dt: float) -> np.ndarray:
        tgt = np.asarray(target, dtype=float)
        prev = self._state_smooth.get(key, None)
        if prev is None or prev.shape != tgt.shape:
            self._state_smooth[key] = tgt.copy()
            return tgt

        if not np.all(np.isfinite(prev)):
            self._state_smooth[key] = tgt.copy()
            return tgt

        a = self._blend_frame(dt, alpha_max=1.0)
        if a >= 0.9999:
            sm = tgt
        else:
            sm = (1.0 - a) * prev + a * tgt
            self._state_smooth[key] = sm
        return sm

    def reset_state(self) -> None:
        """Clear per-episode view/smoothing cache so new episodes start cleanly."""
        self._std_hist_t.clear()
        self._std_hist_val.clear()
        self._std_hist_err.clear()
        self._last_panel_layout = None
        self._pan_iso = None
        self._pan_xy = None
        self._pan_xz = None
        self._pan_std = None
        self._iso_view = None
        self._xy_view = None
        self._xz_view = None
        self._last_draw_t = 0.0
        self._state_smooth = {}

    def close(self) -> None:
        self.reset_state()

    def _append_std_history(self, t: float, std_norm: float, err_norm: float) -> None:
        if np.isfinite(t) and np.isfinite(std_norm) and np.isfinite(err_norm):
            self._std_hist_t.append(float(t))
            self._std_hist_val.append(float(std_norm))
            self._std_hist_err.append(float(err_norm))

    def _draw_std_history(self, *, screen, panel: Panel, font) -> None:
        pygame.draw.rect(screen, (8, 8, 22), (panel.x, panel.y, panel.w, panel.h))
        pygame.draw.rect(screen, (90, 90, 140), (panel.x, panel.y, panel.w, panel.h), 1)
        screen.blit(self._text(font, panel.title), (panel.x + 8, panel.y + 6))

        if len(self._std_hist_t) < 2:
            screen.blit(self._text(font, "waiting for std_norm..."), (panel.x + 8, panel.y + 32))
            return

        t = np.asarray(self._std_hist_t, dtype=float)
        v = np.asarray(self._std_hist_val, dtype=float)
        if t.size == 0 or v.size == 0:
            return

        t0 = float(np.min(t))
        t1 = float(np.max(t))
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            t0 = float(t[0])
            t1 = t0 + 1.0
        ts = t1 - t0

        vmin = float(np.min(v[np.isfinite(v)]))
        vmax = float(np.max(v[np.isfinite(v)]))
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return
        if not (vmax > vmin):
            vmax = vmin + 1.0
        margin = max(0.05 * (vmax - vmin), 0.1)
        vmin -= margin
        vmax += margin

        x = panel.x + self._std_plot_margin
        y = panel.y + 28 + self._std_plot_margin
        w = panel.w - 2 * self._std_plot_margin
        h = panel.h - 46 - 2 * self._std_plot_margin
        if w <= 0 or h <= 0:
            return

        inner = (x, y, w, h)
        x0, y0, ww, hh = inner
        pygame.draw.rect(screen, (14, 14, 30), inner)
        pygame.draw.rect(screen, (110, 110, 150), inner, 1)

        def to_xy(tt: float, vv: float) -> Tuple[int, int]:
            xx = int(x0 + ((tt - t0) / ts) * ww)
            yy = int(y0 + hh - ((vv - vmin) / (vmax - vmin)) * hh)
            return xx, yy

        # horizontal grid + labels
        step_y = self._nice_step(vmin, vmax)
        y0v = math.floor(vmin / step_y) * step_y
        for i in range(-2, 10):
            val = y0v + i * step_y
            if not (vmin <= val <= vmax):
                continue
            yy = int(y0 + hh - ((val - vmin) / (vmax - vmin)) * hh)
            pygame.draw.line(screen, (54, 54, 95), (x0, yy), (x0 + ww, yy), 1)
            screen.blit(self._text(font, f"{val:5.2f}", color=(190, 190, 220)), (panel.x + 4, yy - 8))

        # x grid + labels
        step_x = self._nice_step(t0, t1, n_steps=5)
        x0v = math.floor(t0 / step_x) * step_x
        for i in range(-5, 10):
            val = x0v + i * step_x
            xx = int(x0 + ((val - t0) / ts) * ww)
            if x0 <= xx <= x0 + ww:
                pygame.draw.line(screen, (56, 56, 96), (xx, y0), (xx, y0 + hh), 1)
                screen.blit(self._text(font, f"{val:5.1f}s", color=(190, 190, 220)), (xx - 20, y0 + hh + 2))

        # data polyline
        pts = []
        for ti, vi in zip(t, v):
            if np.isfinite(ti) and np.isfinite(vi):
                pts.append(to_xy(float(ti), float(vi)))
        if len(pts) > 1:
            pygame.draw.lines(screen, (255, 180, 90), False, pts, 2)

        screen.blit(self._text(font, f"now={v[-1]:.2f} m", color=(230, 240, 240)), (panel.x + 8, panel.y + panel.h - 22))

    def draw(self, *, env, screen, font) -> None:
        """Draw one frame."""
        if pygame is None:
            return

        W, H = int(self.cfg.screen_w), int(self.cfg.screen_h)
        pan_iso, pan_xy, pan_xz, pan_std = self._layout(W, H)
        self._camera_keys()
        now = float(pygame.time.get_ticks()) * 0.001
        target_dt = 1.0 / max(1, int(getattr(self.cfg, "render_fps", 30)))
        dt = now - self._last_draw_t
        if self._last_draw_t <= 0.0:
            dt = target_dt
        elif dt < 0.0:
            dt = target_dt
        else:
            dt = float(np.clip(dt, 0.0, self._smooth_dt_cap_s))
            if dt < 1e-4:
                dt = target_dt
        self._last_draw_t = now

        if (
            self._iso_view is None
            or self._xy_view is None
            or self._xz_view is None
            or self._iso_view.panel is not pan_iso
            or self._xy_view.panel is not pan_xy
            or self._xz_view.panel is not pan_xz
            or self._pan_std is not pan_std
        ):
            self._iso_view = IsoView(pan_iso, yaw_deg=self.iso_yaw_deg, pitch_deg=self.iso_pitch_deg)
            self._xy_view = FitView2D(pan_xy, flip_y=True)
            self._xz_view = FitView2D(pan_xz, flip_y=False)


        # --- pull state from env ---
        pL1 = np.asarray(env.pL1, dtype=float)
        pL2 = np.asarray(env.pL2, dtype=float)
        pF = np.asarray(env.pF, dtype=float)
        vL1 = np.asarray(env.vL1, dtype=float)
        vL2 = np.asarray(env.vL2, dtype=float)
        vF = np.asarray(env.vF, dtype=float)
        pF_cam = self._smooth("pF_cam", pF, dt)

        pF_hat = self._smooth("pF_hat", np.asarray(env.pf.mean, dtype=float), dt)
        cov = np.asarray(env.pf.cov, dtype=float)
        pC, pF_des = env._formation_desired()
        pC = np.asarray(pC, dtype=float)
        pF_des = np.asarray(pF_des, dtype=float)

        # Particle arrays with compatibility fallback across PF class variants.
        pf_obj = env.pf
        parts_src = getattr(pf_obj, "particles", getattr(pf_obj, "p", None))
        w_src = getattr(pf_obj, "weights", getattr(pf_obj, "w", None))
        if parts_src is None or w_src is None:
            parts = np.zeros((0, 3), dtype=float)
            w = np.zeros((0,), dtype=float)
        else:
            parts = np.asarray(parts_src, dtype=float)
            if parts.ndim != 2 or parts.shape[1] != 3:
                parts = np.zeros((0, 3), dtype=float)

            w = np.asarray(w_src, dtype=float).reshape(-1)
            if w.size != parts.shape[0]:
                if parts.shape[0] > 0:
                    w = np.ones(parts.shape[0], dtype=float) / float(parts.shape[0])
                else:
                    w = np.zeros((0,), dtype=float)
            elif w.size > 0:
                sw = float(np.sum(w))
                if (not np.isfinite(sw)) or sw <= 1e-18:
                    w[:] = 1.0 / float(w.size)
                else:
                    w = w / sw

        N = int(parts.shape[0])
        K = int(min(220, N))
        if N > 0:
            if K < N:
                idx = np.argpartition(w, -K)[-K:]
                parts_s = parts[idx]
                w_s = w[idx]
            else:
                parts_s = parts
                w_s = w
        else:
            parts_s = parts
            w_s = w
        if parts_s.size > 0:
            parts_s = self._smooth("particles", parts_s, dt)

        # Keep automatic camera/frustum fit anchored only on true geometry
        # to avoid jumps when PF estimate updates after measurements.
        key_pts = np.vstack([pL1, pL2, pF, pF_cam, pF_des, pC])
        pts_all = key_pts
        if parts_s.size > 0:
            # include a smaller fraction for view fitting
            take = int(min(120, parts_s.shape[0]))
            pts_all = np.vstack([pts_all, parts_s[:take]])

        z_min = float(np.min(pts_all[:, 2]))
        z_max = float(np.max(pts_all[:, 2]))
        if abs(z_max - z_min) < 1e-6:
            z_max = z_min + 1.0

        # --- views ---
        iso = self._iso_view
        assert iso is not None
        iso.yaw = self.iso_yaw_deg
        iso.pitch = self.iso_pitch_deg
        iso.fit(pts_all, center_override=pF_cam, lock_scale=True)
        iso.scale = iso.base_scale * float(self.iso_zoom)

        xy = self._xy_view
        assert xy is not None
        xy.fit(pts_all[:, [0, 1]], center_override=pF_cam[:2], lock_scale=True)

        xz = self._xz_view
        assert xz is not None
        xz.fit(
            pts_all[:, [0, 2]],
            center_override=np.array([pF_cam[0], pF_cam[2]], dtype=float),
            lock_scale=True,
        )

        # --- draw backgrounds ---
        screen.fill((6, 6, 16))
        self._draw_panel_bg(screen, pan_iso)
        self._draw_panel_bg(screen, pan_xy)
        self._draw_panel_bg(screen, pan_xz)
        self._draw_panel_bg(screen, pan_std)

        # panel titles
        screen.blit(self._text(font, f"{pan_iso.title}   [cam: A/D yaw, R/F pitch, Z/X zoom]"), (pan_iso.x + 8, pan_iso.y + 6))
        screen.blit(self._text(font, pan_xy.title), (pan_xy.x + 8, pan_xy.y + 6))
        screen.blit(self._text(font, pan_xz.title + "  (z = depth, down)") , (pan_xz.x + 8, pan_xz.y + 6))
        screen.blit(self._text(font, pan_std.title), (pan_std.x + 8, pan_std.y + 6))

        # axes in XZ view: draw surface z=0 line if visible
        try:
            z0y = xz.to_screen([xz.center[0], 0.0])[1]
            if pan_xz.y < z0y < pan_xz.y + pan_xz.h:
                pygame.draw.line(screen, (80, 80, 110), (pan_xz.x, z0y), (pan_xz.x + pan_xz.w, z0y), 1)
                screen.blit(self._text(font, "z=0", (160, 160, 190)), (pan_xz.x + 6, z0y - 16))
        except Exception:
            pass

        # --- particle cloud (painter sort in 3D view) ---
        if parts_s.size > 0:
            # sort by camera depth so far points are drawn first
            depths = np.array([iso.cam_depth(p) for p in parts_s], dtype=float)
            order = np.argsort(depths)  # far->near
            # normalize weights for brightness
            w_s2 = w_s / max(float(np.max(w_s)), 1e-12)
            for j in order:
                p = parts_s[int(j)]
                # alpha-like via brightness (pygame circles do not support per-primitive alpha)
                br = 0.25 + 0.75 * float(w_s2[int(j)])
                col = (int(180 * br), int(180 * br), int(200 * br))
                col = _depth_shade(col, p[2], z_min, z_max)
                pygame.draw.circle(screen, col, iso.to_screen(p), 2)
                pygame.draw.circle(screen, col, xy.to_screen(p[[0, 1]]), 1)
                pygame.draw.circle(screen, col, xz.to_screen(p[[0, 2]]), 1)

        # --- helper for drawing one vehicle ---
        def draw_vehicle(
            name: str,
            pos: np.ndarray,
            vel: np.ndarray,
            base_col: Tuple[int, int, int],
            yaw_deg: float,
            pitch_deg: float = 0.0,
            rad: int = 6,
            body_scale: float = 1.0,
            fill: bool = True,
        ) -> None:
            col = _depth_shade(base_col, float(pos[2]), z_min, z_max)

            # 3D body (isometric view)
            if fill:
                verts = self._vehicle_box(
                    pos,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    length=12.0 * body_scale,
                    width=4.0 * body_scale,
                    height=2.4 * body_scale,
                )
                rot = self._yaw_pitch_rot(yaw_deg, pitch_deg)

                faces = [
                    ([4, 5, 6, 7], np.array([0.0, 0.0, 1.0], dtype=float)),   # top
                    ([0, 1, 2, 3], np.array([0.0, 0.0, -1.0], dtype=float)),  # bottom
                    ([1, 5, 6, 2], np.array([1.0, 0.0, 0.0], dtype=float)),   # front
                    ([0, 3, 7, 4], np.array([-1.0, 0.0, 0.0], dtype=float)),  # back
                    ([2, 3, 7, 6], np.array([0.0, 1.0, 0.0], dtype=float)),   # left
                    ([0, 1, 5, 4], np.array([0.0, -1.0, 0.0], dtype=float)),  # right
                ]

                face_render = []
                for face_ids, n_loc in faces:
                    pts = [verts[i] for i in face_ids]
                    pts2 = [iso.to_screen(pv) for pv in pts]
                    n_w = rot @ n_loc
                    d = float(np.mean([iso.cam_depth(pv) for pv in pts]))
                    c = _face_shade(col, n_w)
                    face_render.append((d, pts2, c))

                face_render.sort(key=lambda it: it[0], reverse=True)
                for d, pts2, c in face_render:
                    pygame.draw.polygon(screen, c, pts2, 0)
                    pygame.draw.polygon(screen, _depth_shade(c, float(pos[2]), z_min, z_max), pts2, 1)

                # nose marker for orientation
                nose = (verts[1] + verts[5] + verts[6] + verts[2]) * 0.25
                stern = (verts[0] + verts[3] + verts[4] + verts[7]) * 0.25
                pygame.draw.circle(screen, (235, 235, 235), iso.to_screen(nose), max(2, rad - 4))
                pygame.draw.line(screen, (30, 30, 30), iso.to_screen(nose), iso.to_screen(stern), 2)
            else:
                pygame.draw.circle(screen, col, iso.to_screen(pos), rad)

            # XY
            pygame.draw.circle(screen, col, xy.to_screen(pos[[0, 1]]), max(3, rad - 2))
            # XZ
            pygame.draw.circle(screen, col, xz.to_screen(pos[[0, 2]]), max(3, rad - 2))

            # velocity arrows
            arrow_time = 10.0
            p1 = pos + vel * arrow_time
            _arrow(screen, iso.to_screen(pos), iso.to_screen(p1), col, width=2, head_len=10)
            _arrow(screen, xy.to_screen(pos[[0, 1]]), xy.to_screen(p1[[0, 1]]), col, width=2, head_len=8)
            _arrow(screen, xz.to_screen(pos[[0, 2]]), xz.to_screen(p1[[0, 2]]), col, width=2, head_len=8)

            # label (only in 3D view)
            px, py = iso.to_screen(pos)
            screen.blit(self._text(font, name, (230, 230, 230)), (px + 8, py - 14))

        yawL1 = float(getattr(env, "yaw_L1", 0.0))
        yawL2 = float(getattr(env, "yaw_L2", 0.0))
        yawF = float(getattr(env, "yaw_F", 0.0))
        pitchF = float(getattr(env, "pitch_F", 0.0))

        # leaders + follower true/est/desired
        draw_vehicle("L1", pL1, vL1, (80, 150, 255), yaw_deg=yawL1, pitch_deg=0.0, rad=7, body_scale=1.1)
        draw_vehicle("L2", pL2, vL2, (130, 190, 255), yaw_deg=yawL2, pitch_deg=0.0, rad=7, body_scale=1.1)
        draw_vehicle("F", pF, vF, (80, 230, 130), yaw_deg=yawF, pitch_deg=pitchF, rad=6, body_scale=1.25)
        draw_vehicle("F_hat", pF_hat, vF, (255, 90, 90), yaw_deg=yawF, pitch_deg=pitchF, rad=5, body_scale=1.0)
        draw_vehicle("F_des", pF_des, np.zeros(3), (255, 210, 100), yaw_deg=yawF, pitch_deg=pitchF, rad=5, body_scale=0.8, fill=False)
        draw_vehicle("C", pC, 0.5 * (vL1 + vL2), (235, 235, 235), yaw_deg=0.0, pitch_deg=0.0, rad=4, body_scale=0.6, fill=False)

        # range lines (true)
        pygame.draw.line(screen, (90, 90, 140), iso.to_screen(pL1), iso.to_screen(pF), 1)
        pygame.draw.line(screen, (90, 90, 140), iso.to_screen(pL2), iso.to_screen(pF), 1)
        pygame.draw.line(screen, (140, 110, 110), iso.to_screen(pL1), iso.to_screen(pF_hat), 1)
        pygame.draw.line(screen, (140, 110, 110), iso.to_screen(pL2), iso.to_screen(pF_hat), 1)

        # PF covariance principal axes (3D view)
        try:
            if np.all(np.isfinite(cov)):
                evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
                evals = np.maximum(evals, 1e-12)
                # ~95% for 3D: sqrt(chi2_{0.95,3}) ~ 2.795
                k = 2.8
                for i in range(3):
                    axis = evecs[:, i]
                    L = k * math.sqrt(float(evals[i]))
                    a0 = pF_hat - L * axis
                    a1 = pF_hat + L * axis
                    pygame.draw.line(screen, (255, 150, 150), iso.to_screen(a0), iso.to_screen(a1), 2)
        except Exception:
            pass

        # --- text HUD (top-left in iso panel) ---
        t = float(getattr(env, "t", 0.0))
        step = int(getattr(env, "step_count", 0))
        pf_obj = getattr(env, "pf", None)
        pf_ess_disp = float("nan")
        pf_nis_ratio_disp = float("nan")
        pf_cons_infl_disp = 1.0
        if pf_obj is not None:
            env_ess = float(getattr(env, "pf_ess_step", float("nan")))
            if np.isfinite(env_ess):
                pf_ess_disp = env_ess
            elif hasattr(pf_obj, "ess") and np.isfinite(float(getattr(pf_obj, "ess", float("nan")))):
                pf_ess_disp = float(getattr(pf_obj, "ess"))
            else:
                w = np.asarray(getattr(pf_obj, "w", []), dtype=float).reshape(-1)
                if w.size > 0 and np.all(np.isfinite(w)):
                    s = float(np.sum(w * w))
                    pf_ess_disp = (1.0 / s) if s > 1e-18 else float(getattr(pf_obj, "N", float("nan")))
            nis_ratio = getattr(env, "pf_nis_ratio_step", float("nan"))
            if np.isfinite(float(nis_ratio)):
                pf_nis_ratio_disp = float(nis_ratio)
            cons_infl = getattr(env, "pf_consistency_infl_step", 1.0)
            if np.isfinite(float(cons_infl)):
                pf_cons_infl_disp = float(cons_infl)

        err_true = float(np.linalg.norm(pF - pF_des))
        err_est = float(np.linalg.norm(pF_hat - pF_des))
        err_pf = float(np.linalg.norm(pF_hat - pF))

        std_x = float(math.sqrt(max(float(cov[0, 0]), 0.0))) if cov.shape == (3, 3) else float("nan")
        std_y = float(math.sqrt(max(float(cov[1, 1]), 0.0))) if cov.shape == (3, 3) else float("nan")
        std_z = float(math.sqrt(max(float(cov[2, 2]), 0.0))) if cov.shape == (3, 3) else float("nan")
        std_norm = float(math.sqrt(max(std_x, 0.0) ** 2 + max(std_y, 0.0) ** 2 + max(std_z, 0.0) ** 2)) if np.isfinite(std_x) and np.isfinite(std_y) and np.isfinite(std_z) else float("nan")
        self._append_std_history(t, std_norm, err_pf)

        # running calibration estimate from rendered history (per current run)
        q20_desc = "n/a"
        q50_desc = "n/a"
        p10_q20 = "n/a"
        p20_q20 = "n/a"
        p50_q20 = "n/a"
        if len(self._std_hist_val) >= 30:
            hs = np.asarray(self._std_hist_val, dtype=float)
            he = np.asarray(self._std_hist_err, dtype=float)
            mask_finite = np.isfinite(hs) & np.isfinite(he)
            if np.any(mask_finite):
                hs = hs[mask_finite]
                he = he[mask_finite]
            if hs.size >= 30:
                q20 = float(np.quantile(hs, 0.20))
                q50 = float(np.quantile(hs, 0.50))
                q20_desc = f"{q20:.2f}"
                q50_desc = f"{q50:.2f}"
                m20 = hs <= q20
                if np.any(m20):
                    e20 = he[m20]
                    def _pct(arr, thr):
                        return float((arr <= thr).mean() * 100.0)
                    p10_q20 = f"{_pct(e20, 10.0):4.1f}%"
                    p20_q20 = f"{_pct(e20, 20.0):4.1f}%"
                    p50_q20 = f"{_pct(e20, 50.0):4.1f}%"

        y0 = pan_iso.y + 28
        x0 = pan_iso.x + 10
        def hud(line: str, col=(230, 230, 230)):
            nonlocal y0
            screen.blit(self._text(font, line, col), (x0, y0))
            y0 += 18

        mode = "MANUAL" if bool(getattr(env, "manual_override", False)) else "POLICY"
        hud(f"t={t:6.1f}s  step={step:4d}  mode={mode}")
        hud(f"F true z={pF[2]:6.1f} m   F_hat z={pF_hat[2]:6.1f} m   (leaders z: {pL1[2]:.1f}, {pL2[2]:.1f})")
        hud(f"err_true(form)={err_true:7.2f} m   err_est(form)={err_est:7.2f} m   ||F_hat-F||={err_pf:6.2f} m")
        hud(f"PF uncertainty norm=σ={std_norm:6.2f} m   (sx,sy,sz): ({std_x:5.2f},{std_y:5.2f},{std_z:5.2f})")
        hud(f"PF std: sx={std_x:5.2f}  sy={std_y:5.2f}  sz={std_z:5.2f}   ESS={pf_ess_disp:6.1f}")
        hud(f"PF consistency: nis_ratio={pf_nis_ratio_disp:5.2f}  infl={pf_cons_infl_disp:4.2f}x")

        # FIM/CRLB (if present)
        if hasattr(env, "fim_win"):
            try:
                eigmin, eigmax, cond = env.fim_win.eig_stats(env.fim_win.I_win)
                crlb = env.fim_win.crlb(use_window=True)
                tr = float(np.trace(crlb)) if np.all(np.isfinite(crlb)) else float("nan")
                hud(f"FIM(win): eig_min={eigmin:8.2e} eig_max={eigmax:8.2e} cond={cond:6.1f}  CRLB_tr={tr:7.2f}")
            except Exception:
                pass

        # controls hint
        hud("Controls: M manual on/off | arrows yaw/speed | W/S pitch | BACKSPACE reset | ESC quit", (200, 200, 140))
        self._draw_std_history(screen=screen, panel=pan_std, font=font)

        # bottom overlay for calibration line (split into two lines, below HUD)
        bottom_lines = [
            f"Std calibration hist: q20={q20_desc}  q50={q50_desc}",
            f"P(err≤10|std≤q20)={p10_q20}  P(err≤20|std≤q20)={p20_q20}  P(err≤50|std≤q20)={p50_q20}",
        ]
        by = max(0, H - 2 - 18 * (len(bottom_lines) + 1))
        bx = 8
        for i, bl in enumerate(bottom_lines):
            screen.blit(self._text(font, bl), (bx, by + i * 18))
