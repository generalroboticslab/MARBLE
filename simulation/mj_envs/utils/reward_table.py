"""Live terminal reward table with rolling per-term statistics.

By default (use_mean=True): averages over all envs regardless of viewer.
When use_mean=False with a viewer: tracks the camera-selected robot (viewer.env_idx),
matching the env shown in the MuJoCo viewer's built-in reward plot overlay.

Columns (all in raw*weight units except RawMean and Ep Avg):
    Name | Weight | RawMean | W.Mean | W.Std | W.Min | W.Med | W.Max | Ep Avg

Rolling window = display_interval / sample_interval (default: 2.0s / 0.05s = 40 samples).
One GPU→CPU transfer per sample: _step_reward[env_idx or :].cpu().numpy() covers all terms.

Ep Avg (from extras["log"]) accumulates every env.step() — most reliable for sparse rewards
where W.Med will be 0 in most windows.

Footer line (below Total row):
  Episodes  N=42  ok=87%  fell=10%  illegal=3%

Episodes: cumulative since viewer start; fell_over and illegal_contact counts are independent
  (a single fall often triggers both; treat illegal% as a sub-rate of fell%).
  Velocity rolling mean omitted: cmd varies per env/episode, so averaged err is misleading.
"""

import sys
import time
import threading
from collections import deque
from datetime import datetime
from typing import Callable

import numpy as np


def _get_viewer_env_idx(viewer) -> int:
    """Return camera-tracked env index from viewer, or -1 (sentinel: use mean over all envs)."""
    if viewer is None:
        return -1
    if hasattr(viewer, "env_idx"):   # NativeMujocoViewer
        return viewer.env_idx
    if hasattr(viewer, "_scene"):    # ViserPlayViewer
        return viewer._scene.env_idx
    return 0


class RewardTablePrinter:
    """Live terminal reward table with rolling statistics.

    Runs a single background daemon thread that:
      - Samples _step_reward every sample_interval_s (20 Hz default)
      - Prints the table every display_interval_s (2 s default)

    Rolling statistics (over the 2-second window) per term:
      W.Mean  — rolling average rate; primary signal
      W.Std   — variability; high relative to W.Mean → sparse/intermittent reward
      W.Min   — worst-case in window
      W.Med   — median; 0.0 = reward is sparse (most powerful sparsity signal)
      W.Max   — best-case in window
      RawMean — W.Mean / weight; shows the function's raw output scale
      Ep Avg  — from extras["log"]; accumulates every step, ground truth for sparse rewards

    Args:
        env_unwrapped: Unwrapped ManagerBasedRlEnv (reward_manager + extras).
        task_name: Shown in table header.
        viewer: NativeMujocoViewer or ViserPlayViewer, or None for train mode.
        use_mean: If True (default), average over all envs. If False, track the
            camera-selected env from the viewer (updates live as the user presses
            , / . to cycle envs). Ignored when viewer is None (always uses mean).
        display_interval_s: Terminal refresh period.
        sample_interval_s: _step_reward read period (sets rolling window density).
    """

    _IS_TTY = sys.stdout.isatty()
    # Column widths: Name Weight RawMean W.Mean W.Std W.Min W.Med W.Max EpAvg
    _W = (24, 7, 8, 8, 8, 8, 8, 8, 9)

    @staticmethod
    def _fmt(v: float, width: int) -> str:
        """Right-aligned fixed or scientific notation depending on magnitude."""
        if v != v:  # NaN
            return f"{'nan':>{width}}"
        if v == 0.0 or abs(v) >= 0.001:
            return f"{v:>{width}.4f}"
        return f"{v:>{width}.1e}"

    def __init__(
        self,
        env_unwrapped,
        task_name: str,
        viewer=None,
        use_mean: bool = True,
        display_interval_s: float = 2.0,
        sample_interval_s: float = 0.05,
        window_s: float = 10.0,
        max_iterations: int | None = None,
        iteration_getter: Callable[[], int] | None = None,
    ):
        self._rm = env_unwrapped.reward_manager
        self._env = env_unwrapped
        self._task = task_name
        self._viewer = viewer
        self._use_mean = use_mean
        self._display_interval = display_interval_s
        self._sample_interval = sample_interval_s
        self._max_iterations = max_iterations
        self._iteration_getter = iteration_getter
        window = max(1, round(window_s / sample_interval_s))
        # Each entry: 1D numpy array of shape (num_terms,) = _step_reward[env] snapshot
        self._buf: deque[np.ndarray] = deque(maxlen=window)
        self._prev_lines = 0

        # Episode reward cache: only present in steps where a reset occurs.
        # Cached here so render always shows last-seen value instead of nan.
        self._ep_cache: dict[str, float] = {}

        # Termination counters (cumulative since start).
        # time_out and fell_over are mutually exclusive per env per episode;
        # illegal_contact often co-fires with fell_over (arm touches ground after fall).
        self._ep_time_out: int = 0
        self._ep_fell: int = 0
        self._ep_illegal: int = 0
        # id() of last log dict processed for termination counts; each step creates
        # a fresh dict so id changes every step, letting us process each step once.
        self._last_log_id: int = -1


    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample(self):
        """One GPU→CPU transfer covering all reward terms at once."""
        env_idx = _get_viewer_env_idx(self._viewer)
        if self._use_mean or env_idx < 0:
            snap = self._rm._step_reward.mean(dim=0).cpu().numpy()
        else:
            snap = self._rm._step_reward[env_idx].cpu().numpy()
        self._buf.append(snap)

        # Process each step's log dict exactly once (id changes every step).
        # Reads Episode_Reward/* for the reward cache, and Episode_Termination/*
        # for cumulative episode counters.
        log = self._env.extras.get("log", {})
        log_id = id(log)
        if log_id != self._last_log_id:
            self._last_log_id = log_id
            for k, v in log.items():
                if k.startswith("Episode_Reward/"):
                    try:
                        self._ep_cache[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                elif k.startswith("Episode_Termination/"):
                    try:
                        count = int(float(v))
                        name = k[len("Episode_Termination/"):]
                        if name == "time_out":
                            self._ep_time_out += count
                        elif name == "fell_over":
                            self._ep_fell += count
                        elif name == "illegal_contact":
                            self._ep_illegal += count
                    except (TypeError, ValueError):
                        pass


    def _render_footer(self, bar_w: int) -> list[str]:
        """Build training-diagnostic footer lines appended after the Total row."""
        lines = []

        # --- Episode statistics ---
        # Total episodes = time_out + fell_over; mutually exclusive per env per episode.
        # illegal_contact often co-fires with fell_over (not counted separately in total).
        total = self._ep_time_out + self._ep_fell
        if total > 0:
            ok_pct = 100.0 * self._ep_time_out / total
            fell_pct = 100.0 * self._ep_fell / total
            illegal_pct = 100.0 * self._ep_illegal / total  # may exceed fell% if co-fires
            ep_str = (
                f"  Episodes  N={total}"
                f"  ok={ok_pct:.0f}%"
                f"  fell={fell_pct:.0f}%"
                f"  illegal={illegal_pct:.0f}%"
            )
            lines.append(ep_str)

        return lines

    def _render(self) -> str:
        names = self._rm._term_names
        cfgs = self._rm._term_cfgs
        log = self._ep_cache
        now = datetime.now().strftime("%H:%M:%S")

        # arr: (T, num_terms), each row a snapshot of raw*weight per term
        arr = np.stack(list(self._buf)) if self._buf else np.zeros((1, len(names)))

        env_idx = _get_viewer_env_idx(self._viewer)
        mode = (
            f"all {self._env.num_envs} envs (mean)"
            if (self._use_mean or env_idx < 0)
            else f"env {env_idx}/{self._env.num_envs - 1} (camera)"
        )

        W = self._W
        bar_w = sum(W) + (len(W) - 1) * 2 + 2
        top = "  " + "═" * (bar_w - 2)
        sep = "  " + "─" * (bar_w - 2)
        hdr = (
            f"  {'Reward Term':<{W[0]}}  {'Weight':>{W[1]}}  "
            f"{'RawMean':>{W[2]}}  {'W.Mean':>{W[3]}}  {'W.Std':>{W[4]}}  "
            f"{'W.Min':>{W[5]}}  {'W.Med':>{W[6]}}  {'W.Max':>{W[7]}}  {'Ep Avg':>{W[8]}}"
        )

        # Build iteration string if available
        iter_str = ""
        if self._iteration_getter is not None and self._max_iterations is not None:
            try:
                current_iter = self._iteration_getter()
                iter_str = f"  Learning iteration {current_iter}/{self._max_iterations}"
            except Exception:
                pass

        # Title line with task and iteration
        title = f"[{self._task}]{iter_str}"
        title_bar_w = bar_w - 4  # accounting for "  " padding
        title_centered = f"  {title:^{title_bar_w}}"

        rows = [
            top,
            title_centered,
            top,
            f"  {mode}  |  [{now}]  window={len(self._buf) * self._sample_interval:.0f}s",
            hdr,
            sep,
        ]

        sum_wmean = sum_ep = 0.0
        for i, (name, cfg) in enumerate(zip(names, cfgs)):
            w = cfg.weight
            col = arr[:, i]
            wmean = float(np.mean(col))
            wstd = float(np.std(col))
            wmin = float(np.min(col))
            wmed = float(np.median(col))
            wmax = float(np.max(col))
            rmean = wmean / w if abs(w) > 1e-9 else 0.0

            ep_raw = log.get(f"Episode_Reward/{name}", float("nan"))
            ep = float(ep_raw) if not isinstance(ep_raw, float) else ep_raw

            sum_wmean += wmean
            if ep == ep:  # not NaN
                sum_ep += ep

            f = self._fmt
            rows.append(
                f"  {name:<{W[0]}}  {w:>{W[1]}.3f}  "
                f"{f(rmean, W[2])}  {f(wmean, W[3])}  {f(wstd, W[4])}  "
                f"{f(wmin, W[5])}  {f(wmed, W[6])}  {f(wmax, W[7])}  {f(ep, W[8])}"
            )

        rows += [
            sep,
            (
                f"  {'Total':<{W[0]}}  {'':>{W[1]}}  {'':>{W[2]}}  "
                f"{self._fmt(sum_wmean, W[3])}  {'':>{W[4]}}  {'':>{W[5]}}  "
                f"{'':>{W[6]}}  {'':>{W[7]}}  {self._fmt(sum_ep, W[8])}"
            ),
        ]

        footer = self._render_footer(bar_w)
        if footer:
            rows.append(sep)
            rows.extend(footer)

        rows.append(top)
        return "\n".join(rows)

    def _print(self, n_terms: int):
        # Skip if reward manager is still initializing (partial _term_names list).
        if len(self._rm._term_names) != n_terms:
            return
        table = self._render()
        new_lines = table.count("\n") + 1
        if self._IS_TTY and self._prev_lines:
            # Over-clear: if the table grew (e.g. first render had fewer terms),
            # moving up by max(prev, new) + clearing to end-of-screen erases stale lines.
            move_up = max(self._prev_lines, new_lines)
            print(f"\033[{move_up}A\033[J", end="", flush=True)
        print(table, flush=True)
        self._prev_lines = new_lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start background daemon thread. Returns immediately; thread exits with process."""
        # Snapshot the expected term count at construction time.
        # The background thread skips renders until all terms are present,
        # and waits for the buffer to be at least 25% full before first print
        # so the env is running and _step_reward reflects real reward values.
        n_terms = len(self._rm._term_names)
        min_samples = max(5, self._buf.maxlen // 4)

        def _loop():
            last_display = 0.0
            while True:
                try:
                    self._sample()
                    t = time.monotonic()
                    buf_ready = len(self._buf) >= min_samples
                    if buf_ready and t - last_display >= self._display_interval:
                        self._print(n_terms)
                        last_display = t
                except Exception:
                    pass
                time.sleep(self._sample_interval)

        threading.Thread(target=_loop, daemon=True).start()
