"""Suppresses duplicate terrain rendering in the native multi-env viewer.

Problem: NativeMujocoViewer._render_other_env_geoms adds every non-selected env's
full dynamic geom set into the scene. Earlier local code filtered by geomgroup 0
to avoid duplicate terrain, but some robots (Ballbot) also put all visual geoms in
group 0. That hid every non-selected Ballbot env in play mode.

Fix: keep mjlab's category-level dynamic-only rendering for other envs. Terrain is
static, so it stays out without filtering by geomgroup. Robot visuals remain visible
regardless of their group assignment.

Mechanism: monkeypatches NativeMujocoViewer._render_other_env_geoms directly on the
shared class object. Needed because two independent call sites construct
NativeMujocoViewer (mj_envs/run.py and mj_envs/tasks/navigation/navigation_run.py) —
a subclass swapped in at only one import site would silently miss the other, so the
method is patched on the class itself rather than replaced via subclassing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from mjlab.viewer.native.viewer import NativeMujocoViewer

if TYPE_CHECKING:
    from mjlab.viewer.native.viewer import _SimDataProtocol, _SimProtocol


def _patched_render_other_env_geoms(
    self: NativeMujocoViewer,
    viewer: mujoco.viewer.Handle,
    sim: "_SimProtocol",
    sim_data: "_SimDataProtocol",
) -> None:
    """Render non-selected environments into the native viewer scene."""
    if self.vd is None:
        return
    assert self.mjm is not None
    assert self.vopt is not None
    assert self.pert is not None

    for i in range(self.env.unwrapped.num_envs):
        if i == self.env_idx:
            continue
        self._sync_env_state_to_mjdata(self.vd, sim_data, i)
        self._sync_model_fields(sim, i)
        mujoco.mj_forward(self.mjm, self.vd)
        mujoco.mjv_addGeoms(
            self.mjm, self.vd, self.vopt, self.pert, self.catmask, viewer.user_scn
        )

    # Restore main env's model fields.
    self._sync_model_fields(sim, self.env_idx)


NativeMujocoViewer._render_other_env_geoms = _patched_render_other_env_geoms
