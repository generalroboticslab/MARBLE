"""Ballbot (underwater sphere-slider) configuration for mjlab RL training.

Single variant: assets/UnderwaterRobotSim/Sim_Model.xml.
3-DOF free-floating sphere with 3 orthogonal prismatic (slide) actuators.
Sliders carry their own weight: spec_fn clears gravcomp on every body, matching the real
belt drive. (ball_linear's MJCF tags its carriages gravcomp="1"; that tag is now stripped.)

Usage:
    from mj_envs.asset_zoo.ballbot import get_ballbot_robot_cfg
    cfg = get_ballbot_robot_cfg()

Standalone viewer:
    python -m mj_envs.asset_zoo.ballbot.ballbot_constants
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Support both `python -m mj_envs.asset_zoo.ballbot.ballbot_constants` and
# direct script invocation `python ballbot_constants.py`.
_REPO_ROOT = Path(__file__).parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mj_envs.asset_zoo.ballbot.shared import (  # noqa: E402
    MotorSpec,
    _apply_joint_properties,
)


# ═══════════════════════════════════════════════════════════
#  ASSET PATH
# ═══════════════════════════════════════════════════════════

_XML_PATH = Path(__file__).parents[3] / "asset" / "ball_linear_complex" / "Sim_Model.xml"  # MARBLE: only ball_linear_complex ships; see _use_complex_hull
# Ring-shell hull for the on-water deployment variant. Taller/heavier hull (AABB half-extent
# ≈[0.203,0.203,0.231] m, base_link inertial mass 3.411058 kg vs ball_linear's 2.193929 kg),
# but the same 3-slider drive with IDENTICAL slider axes + joint names, so the heuristic's _M
# is unchanged. NOTE: not a 0.19 m sphere — the ring-cage buoyancy radius must be overridden for it.
SHELL_XML_PATH = Path(__file__).parents[3] / "asset" / "ball_linear_complex" / "Sim_Model.xml"  # MARBLE: as _XML_PATH
# Refined full-assembly export of the ring-shell hull: same robot, 342 individually-modelled
# welded parts instead of one lumped base_link inertial (total mass 5.2579 kg vs the shell's
# 5.2698 kg, sliders/actuators/IMU sites byte-identical). Two structural differences from
# every other ballbot MJCF: there is NO single `base_link_geom` hull mesh (the surface is 8
# `V4_1_Top/Bottom_*` caps) and `base_link` carries no `<inertial>` of its own. Load it via
# get_ballbot_complex_robot_cfg(), which fuses the parts and installs an analytic hull.
COMPLEX_XML_PATH = Path(__file__).parents[3] / "asset" / "ball_linear_complex" / "Sim_Model.xml"

# Rolling radius of the complex hull, measured over the 8 collidable cap meshes. The sealed
# shell surface sits at 0.1937 m (median vertex radius); on top of it run 12 continuous fin
# walls on 9 great circles (3 principal planes, each a parallel pair straddling the plane at
# +/-0.011 m, plus 6 single walls on the face diagonals), cresting at 0.2024 m.
#
# The fins are RINGS, not isolated bosses, so the hull cannot settle between them: a fin wins
# ground contact whenever the nearest one is within 16.9 deg of straight down, and sampling
# orientations uniformly the nearest-fin angle never exceeds 11.4 deg (median 2.8, p99 10.2).
# The hull surface therefore never touches the ground, and the effective rolling radius stays
# in 0.1984-0.2024 m (mean 0.2018). 0.202 is that mean to within 0.1%, and it also equals the
# ring-cage envelope `shell_radius + RING_TUBE_RADIUS` the aquatic classes already use, so
# land and water now collide at the same radius.
#
# This was 0.194 (the bare shell surface) until 2026-09-15, on the mistaken premise that the
# fins were 8 local bosses at the octant diagonals. That understated the rolling radius by
# 4.0%, so a policy trained against it spins the shell ~4% faster than hardware needs.
# Checkpoints trained before the change (the deployed BallbotVelComplexFlatDRLatency run)
# were trained at 0.194 and need a retrain to match.
COMPLEX_HULL_RADIUS = 0.202


# ═══════════════════════════════════════════════════════════
#  ROBOT GEOMETRY CONSTANTS  (mirror env.py values)
# ═══════════════════════════════════════════════════════════

SPHERE_BODY = "base_link"
FREE_JOINT   = "base_link_free"
SLIDER_JOINT_NAMES = (
    "base_link_Slider-5",
    "base_link_Slider-6",
    "base_link_Slider-7",
)

# Regex matching all 3 slider joints (used for actuator targeting).
_SLIDER_JOINTS_EXPR = (r"base_link_Slider-\d+",)

INIT_BASE_Z = 0.19  # m — shell radius; spec_fn recenters body origin on the shell, so the
                    # sphere rests on the ground at z = radius (ball_linear shell R≈0.19)

# Action scale: policy output ∈ [-1, 1] → joint target in metres.
# ball_linear ctrlrange = [-0.114, 0.106]; use the tighter (+) bound so commands
# stay inside the asymmetric hard range (soft limit 0.9 trims further).
ACTION_SCALE = 0.106  # m


# ═══════════════════════════════════════════════════════════
#  MOTOR SPEC
# ═══════════════════════════════════════════════════════════
# All gains are joint-space (linear: N/m, N·s/m) — slide actuators.
# These values OVERWRITE the compiled spec (_apply_joint_properties / BuiltinPositionActuatorCfg),
# so the XML's own numbers are inert — verify with a compile probe, never by reading the XML.
#
# kp/kd are the MOTOR gains reflected through the drive, not free parameters. Transmission is a
# GT2 belt on a 50-tooth pulley: 2 mm pitch x 50 T = 100 mm per motor revolution (same constant
# as ballbot_control's MM_PER_TURN, and it reproduces the deployed --vel-limit 20 rad/s =
# 0.318 m/s exactly). With x = r*theta and F = tau/r, gains reflect as 1/r^2:
#
#   r = 0.100/(2*pi) = 15.9155 mm/rad        1/r^2 = 3947.8 m^-2
#   kp 20   N·m/rad  ->   78957 N/m
#   kd 1.0  N·m·s/rad->    3948 N·s/m
#
# WAS kp=250 / kd=50 (tuned in commit 061d751 against ball_linear). At kp=250 a slider sags
# under its own weight: 0.7086 kg * 9.81 = 6.95 N over 250 N/m = 27.8 mm, i.e. 26 % of the
# +-106 mm travel, and the lean the policy commands is not the lean it gets. ball_linear's
# gravcomp tag did not save that tune — it covered only the 0.1282 kg carriage (1.26 N of
# 6.95 N), and spec_fn now strips it anyway. (This is the standing explanation for
# joint_pos_limits logging 0.0000 forever — the policy was not declining full travel, it could
# not reach it.) Measured step response, 50 mm command: 27.80 mm steady error at kp=250 vs
# 0.044 mm at the reflected gains. Stable at dt=0.005 under implicitfast; checked at dt 0.002
# and 0.001 with identical results, so the stiffness is not an integrator risk here.
#
# To revert: set kp=250.0, kd=50.0.
#
#  effort = 31.42 N  0.50 N·m reflected through the drive (x 1/r). NOT the datasheet rating:
#                    the drive is run at 2x the 0.25 N·m continuous figure, and the trusted
#                    band tops out at 0.60 N·m rather than the sheet's 0.68 N·m peak. Nominal
#                    sits at the LOW end of that band so classes without the `effort_limits`
#                    DR get the conservative number, and the DR scales (1.0, 1.2) up to 0.60
#                    N·m = 37.70 N. WAS 17.0 N (XML forcerange), unsourced.
#                    For reference the sheet says 0.68 N·m peak at 5.22 A = 42.73 N, which is
#                    where experiments.py's old "robot measured pulling 42 N peak" prose came
#                    from — it was peak torque reflected, not a log artifact.
#  armature = 0.0    Wipes the XML's joint armature="0.03134" kg. That number is EXACT: the
#                    datasheet rotor inertia 79.45 g·cm^2 = 7.945e-6 kg·m^2 over r^2 = 2.533e-4
#                    gives 0.031366 kg. So the drive is currently modelled massless. Restoring
#                    it adds 4.4 % to the 0.7086 kg slider (pulley inertia still unaccounted).
#  joint_damping = 0.0 N·s/m — this one DOES overwrite; the XML asks for damping="1".
#  joint_friction = 0.2 N   (Coulomb rail dry-rub; DELIBERATELY not from XML — the XML has
#                            frictionloss=0. Models the real slider rubbing the linear rail,
#                            which the free-rolling sim lacked. Do NOT reset to XML's 0.)

# Belt/pulley reflection. r = L/2pi with L = 100 mm/rev, so torque reflects as 1/r (N per N·m)
# and gains as 1/r^2 (m^-2).
_MM_PER_TURN = 100.0                                          # GT2 2 mm pitch x 50 T pulley
_R_SLIDE = (_MM_PER_TURN / 1000.0) / (2.0 * np.pi)            # r = 15.9155 mm/rad
_FORCE_REFLECTION = 1.0 / _R_SLIDE                            # 1/r  = 62.832 m^-1
_GAIN_REFLECTION = _FORCE_REFLECTION**2                       # 1/r^2 = 3947.8 m^-2

# Motor datasheet (BLDC, 16 V): rated 0.25 N·m @ 1.88 A, peak 0.68 N·m @ 5.22 A, Kt 0.11 N·m/A,
# rotor inertia 79.45 g·cm^2, rated speed 697 rpm. Rated speed reflects to 1.16 m/s, well above
# the deployed --vel-limit 20 rad/s = 0.318 m/s, so the slider is torque-limited, not speed-limited.
# Deliverable torque band, set from operating experience 2026-09-14, NOT straight off the
# datasheet. The sheet says 0.25 N·m continuous / 0.68 N·m peak, but the drive is run at 2x
# rated continuously and is not trusted to the full peak, so the usable band is 0.5-0.6 N·m.
# Nominal is the LOW end: classes without the effort_limits DR then get the conservative
# figure, and the DR spans up to the optimistic one rather than down from it.
# Raised 0.50 -> 0.60 on 2026-09-15: the drive is saturated essentially all the time (measured
# 96% of control steps on ground, 84% in water, pegged at the limit), so the effort limit is a
# binding constraint on the gait, not a formality. 0.60 N·m is the top of the trusted band.
MOTOR_TORQUE_NOMINAL = 0.60     # N·m, top of the trusted band
MOTOR_TORQUE_MAX = 0.68         # N·m, datasheet peak at 5.22 A — DR ceiling only
MOTOR_ROTOR_INERTIA = 79.45e-7  # kg·m^2 (79.45 g·cm^2); pulley inertia not included

# Position-loop gains programmed into the motor driver. NOT free parameters and NOT tunable at
# runtime — they are flashed with the vendor tool, so sim must follow hardware, never the other
# way round. Set on the robot 2026-09-14, down from 40 / 0.5: half the stiffness and double the
# damping, which takes the slider from zeta 2.89 to 8.17 and the slow pole from 82.6 to 21 rad/s
# (tau 12 ms -> 48 ms). That is a deliberate move against the slider oscillation the stiffer
# gains produced once the phantom XML servos and the deployed software smoother were removed.
MOTOR_KP = 20.0  # N·m/rad
MOTOR_KD = 1.0   # N·m·s/rad
EFFORT_PEAK_SCALE = MOTOR_TORQUE_MAX / MOTOR_TORQUE_NOMINAL  # 1.2 — upper bound for effort DR

# Driver velocity clamp, the `--vel-limit 20 rad/s` the deployed launch passes. The driver runs a
# position loop and rate-limits the commanded target; the carriage therefore cannot slew faster
# than this no matter what the policy asks for. Enforced in sim by RateLimitedJointPositionAction
# (tasks/ballbot_velocity/action.py), NOT by the actuator model — MuJoCo position actuators have a
# force clamp but no speed clamp, so before this the ground gait ran the sliders at 2.6 m/s peak,
# 7x the hardware ceiling and above even the motor's reflected rated speed (1.16 m/s), and the
# ball reached 3.05 m/s on flat ground. Measured 2026-09-15.
MOTOR_VEL_LIMIT = 20.0  # rad/s at the motor, flashed in the driver
SLIDER_VEL_LIMIT = MOTOR_VEL_LIMIT * _R_SLIDE  # 0.3183 m/s at the carriage

_MOTOR_SPEC = MotorSpec(
    motor_type     = "slide",
    effort_limit   = MOTOR_TORQUE_NOMINAL * _FORCE_REFLECTION,  # 0.50 N·m -> 31.416 N
    armature       = MOTOR_ROTOR_INERTIA / _R_SLIDE**2,       # 79.45 g·cm^2 -> 0.031366 kg
    joints         = _SLIDER_JOINTS_EXPR,
    kp             = MOTOR_KP * _GAIN_REFLECTION,   # 20 N·m/rad at the motor  ->  78957 N/m
    kd             = MOTOR_KD * _GAIN_REFLECTION,   # 1.0 N·m·s/rad at the motor -> 3948 N·s/m
    joint_damping  = 0.0,
    joint_friction = 0.2,    # N, Coulomb rail dry-rub (frictionloss); breakaway ~0.2 N.
                             # TUNE to hardware.
)


# ═══════════════════════════════════════════════════════════
#  COLLISION CONFIG
# ═══════════════════════════════════════════════════════════
# Only base_link_geom is collidable in the XML (contype=1, conaffinity=1).
# All sub-component geoms have contype=0 (XML default).

# condim 6, not 4: MuJoCo only applies the third friction entry (ROLLING) at condim 6, so at
# condim 4 the hull free-rolled — a loss-free sphere that out-ran the real robot and, at terminal
# speed, had to carry zero time-averaged drive torque (the slider CoM offset then oscillates about
# zero instead of leading persistently). Measured 2026-09-15.
#
# Rolling coefficient 5e-5 -> 0.002. MuJoCo's rolling entry has units of length, so the equivalent
# rolling-resistance coefficient is coeff / R = 0.002 / 0.202 = 0.01 — textbook hard rubber on
# concrete. It is the smallest value in the condim-6 sweep that is not effectively inert (terminal
# speed 0.712 m/s vs 0.993 at 5e-5). Values above it are NOT monotonic in the sweep (0.005 ->
# 0.881, 0.01 -> 0.730, 0.02 -> 0.423) because gait phase dominates at that point, so a larger
# number would be a guess dressed as a measurement.
#
# NOT calibrated against hardware — no measurement of real hull spin rate exists yet. When one
# does (IMU gyro, both media), this is the knob to fit, and checkpoints trained before this change
# were trained on a loss-free ground.
HULL_COLLISION = CollisionCfg(
    geom_names_expr = (r"^base_link_geom$",),
    contype         = 1,
    conaffinity     = 1,
    condim          = 6,
    priority        = 0,
    friction        = (0.3, 0.002, 0.002),
)


# ═══════════════════════════════════════════════════════════
#  ACTION SCALE MAP
# ═══════════════════════════════════════════════════════════

def get_action_scale() -> dict[str, float]:
    """Map joint-name regex → action scale (metres)."""
    return {pat: ACTION_SCALE for pat in _SLIDER_JOINTS_EXPR}


# ═══════════════════════════════════════════════════════════
#  PUBLIC FACTORY
# ═══════════════════════════════════════════════════════════

def _has_geom(model, name: str) -> bool:
    """True if the compiled model defines a geom called `name`."""
    return any(model.geom(i).name == name for i in range(model.ngeom))


def _fuse_static_children(spec: mujoco.MjSpec, xml: Path, body_name: str) -> None:
    """Fold `body_name`'s rigid leaf children into it, in place.

    A full-assembly CAD export emits every bracket, screw and PCB as its own MuJoCo body
    welded to the hull — ball_linear_complex has 342 of them, giving nbody 399 against the
    lumped shell export's 57 and stepping 4.2x slower (94 kHz -> 22 kHz single-instance).
    None of them carries a DOF, so the split is pure bookkeeping: summing their mass and
    inertia into the parent and reparenting their geoms is EXACT, not an approximation.
    Verified against the unfused model: mass matrix agrees to 8.9e-15 and a 2000-step
    free-flight trajectory to 1.6e-15. (Rolling-contact trajectories do drift, because geom
    ids renumber and the contact solver is order-sensitive — irrelevant here, since callers
    replace the 8 cap meshes with a single analytic hull sphere anyway.)

    Only joint-free, site-free, childless bodies are fused. `sphere_center` owns the IMU
    sites that the <sensor> block references and the three `base-*` bodies own the slider
    joints, so all four are left standing.

    Inertials are read from a throwaway compile rather than the spec: the compiler has
    already diagonalised every `fullinertia` into a consistent (inertia, iquat) pair, so the
    rigid-body sum below does not have to re-derive them. `MjsBody.to_frame()` moves the
    geoms up for us but DISCARDS the body's explicit inertial, which is why the parent's
    inertial is recomputed here — without it the compile fails with "mass and inertia of
    moving bodies must be larger than mjMINVAL".
    """
    probe = mujoco.MjSpec.from_file(str(xml)).compile()
    base = spec.body(body_name)
    base_id = probe.body(body_name).id

    fused_ids = [base_id]
    for child in list(base.bodies):
        if child.joints or child.sites or child.bodies:
            continue
        fused_ids.append(probe.body(child.name).id)
        child.to_frame()

    masses, coms, inertias = [], [], []
    for i in fused_ids:
        # Child frame -> base frame. The parent itself contributes at identity.
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if i == base_id else probe.body_quat[i]
        pos = np.zeros(3) if i == base_id else probe.body_pos[i]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        irot = np.zeros(9)
        mujoco.mju_quat2Mat(irot, probe.body_iquat[i])
        principal = rot @ irot.reshape(3, 3)
        masses.append(probe.body_mass[i])
        coms.append(pos + rot @ probe.body_ipos[i])
        inertias.append(principal @ np.diag(probe.body_inertia[i]) @ principal.T)

    masses = np.array(masses)
    coms = np.array(coms)
    total_mass = masses.sum()
    com = (masses[:, None] * coms).sum(axis=0) / total_mass

    # Parallel-axis shift of every part onto the fused COM.
    inertia = np.zeros((3, 3))
    for mass, part_com, part_inertia in zip(masses, coms, inertias):
        d = part_com - com
        inertia += part_inertia + mass * (d @ d * np.eye(3) - np.outer(d, d))

    eigvals, eigvecs = np.linalg.eigh(inertia)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1  # eigh may return a reflection; MuJoCo needs a rotation
    iquat = np.zeros(4)
    mujoco.mju_mat2Quat(iquat, eigvecs.reshape(9))

    base.mass = float(total_mass)
    base.ipos = com
    base.iquat = iquat
    base.inertia = eigvals
    base.explicitinertial = True


def _scale_subtree_mass(body, scale: float) -> None:
    """Recursively scale mass + fullinertia of body and all descendants."""
    body.mass *= scale
    body.fullinertia = body.fullinertia * scale
    for child in body.bodies:
        _scale_subtree_mass(child, scale)


def get_ballbot_robot_cfg(
    collisions: tuple = (HULL_COLLISION,),
    init_base_z: float = INIT_BASE_Z,
    slider_mass_scale: float = 1.0,
    xml_path: Path | None = None,
) -> EntityCfg:
    """Return EntityCfg for the underwater ballbot.

    Args:
        collisions:  Tuple of CollisionCfg. Default: HULL_COLLISION.
        init_base_z: Override sphere-centre init height (metres).
        slider_mass_scale: Scale factor applied to each slider subtree's mass
            and inertia (default 1.0 = XML as-authored, 0.7086 kg/slider).
            Full open-loop sweep (1x-10x) found
            crossover at 5x (0.240 m/s); 1x-4x all fail (<0.04 m/s). 2x
            forcerange has zero effect. CAVEAT: that sweep ran under the
            "sliders are gravcomp=1, so only inertial reaction drives the
            sphere" premise, which was wrong twice over — gravcomp covered
            18 % of slider mass where it existed at all, and is now stripped
            entirely. Quasi-static lean DOES produce drive; re-probe before
            leaning on these crossover numbers.

    Returns:
        EntityCfg ready for mjlab Entity construction.

    Raises:
        AssertionError if MJCF not found at expected path.
    """
    xml = Path(xml_path) if xml_path is not None else _XML_PATH
    assert xml.exists(), (
        f"MJCF not found: {xml}\n"
        f"Expected: asset/ball_linear/Sim_Model.xml (or ball_linear_shell for the shell variant)"
    )

    motors = [_MOTOR_SPEC]
    actuator_cfgs = (
        BuiltinPositionActuatorCfg(
            target_names_expr = _SLIDER_JOINTS_EXPR,
            stiffness         = _MOTOR_SPEC.kp,
            damping           = _MOTOR_SPEC.kd,
            effort_limit      = _MOTOR_SPEC.effort_limit,
            armature          = _MOTOR_SPEC.armature,
        ),
    )

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(xml))
        # Auto-recenter the body frame on the hull shell, when the model HAS a lumped hull
        # mesh to measure. ball_linear_complex does not (its surface is 8 separate cap meshes)
        # and its export is already centred on the hull, so the probe below is skipped there
        # rather than made to guess a centre from an arbitrary geom.
        # A CAD export can place the
        # freejoint/body origin at an arbitrary datum rather than the sphere's geometric
        # centre (ball_linear: origin is ~0.19 m below the shell centre — the COM was
        # ballasted to the centre but the frame origin was not moved with it; the old
        # UnderwaterRobotSim export happened to coincide). ALL ballbot task code assumes the
        # shell sits at the body origin (collision sphere, ring-cage, buoyancy point,
        # INIT_BASE_Z), so a non-zero offset detaches the cage/visual from the frame and
        # mis-locates the buoyancy wrench. Measure the hull-mesh AABB centre (throwaway
        # compile of a SEPARATE spec instance — the only pre-compile way to get mesh bounds;
        # compiling the working spec would finalize it and block the mutations below) and shift
        # base_link's direct children by -centre so origin == shell centre, regardless of export.
        probe = mujoco.MjSpec.from_file(str(xml)).compile()
        if _has_geom(probe, "base_link_geom"):
            _hg = probe.geom("base_link_geom")
            off = np.array(probe.geom_pos[_hg.id]) + np.array(probe.geom_aabb[_hg.id][:3])
            base = spec.body(SPHERE_BODY)
            for geom in base.geoms:
                geom.pos = np.array(geom.pos) - off
            base.ipos = np.array(base.ipos) - off      # COM was at the shell centre → ~origin
            for child in base.bodies:                   # slider subtrees move with their parent
                child.pos = np.array(child.pos) - off
        # (base_link's freejoint pos is irrelevant; slider joints live on the child bodies above)
        # Strip gravity compensation. ball_linear tags its 3 slider carriages gravcomp="1";
        # ball_linear_shell and ball_linear_complex tag nothing. The real belt drive has no
        # such term — the motor holds the load — so it is cleared here for every asset rather
        # than edited out of a regenerated CAD export.
        # NOTE the tag was near-inert even where present: MuJoCo's gravcomp is per-body, not
        # per-subtree, and the carriage is 0.1282 kg of the 0.7086 kg slider, so it returned
        # 1.258 N of the 6.951 N the vertical slider actually carries (18 %). Any claim
        # elsewhere that these sliders are "neutrally buoyant" predates that measurement.
        for body in spec.bodies:
            body.gravcomp = 0.0
        # Delete the MJCF's own <actuator> block. BuiltinPositionActuatorCfg APPENDS its three
        # position servos (builtin_actuator.py:450 add_actuator) rather than replacing these, so
        # leaving them gives nu=6 on 3 joints: the action manager writes ctrl[3:6] and ctrl[0:3]
        # stays 0, i.e. three XML servos holding every slider at centre with kp 250 / kd 50 /
        # +-17 N. Measured: a full-travel command (0.106 m) reached 0.0676 m = 63.8 %.
        # Every ballbot result before 2026-09-14 was trained against that hidden spring.
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        # Neutralize worldbody-level floor — Sim_Model.xml bundles it for
        # standalone use; disable collision and hide so it doesn't conflict
        # with the scene's own floor. MjsGeom has no delete() in these bindings.
        for geom in spec.worldbody.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                geom.contype = 0
                geom.conaffinity = 0
                geom.group = 99  # invisible
        _apply_joint_properties(spec, motors)
        if slider_mass_scale != 1.0:
            for jname in SLIDER_JOINT_NAMES:
                _scale_subtree_mass(spec.joint(jname).parent, slider_mass_scale)
        return spec

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, init_base_z),
            joint_pos={r"base_link_Slider-\d+": 0.0},
            joint_vel={".*": 0.0},
        ),
        collisions=collisions,
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=actuator_cfgs,
            soft_joint_pos_limit_factor=0.9,  # soft from [-0.114, 0.106] m hard (ball_linear)
        ),
    )


def get_ballbot_complex_robot_cfg(collision_radius: float = COMPLEX_HULL_RADIUS) -> EntityCfg:
    """Return EntityCfg for the refined full-assembly ballbot (`ball_linear_complex`).

    Three things the lumped-shell path cannot do for this MJCF, done here so land and water
    experiments share one loader:

    1. **Fuse the 342 welded parts** (`_fuse_static_children`) — exact, and recovers the
       4.2x step-rate cost of the detailed export.
    2. **Install an analytic hull sphere named `base_link_geom`.** The export's collidable
       surface is 8 separate cap meshes, which MuJoCo collides by convex hull; the union of
       8 hulls is neither the sphere the ring-cage drag model assumes nor a shape whose
       radius is under our control. Same proxy decision as BallbotVelRingCageShellDR. Keeping the canonical name means HULL_COLLISION, the
       `hull_friction` DR event and the viewer's `foot_geom_names` all match unchanged.
       `density=0` so the geom cannot perturb total mass — BallbotVelRingCage.configure()
       solves float equilibrium from it.
    3. **Skip the hull recentre**, which has no `base_link_geom` to measure pre-fusion.

    Args:
        collision_radius: hull sphere radius (m). Defaults to the rolling radius; the
            ring-cage variants pass `shell_radius + RING_TUBE_RADIUS` so the contact
            envelope matches the ring envelope the drag term models.

    Design note: `disable_other_geoms` defaults True on CollisionCfg, so the 8 cap meshes
    drop to contype/conaffinity 0 and survive as visuals only.
    """
    cfg = get_ballbot_robot_cfg(
        xml_path=COMPLEX_XML_PATH,
        init_base_z=collision_radius,
        collisions=(HULL_COLLISION,),
    )
    inner_spec_fn = cfg.spec_fn

    def spec_fn() -> mujoco.MjSpec:
        spec = inner_spec_fn()
        _fuse_static_children(spec, COMPLEX_XML_PATH, SPHERE_BODY)
        spec.body(SPHERE_BODY).add_geom(
            name="base_link_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[collision_radius, 0.0, 0.0],
            density=0.0,
            group=3,                        # collision group; hidden in default render
            rgba=[0.5, 0.5, 0.5, 0.3],
        )
        return spec

    cfg.spec_fn = spec_fn
    return cfg

