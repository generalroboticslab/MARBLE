# Policy checkpoints

The six TorchScript velocity policies for `--policy trained`. Launch them as in
[`docs/OPERATIONS.md`](../docs/OPERATIONS.md#learned-policy); the export
contract is in [`docs/DEVELOPING.md`](../docs/DEVELOPING.md#exporting-a-learned-policy).

## Which one to run

| Folder | Medium | Use |
|---|---|---|
| `BallbotVelComplexFlatDRLatency` | land | Default on land. `run.sh` preselects it. |
| `BallbotVelRingCageComplexDR` | water | Default on water. |
| `BallbotVelRingCageComplexSmooth` | water | Only if the water default visibly chatters. |
| `BallbotVelComplexFlatSmooth` | land | Not for use: too slow on hardware. |
| `BallbotVelRingCageFlatDRLatency` | land | Retired. A/B comparison only. |
| `BallbotVelRingCageShellDR` | water | Retired. A/B comparison only. The code default `DEFAULT_POLICY_PATH` still names it. |

Folder names are training-task names and do not always match the medium:
`BallbotVelRingCageFlatDRLatency` is a land policy.

## How they differ

**Defaults.** Their `env_config.yaml` files are identical except for the initial
root height: 0.194 m on land, 1.092 m in the water scene. The water `robot.xml`
adds 13 ring geoms (`ring_torus_*`) around the shell.

**Smooth siblings.** Each `Smooth` folder is its parent retrained with a heavier
action-rate penalty (-0.6 on land, -0.35 on water, against -0.2); the files are
otherwise identical to the parent's.

**Retired.** The two retired checkpoints were trained on an earlier plant whose
slider servo (kp 250, kd 50) behaved like the software trajectory smoother, so
they must run with it on
([`docs/OPERATIONS.md`](../docs/OPERATIONS.md#trajectory-smoothing)); their
`requires_smoothing` file tells `run.sh`. The current plant (motor gains,
effort limit, armature) is in the `actuators` block of each `env_config.yaml`.
Its rail friction (0.2 N) is well below the real breakaway force (roughly a
slider's 6.9 N weight, unmeasured).

## Folder contents

| File | Read by | Contents |
|---|---|---|
| `policy_deployed.pt` | `TrainedPolicy` | TorchScript actor, CPU. Input `(1, 3, 23)` float32, output `(1, 3)`. |
| `env_config.yaml` | `PolicyDeployConfig` | A few fields are checked at startup. The rest records the training plant. |
| `robot.xml` | `scripts/policy_sim_check.py` only | Exported MuJoCo model. Does not compile as shipped; see [`PROVENANCE.md`](../PROVENANCE.md#the-shipped-checkpoints). |
| `requires_smoothing` | `run.sh` | Retired folders only. |

## Checksums

MD5 of each `policy_deployed.pt` as shipped:

| Folder | md5 |
|---|---|
| `BallbotVelComplexFlatDRLatency` | `30a67996e22d30d22799227220704c29` |
| `BallbotVelComplexFlatSmooth` | `42ff6e060d421063a629479725128d08` |
| `BallbotVelRingCageComplexDR` | `3e25e90b2496beff642e8ad691035eb0` |
| `BallbotVelRingCageComplexSmooth` | `ad85b4aa03b7abe5c9bb7960b2d4da62` |
| `BallbotVelRingCageFlatDRLatency` | `2b3744860dfa68e3895dd0a25ab385fc` |
| `BallbotVelRingCageShellDR` | `de4482461c2c5a569682f6172df1da0c` |

```bash
md5sum policies/*/policy_deployed.pt   # Linux
md5 policies/*/policy_deployed.pt      # macOS
```

## Adding a checkpoint

Put `policy_deployed.pt` and its `env_config.yaml` in `policies/<name>/` and
check it as in [`docs/DEVELOPING.md`](../docs/DEVELOPING.md#exporting-a-learned-policy).
`run.sh` lists it automatically.
