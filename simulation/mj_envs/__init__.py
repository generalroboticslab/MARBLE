# No package-wide side-effect imports here: every mjlab-env-building entry point
# (run.py, ...)
# already imports mjlab_util.patched_observation_manager itself at env-construction
# time and applies ChronologicalObservationManager per-instance. Forcing it here
# instead drags mjlab (+ prettytable/wandb/tensorboard/...) into every consumer of
# mj_envs.*, including deploy scripts that only need mj_envs.utils and have no
# mjlab dependency at all.
