"""RI-4438 HIM + LP-ACRL random-process birth-task curriculum.

Rough uses 832 world-heading random processes: 52 birth terrain/level pairs x
four independent vx ranges x four independent vy ranges. Flat uses 16 groups.
Every training episode is sampled by the teacher, without fixed-command tasks
or a separate forward-stair branch. Groups start uniformly; signed LP,
temperature .025 and uniform mix .20 update every 50 PPO updates, requiring
eight episodes per group.

The static 7 x 13 terrain grid has 55 core spawn tiles for 52 birth conditions,
one surrounding ring and a 16 m outer flat frame. Only the whole-map edge
truncates an otherwise live 20 s episode. Commands, tile crossings and the
latched four-foot pass event never reset physical state or HIM history.

Random processes resample vx, vy and world heading together every U(2,10) s.
Clipped .5 * wrapped heading error supplies policy wz.
Current presets use command-dependent low-speed tracking scales, separate
translation/yaw motion gates and sustained low-speed progress penalties on all
terrains. Whole-episode tracking retains its 2:1 score and failure-budget penalty;
HIM dimensions, PPO, robot control and randomization stay unchanged. Historical
A0–A3 and saved snapshots retain the original reward/observation protocol.

Common entry points: scripts.train / scripts.play. Rough-A0/A1/A2/A3 preserve
the historical ablations, including fixed tasks and 25% forward-stair coverage.
Strict resume requires matching map/task/scoring fingerprints; switching from
fixed/mixed training to the default random-only sampler uses warm-start.
Offline rl.evaluate separately reports paper EPTE-SP, all-axis tracking,
explicit switch outcomes, complete terrain exits and actual support exposure.
"""
