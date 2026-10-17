"""RI-4438 HIM + LP-ACRL V7 mixed birth-task curriculum.

Rough keeps the 1560 fixed command tasks (520 directional teacher groups) and
appends 832 world-heading random processes: 52 birth terrain/level pairs x
four independent vx ranges x four independent vy ranges. The 1352 regular
teacher groups start uniformly; signed LP, temperature .025 and uniform mix
.20 update every 50 PPO updates, requiring eight episodes per group. The
separate 25% forward-stair branch never enters regular LP scores.

The static 7 x 13 terrain grid has 55 core spawn tiles for 52 birth conditions,
one surrounding ring and a 16 m outer flat frame. Only the whole-map edge
truncates an otherwise live 20 s episode. Commands, tile crossings and the
latched four-foot pass event never reset physical state or HIM history.

Random processes resample vx, vy and world heading together every U(2,10) s.
Fixed tasks retain their original bins, within-bin sampling and spawn-relative
heading offsets. Both use clipped .5 * wrapped heading error as policy wz.
Whole-episode linear/angular tracking kernels retain their 2:1 score and
failure-budget penalty; rewards, HIM, PPO and randomization stay unchanged.

Common entry points: scripts.train / scripts.play. Rough defaults to A3;
Rough-A0/A1/A2/A3 expose the document's ablations. Strict resume requires V7
state and matching map/task/scoring fingerprints; V6 weights use warm-start.
Offline rl.evaluate separately reports paper EPTE-SP, all-axis tracking,
explicit switch outcomes, complete terrain exits and actual support exposure.
"""
