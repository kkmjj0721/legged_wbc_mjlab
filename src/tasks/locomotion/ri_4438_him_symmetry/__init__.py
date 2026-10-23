"""RI-4438 HIM symmetry task with task-owned environment and training code.

Environment terms live in ``mdp``; PPO and mirror losses live in ``rl.him_ppo``.
Robot assets, observation layouts and the RSL-RL model/storage APIs stay shared.
"""
