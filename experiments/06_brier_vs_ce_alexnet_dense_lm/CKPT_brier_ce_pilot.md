# CKPT brier vs cross-entropy

## State

- Status: CPU_CHECKS_PASS (123551232 LM parameters). GPU smoke not started
- Branch: cursor/brier-ce-pilot-076e
- Worktree: ~/.worktrees/ultimate-utils/brier-ce-pilot
- Canonical repo left untouched: ~/ultimate-utils (other session's dirty tree)
- Host plan: mercury2, one idle A4000, recheck at launch
- Clock: 12h from the full run's first start. Smoke first.
- Disk: 15596736512 bytes free on mercury2 LFS at 22:44 PT 11-06-2026. Cap 6GB new files.
- Job dir: /lfs/mercury2/0/brando9/brier_ce_pilot_076e
- Python: /lfs/mercury2/0/brando9/uv_envs/veribench/bin/python
- Deadline field: stored in the job state.json, not reset on resume
- Test split: closed until selection for that dataset is saved
- Do not edit ~/ultimate-utils or the shared bashrc

## Log

- 11-06-2026 22:43 PT: authorization changed from a 1 GPU-hour pilot to the matched multi-seed run, up to 12h on one GPU.
- 11-06-2026 22:41 PT: mercury2 ten A4000s idle; skampere2 full; skampere1 GPU 1 and 2 idle but not used; skampere3 GPU 2 and 5 idle but mercury is the placement; mercury1 has own vLLM jobs left alone.
- 11-06-2026 22:50 PT: CPU_CHECKS_PASS. Full model 123551232 parameters. Collapsed class-0 bias gradient is the shared-bias gradient of the mean loss.
- 11-06-2026 22:58 PT: mercury2 rechecked. All 10 A4000s idle, no compute apps. Free 15587700736 bytes. Pin GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697 if still idle at launch. No duplicate job.
