# Checkpoint: finished Brier-versus-cross-entropy study
**Doc link:** <https://github.com/brando90/ultimate-utils/blob/cursor/brier-ce-pilot-076e/experiments/06_brier_vs_ce_alexnet_dense_lm/CKPT_brier_ce_pilot.md>
**TLDR:** All five study components are finished; the takeover verified the supplied artifacts and does not reopen any run.
Created: 11-06-2026 22:57 PST (first Git commit; original creation time was not recorded)
Last updated: 11-08-2026 17:53 PST

## Goal
- Compare Brier-score and cross-entropy training in `experiments/06_brier_vs_ce_alexnet_dense_lm`; preserve the completed evidence and its scope.
## What ran
- Inherited results revision: `a1fd6973bff8d9da770284050970714d94bb266c` on `cursor/brier-ce-pilot-076e`; local and remote branch matched before this checkpoint update.
- DONE: WikiText-103 (`5db52dd`), FineWeb-Edu (`d8bcb86`, provenance/state `a1fd697`), CIFAR-10 optimizer follow-up (`07b4d76`), collapsed-bias seeds 1–4 (`73085d0`), and the earlier six-epoch AlexNet comparison. No training was started during takeover.
- Working directory: `/Users/brandomiranda/.worktrees/ultimate-utils/brier-ce-pilot`; retained as the user-designated continuation tree. Codex owns this checkpoint update; no temporary checkout was created.
## Results
- Frozen records: [WikiText-103](lm_fixed/results.md), [FineWeb-Edu](fineweb_edu/results.md), [CIFAR-10 follow-up](stability_cifar/results.md), [collapsed-bias measurements](collapsed_replication/MEASURED.md), [collapsed-bias table](collapsed_replication/results.md), and [early AlexNet comparison](results.md).
- All nine user-listed files, including FineWeb-Edu protocol, token metadata, and state, matched their committed bytes. FineWeb-Edu state is DONE; its protocol matches `protocol.json` exactly. Trainer outputs remain unchanged.
- Metadata caveat: FineWeb-Edu records retain the hardcoded `dataset: wikitext103` from reused `train_lm_fixed.py:train_one`; `train_fineweb.py`, the protocol, and token provenance identify FineWeb-Edu. The old top-level language-model BLOCKED rows describe the earlier download failure, superseded by `lm_fixed/results.md`.
## Stuck / blocked
- Nothing blocks this takeover. No new run or research deliverable was requested. Cluster completion and email receipts are inherited from Brando's handoff; no fresh cluster or mail check was performed.
## Next steps
- [p 10/10, completed] Adopt the finished state. To inspect later: `git -C /Users/brandomiranda/.worktrees/ultimate-utils/brier-ce-pilot status --short --branch`; then read the relevant frozen result file before proposing a new run and name the unanswered question.
- Preserve the main checkout `/Users/brandomiranda/ultimate-utils`, both `experiments/06_*` directory names, and every finished job. Do not rerun, restart, duplicate, or delete finished jobs. Do not open a pull request; the refusal is final.
- Preserve `/lfs/mercury2/0/brando9/brier_ce_pilot_076e/` (`lm_fixed`, `full`, `data`, `code`) and `/lfs/mercury1/0/brando9/brier_ce_pilot_076e/fineweb_edu`. No new mercury2 `/lfs` data: the handoff reports about 13 GB free. No purchases, new paid graphics processors, cluster-file deletion, or authentication bypass.
- Before fresh Secure Shell (SSH), run `python3 ~/agents-config/scripts/snap_reconnect.py --status`; reuse existing ControlMaster sockets for mercury1/2 and skampere1–3, with at most one fresh target connection per 10 minutes. No connection is needed to resume reading these local results.
- Do not resend WikiText email `1a1ba44eca337ce4`, CIFAR-10 email, or collapsed-bias email. FineWeb-Edu email remains unsent unless Brando asks. No email was sent during takeover.
- Preserve interpretation: six AlexNet epochs are not a converged minimum; the historical weight-decay explanation for cross-entropy's popularity was dropped; experiment 05 uses AdamW with weight decay on weights only and is not the stochastic-gradient-descent stall. Seed 0 of `experiments/06_sgd_vs_adam_collapsed_bias` was not rerun.
## Brando to-do
- None for the takeover.

## Log

- 11-06-2026 22:43 PT: authorization changed from a 1 GPU-hour pilot to the matched multi-seed run, up to 12h on one GPU.
- 11-06-2026 22:41 PT: mercury2 ten A4000s idle; skampere2 full; skampere1 GPU 1 and 2 idle but not used; skampere3 GPU 2 and 5 idle but mercury is the placement; mercury1 has own vLLM jobs left alone.
- 11-06-2026 22:50 PT: CPU_CHECKS_PASS. Full model 123551232 parameters. Collapsed class-0 bias gradient is the shared-bias gradient of the mean loss.
- 11-06-2026 22:58 PT: mercury2 rechecked. All 10 A4000s idle, no compute apps. Free 15587700736 bytes. Pin GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697 if still idle at launch. No duplicate job.
- 11-06-2026 22:59 PT: smoke SMOKE_PASS on that GPU. MNIST cross-entropy at learning rate 0.001 fell from 2.3031 to 1.1571 over 60 steps. Language model 123551232 parameters, 10 steps finite.
- 11-06-2026 23:01 PT: full job started. tmux brier-ce-pilot-076e, driver PID 860544, train PID 861294, GPU GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697. MNIST tune in progress at 23:03 PT (cross-entropy train losses 0.2896, 0.2284, 0.2308, 0.3803 at the four learning rates).
- 11-08-2026 17:53 PST -- Codex took over the finished study at `a1fd697`; verified all nine supplied artifacts against Git and the remote branch, replaced stale RUNNING state, retained the prior log, and changed no trainer output. No new run, cluster connection, email, or pull request.
