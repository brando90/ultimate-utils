# Checkpoint: finished Brier-versus-cross-entropy study
**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/06_brier_vs_ce_alexnet_dense_lm/CKPT_brier_ce_pilot.md>
**TLDR:** All five study components and paired analyses are published on main at `76ee3fc`. The requested saved-curve comparison is in progress; no new training or evaluation is authorized.
Created: 11-06-2026 22:57 PST (first Git commit; original creation time was not recorded)
Last updated: 11-09-2026 11:33 PST

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
- Saved-curve comparison: FineWeb-Edu figures/statistics complete; WikiText-103 retrieval pending. The shared guard reserved again at 11:12:15 PST on 11-09-2026, with next eligibility 12:12:15 PST. A separate local agent task's one-shot mercury1 read owns that reservation, verified by its saved receipt; its process has exited. The failure layer remains unverified. Our waiter failed locally on a busy status lock before SSH; inspected desktop reconnects were also locally rejected. The helper handles transient contention and preserves complete private diagnostics; nine mocked regression checks passed. One root-owned retrieval waits in exec session `68842` (`retrieve_wikitext_state.py --wait-seconds 4000 --diagnostics-dir /Users/brandomiranda/.agent-tasks/11-09-2026-brier-curves-connection/attempt_12-12`); sessions `29073` and `11616` have exited. Do not duplicate it. No new training or evaluation; nine original artifacts unchanged.
## Next steps
- [p 10/10, active] Retrieve saved WikiText-103 curves after cooldown, compare late loss trends with FineWeb-Edu, publish figures/report to main, and sync the Google findings. Root owns retrieval, report, and Git; worker completed `curve_analysis/analyze_curves.py` and partial figures/statistics. [Current report](curve_analysis/README.md) explicitly marks missing WikiText data.
- [p 10/10, completed] Summarize all five parts and compute saved-data differences: [synthesis](SUMMARY_RESULTS_11-08-2026.md), [FineWeb-Edu paired report](fineweb_edu/paired_differences.md), and [Google Assistant Results](https://docs.google.com/document/d/1xkEn8vW800jBucuJANuXD8ph-sMQMofwNZw_nJzjJls/edit?tab=t.ty5se4uscfip). Exact next verification: `python3 experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu/compute_paired_differences.py --check`. No new runs requested.
- Preserve the main checkout `/Users/brandomiranda/ultimate-utils`, both `experiments/06_*` directory names, and every finished job. Do not rerun, restart, duplicate, or delete finished jobs. Do not open a pull request; the refusal is final.
- Preserve `/lfs/mercury2/0/brando9/brier_ce_pilot_076e/` (`lm_fixed`, `full`, `data`, `code`) and `/lfs/mercury1/0/brando9/brier_ce_pilot_076e/fineweb_edu`. No new mercury2 `/lfs` data: the handoff reports about 13 GB free. No purchases, new paid graphics processors, cluster-file deletion, or authentication bypass.
- Before fresh Secure Shell (SSH), run `python3 ~/agents-config/scripts/snap_reconnect.py --status`; reuse existing ControlMaster sockets for mercury1/2 and skampere1–3, with at most one fresh target connection per 10 minutes. One guarded mercury2 read will retrieve the missing WikiText-103 state; no remote writes or job execution.
- Do not resend WikiText email `1a1ba44eca337ce4`, CIFAR-10 email, or collapsed-bias email. The handoff said FineWeb-Edu email was unsent; a later read found an existing 11-08-2026 result email (connected-mailbox copy `1a1bccbd8f531a74`) and consolidated archive. Do not resend them. No email was sent during this work.
- Preserve interpretation: six AlexNet epochs are not a converged minimum; the historical weight-decay explanation for cross-entropy's popularity was dropped; experiment 05 uses AdamW with weight decay on weights only and is not the stochastic-gradient-descent stall. Seed 0 of `experiments/06_sgd_vs_adam_collapsed_bias` was not rerun.
## Brando to-do
- None for this synthesis.

## Log

- 11-06-2026 22:43 PT: authorization changed from a 1 GPU-hour pilot to the matched multi-seed run, up to 12h on one GPU.
- 11-06-2026 22:41 PT: mercury2 ten A4000s idle; skampere2 full; skampere1 GPU 1 and 2 idle but not used; skampere3 GPU 2 and 5 idle but mercury is the placement; mercury1 has own vLLM jobs left alone.
- 11-06-2026 22:50 PT: CPU_CHECKS_PASS. Full model 123551232 parameters. Collapsed class-0 bias gradient is the shared-bias gradient of the mean loss.
- 11-06-2026 22:58 PT: mercury2 rechecked. All 10 A4000s idle, no compute apps. Free 15587700736 bytes. Pin GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697 if still idle at launch. No duplicate job.
- 11-06-2026 22:59 PT: smoke SMOKE_PASS on that GPU. MNIST cross-entropy at learning rate 0.001 fell from 2.3031 to 1.1571 over 60 steps. Language model 123551232 parameters, 10 steps finite.
- 11-06-2026 23:01 PT: full job started. tmux brier-ce-pilot-076e, driver PID 860544, train PID 861294, GPU GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697. MNIST tune in progress at 23:03 PT (cross-entropy train losses 0.2896, 0.2284, 0.2308, 0.3803 at the four learning rates).
- 11-08-2026 17:53 PST -- Codex took over the finished study at `a1fd697`; verified all nine supplied artifacts against Git and the remote branch, replaced stale RUNNING state, retained the prior log, and changed no trainer output. No new run, cluster connection, email, or pull request.
- 11-08-2026 20:49 PST -- Completed the requested five-part synthesis, exact saved-record FineWeb-Edu paired statistics and collapsed-recall intervals; verified 84 native Google table entries and preserved existing document text. Added reproducible analysis scripts; no new runs, original-output changes, cluster access, or email.

- 11-09-2026 10:10 PST -- Brando requested saved-curve figures and publication to main. Merged current origin/main in the designated study tree and pushed `76ee3fc`; remote main verified. All 39 study additions are confined to this experiment; main-only paths and nine frozen evidence files are preserved. Original local main checkout remains unchanged.

- 11-09-2026 10:22 PST -- Added reproducible saved-curve analysis, FineWeb-Edu overview/late figures, three exploratory window contrasts and descriptive slopes. All six available curves validated; numerical reproduction and deterministic constant/linear controls passed. WikiText panels remain explicitly missing while guarded retrieval waits; no new training or evaluation.

- 11-09-2026 10:27 PST -- Partial curve analysis published to main at `c5454c2`; Google Assistant Results now contains the matching pending-WikiText status, methods, sensitivity statistics, figure links and a native 15-entry table. Readback verified all entries and preserved all 929 pre-existing paragraphs. Corrected the derived analysis-script hash after SVG whitespace normalization; no numerical value changed. Retrieval in session `29073` remains pending the shared guard.

- 11-09-2026 10:55 PST -- Claude assisted with local connection diagnosis using the existing subscription and full permissions; its findings were written before the CLI turn cap. Root and a native evidence worker verified the logs and corrected the earlier automatic-reconnect attribution: the admitted client is unknown, and stale previousError fields do not prove a prolonged handshake. The ordinary gateway route and local Kerberos ticket are present. One root retrieval remains active; no extra target connection or model run was started.

- 11-09-2026 11:15 PST -- The 11:12 window produced another unidentified reservation and no reusable socket. Root waiter 29073 failed locally on status-lock contention before SSH. Fixed this race without changing the shared guard or settings; five network-free regression checks passed. Replacement waiter 11616 is active for 12:12:15 PST. A read-only worker is tracing competing local owners; no extra target attempt was made.

- 11-09-2026 11:20 PST -- Identified the 11:12 reservation owner from another local task's saved one-shot receipt: its mercury1 SSH exited 255 after about 5.3 seconds, with no stdout and only an UNKNOWN-port closure message. No persistent competing waiter was found. The existing private Sherlock bridge reports an online, idle brando9 runner; a bounded identity/host-policy read is being verified before considering it as an alternative read-only access route. The shared SNAP guard remains unchanged.

- 11-09-2026 11:28 PST -- Verified the existing private Sherlock bridge as brando9 on compute node sh03-01n70.int, Slurm allocation 48402562, and read current host policy. Local metadata checks found no SNAP socket, no home keytab or SSH config, and no kinit/klist on the current PATH. No fresh SNAP connection or credential transfer was attempted through it. The corrected Mac waiter 11616 remains the only retrieval owner.

- 11-09-2026 11:33 PST -- Added optional full private SSH diagnostics, including partial output on timeout. The task-local proxy wrapper executes the unchanged canonical guard and adds only gateway verbosity; checked its child arguments against the canonical command. Nine network-free regression checks passed. Replaced only the owned waiting process (11616, exit 143) with diagnostic waiter 68842 for the same 12:12:15 window; no connection was in progress and no fresh target attempt was made.
