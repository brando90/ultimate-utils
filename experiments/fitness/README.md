# Fitness Tracking

Personal fitness / workout tracking experiment.

For now this lives inside `ultimate-utils` under `experiments/fitness/`. It may be
split out into its own dedicated repo later — keeping it here is just a convenient
starting point.

## Structure

- `logs/` — one Markdown file per workout day (`YYYY-MM-DD.md`), transcribed from
  handwritten notes or logged directly.
- `images/` — original photos / scans of handwritten logs, named `YYYY-MM-DD_*.jpg`.

## Log format

Each workout log records, per exercise:

| Field | Meaning |
| --- | --- |
| Exercise | Movement / machine |
| Sets x Reps | e.g. `3 x 12` |
| Weight | Load (lb unless noted); `?` marks uncertain values from the photo |

A `?` next to a transcribed value means the handwriting was ambiguous — double-check
against the source image in `images/`.
