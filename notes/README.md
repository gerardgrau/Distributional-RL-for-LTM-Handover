# notes/

Research journal for the project: the running record of experiments, focused
studies, meetings, and planning that sits behind the manuscript in `paper/` and
the reference docs in `docs/`. These are working documents kept in roughly
chronological form; they capture *why* decisions were made, not the final
architecture (that lives in `CLAUDE.md` and `docs/`).

## Layout

| Folder | Contents |
| --- | --- |
| `experiments/` | Dated run-logs and result write-ups (overnight queues, `RESULTS_*` summaries, the running experiments log). |
| `studies/` | Focused analyses that fed the paper: quantile-count and quantile-mode studies, the hyperparameter search, the ablation/risk findings, and the ranking-metric analysis. |
| `meetings/` | Tutor-meeting notes and the `2026-05-18` meeting report (with its plots and behaviour animations). |
| `planning/` | Forward-looking lists: the live task list, the paper-revision to-do, and the deferred / to-repeat experiment backlog. |
| `figures/` | Figure outputs from the analysis tools (`plot_risk_frontier.py`, `plot_per_ue_tails.py`, `plot_return_distributions.py`). |

## Not tracked

`conversation-summary.md` and `meeting-notes.md` are personal scratch files,
kept out of version control via `.gitignore`. The structured, reusable
documentation (architecture, parity audit, future improvements) lives in
`docs/`; the manuscript source lives in `paper/`.
