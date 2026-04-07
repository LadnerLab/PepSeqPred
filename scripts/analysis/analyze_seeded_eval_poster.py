"""analyze_seeded_eval_poster.py

Generates statistical results and plots for poster presentation.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Tuple, List, Dict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _f1(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall):
        return float("nan")
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / denom


def _bootstrap_ci_mean(
    values: List[float],
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    arr = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        x = float(arr[0])
        return x, x
    idx = rng.integers(0, arr.size, size=(
        n_bootstrap, arr.size), endpoint=False)
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def _two_sided_sign_test_pvalue(diffs: List[float]) -> float:
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, p))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_name(index: int) -> str:
    return f"set_{index:02d}"


def _iter_seed_rows(
    base_dir: Path,
    models: List[str],
    set_start: int,
    set_end: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    seed_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for model in models:
        for set_index in range(set_start, set_end + 1):
            set_dir = base_dir / model / _set_name(set_index) / "combined"
            eval_json = set_dir / "evaluation" / "ffnn_eval_summary.json"
            compare_json = set_dir / "peptide_compare" / "peptide_comparison_summary.json"
            if not eval_json.exists():
                warnings.append(f"missing eval summary: {eval_json}")
                continue

            payload = _load_json(eval_json)
            evaluation = payload.get("evaluation", {}) or {}
            metrics = evaluation.get("metrics", {}) or {}
            best_fold = evaluation.get("best_fold", {}) or {}
            best_metrics = best_fold.get("metrics", {}) or {}

            support_pos = _safe_float(metrics.get("support_pos"))
            support_neg = _safe_float(metrics.get("support_neg"))
            support_total = support_pos + support_neg
            prevalence = (
                support_pos /
                support_total if support_total > 0.0 else float("nan")
            )

            overall_pr_auc = _safe_float(metrics.get("pr_auc"))
            best_pr_auc = _safe_float(best_metrics.get("pr_auc"))

            row: Dict[str, Any] = {
                "model": model,
                "set_index": set_index,
                "n_members": _safe_float(payload.get("n_members")),
                "best_fold_index": _safe_float(best_fold.get("fold_index")),
                "valid_residues": _safe_float(evaluation.get("valid_residues")),
                "pos_residues": _safe_float(evaluation.get("pos_residues")),
                "neg_residues": _safe_float(evaluation.get("neg_residues")),
                "prevalence": prevalence,
                "overall_auc": _safe_float(metrics.get("auc")),
                "overall_pr_auc": overall_pr_auc,
                "overall_pr_auc_trapz": _safe_float(metrics.get("pr_auc_trapz")),
                "overall_pr_lift": (
                    overall_pr_auc / prevalence
                    if prevalence > 0.0 and math.isfinite(overall_pr_auc)
                    else float("nan")
                ),
                "best_fold_auc": _safe_float(best_metrics.get("auc")),
                "best_fold_pr_auc": best_pr_auc,
                "best_fold_pr_auc_trapz": _safe_float(best_metrics.get("pr_auc_trapz")),
                "best_fold_pr_lift": (
                    best_pr_auc / prevalence
                    if prevalence > 0.0 and math.isfinite(best_pr_auc)
                    else float("nan")
                ),
                "overall_precision": _safe_float(metrics.get("precision")),
                "overall_recall": _safe_float(metrics.get("recall")),
                "overall_f1": _safe_float(metrics.get("f1")),
                "overall_mcc": _safe_float(metrics.get("mcc"))
            }

            if compare_json.exists():
                compare_payload = _load_json(compare_json)
                conf = compare_payload.get(
                    "confusion_peptide_any_positive", {}) or {}
                pep_precision = _safe_float(conf.get("precision"))
                pep_recall = _safe_float(conf.get("recall"))
                row.update(
                    {
                        "peptide_n_rows": _safe_float(compare_payload.get("n_rows_compared")),
                        "peptide_reactive": _safe_float(compare_payload.get("n_reactive")),
                        "peptide_nonreactive": _safe_float(
                            compare_payload.get("n_nonreactive")
                        ),
                        "peptide_tp": _safe_float(conf.get("tp")),
                        "peptide_fp": _safe_float(conf.get("fp")),
                        "peptide_tn": _safe_float(conf.get("tn")),
                        "peptide_fn": _safe_float(conf.get("fn")),
                        "peptide_precision": pep_precision,
                        "peptide_recall": pep_recall,
                        "peptide_f1": _f1(pep_precision, pep_recall),
                        "peptide_accuracy": _safe_float(conf.get("accuracy")),
                        "peptide_exact_match_rate": _safe_float(
                            compare_payload.get("exact_match_rate")
                        ),
                        "peptide_mean_pred_ones": _safe_float(
                            compare_payload.get("mean_pred_ones_overall")
                        )
                    }
                )
            else:
                warnings.append(f"missing peptide summary: {compare_json}")

            seed_rows.append(row)

            for fold in evaluation.get("folds", []) or []:
                fold_metrics = fold.get("metrics", {}) or {}
                fold_rows.append(
                    {
                        "model": model,
                        "set_index": set_index,
                        "fold_index": _safe_float(fold.get("fold_index")),
                        "auc": _safe_float(fold_metrics.get("auc")),
                        "pr_auc": _safe_float(fold_metrics.get("pr_auc")),
                        "pr_auc_trapz": _safe_float(fold_metrics.get("pr_auc_trapz")),
                        "precision": _safe_float(fold_metrics.get("precision")),
                        "recall": _safe_float(fold_metrics.get("recall")),
                        "f1": _safe_float(fold_metrics.get("f1")),
                        "mcc": _safe_float(fold_metrics.get("mcc"))
                    }
                )

    return seed_rows, fold_rows, warnings


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _model_values(rows: List[Dict[str, Any]], models: List[str], metric: str) -> List[List[float]]:
    out: List[List[float]] = []
    for model in models:
        vals = [
            _safe_float(r.get(metric))
            for r in rows
            if r.get("model") == model and math.isfinite(_safe_float(r.get(metric)))
        ]
        out.append(vals)
    return out


def _plot_box_scatter(
    rows: List[Dict[str, Any]],
    models: List[str],
    metric: str,
    title: str,
    ylabel: str,
    out_base: Path,
    rng: np.random.Generator
) -> None:
    series = _model_values(rows, models=models, metric=metric)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    positions = np.arange(1, len(models) + 1, dtype=float)
    valid_data = [vals if len(vals) > 0 else [float("nan")] for vals in series]
    bp = ax.boxplot(
        valid_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=False,
    )

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.35)

    for i, vals in enumerate(series):
        if len(vals) == 0:
            continue
        x = positions[i] + rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=40,
            alpha=0.9,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

    ax.set_xticks(positions, labels=models)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(str(out_base) + f".{ext}", dpi=240)
    plt.close(fig)


def _plot_seed_paired_lines(
    rows: List[Dict[str, Any]],
    model_left: str,
    model_right: str,
    metric: str,
    title: str,
    ylabel: str,
    out_base: Path
) -> None:
    by_model_set: Dict[Tuple[str, int], float] = {}
    for row in rows:
        key = (str(row.get("model")), int(_safe_float(row.get("set_index"))))
        val = _safe_float(row.get(metric))
        if math.isfinite(val):
            by_model_set[key] = val

    paired: List[Tuple[int, float, float]] = []
    for set_index in sorted({k[1] for k in by_model_set.keys()}):
        a = by_model_set.get((model_left, set_index))
        b = by_model_set.get((model_right, set_index))
        if a is None or b is None:
            continue
        paired.append((set_index, a, b))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    x_left = 1.0
    x_right = 2.0
    for set_index, a, b in paired:
        color = "#54A24B" if b >= a else "#E45756"
        ax.plot([x_left, x_right], [a, b],
                color=color, alpha=0.75, linewidth=1.4)
        ax.scatter([x_left, x_right], [a, b], color=[
                   "#4C78A8", "#F58518"], s=24)
        ax.text(x_right + 0.03, b, f"{set_index:02d}", fontsize=7, va="center")

    ax.set_xticks([x_left, x_right], labels=[model_left, model_right])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(str(out_base) + f".{ext}", dpi=240)
    plt.close(fig)


def _plot_fold_metric(
    rows: List[Dict[str, Any]],
    models: List[str],
    metric: str,
    title: str,
    ylabel: str,
    out_base: Path,
    rng: np.random.Generator
) -> None:
    _plot_box_scatter(
        rows=rows,
        models=models,
        metric=metric,
        title=title,
        ylabel=ylabel,
        out_base=out_base,
        rng=rng
    )


def _plot_peptide_grouped_bars(
    rows: List[Dict[str, Any]],
    models: List[str],
    out_base: Path
) -> None:
    metrics = ["peptide_precision", "peptide_recall", "peptide_f1"]
    labels = ["Precision", "Recall", "F1"]

    means: Dict[str, List[float]] = {m: [] for m in models}
    stds: Dict[str, List[float]] = {m: [] for m in models}

    for model in models:
        model_rows = [r for r in rows if r.get("model") == model]
        for metric in metrics:
            vals = np.asarray(
                [
                    _safe_float(r.get(metric))
                    for r in model_rows
                    if math.isfinite(_safe_float(r.get(metric)))
                ],
                dtype=float,
            )
            if vals.size == 0:
                means[model].append(float("nan"))
                stds[model].append(float("nan"))
            else:
                means[model].append(float(vals.mean()))
                stds[model].append(float(vals.std(ddof=1))
                                   if vals.size > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(metrics), dtype=float)
    width = 0.34
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means[model],
            width=width,
            yerr=stds[model],
            label=model,
            color=colors[i % len(colors)],
            alpha=0.85,
            capsize=4
        )

    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Peptide-level score")
    ax.set_title("Peptide Metrics Across Seeded Runs (any-positive rule)")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(str(out_base) + f".{ext}", dpi=240)
    plt.close(fig)


def _paired_metric_summary(
    rows: List[Dict[str, Any]],
    model_a: str,
    model_b: str,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int
) -> Dict[str, Any]:
    map_a: Dict[int, float] = {}
    map_b: Dict[int, float] = {}

    for row in rows:
        model = str(row.get("model"))
        set_index = int(_safe_float(row.get("set_index")))
        value = _safe_float(row.get(metric))
        if not math.isfinite(value):
            continue
        if model == model_a:
            map_a[set_index] = value
        elif model == model_b:
            map_b[set_index] = value

    common_sets = sorted(set(map_a.keys()) & set(map_b.keys()))
    diffs = [map_b[s] - map_a[s] for s in common_sets]

    mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else float("nan")
    median_diff = float(np.median(diffs)) if len(diffs) > 0 else float("nan")
    ci_lo, ci_hi = _bootstrap_ci_mean(diffs, rng=rng, n_bootstrap=n_bootstrap)
    p_sign = _two_sided_sign_test_pvalue(diffs)

    return {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_pairs": len(diffs),
        "mean_diff_b_minus_a": mean_diff,
        "median_diff_b_minus_a": median_diff,
        "bootstrap_ci95_low": ci_lo,
        "bootstrap_ci95_high": ci_hi,
        "sign_test_pvalue": p_sign,
        "paired_sets": common_sets
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate seeded FFNN evaluation outputs and generate poster-ready plots."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("localdata/evals/cocci_eval/seeded_runs"),
        help="Seeded evaluation base directory."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="flagship1,flagship2",
        help="Comma-separated model directory names."
    )
    parser.add_argument(
        "--set-start",
        type=int,
        default=1,
        help="Starting set index (inclusive)."
    )
    parser.add_argument(
        "--set-end",
        type=int,
        default=10,
        help="Ending set index (inclusive)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <base-dir>/poster_analysis)."
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Bootstrap draws for paired CI estimates."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jitter and bootstrap sampling."
    )
    args = parser.parse_args()

    if args.set_start < 1 or args.set_end < args.set_start:
        raise ValueError("Invalid --set-start/--set-end range")

    models = [token.strip() for token in str(
        args.models).split(",") if token.strip()]
    if len(models) < 1:
        raise ValueError("--models resolved to empty List")

    base_dir = args.base_dir
    out_dir = args.out_dir or (base_dir / "poster_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    seed_rows, fold_rows, warnings = _iter_seed_rows(
        base_dir=base_dir,
        models=models,
        set_start=args.set_start,
        set_end=args.set_end
    )

    if len(seed_rows) == 0:
        raise SystemExit("No seeded evaluation rows found. Nothing to plot.")

    _write_csv(out_dir / "seed_metrics.csv", seed_rows)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)

    _plot_box_scatter(
        rows=seed_rows,
        models=models,
        metric="best_fold_auc",
        title="Best-Fold ROC AUC by Model (10 seeded runs)",
        ylabel="ROC AUC",
        out_base=out_dir / "best_fold_roc_auc_by_model",
        rng=rng
    )
    _plot_box_scatter(
        rows=seed_rows,
        models=models,
        metric="best_fold_pr_auc",
        title="Best-Fold PR AUC by Model (10 seeded runs)",
        ylabel="PR AUC",
        out_base=out_dir / "best_fold_pr_auc_by_model",
        rng=rng
    )
    _plot_box_scatter(
        rows=seed_rows,
        models=models,
        metric="best_fold_pr_lift",
        title="Best-Fold PR Lift vs Prevalence by Model",
        ylabel="PR AUC / prevalence",
        out_base=out_dir / "best_fold_pr_lift_by_model",
        rng=rng
    )

    _plot_fold_metric(
        rows=fold_rows,
        models=models,
        metric="auc",
        title="Fold-Level ROC AUC Distribution (all folds)",
        ylabel="ROC AUC",
        out_base=out_dir / "fold_roc_auc_by_model",
        rng=rng
    )
    _plot_fold_metric(
        rows=fold_rows,
        models=models,
        metric="pr_auc",
        title="Fold-Level PR AUC Distribution (all folds)",
        ylabel="PR AUC",
        out_base=out_dir / "fold_pr_auc_by_model",
        rng=rng
    )

    if len(models) >= 2:
        left = models[0]
        right = models[1]
        _plot_seed_paired_lines(
            rows=seed_rows,
            model_left=left,
            model_right=right,
            metric="best_fold_auc",
            title=f"Paired Seed Comparison: Best-Fold ROC AUC ({left} -> {right})",
            ylabel="ROC AUC",
            out_base=out_dir / "paired_seed_best_fold_roc_auc"
        )
        _plot_seed_paired_lines(
            rows=seed_rows,
            model_left=left,
            model_right=right,
            metric="best_fold_pr_auc",
            title=f"Paired Seed Comparison: Best-Fold PR AUC ({left} -> {right})",
            ylabel="PR AUC",
            out_base=out_dir / "paired_seed_best_fold_pr_auc"
        )

    _plot_peptide_grouped_bars(
        rows=seed_rows,
        models=models,
        out_base=out_dir / "peptide_metrics_by_model"
    )

    stats_rows: List[Dict[str, Any]] = []
    if len(models) >= 2:
        a = models[0]
        b = models[1]
        for metric in (
            "best_fold_auc",
            "best_fold_pr_auc",
            "best_fold_pr_lift",
            "overall_auc",
            "overall_pr_auc",
            "peptide_precision",
            "peptide_recall",
            "peptide_f1"
        ):
            stats_rows.append(
                _paired_metric_summary(
                    rows=seed_rows,
                    model_a=a,
                    model_b=b,
                    metric=metric,
                    rng=rng,
                    n_bootstrap=args.n_bootstrap
                )
            )

    _write_csv(out_dir / "paired_stats.csv", stats_rows)
    (out_dir / "warnings.txt").write_text("\n".join(warnings), encoding="utf-8")

    summary_payload = {
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "models": models,
        "set_range": [args.set_start, args.set_end],
        "n_seed_rows": len(seed_rows),
        "n_fold_rows": len(fold_rows),
        "n_warnings": len(warnings),
        "files": {
            "seed_metrics_csv": str(out_dir / "seed_metrics.csv"),
            "fold_metrics_csv": str(out_dir / "fold_metrics.csv"),
            "paired_stats_csv": str(out_dir / "paired_stats.csv"),
            "warnings_txt": str(out_dir / "warnings.txt")
        }
    }
    (out_dir / "poster_analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8"
    )

    print(
        "poster-analysis complete "
        f"seed_rows={len(seed_rows)} fold_rows={len(fold_rows)} "
        f"warnings={len(warnings)} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
