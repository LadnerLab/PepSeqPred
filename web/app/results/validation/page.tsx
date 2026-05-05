import { assetPath, formatFixed, formatPercent } from "@/lib/format";
import { phaseBValidationData } from "@/lib/manualData";

export default function ValidationPage(): JSX.Element {
  const { summaries } = phaseBValidationData;

  return (
    <section className="section">
      <div className="wrap">
        <h1>Latest PhaseB Validation</h1>
        <p className="lead">
          This page intentionally uses the most recent PhaseB validation run only:
          <code> {phaseBValidationData.runId}</code> (last modified {phaseBValidationData.lastModifiedDate}).
          Multi-dataset PhaseC validation is intentionally excluded until stability is established.
        </p>

        <div className="metric-grid">
          <article className="metric">
            <p className="label">Train mode</p>
            <p className="value">{phaseBValidationData.trainMode}</p>
          </article>
          <article className="metric">
            <p className="label">Split type</p>
            <p className="value">{phaseBValidationData.splitType}</p>
          </article>
          <article className="metric">
            <p className="label">Runs / Folds</p>
            <p className="value">
              {phaseBValidationData.nRuns} / {phaseBValidationData.nFolds}
            </p>
          </article>
          <article className="metric">
            <p className="label">PR baseline mean</p>
            <p className="value">{formatPercent(phaseBValidationData.prBaselineMean, 3)}</p>
          </article>
        </div>

        <div className="note">
          Split seed(s): {phaseBValidationData.splitSeeds.join(", ")} | Train seed(s):{" "}
          {phaseBValidationData.trainSeeds.join(", ")} | Best model metric: {phaseBValidationData.bestModelMetric}
        </div>

        <h2>Run-Level Summary (n=5 folds)</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td data-label="Metric">PR-AUC</td>
                <td data-label="Mean">{formatFixed(summaries.prAuc.mean, 6)}</td>
                <td data-label="Std">{formatFixed(summaries.prAuc.std, 6)}</td>
                <td data-label="Min">{formatFixed(summaries.prAuc.min, 6)}</td>
                <td data-label="Max">{formatFixed(summaries.prAuc.max, 6)}</td>
              </tr>
              <tr>
                <td data-label="Metric">F1</td>
                <td data-label="Mean">{formatFixed(summaries.f1.mean, 6)}</td>
                <td data-label="Std">{formatFixed(summaries.f1.std, 6)}</td>
                <td data-label="Min">{formatFixed(summaries.f1.min, 6)}</td>
                <td data-label="Max">{formatFixed(summaries.f1.max, 6)}</td>
              </tr>
              <tr>
                <td data-label="Metric">MCC</td>
                <td data-label="Mean">{formatFixed(summaries.mcc.mean, 6)}</td>
                <td data-label="Std">{formatFixed(summaries.mcc.std, 6)}</td>
                <td data-label="Min">{formatFixed(summaries.mcc.min, 6)}</td>
                <td data-label="Max">{formatFixed(summaries.mcc.max, 6)}</td>
              </tr>
              <tr>
                <td data-label="Metric">ROC AUC</td>
                <td data-label="Mean">{formatFixed(summaries.auc.mean, 6)}</td>
                <td data-label="Std">{formatFixed(summaries.auc.std, 6)}</td>
                <td data-label="Min">{formatFixed(summaries.auc.min, 6)}</td>
                <td data-label="Max">{formatFixed(summaries.auc.max, 6)}</td>
              </tr>
              <tr>
                <td data-label="Metric">AUC10</td>
                <td data-label="Mean">{formatFixed(summaries.auc10.mean, 6)}</td>
                <td data-label="Std">{formatFixed(summaries.auc10.std, 6)}</td>
                <td data-label="Min">{formatFixed(summaries.auc10.min, 6)}</td>
                <td data-label="Max">{formatFixed(summaries.auc10.max, 6)}</td>
              </tr>
              <tr>
                <td data-label="Metric">Balanced accuracy</td>
                <td data-label="Mean">{formatFixed(summaries.balancedAcc.mean, 6)}</td>
                <td data-label="Std">{formatFixed(summaries.balancedAcc.std, 6)}</td>
                <td data-label="Min">{formatFixed(summaries.balancedAcc.min, 6)}</td>
                <td data-label="Max">{formatFixed(summaries.balancedAcc.max, 6)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h2>Fold-Level Best Checkpoints</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Fold</th>
                <th>Best epoch</th>
                <th>PR-AUC</th>
                <th>F1</th>
                <th>MCC</th>
                <th>AUC</th>
                <th>AUC10</th>
                <th>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {phaseBValidationData.foldMetrics.map((row) => (
                <tr key={row.fold}>
                  <td data-label="Fold">{row.fold}</td>
                  <td data-label="Best epoch">{row.bestEpoch}</td>
                  <td data-label="PR-AUC">{formatFixed(row.prAuc, 6)}</td>
                  <td data-label="F1">{formatFixed(row.f1, 6)}</td>
                  <td data-label="MCC">{formatFixed(row.mcc, 6)}</td>
                  <td data-label="AUC">{formatFixed(row.auc, 6)}</td>
                  <td data-label="AUC10">{formatFixed(row.auc10, 6)}</td>
                  <td data-label="Threshold">{formatFixed(row.threshold, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="gallery">
          {phaseBValidationData.plots.map((plot) => (
            <figure key={plot.src}>
              <img src={assetPath(plot.src)} alt={plot.alt} />
              <figcaption>{plot.caption}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </section>
  );
}
