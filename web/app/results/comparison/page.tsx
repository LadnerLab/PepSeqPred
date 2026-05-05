import { assetPath, formatFixed, formatPercent, formatSci, formatSigned } from "@/lib/format";
import { modelComparisonData } from "@/lib/manualData";

export default function ComparisonPage() {
  return (
    <section className="section alt">
      <div className="wrap">
        <h1>Flagship2 vs Flagship1 Results</h1>
        <p className="lead">
          Headline metrics below are means across seeded evaluation sets and paired deltas (flagship2 - flagship1).
          PR metrics are primary because the positive class is rare.
        </p>

        <div className="note">
          Positive-class prevalence in this snapshot: <strong>{formatPercent(modelComparisonData.prevalenceMean, 4)}</strong>.
          Accuracy can look high under imbalance; PR-AUC and PR lift are more informative for epitope recovery quality.
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Flagship1 mean</th>
                <th>Flagship2 mean</th>
                <th>Delta (F2 - F1)</th>
                <th>95% bootstrap CI</th>
                <th>Sign-test p</th>
              </tr>
            </thead>
            <tbody>
              {modelComparisonData.metrics.map((metric) => (
                <tr key={metric.id}>
                  <td data-label="Metric">
                    {metric.label}
                    {metric.primary ? " *" : ""}
                  </td>
                  <td data-label="Flagship1 mean">{formatFixed(metric.flagship1Mean, metric.id.includes("lift") ? 3 : 6)}</td>
                  <td data-label="Flagship2 mean">{formatFixed(metric.flagship2Mean, metric.id.includes("lift") ? 3 : 6)}</td>
                  <td data-label="Delta (F2 - F1)">
                    {formatSigned(metric.delta, metric.id.includes("lift") ? 3 : 6)}
                  </td>
                  <td data-label="95% bootstrap CI">
                    {formatSigned(metric.ciLow, metric.id.includes("lift") ? 3 : 6)} to{" "}
                    {formatSigned(metric.ciHigh, metric.id.includes("lift") ? 3 : 6)}
                  </td>
                  <td data-label="Sign-test p">{formatSci(metric.signTestPValue, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="gallery">
          {modelComparisonData.gallery.map((plot) => (
            <figure key={plot.src}>
              <img src={assetPath(plot.src)} alt={plot.alt} />
              <figcaption>{plot.caption}</figcaption>
            </figure>
          ))}
        </div>

        <p className="muted">
          Snapshot date: {modelComparisonData.snapshotDate}. Generated: {modelComparisonData.generatedAtUtc}.
        </p>
      </div>
    </section>
  );
}
