import { formatFixed, formatSci, formatSigned } from "@/lib/format";
import { externalCocciData } from "@/lib/manualData";

const metricLabels: Record<string, string> = {
  best_fold_auc: "Best-fold ROC AUC",
  best_fold_pr_auc: "Best-fold PR-AUC",
  best_fold_pr_lift: "Best-fold PR lift",
  overall_auc: "Overall ROC AUC",
  overall_pr_auc: "Overall PR-AUC",
  peptide_precision: "Peptide precision",
  peptide_recall: "Peptide recall",
  peptide_f1: "Peptide F1"
};

export default function ExternalCocciPage(): JSX.Element {
  return (
    <section className="section alt">
      <div className="wrap">
        <h1>{externalCocciData.title}</h1>
        <p className="lead">{externalCocciData.description}</p>

        <div className="note">
          Set range: {externalCocciData.setRange[0]}-{externalCocciData.setRange[1]} | Seed rows:{" "}
          {externalCocciData.nSeedRows} | Fold rows: {externalCocciData.nFoldRows}
        </div>

        <div className="two-col">
          <article className="card">
            <h3>Flagship1 means</h3>
            <ul>
              <li>Best-fold PR-AUC: {formatFixed(externalCocciData.models.flagship1.bestFoldPrAucMean, 6)}</li>
              <li>Best-fold PR lift: {formatFixed(externalCocciData.models.flagship1.bestFoldPrLiftMean, 3)}</li>
              <li>Best-fold ROC AUC: {formatFixed(externalCocciData.models.flagship1.bestFoldAucMean, 6)}</li>
              <li>Overall PR-AUC: {formatFixed(externalCocciData.models.flagship1.overallPrAucMean, 6)}</li>
            </ul>
          </article>
          <article className="card">
            <h3>Flagship2 means</h3>
            <ul>
              <li>Best-fold PR-AUC: {formatFixed(externalCocciData.models.flagship2.bestFoldPrAucMean, 6)}</li>
              <li>Best-fold PR lift: {formatFixed(externalCocciData.models.flagship2.bestFoldPrLiftMean, 3)}</li>
              <li>Best-fold ROC AUC: {formatFixed(externalCocciData.models.flagship2.bestFoldAucMean, 6)}</li>
              <li>Overall PR-AUC: {formatFixed(externalCocciData.models.flagship2.overallPrAucMean, 6)}</li>
            </ul>
          </article>
        </div>

        <h2>Paired Statistics (flagship2 - flagship1)</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>n pairs</th>
                <th>Mean delta</th>
                <th>Median delta</th>
                <th>95% bootstrap CI</th>
                <th>Sign-test p</th>
              </tr>
            </thead>
            <tbody>
              {externalCocciData.pairedStats.map((row) => (
                <tr key={row.metric}>
                  <td data-label="Metric">{metricLabels[row.metric] ?? row.metric}</td>
                  <td data-label="n pairs">{row.nPairs}</td>
                  <td data-label="Mean delta">{formatSigned(row.meanDiffBMinusA, 6)}</td>
                  <td data-label="Median delta">{formatSigned(row.medianDiffBMinusA, 6)}</td>
                  <td data-label="95% bootstrap CI">
                    {formatSigned(row.ciLow, 6)} to {formatSigned(row.ciHigh, 6)}
                  </td>
                  <td data-label="Sign-test p">{formatSci(row.signTestPValue, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
