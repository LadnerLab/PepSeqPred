import Link from "next/link";

export default function HomePage(): JSX.Element {
  return (
    <>
      <section className="hero">
        <div className="wrap hero-grid">
          <div>
            <p className="eyebrow">PepSeqPred</p>
            <h1>Residue-level epitope prediction with reproducible evidence.</h1>
            <p className="lead">
              PepSeqPred predicts epitope masks for protein sequences and supports full developer workflows for
              preprocessing, training, evaluation, and HPC execution.
            </p>
            <div className="chip-row">
              <span className="chip">Python 3.12+</span>
              <span className="chip">ESM2 embeddings</span>
              <span className="chip">DDP-ready training</span>
              <span className="chip">Seeded evaluation snapshots</span>
            </div>
            <p className="hero-links">
              <a href="https://pypi.org/project/pepseqpred/" target="_blank" rel="noreferrer">
                PyPI
              </a>
              <a href="https://github.com/LadnerLab/PepSeqPred" target="_blank" rel="noreferrer">
                GitHub
              </a>
              <a href="https://github.com/LadnerLab/PepSeqPred/blob/main/README.md" target="_blank" rel="noreferrer">
                Developer README
              </a>
            </p>
            <p className="hero-links">
              <Link href="/results/comparison">Model Comparison</Link>
              <Link href="/results/validation">PhaseB Validation</Link>
              <Link href="/results/external-cocci">External Cocci</Link>
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <h2>Install</h2>
          <pre>
            <code>pip install pepseqpred</code>
          </pre>
        </div>
      </section>

      <section className="section alt">
        <div className="wrap">
          <h2>Quickstart APIs</h2>
          <div className="two-col">
            <article>
              <h3>Pretrained API</h3>
              <pre>
                <code>{`from pepseqpred import load_pretrained_predictor

predictor = load_pretrained_predictor(
    model_id="default",
    device="auto"
)
result = predictor.predict_sequence(
    "ACDEFGHIKLMNPQRSTVWY",
    header="example_protein"
)
print(result.binary_mask)`}</code>
              </pre>
            </article>
            <article>
              <h3>Artifact-path API</h3>
              <pre>
                <code>{`from pepseqpred import load_predictor

predictor = load_predictor(
    model_artifact="path/to/ensemble_manifest.json",
    device="auto"
)
result = predictor.predict_sequence(
    "ACDEFGHIKLMNPQRSTVWY"
)
print(result.binary_mask)`}</code>
              </pre>
            </article>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <h2>Why PepSeqPred</h2>
          <p className="lead">
            PepSeqPred is built for reproducible residue-level prediction under strong class imbalance. Training and
            validation are documented as protocol-first; numeric scorecards are based on seeded evaluation snapshots.
          </p>
          <div className="three-col">
            <article className="card">
              <h3>Training</h3>
              <ul>
                <li>Ensemble-kfold and seeded runs with deterministic split/train seeds.</li>
                <li>ID-family-aware splitting to reduce leakage risk across related proteins.</li>
                <li>DistributedDataParallel support for multi-GPU HPC workflows.</li>
                <li>Run artifacts include checkpoints, manifests, and run-level CSV/JSON outputs.</li>
              </ul>
            </article>
            <article className="card">
              <h3>Validation</h3>
              <ul>
                <li>Checkpoint selection records threshold, PR-AUC, F1, MCC, AUC, and AUC10.</li>
                <li>Threshold policy maximizes recall subject to minimum precision constraints.</li>
                <li>Validation metrics are captured per run with explicit seed provenance.</li>
                <li>PhaseB validation scorecards are shown directly from manually curated run artifacts.</li>
              </ul>
            </article>
            <article className="card">
              <h3>Evaluation</h3>
              <ul>
                <li>Seeded external Cocci evaluation compares flagship models across sets 1-10.</li>
                <li>Class prevalence is very low, so PR metrics are emphasized over accuracy.</li>
                <li>Paired set statistics include bootstrap confidence intervals and sign tests.</li>
                <li>Frozen benchmark snapshot remains available as a curated web evidence source.</li>
              </ul>
            </article>
          </div>
        </div>
      </section>
    </>
  );
}
