export default function ReproducibilityPage(): JSX.Element {
  return (
    <section className="section">
      <div className="wrap">
        <h1>Reproducibility</h1>
        <p className="lead">
          PepSeqPred is fully open-source. Pipeline code, scripts, and curated website evidence sources are all in the
          same repository for transparent review.
        </p>

        <div className="two-col">
          <article className="card">
            <h3>Repository Source</h3>
            <ul>
              <li>
                <a href="https://github.com/LadnerLab/PepSeqPred" target="_blank" rel="noreferrer">
                  PepSeqPred repository
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/LadnerLab/PepSeqPred/tree/main/src/pepseqpred/core"
                  target="_blank"
                  rel="noreferrer"
                >
                  Core pipeline modules
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/LadnerLab/PepSeqPred/tree/main/src/pepseqpred/apps"
                  target="_blank"
                  rel="noreferrer"
                >
                  CLI application entrypoints
                </a>
              </li>
            </ul>
          </article>

          <article className="card">
            <h3>Curated Web Evidence</h3>
            <ul>
              <li>Flagship comparison: curated benchmark snapshot from seeded Cocci evaluation artifacts.</li>
              <li>
                Validation view: manually curated latest PhaseB run (
                <code>ffnn_ens_2.4_v100_28234651</code>).
              </li>
              <li>External Cocci view: manually curated paired statistics and means across sets 1-10.</li>
            </ul>
          </article>
        </div>
      </div>
    </section>
  );
}
