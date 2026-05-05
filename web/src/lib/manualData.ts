export type ComparisonMetric = {
  id: "best_fold_pr_auc" | "best_fold_pr_lift" | "best_fold_auc" | "overall_pr_auc";
  label: string;
  primary: boolean;
  flagship1Mean: number;
  flagship2Mean: number;
  delta: number;
  ciLow: number;
  ciHigh: number;
  signTestPValue: number;
};

export type ModelComparisonData = {
  snapshotDate: string;
  generatedAtUtc: string;
  prevalenceMean: number;
  validResiduesMean: number;
  posResiduesMean: number;
  negResiduesMean: number;
  metrics: ComparisonMetric[];
  gallery: Array<{ src: string; alt: string; caption: string }>;
};

export type PhaseBRunMetricSummary = {
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
};

export type PhaseBFoldMetric = {
  fold: number;
  bestEpoch: number;
  precision: number;
  recall: number;
  f1: number;
  mcc: number;
  auc: number;
  prAuc: number;
  auc10: number;
  threshold: number;
};

export type PhaseBValidationData = {
  runId: string;
  lastModifiedDate: string;
  trainMode: string;
  splitType: string;
  bestModelMetric: string;
  ensembleSeedMode: string;
  splitSeeds: number[];
  trainSeeds: number[];
  nRuns: number;
  nFolds: number;
  nSets: number;
  prBaselineMean: number;
  plotStatus: string;
  plotFormats: string[];
  summaries: {
    prAuc: PhaseBRunMetricSummary;
    f1: PhaseBRunMetricSummary;
    mcc: PhaseBRunMetricSummary;
    auc: PhaseBRunMetricSummary;
    auc10: PhaseBRunMetricSummary;
    balancedAcc: PhaseBRunMetricSummary;
    bestValLoss: PhaseBRunMetricSummary;
    elapsedSec: PhaseBRunMetricSummary;
  };
  foldMetrics: PhaseBFoldMetric[];
  plots: Array<{ src: string; alt: string; caption: string }>;
};

export type ExternalCocciPairedStat = {
  metric: string;
  nPairs: number;
  meanDiffBMinusA: number;
  medianDiffBMinusA: number;
  ciLow: number;
  ciHigh: number;
  signTestPValue: number;
};

export type ExternalCocciData = {
  title: string;
  description: string;
  models: {
    flagship1: {
      bestFoldPrAucMean: number;
      bestFoldPrLiftMean: number;
      bestFoldAucMean: number;
      overallPrAucMean: number;
    };
    flagship2: {
      bestFoldPrAucMean: number;
      bestFoldPrLiftMean: number;
      bestFoldAucMean: number;
      overallPrAucMean: number;
    };
  };
  setRange: [number, number];
  nSeedRows: number;
  nFoldRows: number;
  pairedStats: ExternalCocciPairedStat[];
};

export const modelComparisonData: ModelComparisonData = {
  snapshotDate: "2026-04-10",
  generatedAtUtc: "2026-04-10T16:31:29.939052+00:00",
  prevalenceMean: 0.0010693776179008976,
  validResiduesMean: 5086136.0,
  posResiduesMean: 5439.0,
  negResiduesMean: 5080697.0,
  metrics: [
    {
      id: "best_fold_pr_auc",
      label: "Best-fold PR-AUC (primary)",
      primary: true,
      flagship1Mean: 0.004415553078200808,
      flagship2Mean: 0.007277773537450639,
      delta: 0.002862220459249832,
      ciLow: 0.0010452323009735695,
      ciHigh: 0.0048898923295385595,
      signTestPValue: 0.021484375
    },
    {
      id: "best_fold_pr_lift",
      label: "Best-fold PR lift",
      primary: false,
      flagship1Mean: 4.129086867245439,
      flagship2Mean: 6.805616103819645,
      delta: 2.6765292365742055,
      ciLow: 0.991165192361917,
      ciHigh: 4.6195552508524145,
      signTestPValue: 0.021484375
    },
    {
      id: "best_fold_auc",
      label: "Best-fold ROC AUC",
      primary: false,
      flagship1Mean: 0.6377073023427999,
      flagship2Mean: 0.6390684676542586,
      delta: 0.0013611653114587786,
      ciLow: -0.0012670152144873994,
      ciHigh: 0.0041660938988901195,
      signTestPValue: 0.34375
    },
    {
      id: "overall_pr_auc",
      label: "Overall PR-AUC",
      primary: false,
      flagship1Mean: 0.0033375050890640497,
      flagship2Mean: 0.004251229352761317,
      delta: 0.0009137242636972677,
      ciLow: 0.0004432262950493327,
      ciHigh: 0.001422448505645043,
      signTestPValue: 0.001953125
    }
  ],
  gallery: [
    {
      src: "/assets/results/best_fold_pr_auc_by_model.png",
      alt: "Best-fold PR-AUC by model",
      caption: "Best-fold PR-AUC distribution by model."
    },
    {
      src: "/assets/results/best_fold_pr_lift_by_model.png",
      alt: "Best-fold PR lift by model",
      caption: "Best-fold PR lift by model."
    },
    {
      src: "/assets/results/best_fold_roc_auc_by_model.png",
      alt: "Best-fold ROC AUC by model",
      caption: "Best-fold ROC AUC distribution by model."
    },
    {
      src: "/assets/results/paired_seed_best_fold_pr_auc.png",
      alt: "Paired seeded best-fold PR-AUC comparison",
      caption: "Paired seeded PR-AUC comparison."
    }
  ]
};

export const phaseBValidationData: PhaseBValidationData = {
  runId: "ffnn_ens_2.4_v100_28234651",
  lastModifiedDate: "2026-04-30",
  trainMode: "ensemble-kfold",
  splitType: "id-family",
  bestModelMetric: "loss",
  ensembleSeedMode: "set-paired",
  splitSeeds: [101],
  trainSeeds: [11],
  nRuns: 5,
  nFolds: 5,
  nSets: 1,
  prBaselineMean: 0.06967984358579282,
  plotStatus: "ok",
  plotFormats: ["png"],
  summaries: {
    prAuc: {
      count: 5,
      mean: 0.2462629488491391,
      std: 0.024042600106578507,
      min: 0.21207421426593107,
      max: 0.2720279392309227
    },
    f1: {
      count: 5,
      mean: 0.3174032329083869,
      std: 0.04438740080329017,
      min: 0.27504656216048423,
      max: 0.37168988491823135
    },
    mcc: {
      count: 5,
      mean: 0.26771526469774926,
      std: 0.036428157109094764,
      min: 0.23554296962157753,
      max: 0.3274213487552121
    },
    auc: {
      count: 5,
      mean: 0.7870453178967496,
      std: 0.037245709043044216,
      min: 0.7445568198569751,
      max: 0.8310811610839786
    },
    auc10: {
      count: 5,
      mean: 0.27522335667817605,
      std: 0.07970414703562351,
      min: 0.14673901959687433,
      max: 0.3658578768539817
    },
    balancedAcc: {
      count: 5,
      mean: 0.672451114654541,
      std: 0.04690270605120357,
      min: 0.6294804811477661,
      max: 0.7449910640716553
    },
    bestValLoss: {
      count: 5,
      mean: 1.1297061191901376,
      std: 0.539895332333516,
      min: 0.7063693824145981,
      max: 2.070721930070978
    },
    elapsedSec: {
      count: 5,
      mean: 7141.516974306107,
      std: 265.26893923759604,
      min: 6889.618050813675,
      max: 7539.753179073334
    }
  },
  foldMetrics: [
    {
      fold: 1,
      bestEpoch: 8,
      precision: 0.25000957073869695,
      recall: 0.6327459963735519,
      f1: 0.3584062470599304,
      mcc: 0.3274213487552121,
      auc: 0.8111752352610528,
      prAuc: 0.25770153001881113,
      auc10: 0.27789161012294705,
      threshold: 0.4739663302898407
    },
    {
      fold: 2,
      bestEpoch: 1,
      precision: 0.2500027159742743,
      recall: 0.7241794379582717,
      f1: 0.37168988491823135,
      mcc: 0.26741669481329794,
      auc: 0.7445568198569751,
      prAuc: 0.2720279392309227,
      auc10: 0.14673901959687433,
      threshold: 0.31692880392074585
    },
    {
      fold: 3,
      bestEpoch: 10,
      precision: 0.25,
      recall: 0.3636629417879418,
      f1: 0.2963051188396591,
      mcc: 0.2670822951888556,
      auc: 0.8310811610839786,
      prAuc: 0.2578722452284484,
      auc10: 0.3658578768539817,
      threshold: 0.758105456829071
    },
    {
      fold: 4,
      bestEpoch: 23,
      precision: 0.25000881740909253,
      recall: 0.3056573670820577,
      f1: 0.27504656216048423,
      mcc: 0.23554296962157753,
      auc: 0.7530805227612958,
      prAuc: 0.21207421426593107,
      auc10: 0.2983896346092831,
      threshold: 0.5711401104927063
    },
    {
      fold: 5,
      bestEpoch: 11,
      precision: 0.25000421848370824,
      recall: 0.33292885713001663,
      f1: 0.28556835156362936,
      mcc: 0.2411130151098029,
      auc: 0.7953328505204458,
      prAuc: 0.23163881550158213,
      auc10: 0.28723864220779416,
      threshold: 0.7127288579940796
    }
  ],
  plots: [
    {
      src: "/assets/validation/phaseb/roc_auc_folds.png",
      alt: "PhaseB fold ROC AUC curves",
      caption: "PhaseB fold ROC AUC curves."
    },
    {
      src: "/assets/validation/phaseb/pr_auc_folds.png",
      alt: "PhaseB fold PR AUC curves",
      caption: "PhaseB fold PR AUC curves."
    }
  ]
};

export const externalCocciData: ExternalCocciData = {
  title: "Seeded External Cocci Evaluation",
  description:
    "Seeded external evaluation compares flagship1 and flagship2 across sets 1-10. Values below are manually curated from the current benchmark snapshot and paired statistics tables.",
  models: {
    flagship1: {
      bestFoldPrAucMean: 0.004415553078200808,
      bestFoldPrLiftMean: 4.129086867245439,
      bestFoldAucMean: 0.6377073023427999,
      overallPrAucMean: 0.0033375050890640497
    },
    flagship2: {
      bestFoldPrAucMean: 0.007277773537450639,
      bestFoldPrLiftMean: 6.805616103819645,
      bestFoldAucMean: 0.6390684676542586,
      overallPrAucMean: 0.004251229352761317
    }
  },
  setRange: [1, 10],
  nSeedRows: 20,
  nFoldRows: 100,
  pairedStats: [
    {
      metric: "best_fold_auc",
      nPairs: 10,
      meanDiffBMinusA: 0.0013611653114587786,
      medianDiffBMinusA: 0.0009279476407004772,
      ciLow: -0.0012670152144873994,
      ciHigh: 0.0041660938988901195,
      signTestPValue: 0.34375
    },
    {
      metric: "best_fold_pr_auc",
      nPairs: 10,
      meanDiffBMinusA: 0.002862220459249832,
      medianDiffBMinusA: 0.0020081863210524064,
      ciLow: 0.0010452323009735695,
      ciHigh: 0.0048898923295385595,
      signTestPValue: 0.021484375
    },
    {
      metric: "best_fold_pr_lift",
      nPairs: 10,
      meanDiffBMinusA: 2.6765292365742055,
      medianDiffBMinusA: 1.8779019566486856,
      ciLow: 0.991165192361917,
      ciHigh: 4.6195552508524145,
      signTestPValue: 0.021484375
    },
    {
      metric: "overall_auc",
      nPairs: 10,
      meanDiffBMinusA: 0.002112123082610595,
      medianDiffBMinusA: 0.0020518275909219286,
      ciLow: 0.00014066679608406158,
      ciHigh: 0.004152330592404762,
      signTestPValue: 0.34375
    },
    {
      metric: "overall_pr_auc",
      nPairs: 10,
      meanDiffBMinusA: 0.0009137242636972677,
      medianDiffBMinusA: 0.0006548439635512353,
      ciLow: 0.0004432262950493327,
      ciHigh: 0.001422448505645043,
      signTestPValue: 0.001953125
    },
    {
      metric: "peptide_precision",
      nPairs: 10,
      meanDiffBMinusA: -0.000002667793326341524,
      medianDiffBMinusA: 0.0000026715793480341617,
      ciLow: -0.0000316001936001862,
      ciHigh: 0.000025785836862170725,
      signTestPValue: 0.75390625
    },
    {
      metric: "peptide_recall",
      nPairs: 10,
      meanDiffBMinusA: -0.0021505376344086113,
      medianDiffBMinusA: 0.005376344086021501,
      ciLow: -0.02526881720430112,
      ciHigh: 0.020430107526881746,
      signTestPValue: 1.0
    },
    {
      metric: "peptide_f1",
      nPairs: 10,
      meanDiffBMinusA: -0.0000053268659698639875,
      medianDiffBMinusA: 0.000005481503158314164,
      ciLow: -0.00006146764440791626,
      ciHigh: 0.00005228523452197981,
      signTestPValue: 0.75390625
    }
  ]
};
