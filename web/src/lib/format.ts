const BASE_PATH = "/PepSeqPred";

export function assetPath(src: string): string {
  if (!src.startsWith("/")) {
    return `${BASE_PATH}/${src}`;
  }
  return `${BASE_PATH}${src}`;
}

export function formatFixed(value: number, digits: number): string {
  return value.toFixed(digits);
}

export function formatSigned(value: number, digits: number): string {
  const out = value.toFixed(digits);
  return value >= 0 ? `+${out}` : out;
}

export function formatPercent(value: number, digits: number): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatSci(value: number, digits: number): string {
  return value.toExponential(digits);
}
