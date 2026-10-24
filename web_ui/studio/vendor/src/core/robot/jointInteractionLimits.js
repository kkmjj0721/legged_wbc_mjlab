export const JOINT_INTERACTION_NEUTRAL_ZERO_EPSILON = 1e-8;

const toFiniteLimitValue = (value) => {
  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value === 'string' && value.trim() === '') {
    return null;
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

export function normalizeJointInteractionLimits(lowerLimit, upperLimit, defaultLimits = {}) {
  const fallbackLower = toFiniteLimitValue(defaultLimits?.lower);
  const fallbackUpper = toFiniteLimitValue(defaultLimits?.upper);

  let lower = toFiniteLimitValue(lowerLimit);
  let upper = toFiniteLimitValue(upperLimit);

  if (lower === null) {
    lower = fallbackLower;
  }

  if (upper === null) {
    upper = fallbackUpper;
  }

  if (lower === null && upper === null) {
    return null;
  }

  if (lower === null) {
    lower = upper;
  }

  if (upper === null) {
    upper = lower;
  }

  if (lower > upper) {
    const previousLower = lower;
    lower = upper;
    upper = previousLower;
  }

  return { lower, upper };
}

export function clampJointInteractionValue(value, lowerLimit, upperLimit, options = {}) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 0;
  }

  const limits = normalizeJointInteractionLimits(lowerLimit, upperLimit, options.defaultLimits);
  if (!limits) {
    return numericValue;
  }

  const neutralZeroEpsilon =
    toFiniteLimitValue(options.neutralZeroEpsilon) ?? JOINT_INTERACTION_NEUTRAL_ZERO_EPSILON;
  if (
    options.preserveNeutralZero === true &&
    Math.abs(numericValue) <= neutralZeroEpsilon &&
    limits.lower <= 0 &&
    limits.upper >= 0
  ) {
    return 0;
  }

  return Math.min(Math.max(numericValue, limits.lower), limits.upper);
}
