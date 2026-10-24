import * as THREE from 'three';

export interface ParsedThreeColor {
  color: THREE.Color;
  opacity: number | null;
}

export type ColorRgbaTuple = [number, number, number, number];

export function createThreeColorFromSRGB(
  red: number,
  green: number,
  blue: number,
): THREE.Color {
  return new THREE.Color().setRGB(red, green, blue, THREE.SRGBColorSpace);
}

export function setThreeColorFromSRGB(
  target: THREE.Color,
  red: number,
  green: number,
  blue: number,
): THREE.Color {
  return target.setRGB(red, green, blue, THREE.SRGBColorSpace);
}

function expandShortHex(hex: string): string {
  return hex
    .split('')
    .map((char) => `${char}${char}`)
    .join('');
}

function clampUnitInterval(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function normalizeColorChannel(value: number): number {
  return Math.abs(value) <= 1 ? value * 255 : value;
}

function toHexChannel(value: number): string {
  return Math.max(0, Math.min(255, Math.round(normalizeColorChannel(value))))
    .toString(16)
    .padStart(2, '0');
}

export function normalizeColorRgbaTuple(
  value: readonly number[] | null | undefined,
): ColorRgbaTuple | null {
  if (!Array.isArray(value) || value.length < 4) {
    return null;
  }

  const channels = value.slice(0, 4).map((channel) => Number(channel));
  if (!channels.every((channel) => Number.isFinite(channel))) {
    return null;
  }

  return [
    clampUnitInterval(channels[0]!),
    clampUnitInterval(channels[1]!),
    clampUnitInterval(channels[2]!),
    clampUnitInterval(channels[3]!),
  ];
}

export function colorRgbaTupleToHex(
  value: readonly number[] | null | undefined,
  options: { includeOpaqueAlpha?: boolean } = {},
): string | null {
  const rgba = normalizeColorRgbaTuple(value);
  if (!rgba) {
    return null;
  }

  const [red, green, blue, alpha] = rgba;
  const rgb = `${toHexChannel(red)}${toHexChannel(green)}${toHexChannel(blue)}`;
  return alpha < 0.999 || options.includeOpaqueAlpha ? `#${rgb}${toHexChannel(alpha)}` : `#${rgb}`;
}

export function colorRgbaTupleToOpacity(
  value: readonly number[] | null | undefined,
): number | undefined {
  const rgba = normalizeColorRgbaTuple(value);
  return rgba ? rgba[3] : undefined;
}

export function parseThreeColorWithOpacity(
  value: THREE.ColorRepresentation | null | undefined,
): ParsedThreeColor | null {
  if (value === null || value === undefined) {
    return null;
  }

  if (value instanceof THREE.Color) {
    return {
      color: value.clone(),
      opacity: null,
    };
  }

  if (typeof value === 'string') {
    const raw = value.trim();
    if (!raw) {
      return null;
    }

    const normalized = raw.startsWith('#') ? raw.slice(1) : raw;
    if (/^[0-9a-f]{3}$/i.test(normalized)) {
      return {
        color: new THREE.Color(`#${expandShortHex(normalized)}`),
        opacity: null,
      };
    }

    if (/^[0-9a-f]{4}$/i.test(normalized)) {
      const expanded = expandShortHex(normalized);
      return {
        color: new THREE.Color(`#${expanded.slice(0, 6)}`),
        opacity: parseInt(expanded.slice(6, 8), 16) / 255,
      };
    }

    if (/^[0-9a-f]{6}$/i.test(normalized)) {
      return {
        color: new THREE.Color(`#${normalized}`),
        opacity: null,
      };
    }

    if (/^[0-9a-f]{8}$/i.test(normalized)) {
      return {
        color: new THREE.Color(`#${normalized.slice(0, 6)}`),
        opacity: parseInt(normalized.slice(6, 8), 16) / 255,
      };
    }
  }

  try {
    return {
      color: new THREE.Color(value),
      opacity: null,
    };
  } catch (e) {
    console.error('[color] parseColorFast failed:', value, e);
    return null;
  }
}
