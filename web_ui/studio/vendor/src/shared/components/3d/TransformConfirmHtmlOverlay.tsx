import React from 'react';
import { Html } from '@react-three/drei';

import { TransformConfirmOverlay } from './TransformConfirmOverlay';

function getTransformAxisColor(axisLabel: string | null | undefined): string {
  if (axisLabel === 'X') return '#ef4444';
  if (axisLabel === 'Y') return '#22c55e';
  if (axisLabel === 'Z') return '#3b82f6';
  return '#94a3b8';
}

interface TransformConfirmHtmlOverlayProps {
  axisLabel: string;
  position: [number, number, number];
  value: string | number;
  step: string | number;
  unitLabel: string;
  deltaDisplay: string;
  onValueChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onConfirm: () => void;
  onCancel: () => void;
  confirmTitle?: string;
  cancelTitle?: string;
  htmlStyle?: React.CSSProperties;
  rootClassName?: string;
  contentClassName?: string;
}

export const TransformConfirmHtmlOverlay: React.FC<TransformConfirmHtmlOverlayProps> = ({
  axisLabel,
  position,
  value,
  step,
  unitLabel,
  deltaDisplay,
  onValueChange,
  onKeyDown,
  onConfirm,
  onCancel,
  confirmTitle,
  cancelTitle,
  htmlStyle,
  rootClassName,
  contentClassName,
}) => {
  return (
    <Html
      position={position}
      style={htmlStyle}
      center
      zIndexRange={[100, 0]}
    >
      <TransformConfirmOverlay
        axisLabel={axisLabel}
        axisColor={getTransformAxisColor(axisLabel)}
        value={value}
        step={step}
        unitLabel={unitLabel}
        deltaDisplay={deltaDisplay}
        onValueChange={onValueChange}
        onKeyDown={onKeyDown}
        onConfirm={onConfirm}
        onCancel={onCancel}
        confirmTitle={confirmTitle}
        cancelTitle={cancelTitle}
        rootClassName={rootClassName}
        contentClassName={contentClassName}
      />
    </Html>
  );
};
