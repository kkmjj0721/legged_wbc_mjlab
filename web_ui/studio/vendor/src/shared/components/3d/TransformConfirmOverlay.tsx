import React from 'react';
import { Check, X } from 'lucide-react';
import { IconButton } from '@/shared/components/ui';

interface TransformConfirmOverlayProps {
  axisLabel: string;
  axisColor: string;
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
  rootClassName?: string;
  contentClassName?: string;
}

export const TransformConfirmOverlay: React.FC<TransformConfirmOverlayProps> = ({
  axisLabel,
  axisColor,
  value,
  step,
  unitLabel,
  deltaDisplay,
  onValueChange,
  onKeyDown,
  onConfirm,
  onCancel,
  confirmTitle = 'Confirm',
  cancelTitle = 'Cancel',
  rootClassName = '',
  contentClassName = '',
}) => {
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div
      className={`flex flex-col items-center gap-1 ${rootClassName}`.trim()}
      role="toolbar"
      aria-label={`${axisLabel} transform confirmation`}
      tabIndex={-1}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <div className={`flex items-center gap-1 ${contentClassName}`.trim()}>
        <span
          className="flex h-5 min-w-5 items-center justify-center rounded px-1.5 text-[10px] font-bold text-white shadow"
          style={{ backgroundColor: axisColor }}
        >
          {axisLabel}
        </span>
        <input
          ref={inputRef}
          type="number"
          step={step}
          value={value}
          onChange={onValueChange}
          onKeyDown={onKeyDown}
          className="w-20 rounded border border-border-strong bg-panel-bg px-1.5 py-0.5 text-xs font-mono text-text-primary shadow-sm focus:border-system-blue focus:outline-none focus-visible:ring-2 focus-visible:ring-system-blue/30"
        />
        <span className="text-[10px] text-text-tertiary">
          {unitLabel} ({deltaDisplay})
        </span>
      </div>

      <div className="flex gap-1">
        <IconButton
          variant="solid"
          tone="success"
          size="sm"
          onClick={onConfirm}
          title={confirmTitle}
        >
          <Check className="h-4 w-4" />
        </IconButton>
        <IconButton
          variant="solid"
          tone="danger"
          size="sm"
          onClick={onCancel}
          title={cancelTitle}
        >
          <X className="h-4 w-4" />
        </IconButton>
      </div>
    </div>
  );
};
