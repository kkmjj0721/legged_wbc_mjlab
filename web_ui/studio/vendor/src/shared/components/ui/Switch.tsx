import React from 'react';

export interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: React.ReactNode;
  ariaLabel?: string;
  disabled?: boolean;
  size?: 'sm' | 'md';
  className?: string;
  labelClassName?: string;
}

export const Switch: React.FC<SwitchProps> = ({
  checked,
  onChange,
  label,
  ariaLabel,
  disabled = false,
  size = 'md',
  className = '',
  labelClassName = '',
}) => {
  const generatedLabelId = React.useId();
  const previousCheckedRef = React.useRef(checked);
  const pendingUserToggleRef = React.useRef(false);
  const handleToggle = () => {
    if (!disabled) {
      pendingUserToggleRef.current = true;
      onChange(!checked);
    }
  };
  const shouldAnimateStateChange =
    pendingUserToggleRef.current && previousCheckedRef.current !== checked;

  React.useLayoutEffect(() => {
    previousCheckedRef.current = checked;
    pendingUserToggleRef.current = false;
  });

  // Desktop-friendly compact sizes with proper alignment
  const desktopSizes = {
    sm: {
      switch: 'h-4 w-8',
      dot: 'h-3 w-3',
      translate: 'translate-x-4',
    },
    md: {
      switch: 'h-6 w-11',
      dot: 'h-5 w-5',
      translate: 'translate-x-5',
    },
  };
  const resolvedAriaLabel = ariaLabel;
  const labelledBy = resolvedAriaLabel ? undefined : label ? generatedLabelId : undefined;
  const trackClassName = `
    relative inline-flex items-center shrink-0 cursor-pointer rounded-full border border-transparent ${shouldAnimateStateChange ? 'transition-[background-color,border-color] duration-200 ease-in-out' : 'transition-none'} focus:outline-none focus-visible:ring-2 focus-visible:ring-system-blue/30 p-0.5
    ${checked ? 'bg-system-blue-solid' : 'bg-switch-off'}
    ${checked ? 'border-system-blue-solid' : 'border-border-strong/70'}
    ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
    ${desktopSizes[size].switch}
  `;
  const thumbClassName = `
    pointer-events-none inline-block transform rounded-full border border-border-black/10 bg-switch-thumb shadow-sm ring-0
    ${shouldAnimateStateChange ? 'transition duration-200 ease-in-out' : 'transition-none'}
    ${checked ? desktopSizes[size].translate : 'translate-x-0'}
    ${desktopSizes[size].dot}
  `;

  if (label) {
    return (
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={resolvedAriaLabel}
        aria-labelledby={labelledBy}
        onClick={handleToggle}
        disabled={disabled}
        className={`flex items-center justify-between gap-3 rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-system-blue/30 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      >
        <span
          id={generatedLabelId}
          className={`select-none text-left text-xs font-medium text-text-secondary ${labelClassName}`}
        >
          {label}
        </span>
        <span className={trackClassName}>
          <span className={thumbClassName} />
        </span>
      </button>
    );
  }

  return (
    <div className={`flex items-center justify-between ${className}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={resolvedAriaLabel}
        onClick={handleToggle}
        disabled={disabled}
        className={trackClassName}
      >
        <span className={thumbClassName} />
      </button>
    </div>
  );
};
