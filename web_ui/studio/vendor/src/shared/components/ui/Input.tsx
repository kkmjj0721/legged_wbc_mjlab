import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: React.ReactNode;
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  icon,
  className = '',
  id,
  ...props
}) => {
  const generatedInputId = React.useId();
  const inputId = id || generatedInputId;

  return (
    <div className="w-full">
      {label && (
        <label htmlFor={inputId} className="block text-xs font-semibold text-text-secondary mb-1.5 ml-0.5">
          {label}
        </label>
      )}
      <div className="relative">
        {icon && (
          <div className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none">
            {icon}
          </div>
        )}
        <input
          id={inputId}
          className={`
            w-full bg-input-bg
            border border-border-black
            text-text-primary
            placeholder:text-text-tertiary
            rounded-lg
            ${icon ? 'pl-9 pr-3' : 'px-3'} py-1.5
            text-sm
            focus:outline-none focus:ring-2 focus:ring-system-blue/30 focus:border-system-blue
            transition-all duration-200
            disabled:opacity-50 disabled:cursor-not-allowed
            ${error ? 'border-danger focus:border-danger focus:ring-danger/30' : ''}
            ${className}
          `}
          {...props}
        />
      </div>
      {error && <p className="mt-1 ml-0.5 text-xs text-danger">{error}</p>}
    </div>
  );
};
