import React from 'react';

interface LabelProps {
  children: React.ReactNode;
  className?: string;
  required?: boolean;
  htmlFor?: string;
}

export const Label: React.FC<LabelProps> = ({
  children,
  className = '',
  required = false,
  htmlFor,
}) => {
  return (
    <label 
      htmlFor={htmlFor}
      className={`block text-xs font-semibold text-text-secondary mb-1.5 ml-0.5 ${className}`}
    >
      {children}
      {required && <span className="ml-0.5 text-danger">*</span>}
    </label>
  );
};
