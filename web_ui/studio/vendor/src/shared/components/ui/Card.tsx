import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  bordered?: boolean;
  glass?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  padding = 'md',
  bordered = true,
  glass = false,
}) => {
  const paddings = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  };

  void glass;
  const bgClass = 'bg-panel-bg';
    
  const borderClass = bordered 
    ? 'border border-border-black' 
    : '';

  return (
    <div className={`rounded-xl shadow-sm ${bgClass} ${borderClass} ${paddings[padding]} ${className}`}>
      {children}
    </div>
  );
};
