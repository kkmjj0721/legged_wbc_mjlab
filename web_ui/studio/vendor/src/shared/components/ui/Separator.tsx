import React from 'react';

interface SeparatorProps {
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export const Separator: React.FC<SeparatorProps> = ({
  orientation = 'horizontal',
  className = '',
}) => {
  return (
    <div
      className={`
        bg-border-black
        ${orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]'}
        ${className}
      `}
    />
  );
};
