import { useState, useEffect } from 'react';
import type { Theme } from '@/types';
import { resolveTheme, type ResolvedTheme } from '@/shared/utils/theme';

export function useResolvedTheme(theme: Theme): ResolvedTheme {
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => resolveTheme('system'));

  useEffect(() => {
    if (theme !== 'system') {
      return;
    }
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      setSystemTheme('light');
      return;
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    setSystemTheme(mediaQuery.matches ? 'dark' : 'light');

    const handleChange = (event: MediaQueryListEvent) => {
      setSystemTheme(event.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  return theme === 'system' ? systemTheme : theme;
}
