import type { Theme } from '@/types';
import { useResolvedTheme } from '@/shared/hooks/useTheme';

export interface WorkspaceCanvasEnvironmentIntensityByTheme {
  light: number;
  dark: number;
}

export function useWorkspaceCanvasTheme(theme: Theme): 'light' | 'dark' {
  return useResolvedTheme(theme);
}

export function resolveWorkspaceCanvasEnvironmentIntensity({
  effectiveTheme,
  environmentIntensity,
  environmentIntensityByTheme,
}: {
  effectiveTheme: 'light' | 'dark';
  environmentIntensity: number;
  environmentIntensityByTheme?: WorkspaceCanvasEnvironmentIntensityByTheme;
}): number {
  return environmentIntensityByTheme?.[effectiveTheme] ?? environmentIntensity;
}
