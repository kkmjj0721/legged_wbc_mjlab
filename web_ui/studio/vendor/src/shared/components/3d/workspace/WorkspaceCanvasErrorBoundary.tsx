import React from 'react';

interface WorkspaceCanvasErrorBoundaryProps {
  children: React.ReactNode;
  fallback: React.ReactNode;
  onError?: (error: unknown) => void;
  resetKey?: string;
}

interface WorkspaceCanvasErrorBoundaryState {
  hasError: boolean;
}

export class WorkspaceCanvasErrorBoundary extends React.Component<
  WorkspaceCanvasErrorBoundaryProps,
  WorkspaceCanvasErrorBoundaryState
> {
  state: WorkspaceCanvasErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): WorkspaceCanvasErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: unknown): void {
    this.props.onError?.(error);
  }

  componentDidUpdate(prevProps: WorkspaceCanvasErrorBoundaryProps): void {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false });
    }
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return this.props.fallback;
    }

    return this.props.children;
  }
}
