export type UpdateCommitMode = 'debounced' | 'immediate' | 'manual';

export interface UpdateCommitOptions {
  historyKey?: string;
  historyLabel?: string;
  commitMode?: UpdateCommitMode;
  debounceMs?: number;
  skipHistory?: boolean;
}
