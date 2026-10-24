/**
 * AI inspection related types
 */

export interface InspectionIssue {
  type: 'error' | 'warning' | 'suggestion' | 'pass';
  title: string;
  description: string;
  relatedIds?: string[]; // IDs of links/joints involved
  profileId: string; // Profile ID
  itemId: string; // Profile item ID
  evidenceLevel?: 'L1' | 'L2' | 'L3' | 'L4'; // Evidence confidence level
  evidenceSource?: string; // Evidence source label
  score?: number; // Score (0-10)
}

export interface InspectionReport {
  summary: string;
  issues: InspectionIssue[];
  overallScore?: number; // Total awarded score across scored inspection items
  profileScores?: Record<string, number>; // Per-profile average scores (0-10)
  maxScore?: number; // Max score across scored inspection items
}
