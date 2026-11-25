/**
 * Digital abuse incident case types for SafeHER.
 */

export interface Case {
  id: string;
  owner_id: string;
  title: string;
  description: string | null;
  status: 'open' | 'investigating' | 'preserved' | 'closed';
  created_at: string;
}
