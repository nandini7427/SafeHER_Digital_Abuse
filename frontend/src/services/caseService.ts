/** Case management API service. */

import { apiRequest } from './api';
import { Case } from '../types/case';

export interface CaseCreatePayload {
  title: string;
  description?: string;
}

export async function fetchUserCases(): Promise<Case[]> {
  return apiRequest<Case[]>('/cases');
}

export async function createUserCase(payload: CaseCreatePayload): Promise<Case> {
  return apiRequest<Case>('/cases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
