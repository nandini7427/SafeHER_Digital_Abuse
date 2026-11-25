/**
 * Image Intelligence API Service.
 * Transmits multipart uploads to backend endpoints with optional case association.
 */

import { apiRequest, ApiError } from './api';
import { ImageAssessmentResponse } from '../types/imageAnalysis';

export async function assessImageEvidence(
  file: File,
  caseId?: string,
  timeoutMs: number = 30000
): Promise<ImageAssessmentResponse> {
  const formData = new FormData();
  formData.append('file', file);

  if (caseId && caseId.trim()) {
    formData.append('case_id', caseId.trim());
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await apiRequest<ImageAssessmentResponse>('/ai/image-assessment', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError(504, 'The image analysis request timed out. Please try again with an optimized image.');
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}
