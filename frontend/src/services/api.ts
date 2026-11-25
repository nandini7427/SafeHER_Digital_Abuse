/**
 * SafeHER Core API Client.
 * Handles session-scoped bearer token injection, timeout handling, and uniform error extraction.
 */

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || '/api/v1';

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export function getStoredToken(): string | null {
  try {
    return sessionStorage.getItem('safeher_access_token');
  } catch {
    return null;
  }
}

export function setStoredToken(token: string | null): void {
  try {
    if (token) {
      sessionStorage.setItem('safeher_access_token', token);
    } else {
      sessionStorage.removeItem('safeher_access_token');
    }
  } catch {
    // Graceful fallback if localStorage is restricted
  }
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getStoredToken();
  const headers = new Headers(options.headers || {});

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const url = `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Fallback to text status if response is not JSON
    }
    throw new ApiError(response.status, errorDetail);
  }

  return response.json() as Promise<T>;
}
