import { apiRequest, getStoredToken, setStoredToken } from './api';
import { AuthToken, User } from '../types/auth';

export interface Credentials {
  email: string;
  password: string;
}

export async function register(credentials: Credentials): Promise<User> {
  return apiRequest<User>('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
}

export async function login(credentials: Credentials): Promise<User> {
  const token = await apiRequest<AuthToken>('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
  setStoredToken(token.access_token);
  try {
    return await apiRequest<User>('/auth/me');
  } catch (error) {
    setStoredToken(null);
    throw error;
  }
}

export async function restoreSession(): Promise<User | null> {
  if (!getStoredToken()) return null;
  try {
    return await apiRequest<User>('/auth/me');
  } catch {
    setStoredToken(null);
    return null;
  }
}

export function logout(): void {
  setStoredToken(null);
}
