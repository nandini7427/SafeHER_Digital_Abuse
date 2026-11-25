/**
 * Authentication and User types for SafeHER.
 */

export interface User {
  id: string;
  email: string;
  role: 'user' | 'admin' | 'advocate';
  is_active: boolean;
  created_at: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}
