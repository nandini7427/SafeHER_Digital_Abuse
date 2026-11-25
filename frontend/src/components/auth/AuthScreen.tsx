import React, { FormEvent, useState } from 'react';
import { ApiError } from '../../services/api';
import { login, register } from '../../services/authService';
import { User } from '../../types/auth';

interface AuthScreenProps {
  onAuthenticated: (user: User) => void;
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ onAuthenticated }) => {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const credentials = { email: email.trim(), password };
      if (mode === 'register') {
        await register(credentials);
        setNotice('Account created. Signing you in…');
      }
      const user = await login(credentials);
      onAuthenticated(user);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.detail : 'Unable to connect to SafeHER. Check that the API is running.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-shell">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="brand-mark">S</div>
        <p className="eyebrow">SURVIVOR-CENTRED DIGITAL SAFETY</p>
        <h1 id="auth-title">Welcome to SafeHER</h1>
        <p className="muted">Securely organize incident information, assess threatening messages and review image evidence.</p>
        <div className="auth-tabs" role="tablist" aria-label="Account access">
          <button type="button" role="tab" aria-selected={mode === 'login'} className={mode === 'login' ? 'active' : ''} onClick={() => { setMode('login'); setError(null); }}>Sign in</button>
          <button type="button" role="tab" aria-selected={mode === 'register'} className={mode === 'register' ? 'active' : ''} onClick={() => { setMode('register'); setError(null); }}>Create account</button>
        </div>
        <form onSubmit={submit} className="stack-form">
          <label>Email address<input type="email" autoComplete="email" required maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" /></label>
          <label>Password<input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={mode === 'register' ? 12 : 1} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} placeholder={mode === 'register' ? 'At least 12 characters' : 'Your password'} /></label>
          {error && <p className="inline-error" role="alert">{error}</p>}
          {notice && <p className="inline-notice" role="status">{notice}</p>}
          <button className="primary-button" type="submit" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in securely' : 'Create account and continue'}</button>
        </form>
        <p className="fine-print">Use a device you trust. This prototype stores its access token for the current browser session. Avoid uploading highly sensitive material in an untrusted demo deployment.</p>
      </section>
    </main>
  );
};
