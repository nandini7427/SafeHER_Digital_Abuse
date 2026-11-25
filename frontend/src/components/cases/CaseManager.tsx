import React, { FormEvent, useCallback, useEffect, useState } from 'react';
import { ApiError } from '../../services/api';
import { createUserCase, fetchUserCases } from '../../services/caseService';
import { Case } from '../../types/case';

interface CaseManagerProps {
  onSelectCase?: (caseId: string) => void;
}

export const CaseManager: React.FC<CaseManagerProps> = ({ onSelectCase }) => {
  const [cases, setCases] = useState<Case[]>([]);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setCases(await fetchUserCases());
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.detail : 'Could not load your cases.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const created = await createUserCase({ title, description: description.trim() || undefined });
      setCases((current) => [created, ...current]);
      setTitle('');
      setDescription('');
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.detail : 'Could not create this case.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="workspace-card">
      <div className="section-heading"><div><p className="eyebrow">MILESTONE 2 · CASE MANAGEMENT</p><h2>Your incident cases</h2><p className="muted">Cases are private to the account that created them.</p></div><button className="secondary-button" onClick={() => void refresh()} disabled={loading}>Refresh</button></div>
      <form onSubmit={submit} className="case-create-form">
        <label>Case title<input value={title} onChange={(event) => setTitle(event.target.value)} minLength={3} maxLength={200} required placeholder="For example: Suspected image-based abuse" /></label>
        <label>Optional notes<textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={3} maxLength={5000} placeholder="Add a brief description. Do not include passwords or unnecessary sensitive details." /></label>
        <button className="primary-button" type="submit" disabled={saving}>{saving ? 'Creating…' : 'Create incident case'}</button>
      </form>
      {error && <p className="inline-error" role="alert">{error}</p>}
      <div className="case-list" aria-live="polite">
        {loading ? <p className="muted">Loading cases…</p> : cases.length === 0 ? <p className="empty-state">No cases yet. Create your first case above.</p> : cases.map((item) => (
          <article className="case-row" key={item.id}>
            <div><h3>{item.title}</h3><p className="muted">{item.description || 'No notes added'} · {new Date(item.created_at).toLocaleDateString()}</p><span className="status-pill">{item.status}</span></div>
            {onSelectCase && <button className="secondary-button" onClick={() => onSelectCase(item.id)}>Analyze evidence</button>}
          </article>
        ))}
      </div>
    </section>
  );
};
