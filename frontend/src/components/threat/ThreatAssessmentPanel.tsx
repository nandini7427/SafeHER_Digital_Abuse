import React, { FormEvent, useEffect, useState } from 'react';
import { ApiError, apiRequest } from '../../services/api';
import { fetchUserCases } from '../../services/caseService';
import { Case } from '../../types/case';

interface ThreatAssessment {
  analyzed_text: string;
  overall_threat_level: string;
  uncertainty_state: string;
  threat_indicators: Array<{ category: string; indicator_type: string; text: string; explanation: string }>;
  flagged_spans: Array<{ start: number; end: number; text: string; category: string; explanation: string }>;
  explanation: string;
  preprocessing_warnings: string[];
  disclaimer: string;
  is_calibrated: boolean;
}

export const ThreatAssessmentPanel: React.FC = () => {
  const [text, setText] = useState('');
  const [cases, setCases] = useState<Case[]>([]);
  const [caseId, setCaseId] = useState('');
  const [result, setResult] = useState<ThreatAssessment | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { fetchUserCases().then(setCases).catch(() => setCases([])); }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(null); setResult(null);
    try {
      const response = await apiRequest<ThreatAssessment>('/ai/threat-assessment', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, case_id: caseId || null }),
      });
      setResult(response);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.detail : 'Threat assessment could not be completed.');
    } finally { setBusy(false); }
  };

  return (
    <section className="workspace-card">
      <p className="eyebrow">MILESTONE 1 · THREAT INTELLIGENCE</p>
      <h2>Assess a threatening message</h2>
      <p className="muted">Paste a message you have permission to share. SafeHER identifies language patterns that may indicate coercion, blackmail or threats. It cannot determine intent or make legal findings.</p>
      <form onSubmit={submit} className="stack-form">
        <label>Message text<textarea value={text} onChange={(event) => setText(event.target.value)} rows={7} maxLength={10000} required placeholder="Paste the message here…" /></label>
        <label>Associate with a case (optional)<select value={caseId} onChange={(event) => setCaseId(event.target.value)}><option value="">No case selected</option>{cases.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>
        <button className="primary-button" type="submit" disabled={busy || !text.trim()}>{busy ? 'Assessing message…' : 'Assess message'}</button>
      </form>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {result && <div className="assessment-result" aria-live="polite">
        <div className="result-title-row"><div><p className="eyebrow">ASSESSMENT RESULT</p><h3 className={`threat-level threat-${result.overall_threat_level}`}>{result.overall_threat_level.toUpperCase()} THREAT INDICATORS</h3></div><span className="status-pill">Uncertainty: {result.uncertainty_state}</span></div>
        <p>{result.explanation}</p>
        {!result.is_calibrated && <p className="callout-warning">This assessment is not calibrated. No score should be interpreted as a probability.</p>}
        {result.threat_indicators.length > 0 ? <ul className="indicator-list">{result.threat_indicators.map((item, index) => <li key={`${item.category}-${index}`}><strong>{item.category.replace(/_/g, ' ')}</strong><span>{item.text}</span><p>{item.explanation}</p></li>)}</ul> : <p className="empty-state">No supported threat indicators were identified in this text. This does not prove that the situation is safe.</p>}
        {result.preprocessing_warnings.map((warning, index) => <p className="muted" key={index}>{warning}</p>)}
        <p className="disclaimer">{result.disclaimer}</p>
      </div>}
    </section>
  );
};
