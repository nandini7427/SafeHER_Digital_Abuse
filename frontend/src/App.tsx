import React, { useEffect, useState } from 'react';
import { AuthScreen } from './components/auth/AuthScreen';
import { CaseManager } from './components/cases/CaseManager';
import { SafeImageAnalysisWorkflow } from './components/image/SafeImageAnalysisWorkflow';
import { ThreatAssessmentPanel } from './components/threat/ThreatAssessmentPanel';
import { logout, restoreSession } from './services/authService';
import { User } from './types/auth';

type Workspace = 'image' | 'threat' | 'cases';

export const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [workspace, setWorkspace] = useState<Workspace>('image');
  const [initialCaseId, setInitialCaseId] = useState<string | undefined>();
  const [sessionLoading, setSessionLoading] = useState(true);

  useEffect(() => {
    let active = true;
    restoreSession().then((restored) => {
      if (active) setUser(restored);
    }).finally(() => {
      if (active) setSessionLoading(false);
    });
    return () => { active = false; };
  }, []);

  const handleLogout = () => {
    logout();
    setUser(null);
    setWorkspace('image');
    setInitialCaseId(undefined);
  };

  if (sessionLoading) {
    return <div className="loading-screen">Loading your secure workspace…</div>;
  }

  if (!user) {
    return <AuthScreen onAuthenticated={setUser} />;
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-lockup"><div className="brand-mark small">S</div><div><strong>SafeHER</strong><span>Digital safety workspace</span></div></div>
        <div className="header-actions"><span className="account-label">{user.email}</span><button type="button" className="quick-exit" onClick={() => window.location.replace('about:blank')} title="Leaves this page quickly; browser history may remain">Quick exit</button><button type="button" className="logout-button" onClick={handleLogout}>Sign out</button></div>
      </header>

      <div className="workspace-layout">
        <aside className="workspace-nav" aria-label="Main workspace">
          <p className="eyebrow">YOUR WORKSPACE</p>
          <button className={workspace === 'image' ? 'nav-item active' : 'nav-item'} onClick={() => setWorkspace('image')}><span>◎</span> Image evidence</button>
          <button className={workspace === 'threat' ? 'nav-item active' : 'nav-item'} onClick={() => setWorkspace('threat')}><span>⌁</span> Message assessment</button>
          <button className={workspace === 'cases' ? 'nav-item active' : 'nav-item'} onClick={() => setWorkspace('cases')}><span>▤</span> Incident cases</button>
          <div className="nav-note"><strong>Your privacy matters</strong><p>Only upload evidence you are authorized to share. Avoid using a shared or untrusted device for sensitive incidents.</p></div>
        </aside>

        <main className="workspace-main">
          {workspace === 'image' && <SafeImageAnalysisWorkflow initialCaseId={initialCaseId} />}
          {workspace === 'threat' && <ThreatAssessmentPanel />}
          {workspace === 'cases' && <CaseManager onSelectCase={(id) => { setInitialCaseId(id); setWorkspace('image'); }} />}
        </main>
      </div>
      <footer className="app-footer">SafeHER · Survivor-centred digital safety and evidence triage · AI results are not legal or forensic proof</footer>
    </div>
  );
};

export default App;
