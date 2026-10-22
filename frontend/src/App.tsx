import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { CasePage } from './pages/CasePage'
import { ClusterAtlasPage } from './pages/ClusterAtlasPage'
import { FeatureLabPage } from './pages/FeatureLabPage'
import { ModelArenaPage } from './pages/ModelArenaPage'
import { ReportsPage } from './pages/ReportsPage'
import { ReviewQueuePage } from './pages/ReviewQueuePage'
import { NotFoundPage } from './pages/NotFoundPage'
import { ReadinessPage } from './pages/ReadinessPage'
import { SceneLibraryPage } from './pages/SceneLibraryPage'
import { TriageDeskPage } from './pages/TriageDeskPage'

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<ReadinessPage />} />
        <Route path="/library" element={<SceneLibraryPage />} />
        <Route path="/triage" element={<TriageDeskPage />} />
        <Route path="/triage/:sceneId" element={<TriageDeskPage />} />
        <Route path="/case" element={<CasePage />} />
        <Route path="/case/:analysisId/:buildingUid" element={<CasePage />} />
        <Route path="/arena" element={<ModelArenaPage />} />
        <Route path="/features" element={<FeatureLabPage />} />
        <Route path="/clusters" element={<ClusterAtlasPage />} />
        <Route path="/review" element={<ReviewQueuePage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AppShell>
  )
}
