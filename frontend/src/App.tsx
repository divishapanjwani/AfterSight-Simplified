import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { NotFoundPage } from './pages/NotFoundPage'
import { ReadinessPage } from './pages/ReadinessPage'
import { SceneLibraryPage } from './pages/SceneLibraryPage'

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<ReadinessPage />} />
        <Route path="/library" element={<SceneLibraryPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AppShell>
  )
}
