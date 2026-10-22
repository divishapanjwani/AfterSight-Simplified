import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import { ApiError } from './services/api'
import './index.css'
import { WorkflowProvider } from './components/Workflow'
import { ThemeProvider } from './theme/ThemeProvider'

// "Not built yet" (503/404/409) answers are deterministic: retrying only delays the designed state.
// Retry only when the backend could not be reached at all.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      refetchOnWindowFocus: false,
      retry: (failures, error) => error instanceof ApiError && error.status === 0 && failures < 2,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <WorkflowProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </WorkflowProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
)
