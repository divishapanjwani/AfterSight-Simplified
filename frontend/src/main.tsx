import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import './index.css'
import { WorkflowProvider } from './components/Workflow'
import { ThemeProvider } from './theme/ThemeProvider'

const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 60_000, refetchOnWindowFocus: false } } })

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
