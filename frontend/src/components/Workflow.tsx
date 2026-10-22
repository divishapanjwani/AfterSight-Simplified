/** Remembers where the user is in the workflow (scene -> analysis -> building) for the stepper links. */
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'

export interface WorkflowState {
  sceneId?: string
  model?: string
  analysisId?: string
  buildingUid?: string
}

const KEY = 'aftersight.workflow'
const Ctx = createContext<{ state: WorkflowState; update: (s: WorkflowState) => void }>({ state: {}, update: () => {} })

function load(): WorkflowState {
  try {
    return JSON.parse(window.sessionStorage.getItem(KEY) ?? '{}') as WorkflowState
  } catch {
    return {}
  }
}

export function WorkflowProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<WorkflowState>(load)
  const update = useCallback((patch: WorkflowState) => {
    setState((prev) => {
      const next = { ...prev, ...patch }
      try {
        window.sessionStorage.setItem(KEY, JSON.stringify(next))
      } catch {
        /* convenience only */
      }
      return next
    })
  }, [])
  return <Ctx.Provider value={{ state, update }}>{children}</Ctx.Provider>
}

export const useWorkflow = () => useContext(Ctx)
