/** Full-page Building Case File: /case/:analysisId/:buildingUid (also reachable from the stepper). */
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { CaseFile } from '../components/CaseFile'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { useWorkflow } from '../components/Workflow'
import { api } from '../services/api'
import { ProvenanceBanner } from './TriageDeskPage'

export function CasePage() {
  const { analysisId, buildingUid } = useParams()
  const { state } = useWorkflow()
  const analysis = useQuery({ queryKey: ['analysis-by-id', analysisId], queryFn: () => api.analysis(analysisId!), enabled: Boolean(analysisId) })
  if (!analysisId || !buildingUid) {
    return (
      <EmptyState title="No building selected.">
        Run an analysis on the Triage Desk and click a footprint to open its case file.{' '}
        <Link to={state.sceneId ? `/triage/${state.sceneId}` : '/triage'} className="text-ink underline underline-offset-2">Go to the Triage Desk</Link>
      </EmptyState>
    )
  }
  if (analysis.isPending) return <LoadingState label="Loading the case file…" rows={4} />
  if (analysis.isError) return <ErrorState error={analysis.error} what="this analysis" />
  const a = analysis.data
  return (
    <div className="mx-auto max-w-[760px] space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Building case file</p>
          <p className="num mt-1 text-sm text-ink-2">{a.scene_id} · {a.event}</p>
        </div>
        <Link to={`/triage/${a.scene_id}?model=${a.model.name}&run=1&b=${buildingUid}`} className="text-xs text-ink underline underline-offset-2">← back to the Triage Desk</Link>
      </div>
      <ProvenanceBanner warning={a.provenance.warning} note={a.provenance.note} />
      <CaseFile analysis={a} uid={buildingUid} />
    </div>
  )
}
