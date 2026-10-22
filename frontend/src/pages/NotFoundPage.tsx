import { Link } from 'react-router-dom'
import { EmptyState } from '../components/States'

export function NotFoundPage() {
  return (
    <EmptyState title="This page does not exist.">
      <Link to="/" className="text-ink underline underline-offset-2">Back to the Readiness board</Link>
    </EmptyState>
  )
}
