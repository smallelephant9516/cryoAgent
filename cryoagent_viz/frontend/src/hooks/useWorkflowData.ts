import { useEffect, useState } from 'react';
import type { WorkflowData } from '../types/workflow';

const API_BASE = 'http://localhost:8000';

/**
 * Hook for fetching static workflow data via REST API.
 */
export function useWorkflowData(workflowPath: string, enabled: boolean) {
  const [data, setData] = useState<WorkflowData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled || !workflowPath) {
      setData(null);
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_BASE}/api/workflow/${workflowPath}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const workflowData = await response.json();
        setData(workflowData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch workflow data');
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [workflowPath, enabled]);

  return { data, loading, error };
}
