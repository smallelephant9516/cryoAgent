import { useCallback, useEffect, useState } from 'react';
import type { WorkflowListResponse } from '../types/workflow';

const API_BASE = 'http://localhost:8000';

export interface WorkflowListState extends WorkflowListResponse {
  registered_dirs?: string[];
}

/**
 * Hook for fetching and registering workflow folders.
 * Starts empty until the user adds a folder via scanFolder.
 */
export function useWorkflowList() {
  const [workflows, setWorkflows] = useState<WorkflowListState | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkflows = useCallback(async () => {
    try {
      setError(null);
      const response = await fetch(`${API_BASE}/api/workflows`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setWorkflows(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch workflows');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkflows();
  }, [fetchWorkflows]);

  const scanFolder = useCallback(async (baseDir: string) => {
    const trimmed = baseDir.trim();
    if (!trimmed) {
      throw new Error('Please enter a folder path');
    }

    setScanning(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/workflows/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_dir: trimmed }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail;
        const message = Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join('; ')
          : detail || `HTTP ${response.status}: ${response.statusText}`;
        throw new Error(message);
      }

      // After scanning, fetch the complete list from the backend
      // (backend accumulates all registered directories)
      await fetchWorkflows();

      return data as {
        found_count: number;
        workflows: WorkflowListState['workflows'];
        base_dir: string;
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to scan folder';
      setError(message);
      throw err;
    } finally {
      setScanning(false);
    }
  }, []);

  return { workflows, loading, scanning, error, scanFolder, refresh: fetchWorkflows };
}
