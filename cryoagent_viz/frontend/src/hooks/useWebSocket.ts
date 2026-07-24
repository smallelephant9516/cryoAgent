import { useEffect, useState } from 'react';
import type { WorkflowData } from '../types/workflow';

const WS_BASE = 'ws://localhost:8000';

/**
 * Hook for real-time workflow data updates via WebSocket.
 */
export function useWebSocket(workflowPath: string, enabled: boolean) {
  const [data, setData] = useState<WorkflowData | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled || !workflowPath) {
      setConnected(false);
      setData(null);
      return;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/workflow/${workflowPath}`);
    let pingInterval: ReturnType<typeof setInterval>;

    ws.onopen = () => {
      setConnected(true);
      setError(null);

      // Send periodic pings to keep connection alive
      pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 30000); // every 30 seconds
    };

    ws.onmessage = (event) => {
      try {
        const workflowData = JSON.parse(event.data);
        setData(workflowData);
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError('WebSocket connection error');
      setConnected(false);
    };

    ws.onclose = () => {
      setConnected(false);
      clearInterval(pingInterval);
    };

    return () => {
      clearInterval(pingInterval);
      ws.close();
    };
  }, [workflowPath, enabled]);

  return { data, connected, error };
}
