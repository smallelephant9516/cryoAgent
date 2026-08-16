import { useEffect, useRef, useState } from 'react';
import type { WorkflowData } from '../types/workflow';

const WS_BASE = 'ws://localhost:8000';
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 15000;

/**
 * Hook for real-time workflow data updates via WebSocket.
 * Reconnects with exponential backoff if the connection drops.
 */
export function useWebSocket(workflowPath: string, enabled: boolean) {
  const [data, setData] = useState<WorkflowData | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);
  const wsRef = useRef<WebSocket | null>(null);
  const closedByUsRef = useRef(false);

  useEffect(() => {
    if (!enabled || !workflowPath) {
      setConnected(false);
      setData(null);
      setError(null);
      return;
    }

    closedByUsRef.current = false;
    let pingInterval: ReturnType<typeof setInterval> | undefined;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      if (closedByUsRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/ws/workflow/${workflowPath}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setError(null);
        backoffRef.current = INITIAL_BACKOFF_MS;

        pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const workflowData = JSON.parse(event.data);
          if (workflowData?.error) {
            setError(String(workflowData.error));
            return;
          }
          setData(workflowData);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = () => {
        setError('WebSocket connection error');
        setConnected(false);
      };

      ws.onclose = () => {
        setConnected(false);
        if (pingInterval) clearInterval(pingInterval);
        wsRef.current = null;

        if (closedByUsRef.current) return;

        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, MAX_BACKOFF_MS);
        reconnectTimer = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      closedByUsRef.current = true;
      if (pingInterval) clearInterval(pingInterval);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [workflowPath, enabled]);

  return { data, connected, error };
}
