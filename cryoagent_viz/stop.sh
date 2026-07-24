#!/bin/bash
# Stop all CryoAgent Workflow Visualizer servers

echo "🛑 Stopping CryoAgent Workflow Visualizer servers..."

# Find and kill all running servers
BACKEND_PIDS=$(pgrep -f "uvicorn server:app")
FRONTEND_PIDS=$(pgrep -f "vite")

if [ -n "$BACKEND_PIDS" ]; then
    echo "   Stopping backend servers (PIDs: $BACKEND_PIDS)..."
    kill $BACKEND_PIDS 2>/dev/null
    echo "   ✓ Backend stopped"
else
    echo "   No backend servers running"
fi

if [ -n "$FRONTEND_PIDS" ]; then
    echo "   Stopping frontend servers (PIDs: $FRONTEND_PIDS)..."
    kill $FRONTEND_PIDS 2>/dev/null
    echo "   ✓ Frontend stopped"
else
    echo "   No frontend servers running"
fi

echo ""
echo "✅ All servers stopped!"
