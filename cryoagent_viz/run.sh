#!/bin/bash
# One-command launcher for CryoAgent Workflow Visualizer

echo "🚀 Starting CryoAgent Workflow Visualizer..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start backend in background
echo "   Starting backend on port 8000..."
cd "$SCRIPT_DIR/backend"
python -m uvicorn server:app --host 0.0.0.0 --port 8000 > /tmp/cryoagent-backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "   ✗ Backend failed to start. Check /tmp/cryoagent-backend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "   ✓ Backend running (PID: $BACKEND_PID)"

# Start frontend
echo "   Starting frontend on port 3000..."
cd "$SCRIPT_DIR/frontend"
~/anaconda3/bin/node node_modules/.bin/vite > /tmp/cryoagent-frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

echo ""
echo "✅ Both servers started!"
echo ""
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000 (or 3001 if 3000 was busy)"
echo ""
echo "   Backend logs:  /tmp/cryoagent-backend.log"
echo "   Frontend logs: /tmp/cryoagent-frontend.log"
echo ""
echo "Open http://localhost:3000 in your browser to start!"
echo ""
echo "To stop both servers, run:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
