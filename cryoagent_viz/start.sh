#!/bin/bash
# Start the CryoAgent Workflow Visualizer

echo "🚀 Starting CryoAgent Workflow Visualizer..."
echo ""

# Start backend
echo "📦 Starting FastAPI backend on port 8000..."
cd "$(dirname "$0")/backend"
python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 2

# Test backend
if curl -s http://localhost:8000/ > /dev/null; then
    echo "   ✓ Backend is running"
else
    echo "   ✗ Backend failed to start"
    exit 1
fi

echo ""
echo "🌐 Starting React frontend on port 3000..."
cd ../frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "   Installing dependencies (this may take a minute)..."
    ~/anaconda3/bin/node ~/anaconda3/lib/node_modules/npm/bin/npm-cli.js install
fi

# Run vite directly using node
~/anaconda3/bin/node node_modules/.bin/vite &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "✓ Servers started!"
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  Frontend UI:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Handle Ctrl+C to stop both servers
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Wait for processes
wait
