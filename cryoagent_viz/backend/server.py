"""
FastAPI backend for CryoAgent Workflow Visualizer.

Provides:
- REST API for workflow listing and fetching
- WebSocket endpoint for real-time updates
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from workflow_parser import parse_workflow, discover_workflows

app = FastAPI(title="CryoAgent Workflow API")

# CORS configuration - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store registered workflow directories
registered_dirs: List[Path] = []


class ScanRequest(BaseModel):
    base_dir: str


@app.get("/api/workflows")
async def list_workflows():
    """List all discovered workflows from registered directories."""
    all_workflows = []

    for base_dir in registered_dirs:
        if base_dir.exists():
            workflows = discover_workflows(base_dir)
            all_workflows.extend(workflows)

    return {
        "workflows": all_workflows,
        "count": len(all_workflows),
        "registered_dirs": [str(d) for d in registered_dirs]
    }


@app.post("/api/workflows/scan")
async def scan_workflows(request: ScanRequest):
    """Scan a directory for workflows and add to registry."""
    base_dir = Path(request.base_dir).expanduser().resolve()

    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {base_dir}")

    if not base_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {base_dir}")

    # Add to registered dirs if not already there
    if base_dir not in registered_dirs:
        registered_dirs.append(base_dir)

    # Discover workflows in this directory
    workflows = discover_workflows(base_dir)

    return {
        "base_dir": str(base_dir),
        "found_count": len(workflows),
        "workflows": workflows
    }


@app.get("/api/workflow/{workflow_path:path}")
async def get_workflow(workflow_path: str):
    """
    Get detailed workflow data for a specific path.

    The workflow_path should be an absolute path to a workflow directory
    containing workflow_state.json.
    """
    # FastAPI strips the leading / from path parameters, so add it back
    # to treat all paths as absolute
    if not workflow_path.startswith('/'):
        workflow_path = '/' + workflow_path

    workflow_dir = Path(workflow_path).expanduser()

    if not workflow_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Workflow directory not found: {workflow_dir}"
        )

    workflow_state_path = workflow_dir / "workflow_state.json"
    if not workflow_state_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No workflow_state.json found in: {workflow_dir}"
        )

    try:
        workflow_data = parse_workflow(workflow_dir)
        return workflow_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing workflow: {str(e)}"
        )


@app.websocket("/ws/workflow/{workflow_path:path}")
async def websocket_endpoint(websocket: WebSocket, workflow_path: str):
    """
    WebSocket endpoint for real-time workflow updates.

    Watches the workflow directory for changes and pushes updates.
    """
    await websocket.accept()

    # FastAPI strips the leading / from path parameters, so add it back
    if not workflow_path.startswith('/'):
        workflow_path = '/' + workflow_path

    workflow_dir = Path(workflow_path).expanduser()

    try:
        # Send initial data
        if workflow_dir.exists():
            workflow_data = parse_workflow(workflow_dir)
            await websocket.send_json(workflow_data)

        # In a real implementation, we'd watch for file changes
        # For now, just keep the connection alive
        while True:
            # Wait for client messages (could be pings)
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "CryoAgent Workflow API",
        "registered_dirs": [str(d) for d in registered_dirs]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
