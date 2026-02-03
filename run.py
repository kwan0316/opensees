"""
Server startup script for 2D Frame Analysis API
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (Render sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Only use reload in development
    reload = os.environ.get("ENV", "production") == "development"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )


