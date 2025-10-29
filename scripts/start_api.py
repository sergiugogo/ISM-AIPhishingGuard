"""
Production startup script for PhishGuard API.

This script starts the API server with production-ready configuration.
"""
import os
import sys
import argparse
import multiprocessing
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_num_workers():
    """Calculate optimal number of workers."""
    # Formula: (2 x num_cores) + 1
    # But limit to max 8 for memory constraints
    cpu_count = multiprocessing.cpu_count()
    workers = min((2 * cpu_count) + 1, 8)
    return workers


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="Start PhishGuard API server")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", str(get_num_workers()))),
        help=f"Number of worker processes (default: auto-calculated)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_dir = os.getenv("MODEL_DIR", "models/phishguard-model")
    model_path = project_root / model_dir
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using: python src/train.py")
        sys.exit(1)
    
    print("="*70)
    print("PhishGuard API Server")
    print("="*70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Model: {model_path}")
    print(f"Log Level: {args.log_level}")
    print(f"Reload: {args.reload}")
    print("="*70)
    
    # Import and run uvicorn
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True,
        # Production settings
        loop="uvloop" if not args.reload else "asyncio",
        http="httptools" if not args.reload else "h11",
        # SSL support (uncomment and configure for HTTPS)
        # ssl_keyfile="/path/to/key.pem",
        # ssl_certfile="/path/to/cert.pem",
    )


if __name__ == "__main__":
    main()
