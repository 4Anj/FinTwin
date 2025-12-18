#!/usr/bin/env python3
"""
Startup script for the Financial Digital Twin application.
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Start the Financial Digital Twin application."""
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("Starting Financial Digital Twin Server...")
    print("Monte Carlo Simulation Engine Ready")
    print("Yahoo Finance Integration Active")
    print("Investment Strategy Engine Loaded")
    print("\n" + "="*50)
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend Interface: http://localhost:8000")
    print("="*50 + "\n")
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down Financial Digital Twin Server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
