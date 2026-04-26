"""
Launcher for Agentic AI Phishing Detector.

Starts the FastAPI/Uvicorn server and opens the browser automatically.
This is the entry point for the PyInstaller .exe build.
"""

import sys
import os
import webbrowser
import threading
import time


def _get_base_dir():
    """Get the base directory — handles both dev and PyInstaller frozen mode."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _setup_frozen_env():
    """Configure environment when running as a PyInstaller bundle."""
    if not getattr(sys, 'frozen', False):
        return

    base = sys._MEIPASS

    # Tesseract OCR data
    tessdata_path = os.path.join(base, 'tessdata')
    if os.path.exists(tessdata_path):
        os.environ['TESSDATA_PREFIX'] = tessdata_path

    # Ensure we can find our app modules
    if base not in sys.path:
        sys.path.insert(0, base)

    # Set working directory to base so relative paths work
    os.chdir(base)


def open_browser(port=8000):
    """Open the default browser after a short delay."""
    time.sleep(2.5)
    url = f'http://localhost:{port}'
    print(f'\n  Opening browser at {url} ...\n')
    webbrowser.open(url)


def main():
    port = 8000

    print(r"""
    ╔══════════════════════════════════════════════════════╗
    ║     Agentic AI Phishing Detector                    ║
    ║     Dual-Engine MCO Architecture                    ║
    ║                                                     ║
    ║     Server starting on http://localhost:8000         ║
    ║     Press Ctrl+C to stop                            ║
    ╚══════════════════════════════════════════════════════╝
    """)

    _setup_frozen_env()

    # Open browser in background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start Uvicorn
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == '__main__':
    main()
