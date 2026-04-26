# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Agentic AI Phishing Detector.

Build command:
    pyinstaller phishing_detector.spec --noconfirm
"""

import os
import sys

block_cipher = None

# Project root
PROJECT_ROOT = os.path.abspath('.')

# Data files to bundle
datas = [
    # Frontend
    (os.path.join('app', 'static', 'index.html'), os.path.join('app', 'static')),
    # Brand domains config
    (os.path.join('data', 'brand_domains.json'), 'data'),
    # Sample emails
    (os.path.join('data', 'sample_emails'), os.path.join('data', 'sample_emails')),
    # Tesseract OCR data
    (os.path.join('tessdata', 'eng.traineddata'), 'tessdata'),
]

# Hidden imports that PyInstaller can't detect automatically
hiddenimports = [
    # Uvicorn internals
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    # FastAPI / Starlette
    'fastapi',
    'starlette',
    'starlette.responses',
    'starlette.routing',
    'starlette.middleware',
    'starlette.middleware.cors',
    'starlette.staticfiles',
    'anyio._backends._asyncio',
    'multipart',
    'python_multipart',
    # Our app modules
    'app',
    'app.main',
    'app.database',
    'app.parsers',
    'app.parsers.eml_parser',
    'app.parsers.text_parser',
    'app.parsers.ocr_parser',
    'app.models',
    'app.models.classifier',
    'app.models.explainer',
    'app.models.semantic_engine',
    'app.models.structural_engine',
    'app.utils',
    'app.utils.url_analyzer',
    'app.utils.header_analyzer',
    'app.utils.html_analyzer',
    'app.adversarial',
    'app.adversarial.evaluator',
    'app.adversarial.obfuscator',
    'app.adversarial.paraphraser',
    # ML / NLP
    'sklearn',
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._classes',
    'sklearn.utils._typedefs',
    'numpy',
    'transformers',
    'torch',
    # Other deps
    'tldextract',
    'Levenshtein',
    'anthropic',
    'pytesseract',
    'PIL',
    'pydantic',
    'httptools',
    'dotenv',
    'email',
    'sqlite3',
    'h11',
    'httpcore',
    'httpx',
    'sniffio',
    'certifi',
    'idna',
]

a = Analysis(
    ['launcher.py'],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'notebook',
        'IPython',
        'scipy',
        'pandas',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhishingDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console for server logs
    icon=None,  # Add an .ico file path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhishingDetector',
)
