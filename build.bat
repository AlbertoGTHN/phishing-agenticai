@echo off
echo ============================================
echo  Building Agentic AI Phishing Detector .exe
echo ============================================
echo.

:: Install PyInstaller if not present
pip install pyinstaller
echo.

:: Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Run PyInstaller
echo Building with PyInstaller...
pyinstaller phishing_detector.spec --noconfirm

echo.
if exist "dist\PhishingDetector\PhishingDetector.exe" (
    echo ============================================
    echo  BUILD SUCCESSFUL!
    echo  Output: dist\PhishingDetector\
    echo  Run:    dist\PhishingDetector\PhishingDetector.exe
    echo ============================================
) else (
    echo ============================================
    echo  BUILD FAILED - Check errors above
    echo ============================================
)
pause
