@echo off
echo 🚀 Hugging Face Dataset Uploader
echo ==============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo 📦 Installing required packages...
pip install -r upload_requirements.txt

echo.
echo 🎙️ Starting dataset upload process...
python upload_to_huggingface.py

echo.
echo Press any key to exit...
pause >nul