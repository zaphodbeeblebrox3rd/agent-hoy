@echo off
REM Conda setup script for agent-hoy (Windows)

echo Setting up agent-hoy with conda...
echo ==================================

REM Check if conda is installed
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda not found. Please install Miniconda or Anaconda first.
    echo    Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ Conda found

REM Update conda
echo Updating conda...
conda update -n base -c defaults conda -y

REM Create environment
echo Creating conda environment...
conda env create -f environment.yaml

REM Activate environment
echo Activating environment...
call conda activate agent-hoy

REM Test installation
echo Testing installation...
python test_setup.py

echo.
echo 🎉 Setup complete!
echo.
echo To use the application:
echo 1. conda activate agent-hoy
echo 2. python main.py
echo.
echo To remove the environment:
echo conda env remove -n agent-hoy
echo.
pause
