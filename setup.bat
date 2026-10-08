@echo off
rem One-time setup on a new Windows computer. Double-click this file, or run  setup.bat  in the MaterialStack folder.
rem Needs Python 3.12 or newer from python.org (tick "Add python.exe to PATH" when installing).
rem Takes a few minutes (installing packages). The data, the trained model and the built web UI are in the
rem repository, so nothing else is downloaded or built. To rebuild them yourself:
rem   python build_data.py, python train.py, python validate.py  (and: cd frontend, npm ci, npm run build)
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12 or newer is needed, found ' + sys.version.split()[0]" || goto :error

echo == 1/3 creating the virtual environment (.venv)
%PY% -m venv .venv || goto :error
.venv\Scripts\python -m pip install --quiet --upgrade pip || goto :error
echo == 2/3 installing packages (requirements.txt)
.venv\Scripts\python -m pip install --quiet -r requirements.txt || goto :error
echo == 3/3 checking with one prediction
.venv\Scripts\python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD || goto :error

echo.
echo Ready. Start the web UI with:
echo   .venv\Scripts\python -m materialstack serve      (then open http://127.0.0.1:8000)
pause
exit /b 0

:error
echo.
echo Setup stopped because of the error above.
pause
exit /b 1
