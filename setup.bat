@echo off
rem One-time setup on a new Windows computer. Double-click this file, or run  setup.bat  in the MaterialStack folder.
rem Needs Python 3.11 or newer from python.org (tick "Add python.exe to PATH" when installing).
rem Takes ~5 minutes (mostly installing packages).
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11 or newer is needed, found ' + sys.version.split()[0]" || goto :error

echo == 1/5 creating the virtual environment (.venv)
%PY% -m venv .venv || goto :error
.venv\Scripts\python -m pip install --quiet --upgrade pip || goto :error
echo == 2/5 installing packages (requirements.txt)
.venv\Scripts\python -m pip install --quiet -r requirements.txt || goto :error
echo == 3/5 building the data tables
.venv\Scripts\python build_data.py || goto :error
echo == 4/5 training the model
.venv\Scripts\python train.py || goto :error
echo == 5/5 validating
.venv\Scripts\python validate.py > nul || goto :error
echo validation done: results\metrics.csv

if not exist frontend\dist\index.html (
  where npm >nul 2>nul && (
    echo == building the web UI
    pushd frontend && call npm ci && call npm run build & popd
  ) || (
    echo note: the web UI is not built and Node.js is not installed. The command line works;
    echo       install Node.js 20+ and run "cd frontend && npm ci && npm run build" for the web UI.
  )
)

echo.
echo Done. Try:
echo   .venv\Scripts\python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD
echo   .venv\Scripts\python -m materialstack serve      (then open http://127.0.0.1:8000)
pause
exit /b 0

:error
echo.
echo Setup stopped because of the error above.
pause
exit /b 1
