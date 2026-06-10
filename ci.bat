@echo off
echo ========================================
echo   CHU-HEN CI
echo ========================================
echo.

REM 1. import check
echo [1/3] Import check...
python -c "from app.api.app import app; print('OK')"
if %errorlevel% neq 0 goto fail

REM 2. tests
echo.
echo [2/3] Tests...
python -m pytest tests/ -q --tb=short
if %errorlevel% neq 0 goto fail

REM 3. audit
echo.
echo [3/3] Audit...
python scripts/audit.py --quick 2>&1
echo   (done)

echo.
echo ========================================
echo   ALL PASSED
echo ========================================
goto end

:fail
echo.
echo ========================================
echo   FAILED
echo ========================================
exit /b 1

:end
