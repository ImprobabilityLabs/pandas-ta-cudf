@echo off
REM Test script for pandas-ta-cudf conversion (Windows)
REM This script verifies that the cudf conversion is working correctly

echo ==========================================
echo Pandas TA cuDF Conversion Test Script
echo ==========================================
echo.

python test_cudf_conversion.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo All tests completed successfully!
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo Some tests failed. Check output above.
    echo ==========================================
)

pause

