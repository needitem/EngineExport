@echo off
echo Setting up dependencies for EngineExport...

REM Create deps directory structure
if not exist deps mkdir deps
cd deps

REM Copy TensorRT from needaimbot project
if exist "..\..\..\needaimbot\modules\TensorRT-10.8.0.43" (
    echo Copying TensorRT from needaimbot project...
    xcopy /E /I /Y "..\..\..\needaimbot\modules\TensorRT-10.8.0.43" "TensorRT"
) else (
    echo TensorRT not found in needaimbot project!
    echo Please ensure TensorRT is available at:
    echo   needaimbot\modules\TensorRT-10.8.0.43
    echo Or manually copy TensorRT to deps\TensorRT\
)

cd ..

echo Dependencies setup completed!
echo.
echo Next steps:
echo 1. Ensure CUDA Toolkit is installed
echo 2. Run build.bat to compile the project
echo.

pause