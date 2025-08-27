@echo off
echo Setting up dependencies for EngineExport GUI...

REM Create deps directory structure
if not exist deps mkdir deps
cd deps

REM Copy TensorRT from needaimbot project
if exist "..\..\needaimbot\needaimbot\modules\TensorRT-10.8.0.43" (
    echo Copying TensorRT from needaimbot project...
    xcopy /E /I /Y "..\..\needaimbot\needaimbot\modules\TensorRT-10.8.0.43" "TensorRT"
) else (
    echo TensorRT not found in needaimbot project!
    echo Please ensure TensorRT is available at:
    echo   needaimbot\needaimbot\modules\TensorRT-10.8.0.43
    echo Or manually copy TensorRT to deps\TensorRT\
)

REM Download and setup GLFW (pre-compiled binaries)
if not exist glfw (
    echo Downloading GLFW...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip' -OutFile 'glfw.zip'"
    powershell -Command "Expand-Archive -Path 'glfw.zip' -DestinationPath '.'"
    ren glfw-3.4.bin.WIN64 glfw
    del glfw.zip
    echo GLFW downloaded and extracted
) else (
    echo GLFW already exists
)

REM Download and setup ImGui
if not exist imgui (
    echo Downloading ImGui...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ocornut/imgui/archive/refs/heads/docking.zip' -OutFile 'imgui.zip'"
    powershell -Command "Expand-Archive -Path 'imgui.zip' -DestinationPath '.'"
    ren imgui-docking imgui
    del imgui.zip
    echo ImGui downloaded and extracted
) else (
    echo ImGui already exists
)

cd ..

echo Dependencies setup completed!
echo.
echo Next steps:
echo 1. Ensure CUDA Toolkit is installed
echo 2. Run build.bat to compile the project
echo.

pause