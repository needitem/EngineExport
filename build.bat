@echo off
echo Building EngineExport...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64

REM Build Release configuration
cmake --build . --config Release --parallel

echo Build completed!
echo Executable location: build\bin\Release\EngineExport.exe

pause