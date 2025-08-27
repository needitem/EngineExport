@echo off
REM Example script for converting ONNX models to TensorRT engines

REM Set paths
set EXPORTER=..\build\bin\Release\EngineExport.exe
set MODEL_DIR=models
set OUTPUT_DIR=engines

REM Create directories if they don't exist
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Example conversions
echo Converting YOLO model with different configurations...

REM Standard conversion (FP32, 640x640)
%EXPORTER% %MODEL_DIR%\yolo.onnx --resolution 640 --output %OUTPUT_DIR%\yolo_640.engine --verbose

REM FP16 conversion for better performance
%EXPORTER% %MODEL_DIR%\yolo.onnx --resolution 640 --fp16 --output %OUTPUT_DIR%\yolo_640_fp16.engine --verbose

REM Different input resolutions
%EXPORTER% %MODEL_DIR%\yolo.onnx --resolution 320 --fp16 --output %OUTPUT_DIR%\yolo_320_fp16.engine --verbose
%EXPORTER% %MODEL_DIR%\yolo.onnx --resolution 1280 --fp16 --output %OUTPUT_DIR%\yolo_1280_fp16.engine --verbose

REM With detailed profiling
%EXPORTER% %MODEL_DIR%\yolo.onnx --resolution 640 --fp16 --detailed-profiling --verbose

echo Conversion completed!
pause