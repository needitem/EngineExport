# EngineExport

A standalone utility for converting ONNX models to TensorRT engines with optimized settings.

## Features

- ONNX to TensorRT engine conversion
- FP16 precision support
- Configurable input resolution
- Optimized build settings for inference performance
- Command line interface for batch processing

## Requirements

- NVIDIA GPU with CUDA support
- TensorRT 10.x
- CUDA 11.8+

## Usage

```bash
EngineExport.exe <input.onnx> [options]
```

### Options

- `--resolution <size>`: Input resolution (default: 640)
- `--fp16`: Enable FP16 precision
- `--output <path>`: Output engine file path
- `--workspace <mb>`: Workspace size in MB (default: 1024)

## Example

```bash
EngineExport.exe model.onnx --resolution 640 --fp16 --output model_640_fp16.engine
```

## Build

Requires Visual Studio 2022 and CUDA Toolkit.

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```