# EngineExport GUI

A user-friendly GUI application for converting ONNX models to TensorRT engines with optimized settings.

![EngineExport Screenshot](screenshot.png)

## Features

- **🖥️ Intuitive GUI Interface** - Easy-to-use graphical interface
- **📁 File Browser Integration** - Built-in file dialogs for selecting ONNX files
- **⚙️ Flexible Export Options** - Configure resolution, precision, and workspace
- **📊 Real-time Progress** - Live progress tracking with detailed logging
- **🚀 Optimized Performance** - FP16/FP8 precision support for maximum speed
- **📋 Auto-naming** - Intelligent output file naming based on settings

## Requirements

- **NVIDIA GPU** with CUDA support
- **TensorRT 10.x** (automatically copied from needaimbot)
- **CUDA 11.8+**
- **Visual Studio 2022** (for building)

## Quick Start

### 1. Setup Dependencies
```bash
setup_deps.bat
```
This will automatically download GLFW, ImGui, and copy TensorRT from needaimbot.

### 2. Build the Application
```bash
build.bat
```

### 3. Run EngineExport
```bash
build\bin\Release\EngineExport.exe
```

## Usage

### GUI Mode (Default)
1. **Select ONNX File** - Click "Browse" to choose your input ONNX model
2. **Configure Options**:
   - **Resolution**: Choose input size (160, 320, 416, 640, 1280)
   - **FP16**: Enable for ~2x speedup on compatible GPUs
   - **FP8**: Enable for maximum performance (Ada Lovelace+)
   - **Workspace**: GPU memory allocation (256MB - 4GB)
3. **Start Export** - Click "Start Export" and monitor progress
4. **View Logs** - Real-time conversion status and error messages

### Export Options Explained

| Option | Description | Recommended |
|--------|-------------|-------------|
| **Resolution** | Model input size (WxH) | Match your model's training resolution |
| **FP16** | Half-precision floating point | ✅ Enable for most GPUs (RTX 20xx+) |
| **FP8** | 8-bit floating point | ⚠️ RTX 40xx only, experimental |
| **Workspace** | TensorRT workspace memory | 1GB for most models, 2GB+ for large models |

### Output Files

The application automatically generates output filenames based on your settings:
- `model.onnx` → `model_640.engine` (640 resolution, FP32)
- `model.onnx` → `model_640_fp16.engine` (640 resolution, FP16)
- `model.onnx` → `model_320_fp16_fp8.engine` (320 resolution, FP16+FP8)

## Build from Source

### Prerequisites
- Visual Studio 2022 with C++ development tools
- CUDA Toolkit 11.8 or later
- CMake 3.18+

### Build Steps
```bash
# 1. Setup dependencies (downloads GLFW, ImGui)
setup_deps.bat

# 2. Build project
build.bat

# 3. Run application
build\bin\Release\EngineExport.exe
```

## Project Structure

```
EngineExport/
├── src/                    # Source code
│   ├── main.cpp           # Application entry point
│   ├── gui_app.cpp/.h     # Main GUI application
│   ├── engine_exporter.*  # TensorRT conversion logic
│   └── config.*           # Configuration management
├── deps/                   # Dependencies (auto-downloaded)
│   ├── TensorRT/          # Copied from needaimbot
│   ├── glfw/              # Window management
│   └── imgui/             # GUI framework
└── build/                  # Build output
    └── bin/Release/
        └── EngineExport.exe
```

## Troubleshooting

### Common Issues

**"Failed to initialize application"**
- Ensure your GPU supports OpenGL 3.3+
- Update graphics drivers

**"TensorRT not found"**
- Run `setup_deps.bat` first
- Verify needaimbot project exists with TensorRT

**"CUDA out of memory"**
- Reduce workspace size to 512MB or 256MB
- Close other GPU-intensive applications

**"Engine export failed"**
- Check log window for detailed error messages
- Verify ONNX model is valid
- Ensure sufficient disk space

### Getting Help

1. Check the log window for detailed error messages
2. Verify all dependencies are properly installed
3. Ensure CUDA drivers are up to date
4. Try reducing workspace size or disabling precision options

## License

This project is part of the needaimbot ecosystem and follows the same licensing terms.