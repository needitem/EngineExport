# Changelog

## v1.0.0 - Initial Release

### Features
- ONNX to TensorRT engine conversion
- Command-line interface with flexible options
- Support for FP16 and FP8 precision modes
- Configurable input resolution
- Workspace memory configuration
- Detailed logging and verbose output
- Cross-platform CMake build system
- Batch processing examples

### Supported Options
- `--resolution`: Set input resolution (default: 640)
- `--fp16`: Enable FP16 precision
- `--fp8`: Enable FP8 precision
- `--output`: Custom output path
- `--workspace`: Workspace memory size
- `--verbose`: Enable detailed logging
- `--detailed-profiling`: Enable TensorRT profiling

### Performance Optimizations
- GPU fallback support
- Precision constraints
- Optimized tactic sources (CUBLAS, CUBLAS_LT, CUDNN)
- Optimization profiles for dynamic inputs

### Build Requirements
- Visual Studio 2022
- CUDA Toolkit 11.8+
- TensorRT 10.x
- CMake 3.18+