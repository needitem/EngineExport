# Changelog

## [Unreleased]

### Changed
- Updated TensorRT from 10.8.0.43 to 10.14.1.48
- Replaced single `nvinfer_builder_resource_10.dll` with architecture-specific builder resources:
  - `nvinfer_builder_resource_sm75_10.dll` (Turing)
  - `nvinfer_builder_resource_sm80_10.dll` (Ampere)
  - `nvinfer_builder_resource_sm86_10.dll` (Ampere)
  - `nvinfer_builder_resource_sm89_10.dll` (Ada Lovelace)
  - `nvinfer_builder_resource_sm90_10.dll` (Hopper)
  - `nvinfer_builder_resource_sm120_10.dll` (Blackwell)
  - `nvinfer_builder_resource_ptx_10.dll`
- Added new TensorRT DLLs:
  - `nvinfer_dispatch_10.dll`
  - `nvinfer_lean_10.dll`
  - `nvinfer_vc_plugin_10.dll`
- Updated `setup_deps.bat` to reference new TensorRT path
- Updated CMakeLists.txt DLL copy commands for new TensorRT structure

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