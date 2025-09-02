cmake_minimum_required(VERSION 3.18)

project(EngineExport LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# TensorRT
if(WIN32)
    set(TENSORRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/deps/TensorRT)
else()
    set(TENSORRT_ROOT "/usr/src/tensorrt")
endif()

# TensorRT includes
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)

# TensorRT libraries - specify exact names for Windows
if(WIN32)
    set(TENSORRT_LIBRARY ${TENSORRT_ROOT}/lib/nvinfer_10.lib)
    set(TENSORRT_ONNX_PARSER_LIBRARY ${TENSORRT_ROOT}/lib/nvonnxparser_10.lib)
else()
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_ROOT}
        PATH_SUFFIXES lib lib64)
    
    find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
        HINTS ${TENSORRT_ROOT}
        PATH_SUFFIXES lib lib64)
endif()

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW
set(GLFW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw)
set(GLFW_INCLUDE_DIR ${GLFW_ROOT}/include)

if(WIN32)
    set(GLFW_LIBRARY ${GLFW_ROOT}/lib-vc2022/glfw3.lib)
else()
    find_library(GLFW_LIBRARY glfw3
        HINTS ${GLFW_ROOT}
        PATH_SUFFIXES lib)
endif()

# ImGui
set(IMGUI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/deps/imgui)
set(IMGUI_SOURCES
    ${IMGUI_ROOT}/imgui.cpp
    ${IMGUI_ROOT}/imgui_demo.cpp
    ${IMGUI_ROOT}/imgui_draw.cpp
    ${IMGUI_ROOT}/imgui_tables.cpp
    ${IMGUI_ROOT}/imgui_widgets.cpp
    ${IMGUI_ROOT}/backends/imgui_impl_glfw.cpp
    ${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp
)

# Include directories
include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_ROOT}
    ${IMGUI_ROOT}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/deps
)

# Source files
set(SOURCES
    src/main.cpp
    src/engine_exporter.cpp
    src/config.cpp
    src/logger.cpp
    src/gui_app.cpp
    ${IMGUI_SOURCES}
)

# Main executable (console app for now to avoid WinMain issues)
add_executable(${PROJECT_NAME} ${SOURCES})

# Engine tester executable
add_executable(engine_tester 
    src/engine_tester.cpp
    src/config.cpp
    src/logger.cpp
    ${IMGUI_SOURCES}
)

# Link libraries for main executable
target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    ${GLFW_LIBRARY}
    ${OPENGL_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)

# Link libraries for engine tester
target_link_libraries(engine_tester
    ${TENSORRT_LIBRARY}
    ${GLFW_LIBRARY}
    ${OPENGL_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
)

# Compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(engine_tester PRIVATE /W4)
    target_compile_definitions(engine_tester PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(engine_tester PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(engine_tester PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy TensorRT DLLs on Windows
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${TENSORRT_ROOT}/lib
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
    
    # Copy GLFW DLL if it exists
    if(EXISTS ${GLFW_ROOT}/lib-vc2022/glfw3.dll)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GLFW_ROOT}/lib-vc2022/glfw3.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
        )
    endif()
endif()