#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include "config.h"

// STB Image libraries for image loading/saving
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Include GLFW and ImGui for display
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/gl.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct Detection {
    float x, y, w, h;
    float confidence;
    int classId;
};

class TRTEngine {
private:
    Logger gLogger;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    void* buffers[2];
    cudaStream_t stream;
    
    int inputIndex;
    int outputIndex;
    size_t inputSize;
    size_t outputSize;
    
    // Model dimensions
    int inputH = 640;
    int inputW = 640;
    int numClasses = 80;
    int maxDetections = 8400;
    
public:
    TRTEngine() : buffers{nullptr, nullptr} {
        cudaStreamCreate(&stream);
    }
    
    ~TRTEngine() {
        if (buffers[0]) cudaFree(buffers[0]);
        if (buffers[1]) cudaFree(buffers[1]);
        cudaStreamDestroy(stream);
    }
    
    bool loadEngine(const std::string& enginePath) {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Cannot open engine file: " << enginePath << std::endl;
            return false;
        }
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // TensorRT 10 API - find tensor indices
        inputIndex = -1;
        outputIndex = -1;
        
        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* tensorName = engine->getIOTensorName(i);
            if (strcmp(tensorName, "images") == 0) {
                inputIndex = i;
            } else if (strcmp(tensorName, "output0") == 0) {
                outputIndex = i;
            }
        }
        
        if (inputIndex == -1 || outputIndex == -1) {
            std::cerr << "Failed to find input/output tensors" << std::endl;
            return false;
        }
        
        auto inputDims = engine->getTensorShape(engine->getIOTensorName(inputIndex));
        auto outputDims = engine->getTensorShape(engine->getIOTensorName(outputIndex));
        
        inputSize = 1 * 3 * inputH * inputW * sizeof(float);
        outputSize = 1 * (numClasses + 4) * maxDetections * sizeof(float);
        
        cudaMalloc(&buffers[0], inputSize);  // Input buffer
        cudaMalloc(&buffers[1], outputSize); // Output buffer
        
        std::cout << "Engine loaded successfully!" << std::endl;
        std::cout << "Input shape: " << inputDims.d[0] << "x" << inputDims.d[1] 
                  << "x" << inputDims.d[2] << "x" << inputDims.d[3] << std::endl;
        std::cout << "Output shape: " << outputDims.d[0] << "x" << outputDims.d[1] 
                  << "x" << outputDims.d[2] << std::endl;
        
        return true;
    }
    
    std::vector<Detection> infer(unsigned char* imageData, int width, int height, int channels) {
        // Resize and preprocess image
        std::vector<float> inputData(3 * inputH * inputW);
        
        // Simple bilinear resize and normalize
        float scaleX = (float)width / inputW;
        float scaleY = (float)height / inputH;
        
        for (int y = 0; y < inputH; y++) {
            for (int x = 0; x < inputW; x++) {
                int srcX = (int)(x * scaleX);
                int srcY = (int)(y * scaleY);
                srcX = std::min(srcX, width - 1);
                srcY = std::min(srcY, height - 1);
                
                int srcIdx = (srcY * width + srcX) * channels;
                
                // Convert to RGB and normalize to [0, 1]
                for (int c = 0; c < 3; c++) {
                    int dstIdx = c * inputH * inputW + y * inputW + x;
                    if (c < channels) {
                        inputData[dstIdx] = imageData[srcIdx + c] / 255.0f;
                    } else {
                        inputData[dstIdx] = 0.0f;
                    }
                }
            }
        }
        
        // Copy to GPU
        cudaMemcpyAsync(buffers[0], inputData.data(), inputSize, 
                       cudaMemcpyHostToDevice, stream);
        
        // Run inference - TensorRT 10 API
        const char* inputName = engine->getIOTensorName(inputIndex);
        const char* outputName = engine->getIOTensorName(outputIndex);
        
        context->setTensorAddress(inputName, buffers[0]);
        context->setTensorAddress(outputName, buffers[1]);
        
        bool status = context->enqueueV3(stream);
        if (!status) {
            std::cerr << "Failed to run inference" << std::endl;
        }
        
        // Copy output back
        std::vector<float> outputData(outputSize / sizeof(float));
        cudaMemcpyAsync(outputData.data(), buffers[1], outputSize, 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Postprocess
        return postprocess(outputData.data());
    }
    
private:
    std::vector<Detection> postprocess(float* output, float confThreshold = 0.25f) {
        std::vector<Detection> detections;
        
        for (int i = 0; i < maxDetections; i++) {
            float* ptr = output + i * (numClasses + 4);
            
            float maxScore = 0;
            int maxClassId = 0;
            for (int j = 4; j < numClasses + 4; j++) {
                if (ptr[j] > maxScore) {
                    maxScore = ptr[j];
                    maxClassId = j - 4;
                }
            }
            
            if (maxScore > confThreshold) {
                Detection det;
                det.x = ptr[0];
                det.y = ptr[1];
                det.w = ptr[2];
                det.h = ptr[3];
                det.confidence = maxScore;
                det.classId = maxClassId;
                detections.push_back(det);
            }
        }
        
        // Simple NMS
        std::vector<Detection> nmsResult;
        std::vector<bool> suppressed(detections.size(), false);
        
        for (size_t i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            
            nmsResult.push_back(detections[i]);
            
            for (size_t j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                if (detections[i].classId != detections[j].classId) continue;
                
                float iou = calculateIoU(detections[i], detections[j]);
                if (iou > 0.45f) {
                    suppressed[j] = true;
                }
            }
        }
        
        return nmsResult;
    }
    
    float calculateIoU(const Detection& a, const Detection& b) {
        float x1_a = a.x - a.w / 2;
        float y1_a = a.y - a.h / 2;
        float x2_a = a.x + a.w / 2;
        float y2_a = a.y + a.h / 2;
        
        float x1_b = b.x - b.w / 2;
        float y1_b = b.y - b.h / 2;
        float x2_b = b.x + b.w / 2;
        float y2_b = b.y + b.h / 2;
        
        float x1_i = std::max(x1_a, x1_b);
        float y1_i = std::max(y1_a, y1_b);
        float x2_i = std::min(x2_a, x2_b);
        float y2_i = std::min(y2_a, y2_b);
        
        float intersection = std::max(0.0f, x2_i - x1_i) * std::max(0.0f, y2_i - y1_i);
        float area_a = a.w * a.h;
        float area_b = b.w * b.h;
        float unionArea = area_a + area_b - intersection;
        
        return intersection / (unionArea + 1e-6f);
    }
};

// Simple video decoder using FFmpeg (command line)
class SimpleVideoDecoder {
private:
    std::string videoPath;
    std::string tempDir;
    int frameCount;
    int currentFrame;
    
public:
    SimpleVideoDecoder(const std::string& path) : videoPath(path), currentFrame(0) {
        tempDir = "temp_frames";
        extractFrames();
    }
    
    ~SimpleVideoDecoder() {
        cleanup();
    }
    
    void extractFrames() {
        // Create temp directory
        std::string mkdirCmd = "mkdir " + tempDir + " 2>nul";
        system(mkdirCmd.c_str());
        
        // Extract frames using ffmpeg
        std::string cmd = "ffmpeg -i \"" + videoPath + "\" -q:v 2 " + tempDir + "/frame_%04d.jpg -y 2>nul";
        std::cout << "Extracting frames from video..." << std::endl;
        system(cmd.c_str());
        
        // Count frames
        frameCount = 0;
        while (true) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%04d.jpg", tempDir.c_str(), frameCount + 1);
            std::ifstream test(filename);
            if (!test.good()) break;
            frameCount++;
        }
        
        std::cout << "Extracted " << frameCount << " frames" << std::endl;
    }
    
    unsigned char* getNextFrame(int& width, int& height, int& channels) {
        if (currentFrame >= frameCount) return nullptr;
        
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%04d.jpg", tempDir.c_str(), currentFrame + 1);
        currentFrame++;
        
        return stbi_load(filename, &width, &height, &channels, 0);
    }
    
    void reset() {
        currentFrame = 0;
    }
    
    int getTotalFrames() const { return frameCount; }
    int getCurrentFrame() const { return currentFrame; }
    
    void cleanup() {
        std::string cmd = "rmdir /s /q " + tempDir + " 2>nul";
        system(cmd.c_str());
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <engine_file> <video_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " model.engine test/test_det.mp4" << std::endl;
        return -1;
    }
    
    std::string enginePath = argv[1];
    std::string videoPath = argv[2];
    
    // Initialize TensorRT engine
    TRTEngine engine;
    if (!engine.loadEngine(enginePath)) {
        std::cerr << "Failed to load engine" << std::endl;
        return -1;
    }
    
    // Initialize video decoder
    SimpleVideoDecoder decoder(videoPath);
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "TensorRT Engine Test", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    
    // Performance metrics
    std::vector<double> inferTimes;
    const int avgWindow = 30;
    bool isPaused = false;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        if (!isPaused) {
            int width, height, channels;
            unsigned char* frameData = decoder.getNextFrame(width, height, channels);
            
            if (frameData) {
                auto startTime = std::chrono::high_resolution_clock::now();
                
                // Run inference
                auto detections = engine.infer(frameData, width, height, channels);
                
                auto endTime = std::chrono::high_resolution_clock::now();
                double inferTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
                
                inferTimes.push_back(inferTime);
                if (inferTimes.size() > avgWindow) {
                    inferTimes.erase(inferTimes.begin());
                }
                
                // Draw frame with detections
                GLuint texture;
                glGenTextures(1, &texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, 
                           channels == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, frameData);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                
                // Display in ImGui window
                ImGui::SetNextWindowPos(ImVec2(0, 0));
                ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
                ImGui::Begin("Video", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                
                // Draw performance metrics
                double avgInferTime = 0;
                if (!inferTimes.empty()) {
                    avgInferTime = std::accumulate(inferTimes.begin(), inferTimes.end(), 0.0) / inferTimes.size();
                }
                double avgFPS = avgInferTime > 0 ? 1000.0 / avgInferTime : 0;
                
                ImGui::SetCursorPos(ImVec2(10, 10));
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0.7f));
                ImGui::BeginChild("Stats", ImVec2(200, 80), true);
                ImGui::Text("FPS: %.1f", avgFPS);
                ImGui::Text("Latency: %.1f ms", avgInferTime);
                ImGui::Text("Frame: %d/%d", decoder.getCurrentFrame(), decoder.getTotalFrames());
                ImGui::Text("Detections: %zu", detections.size());
                ImGui::EndChild();
                ImGui::PopStyleColor();
                
                // Draw video frame
                ImVec2 imageSize(width, height);
                ImGui::SetCursorPos(ImVec2((ImGui::GetWindowWidth() - imageSize.x) * 0.5f,
                                          (ImGui::GetWindowHeight() - imageSize.y) * 0.5f));
                ImGui::Image((void*)(intptr_t)texture, imageSize);
                
                // Draw detections overlay
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                ImVec2 imagePos = ImGui::GetCursorScreenPos();
                imagePos.y -= imageSize.y;
                
                for (const auto& det : detections) {
                    float x1 = (det.x - det.w/2) * width;
                    float y1 = (det.y - det.h/2) * height;
                    float x2 = (det.x + det.w/2) * width;
                    float y2 = (det.y + det.h/2) * height;
                    
                    ImVec2 p1(imagePos.x + x1, imagePos.y + y1);
                    ImVec2 p2(imagePos.x + x2, imagePos.y + y2);
                    
                    drawList->AddRect(p1, p2, IM_COL32(0, 255, 0, 255), 0.0f, 0, 2.0f);
                    
                    char label[64];
                    snprintf(label, sizeof(label), "Class %d: %.0f%%", 
                            det.classId, det.confidence * 100);
                    drawList->AddText(p1, IM_COL32(255, 255, 0, 255), label);
                }
                
                ImGui::End();
                
                // Controls
                ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 60));
                ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize);
                if (ImGui::Button(isPaused ? "Resume" : "Pause")) {
                    isPaused = !isPaused;
                }
                ImGui::SameLine();
                if (ImGui::Button("Restart")) {
                    decoder.reset();
                }
                ImGui::SameLine();
                if (ImGui::Button("Quit")) {
                    glfwSetWindowShouldClose(window, true);
                }
                ImGui::End();
                
                glDeleteTextures(1, &texture);
                stbi_image_free(frameData);
            } else {
                // End of video, restart
                decoder.reset();
            }
        }
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    std::cout << "\nTest completed!" << std::endl;
    
    return 0;
}