#include "gui_app.h"
#include "engine_exporter.h"
#include "config.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

GuiApp::GuiApp() {
}

GuiApp::~GuiApp() {
    shutdown();
}

bool GuiApp::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // GL 3.3 + GLSL 330
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    m_window = glfwCreateWindow(800, 600, "EngineExport - ONNX to TensorRT Converter", NULL, NULL);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Custom styling
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 2.0f;
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    addLog("EngineExport GUI initialized successfully");
    return true;
}

void GuiApp::run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Process log messages from export thread
        processLogQueue();

        // Render main window
        renderMainWindow();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(m_window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window);
    }
}

void GuiApp::shutdown() {
    // Wait for export thread to finish
    if (m_exportThread && m_exportThread->joinable()) {
        m_exportThread->join();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

void GuiApp::renderMainWindow() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | 
                                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    
    ImGui::Begin("EngineExport", nullptr, window_flags);

    // Header
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 10));
    ImGui::Text("EngineExport - ONNX to TensorRT Engine Converter");
    ImGui::Separator();
    ImGui::PopStyleVar();

    // File selection section
    if (ImGui::CollapsingHeader("File Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderFileSelection();
    }

    ImGui::Spacing();

    // Export options section
    if (ImGui::CollapsingHeader("Export Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderExportOptions();
    }

    ImGui::Spacing();

    // Export button and progress
    renderExportButton();
    renderProgressBar();

    ImGui::Spacing();

    // Log window
    if (ImGui::CollapsingHeader("Log Output", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderLogWindow();
    }

    ImGui::End();
}

void GuiApp::renderFileSelection() {
    // Input ONNX file
    ImGui::Text("Input ONNX File:");
    ImGui::PushItemWidth(-100);
    ImGui::InputText("##InputPath", m_inputPath, sizeof(m_inputPath));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Browse##Input")) {
        std::string path;
        if (openFileDialog(path, "ONNX Files\0*.onnx\0All Files\0*.*\0")) {
            strncpy_s(m_inputPath, path.c_str(), sizeof(m_inputPath) - 1);
            std::string outputPath = generateOutputPath();
            strncpy_s(m_outputPath, outputPath.c_str(), sizeof(m_outputPath) - 1);
        }
    }

    ImGui::Spacing();

    // Output engine file
    ImGui::Text("Output Engine File:");
    ImGui::PushItemWidth(-100);
    ImGui::InputText("##OutputPath", m_outputPath, sizeof(m_outputPath));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Browse##Output")) {
        std::string path;
        if (saveFileDialog(path, "Engine Files\0*.engine\0All Files\0*.*\0")) {
            strncpy_s(m_outputPath, path.c_str(), sizeof(m_outputPath) - 1);
        }
    }
}

void GuiApp::renderExportOptions() {
    // Resolution selection
    ImGui::Text("Input Resolution:");
    ImGui::SameLine();
    helpMarker("The input resolution for the model (e.g., 640 for 640x640)");
    
    const char* resolution_items[] = { "160", "320", "416", "640", "1280" };
    int current_resolution_index = 2; // default to 640
    if (m_resolution == 160) current_resolution_index = 0;
    else if (m_resolution == 320) current_resolution_index = 1;
    else if (m_resolution == 416) current_resolution_index = 2;
    else if (m_resolution == 640) current_resolution_index = 3;
    else if (m_resolution == 1280) current_resolution_index = 4;
    
    if (ImGui::Combo("##Resolution", &current_resolution_index, resolution_items, IM_ARRAYSIZE(resolution_items))) {
        switch (current_resolution_index) {
            case 0: m_resolution = 160; break;
            case 1: m_resolution = 320; break;
            case 2: m_resolution = 416; break;
            case 3: m_resolution = 640; break;
            case 4: m_resolution = 1280; break;
        }
        if (strlen(m_inputPath) > 0) {
            std::string outputPath = generateOutputPath();
            strncpy_s(m_outputPath, outputPath.c_str(), sizeof(m_outputPath) - 1);
        }
    }

    ImGui::Spacing();

    // Precision options
    ImGui::Text("Precision Options:");
    ImGui::Checkbox("Enable FP16", &m_enableFp16);
    ImGui::SameLine();
    helpMarker("Enable FP16 precision for better performance on supported GPUs");
    
    ImGui::Checkbox("Enable FP8", &m_enableFp8);
    ImGui::SameLine();
    helpMarker("Enable FP8 precision (experimental, requires Ada Lovelace or newer)");

    ImGui::Spacing();

    // Workspace size
    ImGui::Text("Workspace Size:");
    ImGui::SliderInt("MB##Workspace", &m_workspaceMb, 256, 4096, "%d MB");
    helpMarker("Amount of GPU memory to use for TensorRT workspace");

    ImGui::Spacing();

    // Verbose output
    ImGui::Checkbox("Verbose Output", &m_verbose);
    ImGui::SameLine();
    helpMarker("Enable detailed logging during conversion");
}

void GuiApp::renderExportButton() {
    bool canExport = m_exportStatus == ExportStatus::IDLE && validateInputs();
    
    if (!canExport) {
        ImGui::BeginDisabled();
    }
    
    if (ImGui::Button("Start Export", ImVec2(150, 40))) {
        startExport();
    }
    
    if (!canExport) {
        ImGui::EndDisabled();
    }
    
    if (m_exportStatus == ExportStatus::RUNNING) {
        ImGui::SameLine();
        ImGui::Text("Converting...");
    } else if (m_exportStatus == ExportStatus::COMPLETED) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Conversion Completed!");
    } else if (m_exportStatus == ExportStatus::FAILED) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Conversion Failed");
    }
}

void GuiApp::renderProgressBar() {
    if (m_exportStatus == ExportStatus::RUNNING) {
        ImGui::ProgressBar(m_exportProgress.load(), ImVec2(-1.0f, 0.0f));
    }
}

void GuiApp::renderLogWindow() {
    static std::vector<LogEntry> displayLogs;
    
    // Process new log entries
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        while (!m_logQueue.empty()) {
            displayLogs.push_back(m_logQueue.front());
            m_logQueue.pop();
        }
    }
    
    // Limit log entries to prevent memory growth
    if (displayLogs.size() > 1000) {
        displayLogs.erase(displayLogs.begin(), displayLogs.begin() + 500);
    }
    
    ImGui::BeginChild("LogWindow", ImVec2(0, 200), true);
    
    for (const auto& log : displayLogs) {
        ImVec4 color = log.isError ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        ImGui::TextColored(color, "%s", log.message.c_str());
    }
    
    // Auto-scroll to bottom
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }
    
    ImGui::EndChild();
}

#ifdef _WIN32
bool GuiApp::openFileDialog(std::string& path, const char* filter) {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwGetWin32Window(m_window);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileName(&ofn)) {
        path = szFile;
        return true;
    }
    return false;
}

bool GuiApp::saveFileDialog(std::string& path, const char* filter) {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwGetWin32Window(m_window);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetSaveFileName(&ofn)) {
        path = szFile;
        return true;
    }
    return false;
}
#else
bool GuiApp::openFileDialog(std::string& path, const char* filter) {
    // For non-Windows platforms, would need to implement native dialogs
    // or use a cross-platform library
    return false;
}

bool GuiApp::saveFileDialog(std::string& path, const char* filter) {
    return false;
}
#endif

void GuiApp::startExport() {
    if (m_exportThread && m_exportThread->joinable()) {
        m_exportThread->join();
    }
    
    m_exportStatus = ExportStatus::RUNNING;
    m_exportProgress = 0.0f;
    m_exportError.clear();
    
    m_exportThread = std::make_unique<std::thread>(&GuiApp::exportThreadFunc, this);
}

void GuiApp::exportThreadFunc() {
    try {
        // Create config from GUI settings
        ExportConfig config;
        config.input_onnx_path = std::string(m_inputPath);
        config.output_engine_path = std::string(m_outputPath);
        config.input_resolution = m_resolution;
        config.enable_fp16 = m_enableFp16;
        config.enable_fp8 = m_enableFp8;
        config.workspace_mb = m_workspaceMb;
        config.verbose = m_verbose;
        
        addLog("Starting engine export...");
        addLog("Input: " + config.input_onnx_path);
        addLog("Output: " + config.get_output_path());
        
        m_exportProgress = 0.1f;
        
        // Create and run exporter
        EngineExporter exporter(config);
        
        m_exportProgress = 0.2f;
        
        bool success = exporter.exportEngine();
        
        if (success) {
            m_exportProgress = 1.0f;
            m_exportStatus = ExportStatus::COMPLETED;
            addLog("Engine export completed successfully!");
        } else {
            m_exportStatus = ExportStatus::FAILED;
            addLog("Engine export failed", true);
        }
        
    } catch (const std::exception& e) {
        m_exportStatus = ExportStatus::FAILED;
        addLog("Exception during export: " + std::string(e.what()), true);
    } catch (...) {
        m_exportStatus = ExportStatus::FAILED;
        addLog("Unknown error during export", true);
    }
}

void GuiApp::addLog(const std::string& message, bool isError) {
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_logQueue.emplace(message, isError);
}

void GuiApp::processLogQueue() {
    // This is called from the main thread to update the UI
    // The actual log processing happens in renderLogWindow()
}

bool GuiApp::validateInputs() {
    if (strlen(m_inputPath) == 0) return false;
    if (!fileExists(std::string(m_inputPath))) return false;
    if (strlen(m_outputPath) == 0) return false;
    return true;
}

std::string GuiApp::generateOutputPath() {
    if (strlen(m_inputPath) == 0) return "";
    
    std::filesystem::path inputPath(m_inputPath);
    std::string baseName = inputPath.stem().string();
    
    // Add resolution and precision suffixes
    baseName += "_" + std::to_string(m_resolution);
    if (m_enableFp16) baseName += "_fp16";
    if (m_enableFp8) baseName += "_fp8";
    
    std::filesystem::path outputPath = inputPath.parent_path() / (baseName + ".engine");
    return outputPath.string();
}

void GuiApp::helpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

bool GuiApp::fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}