#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>

struct GLFWwindow;
class EngineExporter;
struct ExportConfig;

enum class ExportStatus {
    IDLE,
    RUNNING,
    COMPLETED,
    FAILED
};

struct LogEntry {
    std::string message;
    bool isError;
    
    LogEntry(const std::string& msg, bool error = false) 
        : message(msg), isError(error) {}
};

class GuiApp {
public:
    GuiApp();
    ~GuiApp();
    
    bool initialize();
    void run();
    void shutdown();
    
private:
    // Window management
    GLFWwindow* m_window = nullptr;
    
    // GUI state
    char m_inputPath[512] = {};
    char m_outputPath[512] = {};
    int m_resolution = 640;
    bool m_enableFp16 = true;
    bool m_enableFp8 = false;
    int m_workspaceMb = 1024;
    bool m_verbose = true;
    bool m_fixNmsOutput = false;
    int m_nmsMaxDetections = 300;
    
    // Export state
    std::atomic<ExportStatus> m_exportStatus{ExportStatus::IDLE};
    std::atomic<float> m_exportProgress{0.0f};
    std::string m_exportError;
    
    // Threading
    std::unique_ptr<std::thread> m_exportThread;
    std::mutex m_logMutex;
    std::queue<LogEntry> m_logQueue;
    
    // UI methods
    void renderMainWindow();
    void renderFileSelection();
    void renderExportOptions();
    void renderExportButton();
    void renderProgressBar();
    void renderLogWindow();
    
    // File dialogs
    bool openFileDialog(std::string& path, const char* filter);
    bool saveFileDialog(std::string& path, const char* filter);
    
    // Export functionality
    void startExport();
    void exportThreadFunc();
    void addLog(const std::string& message, bool isError = false);
    void processLogQueue();
    
    // Validation
    bool validateInputs();
    std::string generateOutputPath();
    
    // UI helpers
    void helpMarker(const char* desc);
    bool fileExists(const std::string& path);
};