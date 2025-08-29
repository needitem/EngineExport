#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>

struct GLFWwindow;
class EngineExporter;
struct ExportConfig;
struct PluginInfo;
struct CustomPluginInfo;

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
    
    // Plugin selection state
    std::vector<PluginInfo> m_availablePlugins;
    std::vector<CustomPluginInfo> m_customPlugins;
    bool m_showPluginWindow = false;
    
    // Custom plugin dialog state
    char m_newPluginName[256] = {};
    char m_newPluginPath[512] = {};
    char m_newPluginDesc[512] = {};
    bool m_showAddPluginDialog = false;
    
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
    void renderPluginSelection();
    void renderAddPluginDialog();
    
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
    
    // Plugin management
    void initializePlugins();
    void updateSelectedPlugins();
    void addCustomPlugin(const std::string& name, const std::string& path, const std::string& description);
    void removeCustomPlugin(size_t index);
    bool validatePluginPath(const std::string& path);
};