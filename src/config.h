#pragma once

#include <string>
#include <vector>

struct ExportConfig {
    std::string input_onnx_path;
    std::string output_engine_path;
    int input_resolution = 640;
    bool enable_fp16 = false;
    bool enable_fp8 = false;
    int workspace_mb = 1024;
    bool verbose = false;
    
    // TensorRT optimization settings
    bool enable_gpu_fallback = true;
    bool enable_precision_constraints = true;
    bool enable_detailed_profiling = false;
    
    // NMS settings
    bool fix_nms_output = false;
    int nms_max_detections = 300;
    
    // Validation
    bool is_valid() const {
        return !input_onnx_path.empty();
    }
    
    // Generate output path if not specified
    std::string get_output_path() const {
        if (!output_engine_path.empty()) {
            return output_engine_path;
        }
        
        // Generate based on input path and settings
        std::string base = input_onnx_path.substr(0, input_onnx_path.find_last_of('.'));
        base += "_" + std::to_string(input_resolution);
        if (enable_fp16) base += "_fp16";
        if (enable_fp8) base += "_fp8";
        return base + ".engine";
    }
};

class ConfigParser {
public:
    static ExportConfig parseCommandLine(int argc, char* argv[]);
    static void printUsage(const std::string& program_name);
    static void printVersion();
    
private:
    static bool isOption(const std::string& arg);
    static std::string getOptionValue(const std::vector<std::string>& args, size_t& index);
};