#include "config.h"
#include <iostream>
#include <algorithm>

ExportConfig ConfigParser::parseCommandLine(int argc, char* argv[]) {
    ExportConfig config;
    std::vector<std::string> args(argv, argv + argc);
    
    if (argc < 2) {
        printUsage(args[0]);
        return config;
    }
    
    // First non-option argument is input file
    bool found_input = false;
    for (size_t i = 1; i < args.size(); ++i) {
        if (!isOption(args[i]) && !found_input) {
            config.input_onnx_path = args[i];
            found_input = true;
            continue;
        }
        
        if (args[i] == "--help" || args[i] == "-h") {
            printUsage(args[0]);
            return config;
        }
        else if (args[i] == "--version" || args[i] == "-v") {
            printVersion();
            return config;
        }
        else if (args[i] == "--resolution" || args[i] == "-r") {
            std::string value = getOptionValue(args, i);
            config.input_resolution = std::stoi(value);
        }
        else if (args[i] == "--output" || args[i] == "-o") {
            config.output_engine_path = getOptionValue(args, i);
        }
        else if (args[i] == "--workspace" || args[i] == "-w") {
            std::string value = getOptionValue(args, i);
            config.workspace_mb = std::stoi(value);
        }
        else if (args[i] == "--fp16") {
            config.enable_fp16 = true;
        }
        else if (args[i] == "--fp8") {
            config.enable_fp8 = true;
        }
        else if (args[i] == "--verbose") {
            config.verbose = true;
        }
        else if (args[i] == "--no-gpu-fallback") {
            config.enable_gpu_fallback = false;
        }
        else if (args[i] == "--no-precision-constraints") {
            config.enable_precision_constraints = false;
        }
        else if (args[i] == "--detailed-profiling") {
            config.enable_detailed_profiling = true;
        }
    }
    
    return config;
}

void ConfigParser::printUsage(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " <input.onnx> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help                    Show this help message\n";
    std::cout << "  -v, --version                 Show version information\n";
    std::cout << "  -r, --resolution <size>       Input resolution (default: 640)\n";
    std::cout << "  -o, --output <path>           Output engine file path\n";
    std::cout << "  -w, --workspace <mb>          Workspace size in MB (default: 1024)\n";
    std::cout << "  --fp16                        Enable FP16 precision\n";
    std::cout << "  --fp8                         Enable FP8 precision\n";
    std::cout << "  --verbose                     Enable verbose output\n";
    std::cout << "  --no-gpu-fallback             Disable GPU fallback\n";
    std::cout << "  --no-precision-constraints    Disable precision constraints\n";
    std::cout << "  --detailed-profiling          Enable detailed profiling\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " model.onnx\n";
    std::cout << "  " << program_name << " model.onnx --resolution 640 --fp16\n";
    std::cout << "  " << program_name << " model.onnx --output custom_name.engine --verbose\n";
}

void ConfigParser::printVersion() {
    std::cout << "EngineExport v1.0.0\n";
    std::cout << "TensorRT ONNX to Engine Converter\n";
}

bool ConfigParser::isOption(const std::string& arg) {
    return arg.size() > 0 && arg[0] == '-';
}

std::string ConfigParser::getOptionValue(const std::vector<std::string>& args, size_t& index) {
    if (index + 1 >= args.size() || isOption(args[index + 1])) {
        std::cerr << "Error: Option " << args[index] << " requires a value\n";
        return "";
    }
    return args[++index];
}