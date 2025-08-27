#include <iostream>
#include <exception>
#include "config.h"
#include "engine_exporter.h"

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        ExportConfig config = ConfigParser::parseCommandLine(argc, argv);
        
        // Check if we need to exit early (help, version, etc.)
        if (!config.is_valid()) {
            return config.input_onnx_path.empty() ? 1 : 0;
        }
        
        // Print configuration
        std::cout << "EngineExport - ONNX to TensorRT Engine Converter\n";
        std::cout << "================================================\n";
        std::cout << "Input ONNX: " << config.input_onnx_path << "\n";
        std::cout << "Output Engine: " << config.get_output_path() << "\n";
        std::cout << "\n";
        
        // Create exporter and run conversion
        EngineExporter exporter(config);
        bool success = exporter.exportEngine();
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}