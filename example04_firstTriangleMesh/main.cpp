#include "SampleWindow.h"

#include <spdlog/spdlog.h>

int main() {
    try {
        SampleWindow* window = new SampleWindow("Optix 7 Course Example03");
        window->run();
    } catch (std::runtime_error& e) {
        spdlog::error("FATAL ERROR: {}", e.what());
        exit(1);
    }
    return 0;
}
