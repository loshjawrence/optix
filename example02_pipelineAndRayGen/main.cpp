#include "SampleRenderer.h"

#include <spdlog/spdlog.h>

int main()
{
    SampleRenderer renderer;
    try {
      SampleRenderer sample;

      sample.init();
      sample.render();
      sample.saveFramebuffer();

    } catch (std::runtime_error& e) {
		spdlog::error("FATAL ERROR: {}", e.what());
		exit(1);
    }

    return 0;
}
