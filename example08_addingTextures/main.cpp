#include "SampleWindow.h"

#include "CMakeGenerated.h"
#include "Camera.h"
#include "Model.h"
#include <filesystem>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
namespace fs = std::filesystem;

int main() {
    try {
        fs::path modelPath = g_modelsPath / "sponza/sponza.obj";
        Model* model = loadOBJ(modelPath.string());
        Camera camera = {/*from*/ glm::vec3(-1293.07f, 154.681f, -0.7304f),
                         /* at */ model->bounds.center() - glm::vec3(0, 400, 0),
                         /* up */ glm::vec3(0.f, 1.f, 0.f)};
        // something approximating the scale of the world, so the
        // camera knows how much to move for any given user interaction:
        const float worldScale = length(model->bounds.diag());

        SampleWindow* window = new SampleWindow("Optix 7 Course Example 07",
                                                model,
                                                camera,
                                                worldScale);
        window->run();

    } catch (std::runtime_error& e) {
        spdlog::error("FATAL ERROR: {}", e.what());
        exit(1);
    }

    return 0;
}
