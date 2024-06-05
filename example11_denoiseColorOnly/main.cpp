#include "SampleWindow.h"

#include "CMakeGenerated.h"
#include "Camera.h"
#include "Model.h"
#include "QuadLight.h"
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

        // some simple, hard-coded light ... obviously, only works for sponza
        const float light_size = 200.0f;
        QuadLight light = {
            /* origin */ glm::vec3(-1000 - light_size, 800, -light_size),
            /* edge 1 */ glm::vec3(2.f * light_size, 0, 0),
            /* edge 2 */ glm::vec3(0, 0, 2.f * light_size),
            /* power */ glm::vec3(3000000.f)};

        // something approximating the scale of the world, so the
        // camera knows how much to move for any given user interaction:
        const float worldScale = length(model->bounds.diag());

        SampleWindow* window = new SampleWindow("Optix 7 Course Example 07",
                                                model,
                                                camera,
                                                light,
                                                worldScale);
        window->run();

    } catch (std::runtime_error& e) {
        spdlog::error("FATAL ERROR: {}", e.what());
        exit(1);
    }

    return 0;
}
