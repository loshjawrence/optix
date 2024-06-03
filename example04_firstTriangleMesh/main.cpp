#include "SampleWindow.h"

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include "TriangleMesh.h"
#include "Camera.h"

int main() {
    try {
      TriangleMesh model;
      // 100x100 thin ground plane
      model.addCube(glm::vec3(0.f,-1.5f,0.f),glm::vec3(10.f,.1f,10.f));
      // a unit cube centered on top of that
      model.addCube(glm::vec3(0.f,0.f,0.f),glm::vec3(2.f,2.f,2.f));

      Camera camera = { /*from*/glm::vec3(-10.f,2.f,-12.f),
                        /* at */glm::vec3(0.f,0.f,0.f),
                        /* up */glm::vec3(0.f,1.f,0.f) };

      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = 10.f;

      SampleWindow *window = new SampleWindow("Optix 7 Course Example04",
                                              model,camera,worldScale);
      window->run();
      
    } catch (std::runtime_error& e) {
        spdlog::error("FATAL ERROR: {}", e.what());
        exit(1);
    }

    return 0;
}
