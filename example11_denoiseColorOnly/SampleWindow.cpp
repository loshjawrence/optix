#define NOMINMAX

#include "SampleWindow.h"

#include <assert.h>
#include <algorithm>

#include <spdlog/spdlog.h>

#include "Camera.h"

SampleWindow::SampleWindow(const std::string& title,
                           const Model* model,
                           const Camera& camera,
                           const QuadLight& light,
                           const float worldScale)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale)
    , sample(model, light) {
}

void SampleWindow::render() {
    if (cameraFrame.modified) {
        sample.setCamera(Camera{cameraFrame.get_from(),
                                cameraFrame.get_at(),
                                cameraFrame.get_up()});
        cameraFrame.modified = false;
    }
    sample.render();
}

void SampleWindow::draw() {
    sample.downloadFramebuffer(pixels);
    if (!fbTexture) {
        try {
            glGenTextures(1, &fbTexture);
        } catch (std::runtime_error& e) {
            spdlog::error("FATAL ERROR: {}", e.what());
            GLenum result = glGetError();
            spdlog::error("glGenTextures did not create a texture. Got {}",
                          result);
            exit(1);
        }
    }

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_FLOAT;
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 texFormat,
                 fbSize.x,
                 fbSize.y,
                 0,
                 GL_RGBA,
                 texelType,
                 pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();
}

void SampleWindow::resize(const glm::ivec2 newSize) {
    fbSize = newSize;
    sample.resizeFramebuffer(newSize);
    pixels.resize(newSize.x * newSize.y);
}

void SampleWindow::key(int key, int mods) {
    if (key == 'D' || key == ' ' || key == 'd') {
        sample.denoiserOn = !sample.denoiserOn;
		spdlog::info("denoising now: {}", sample.denoiserOn);
    }
    if (key == 'A' || key == 'a') {
        sample.accumulate = !sample.accumulate;
		spdlog::info("accumulation/progressive refinement now: {}", sample.accumulate);
    }
    if (key == ',') {
        sample.launchParams.numPixelSamples =
            std::max(1, sample.launchParams.numPixelSamples - 1);
		spdlog::info("num samples/pixel now: {}", sample.launchParams.numPixelSamples);
    }
    if (key == '.') {
        sample.launchParams.numPixelSamples =
            std::max(1, sample.launchParams.numPixelSamples + 1);
		spdlog::info("num samples/pixel now: {}", sample.launchParams.numPixelSamples);
    }
}
