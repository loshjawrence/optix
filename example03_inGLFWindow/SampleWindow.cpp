#include "SampleWindow.h"

#include <assert.h>
#include <spdlog/spdlog.h>

SampleWindow::SampleWindow(const std::string& title)
    : GLFWindow(title) {
    sample.init();
}

void SampleWindow::render() {
    sample.render();
}

void SampleWindow::draw() {
    sample.downloadFramebuffer(pixels);
    if (!fbTexture)
    {
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
    GLenum texelType = GL_UNSIGNED_BYTE;
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
