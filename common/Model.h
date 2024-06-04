#pragma once

#include <vector>
#include <string>

#include "Box.h"
struct TriangleMesh;
struct Texture;

struct Model {
    ~Model() {
        for (auto mesh : meshes)
        {
            delete mesh;
        }

        for (auto texture : textures)
        {
            delete texture;
        }
    }

    std::vector<TriangleMesh*> meshes{};
    std::vector<Texture*> textures{};

    Box bounds{};
};

Model* loadOBJ(const std::string& objFile);
