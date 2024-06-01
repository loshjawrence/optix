#pragma once

#include <filesystem>
#include <vector>

std::vector<char> getBinaryDataFromFile(const std::filesystem::path& file);
