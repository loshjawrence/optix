#include "IOUtil.h"

#include <fstream>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

std::vector<char> getBinaryDataFromFile(const fs::path& file)
{
	const std::string filename = file.string();
	std::ifstream inputData(filename, std::ios::binary);

	if (inputData.fail())
	{
		spdlog::error("ERROR: readData() Failed to open file {}", filename);
		return std::vector<char>();
	}

	// Copy the input buffer to a char vector.
	std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

	if (inputData.fail())
	{
		spdlog::error("ERROR: readData() Failed to open file {}", filename);
		return std::vector<char>();
	}

	return data;
}
