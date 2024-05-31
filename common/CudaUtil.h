#pragma once

#include <source_location>

void cudaCheck(int result, std::source_location sl = std::source_location::current());
void cudaSyncCheck(std::source_location sl = std::source_location::current());
