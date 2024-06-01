#pragma once

#include <source_location>

#include <optix_types.h>


void optixCheck(std::source_location sl);
void optixCheck(OptixResult result,
                std::source_location sl = std::source_location::current());

void initOptix();

void optixSetLogCallback(const OptixDeviceContext& optixContext);
