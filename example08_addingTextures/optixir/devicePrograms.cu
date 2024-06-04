#include <optix_device.h>

#include "LaunchParams.h"
#include "TriangleMeshSBTData.h"

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

static __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __device__ T* getPerRayData() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

static __device__ float3 asFloat3(glm::vec3 input) {
    return float3{input.x, input.y, input.z};
}

static __device__ glm::vec3 asVec3(float3 input) {
    return glm::vec3{input.x, input.y, input.z};
}
static __device__ glm::vec3 asVec3(float4 input) {
    return glm::vec3{input.x, input.y, input.z};
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

/*! helper function that creates a semi-random color from an ID */
__device__ glm::vec3 randomColor(size_t idx) {
    unsigned int r = (unsigned int)(idx * 13 * 17 + 0x234235);
    unsigned int g = (unsigned int)(idx * 7 * 3 * 5 + 0x773477);
    unsigned int b = (unsigned int)(idx * 11 * 19 + 0x223766);
    return glm::vec3((r & 255) / 255.f, (g & 255) / 255.f, (b & 255) / 255.f);
}

extern "C" __global__ void __closesthit__radiance() {
    const TriangleMeshSBTData& sbtData =
        *reinterpret_cast<const TriangleMeshSBTData*>(optixGetSbtDataPointer());

    // gather some basic hit info
    const int primID = optixGetPrimitiveIndex();
    const glm::ivec3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    glm::vec3 N{};
    if (sbtData.normal) {
        // barycentric weighting on the 3 vertex normals
        const glm::vec3 shadingNormal = ((1.0f - u - v) * sbtData.normal[index.x]) +
            (u * sbtData.normal[index.y]) + (v * sbtData.normal[index.z]);
        N = shadingNormal;
    } else {
		const glm::vec3& A = sbtData.vertex[index.x];
		const glm::vec3& B = sbtData.vertex[index.y];
		const glm::vec3& C = sbtData.vertex[index.z];
		const glm::vec3& geomNormal = glm::normalize(glm::cross(B - A, C - A));
        N = geomNormal;
    }

    // compute diffuse
    glm::vec3 diffuseColor = sbtData.diffuse;
    if (sbtData.hasTexture && sbtData.texcoord)
    {
        const glm::vec2 tc = ((1.0f - u - v) * sbtData.texcoord[index.x]) +
            (u * sbtData.texcoord[index.y]) + (v * sbtData.texcoord[index.z]);
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= asVec3(fromTexture);
    }

    // compute lambertian coeff
    const glm::vec3 rayDir = asVec3(optixGetWorldRayDirection());
    const float cosDN = 0.2f + 0.8f * std::fabsf(glm::dot(rayDir, N));

    glm::vec3& prd = *getPerRayData<glm::vec3>();
    prd = cosDN * diffuseColor;
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    glm::vec3& prd = *getPerRayData<glm::vec3>();
    prd = glm::vec3(1.0f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto& camera = optixLaunchParams.camera;

    // one per-ray data for this example.
    // what we intitialize it to won't matter,
    // since this value will be overwritten by either the miss or hit program, anyway.
    glm::vec3 pixelColorPRD = glm::vec3(0.0f);

    // store the u64 pointer to the pixelColorPRD var on our stack
    // in 2 u32s so that those u32s can be passed to optixTrace
    // and downstream calls can manipulate pixelColorPRD
    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalize screen plane position, in [0, 1]^2
    const glm::vec2 screen(glm::vec2(ix + 0.5f, iy + 0.5f) /
                           glm::vec2(optixLaunchParams.frame.size));

    // generate ray direction
    glm::vec3 rayDir = glm::normalize(camera.direction +
                                      ((screen.x - 0.5f) * camera.horizontal) +
                                      ((screen.y - 0.5f) * camera.vertical));

    const float tmin = 0.0f;
    const float tmax = 1e20f;
    const float rayTime = 0.0f;
    optixTrace(optixLaunchParams.traversable,
               asFloat3(camera.position),
               asFloat3(rayDir),
               tmin,
               tmax,
               rayTime,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,
               RAY_TYPE_COUNT,
               SURFACE_RAY_TYPE,
               u0,
               u1);

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + (iy * optixLaunchParams.frame.size.x);
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
