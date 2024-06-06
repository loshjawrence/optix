#include <optix_device.h>

#include "EnumRayType.h"
#include "LaunchParams.h"
#include "Random.h"
#include "TriangleMeshSBTData.h"

#define NUM_LIGHT_SAMPLES 4

using Random = LCG<16>;

struct PRD {
    Random random{};
    glm::vec3 pixelColor{};
};

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

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

extern "C" __global__ void __closesthit__shadow() {
    // not going to be used
}

extern "C" __global__ void __closesthit__radiance() {
    const TriangleMeshSBTData& sbtData =
        *reinterpret_cast<const TriangleMeshSBTData*>(optixGetSbtDataPointer());
    PRD& prd = *getPerRayData<PRD>();

    // gather some basic hit info
    const int primID = optixGetPrimitiveIndex();
    const glm::ivec3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const glm::vec3& A = sbtData.vertex[index.x];
    const glm::vec3& B = sbtData.vertex[index.y];
    const glm::vec3& C = sbtData.vertex[index.z];
    glm::vec3 Ng = glm::cross(B - A, C - A);
    // barycentric weighting on the 3 vertex normals
    glm::vec3 Ns = sbtData.normal ? ((1.0f - u - v) * sbtData.normal[index.x]) +
            (u * sbtData.normal[index.y]) + (v * sbtData.normal[index.z])
                                  : Ng;

    const glm::vec3 rayDir = asVec3(optixGetWorldRayDirection());

    // face-forward and normalize normals
    // not sure what this is for, commenting out the if's makes no difference
    if (dot(rayDir, Ng) > 0.0f) {
        Ng = -Ng;
    }
    Ng = normalize(Ng);
    if (dot(Ng, Ns) < 0.0f) {
        Ns -= 2.0f * dot(Ng, Ns) * Ng;
    }
    Ns = normalize(Ns);

    // compute diffuse
    glm::vec3 diffuseColor = sbtData.diffuse;
    if (sbtData.hasTexture && sbtData.texcoord) {
        const glm::vec2 tc = ((1.0f - u - v) * sbtData.texcoord[index.x]) +
            (u * sbtData.texcoord[index.y]) + (v * sbtData.texcoord[index.z]);
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= asVec3(fromTexture);
    }

    // start with some ambient term
    glm::vec3 pixelColor =
        (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;

    // compute shadow
    const glm::vec3 surfPos = (1.f - u - v) * sbtData.vertex[index.x] +
        u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];

    const int numLightSamples = NUM_LIGHT_SAMPLES;
    for (int lightSampleID = 0; lightSampleID < numLightSamples;
         ++lightSampleID) {
        const glm::vec3 lightPos = optixLaunchParams.light.origin +
            prd.random() * optixLaunchParams.light.du +
            prd.random() * optixLaunchParams.light.dv;

        glm::vec3 lightDir = lightPos - surfPos;
        float lightDist = glm::length(lightDir);
        lightDir = normalize(lightDir);

        // trace shadow ray
        const float NdotL = dot(lightDir, Ns);
        if (NdotL < 0.0f) {
            continue;
        }

        // correct side of surface hemisphere, light possibly visible.
        glm::vec3 lightVisibility{};
        uint32_t u0, u1;
        packPointer(&lightVisibility, u0, u1);
        optixTrace(
            optixLaunchParams.traversable,
            asFloat3(surfPos + 1e-3f * Ng),
            asFloat3(lightDir),
            1e-3f,                     // tmin
            lightDist * (1.f - 1e-3f), // tmax
            0.0f,                      // rayTime
            OptixVisibilityMask(255),
            // For shadow rays: skip any/closest hit shaders and terminate on first
            // intersection with anything. The miss shader is used to mark if the
            // light was visible.
            OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            SHADOW_RAY_TYPE, // SBT offset
            RAY_TYPE_COUNT,  // SBT stride
            SHADOW_RAY_TYPE, // missSBTIndex
            u0,
            u1);
        pixelColor += lightVisibility * optixLaunchParams.light.power *
            diffuseColor * NdotL / (lightDist * lightDist * numLightSamples);
    }
    prd.pixelColor = pixelColor;
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void
__anyhit__shadow() { /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    PRD& prd = *getPerRayData<PRD>();
    prd.pixelColor = glm::vec3(1.0f);
}

extern "C" __global__ void __miss__shadow() {
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

    PRD prd;
    prd.random.init(ix + iy * optixLaunchParams.frame.size.x,
                   optixLaunchParams.frame.frameID);
    prd.pixelColor = glm::vec3(0.f);

    // store the u64 pointer to the pixelColorPRD var on our stack
    // in 2 u32s so that those u32s can be passed to optixTrace
    // and downstream calls can manipulate pixelColorPRD
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    int numPixelSamples = optixLaunchParams.numPixelSamples;

    glm::vec3 pixelColor{};

    for (int sampleID = 0; sampleID < numPixelSamples; ++sampleID) {
        // normalized screen plane position, in [0,1]^2
        const glm::vec2 screen(glm::vec2(ix + prd.random(), iy + prd.random()) /
                               glm::vec2(optixLaunchParams.frame.size));

        // generate ray direction
        glm::vec3 rayDir = normalize(camera.direction +
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
                   RADIANCE_RAY_TYPE,
                   RAY_TYPE_COUNT,
                   RADIANCE_RAY_TYPE,
                   u0,
                   u1);
        pixelColor += prd.pixelColor;
    }
    glm::vec4 rgba(pixelColor / float(numPixelSamples), 1.f);

    // and write/accumulate to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
	rgba += float(optixLaunchParams.frame.frameID) *
		glm::vec4(optixLaunchParams.frame.renderBuffer[fbIndex]);
	rgba /= (optixLaunchParams.frame.frameID + 1.0f);
    optixLaunchParams.frame.renderBuffer[fbIndex] = rgba;
}
