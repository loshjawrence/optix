#define NOMINMAX

#include "SampleRenderer.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <source_location>
namespace fs = std::filesystem;

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <optix_stack_size.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// NOTE: this include may only appear in a single .cpp
// TODO: Should probably wrap it better then...
// #include <optix_function_table_definition.h>

#include "CMakeGenerated.h"
#include "CudaUtil.h"
#include "EnumRayType.h"
#include "IOUtil.h"
#include "Model.h"
#include "OptixUtil.h"
#include "QuadLight.h"
#include "Texture.h"
#include "TriangleMesh.h"
#include "TriangleMeshSBTData.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

static void context_log_cb(unsigned int level,
                           const char* tag,
                           const char* message,
                           [[maybe_unused]] void*) {
    spdlog::warn("\n[{}][{}]: {}", level, tag, message);
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    // sbt record for raygen program
    // TODO: probably want to extract these two to SBTRecord
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    void* data{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    // sbt record for miss program
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    void* data{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    // sbt record for miss program
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    TriangleMeshSBTData data{};
};

static void printSuccess(
    std::source_location sl = std::source_location::current()) {
    spdlog::info("{} successfully ran", sl.function_name());
}

SampleRenderer::SampleRenderer(const Model* model, const QuadLight& light)
    : model(model) {
    initOptix();

    launchParams.light.origin = light.origin;
    launchParams.light.du = light.du;
    launchParams.light.dv = light.dv;
    launchParams.light.power = light.power;

    createContext();
    createModule();
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();

    launchParams.traversable = buildAccel();

    createPipeline();
    createTextures();
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    printSuccess();
}
void SampleRenderer::createTextures() {
    int numTextures = (int)model->textures.size();
    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);
    for (int textureID = 0; textureID < numTextures; ++textureID) {
        auto texture = model->textures[textureID];

        cudaChannelFormatDesc channel_desc{};
        int width = texture->resolution.x;
        int height = texture->resolution.y;
        int numComponents = 4;
        int pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t& pixelArray = textureArrays[textureID];
        cudaCheck(cudaMallocArray(&pixelArray, &channel_desc, width, height));

        int wOffset{};
        int hOffset{};
        cudaCheck(cudaMemcpy2DToArray(pixelArray,
                                      wOffset,
                                      hOffset,
                                      texture->pixel,
                                      pitch,
                                      pitch,
                                      height,
                                      cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc{};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc{};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        cudaTextureObject_t cuda_tex{};
        cudaCheck(
            cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        textureObjects[textureID] = cuda_tex;
    }
}

OptixTraversableHandle SampleRenderer::buildAccel() {
    const int numMeshes = (int)model->meshes.size();
    vertexBuffer.resize(numMeshes);
    normalBuffer.resize(numMeshes);
    texcoordBuffer.resize(numMeshes);
    indexBuffer.resize(numMeshes);

    OptixTraversableHandle asHandle{};

    // triangle inputs
    std::vector<OptixBuildInput> triangleInput(numMeshes);
    std::vector<CUdeviceptr> d_vertices(numMeshes);
    std::vector<CUdeviceptr> d_indices(numMeshes);
    std::vector<uint32_t> triangleInputFlags(numMeshes);

    for (int meshID = 0; meshID < numMeshes; meshID++) {
        // upload the model to the device: the builder
        TriangleMesh& mesh = *(model->meshes[meshID]);
        vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
        indexBuffer[meshID].alloc_and_upload(mesh.index);
        if (!mesh.normal.empty()) {
            normalBuffer[meshID].alloc_and_upload(mesh.normal);
        }
        if (!mesh.texcoord.empty()) {
            texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);
        }

        triangleInput[meshID] = {};
        triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat =
            OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes =
            sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices =
            (int)mesh.vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat =
            OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes =
            sizeof(glm::ivec3);
        triangleInput[meshID].triangleArray.numIndexTriplets =
            (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    // BLAS setup
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    optixCheck(optixAccelComputeMemoryUsage(optixContext,
                                            &accelOptions,
                                            triangleInput.data(),
                                            numMeshes, // num_build_inputs
                                            &blasBufferSizes));

    // prepare compaction
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // execute build (main stage)
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixCheck(optixAccelBuild(optixContext,
                               /* stream */ 0,
                               &accelOptions,
                               triangleInput.data(),
                               numMeshes,
                               tempBuffer.d_pointer(),
                               tempBuffer.byteSize(),

                               outputBuffer.d_pointer(),
                               outputBuffer.byteSize(),

                               &asHandle,

                               &emitDesc,
                               1));
    cudaSyncCheck();

    // perform compaction
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    optixCheck(optixAccelCompact(optixContext,
                                 /*stream:*/ 0,
                                 asHandle,
                                 asBuffer.d_pointer(),
                                 asBuffer.byteSize(),
                                 &asHandle));
    cudaSyncCheck();

    // clean up
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

void SampleRenderer::createContext() {
    const int deviceID{};
    cudaCheck(cudaSetDevice(deviceID));
    cudaCheck(cudaStreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    spdlog::info("deviceProps.name: {}", deviceProps.name);

    cudaCheck(cuCtxGetCurrent(&cudaContext));
    optixCheck(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    optixCheck(optixDeviceContextSetLogCallback(optixContext,
                                                context_log_cb,
                                                nullptr,
                                                4));
    printSuccess();
}

void SampleRenderer::createModule() {
    moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

#if DEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;
    // NOTE: OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW causes issues with 
    //pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

    const fs::path moduleFilename = g_optixIRPath / "devicePrograms.optixir";
    const std::vector<char> ptxCode = getBinaryDataFromFile(moduleFilename);
    optixCheck(optixModuleCreate(optixContext,
                                 &moduleCompileOptions,
                                 &pipelineCompileOptions,
                                 ptxCode.data(),
                                 ptxCode.size(),
                                 nullptr,
                                 nullptr,
                                 &module));
    printSuccess();
}

void SampleRenderer::createRaygenPrograms() {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions{};
    OptixProgramGroupDesc pgDesc{};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       1,
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &raygenPGs[0]));
    printSuccess();
}

void SampleRenderer::createMissPrograms() {
    // we do a single ray gen program in this example:
    missPGs.resize(RAY_TYPE_COUNT);

    OptixProgramGroupOptions pgOptions{};
    OptixProgramGroupDesc pgDesc{};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;

    // radiance
    pgDesc.miss.entryFunctionName = "__miss__radiance";
    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       1,
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &missPGs[RADIANCE_RAY_TYPE]));

    // shadow
    pgDesc.miss.entryFunctionName = "__miss__shadow";
    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       1,
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &missPGs[SHADOW_RAY_TYPE]));
    printSuccess();
}

void SampleRenderer::createHitgroupPrograms() {
    // we do a single ray gen program in this example:
    hitgroupPGs.resize(RAY_TYPE_COUNT);

    OptixProgramGroupOptions pgOptions{};
    OptixProgramGroupDesc pgDesc{};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.moduleAH = module;

    // radiance
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       1,
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &hitgroupPGs[RADIANCE_RAY_TYPE]));

    // shadow rays: technically we don't need this hit group,
    // since we just use the miss shader to check if we were not
    // in shadow
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       1,
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &hitgroupPGs[SHADOW_RAY_TYPE]));
    printSuccess();
}

void SampleRenderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    programGroups.insert(programGroups.end(),
                         raygenPGs.begin(),
                         raygenPGs.end());
    programGroups.insert(programGroups.end(), missPGs.begin(), missPGs.end());
    programGroups.insert(programGroups.end(),
                         hitgroupPGs.begin(),
                         hitgroupPGs.end());

    optixCheck(optixPipelineCreate(optixContext,
                                   &pipelineCompileOptions,
                                   &pipelineLinkOptions,
                                   programGroups.data(),
                                   int(programGroups.size()),
                                   nullptr,
                                   nullptr,
                                   &pipeline));
    constexpr int KB = 1024;
    constexpr int many_KB = 4 * KB;
    optixCheck(optixPipelineSetStackSize(
        // [in] The pipeline to configure the stack size for
        pipeline,
        // [in] The direct stack size requirement for direct callables invoked from IS(intersection) or AH(anyhit).
        many_KB,
        // [in] The direct stack size requirement for direct callables invoked from RG(raygen), MS(miss), or CH(closesthit).
        many_KB,
        // [in] The continuation stack requirement.
        many_KB,
        // [in] The maximum depth of a traversable graph passed to trace.
        1));

    printSuccess();
}

void SampleRenderer::buildSBT() {
    // RAYGEN RECORDS
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); ++i) {
        RaygenRecord rec{};
        optixCheck(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        //rec.data = ...later;
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // MISS RECORDS
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); ++i) {
        MissRecord rec{};
        optixCheck(optixSbtRecordPackHeader(missPGs[i], &rec));
        //rec.data = ...later;
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = int(missRecords.size());

    // HITGROUP RECORDS
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < model->meshes.size(); meshID++) {
        for (int rayID = 0; rayID < RAY_TYPE_COUNT; ++rayID) {
            auto mesh = model->meshes[meshID];
            HitgroupRecord rec;
            // all meshes use the same code, so all same hit group
            optixCheck(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
            rec.data.diffuse = mesh->diffuse;
            if (mesh->diffuseTextureID >= 0) {
                rec.data.hasTexture = true;
                rec.data.texture = textureObjects[mesh->diffuseTextureID];
            } else {
                rec.data.hasTexture = false;
            }
            rec.data.index = (glm::ivec3*)indexBuffer[meshID].d_pointer();
            rec.data.vertex = (glm::vec3*)vertexBuffer[meshID].d_pointer();
            rec.data.normal = (glm::vec3*)normalBuffer[meshID].d_pointer();
            rec.data.texcoord = (glm::vec2*)texcoordBuffer[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
        }
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

    printSuccess();
}

void SampleRenderer::render() {
    if (!launchParams.frame.renderBuffer || launchParams.frame.size.x == 0) {
        spdlog::warn(
            "Ran with invalid launch params, launchParamsfbSize.x is 0 or "
            "launchParams.frame.renderBuffer is nullptr. Need to init first.");
        return;
    }

    if (!accumulate) {
        launchParams.frame.frameID = 0;
    }

    launchParamsBuffer.upload(&launchParams, 1);
    launchParams.frame.frameID++;
    const int depth = 1;
    optixCheck(optixLaunch(pipeline,
                           stream,
                           launchParamsBuffer.d_pointer(),
                           launchParamsBuffer.byteSize(),
                           &sbt,
                           launchParams.frame.size.x,
                           launchParams.frame.size.y,
                           depth));

    denoiserIntensity.resize(sizeof(float));

    OptixDenoiserParams denoiserParams{};
    denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
    if (accumulate) {
        denoiserParams.blendFactor = 1.0f / launchParams.frame.frameID;
    }

    OptixImage2D inputLayer[3]{};
    // fbRender
    inputLayer[0].data = fbRender.d_pointer();
    inputLayer[0].width = launchParams.frame.size.x;
    inputLayer[0].height = launchParams.frame.size.y;
    inputLayer[0].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;
    // fbAlbedo
    inputLayer[1].data = fbAlbedo.d_pointer();
    inputLayer[1].width = launchParams.frame.size.x;
    inputLayer[1].height = launchParams.frame.size.y;
    inputLayer[1].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;
    // fbNormal
    inputLayer[2].data = fbNormal.d_pointer();
    inputLayer[2].width = launchParams.frame.size.x;
    inputLayer[2].height = launchParams.frame.size.y;
    inputLayer[2].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D outputLayer{};
    outputLayer.data = denoisedBuffer.d_pointer();
    outputLayer.width = launchParams.frame.size.x;
    outputLayer.height = launchParams.frame.size.y;
    outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    outputLayer.pixelStrideInBytes = sizeof(float4);
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    if (denoiserOn) {
        optixCheck(optixDenoiserComputeIntensity(denoiser,
                                                 0,
                                                 &inputLayer[0],
                                                 denoiserIntensity.d_pointer(),
                                                 denoiserScratch.d_pointer(),
                                                 denoiserScratch.byteSize()));
        OptixDenoiserGuideLayer denoiserGuideLayer{};
        denoiserGuideLayer.albedo = inputLayer[1];
        denoiserGuideLayer.normal = inputLayer[2];

        OptixDenoiserLayer denoiserLayer{};
        denoiserLayer.input = inputLayer[0];
        denoiserLayer.output = outputLayer;

        int inputOffsetX{};
        int inputOffsetY{};
        int numLayers{1};
        optixCheck(optixDenoiserInvoke(denoiser,
                                       0,
                                       &denoiserParams,
                                       denoiserState.d_pointer(),
                                       denoiserState.byteSize(),
                                       &denoiserGuideLayer,
                                       &denoiserLayer,
                                       numLayers,
                                       inputOffsetX,
                                       inputOffsetY,
                                       denoiserScratch.d_pointer(),
                                       denoiserScratch.byteSize()));
    } else {
        cudaMemcpy((void*)outputLayer.data,
                   (void*)inputLayer[0].data,
                   outputLayer.width * outputLayer.height * sizeof(float4),
                   cudaMemcpyDeviceToDevice);
    }

    computeFinalPixelColors();

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    cudaSyncCheck();
}

void SampleRenderer::setCamera(const Camera& camera) {
    lastSetCamera = camera;
    launchParams.frame.frameID = 0;
    launchParams.camera.position = camera.from;
    launchParams.camera.direction = glm::normalize(camera.at - camera.from);
    const float cosFovy = 0.66f;
    const float aspect =
        launchParams.frame.size.x / float(launchParams.frame.size.y);
    launchParams.camera.horizontal = cosFovy * aspect *
        glm::normalize(glm::cross(launchParams.camera.direction, camera.up));
    launchParams.camera.vertical = cosFovy *
        glm::normalize(glm::cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
}

void SampleRenderer::resizeFramebuffer(const glm::ivec2& newSize) {
    if (newSize.x <= 0 || newSize.y <= 0) {
        return;
    }
    if (denoiser) {
        optixCheck(optixDenoiserDestroy(denoiser));
    }

    // create denoiser
    OptixDenoiserOptions denoiserOptions{};
    optixCheck(optixDenoiserCreate(optixContext,
                                   OPTIX_DENOISER_MODEL_KIND_LDR,
                                   &denoiserOptions,
                                   &denoiser));

    OptixDenoiserSizes denoiserReturnSizes{};
    optixCheck(optixDenoiserComputeMemoryResources(denoiser,
                                                   newSize.x,
                                                   newSize.y,
                                                   &denoiserReturnSizes));

    size_t max = std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                          denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
    denoiserScratch.resize(max);

    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

    denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));
    fbRender.resize(newSize.x * newSize.y * sizeof(glm::vec4));
    fbNormal.resize(newSize.x * newSize.y * sizeof(glm::vec4));
    fbAlbedo.resize(newSize.x * newSize.y * sizeof(glm::vec4));
    finalColorBuffer.resize(newSize.x * newSize.y * sizeof(glm::vec4));

    launchParams.frame.size = newSize;
    launchParams.frame.renderBuffer =
        reinterpret_cast<glm::vec4*>(fbRender.d_pointer());
    launchParams.frame.normalBuffer =
        reinterpret_cast<glm::vec4*>(fbNormal.d_pointer());
    launchParams.frame.albedoBuffer =
        reinterpret_cast<glm::vec4*>(fbAlbedo.d_pointer());

    setCamera(lastSetCamera);

    optixCheck(optixDenoiserSetup(denoiser,
                                  0,
                                  newSize.x,
                                  newSize.y,
                                  denoiserState.d_pointer(),
                                  denoiserState.byteSize(),
                                  denoiserScratch.d_pointer(),
                                  denoiserScratch.byteSize()));

    printSuccess();
}

void SampleRenderer::downloadFramebuffer(std::vector<glm::vec4>& outPayload) {
    outPayload.resize(launchParams.frame.size.x * launchParams.frame.size.y);
    finalColorBuffer.download(&outPayload[0], outPayload.size());
}

void SampleRenderer::saveFramebuffer() {
    std::vector<glm::vec4> pixels{};
    downloadFramebuffer(pixels);
    const fs::path filename = g_debugImagesPath / "example12.png";
    if (!stbi_write_png(filename.string().c_str(),
                        launchParams.frame.size.x,
                        launchParams.frame.size.y,
                        4,
                        reinterpret_cast<const void*>(pixels.data()),
                        launchParams.frame.size.x * sizeof(glm::vec4))) {
        spdlog::error("Failed to save framebuffer to {}.", filename.string());
    }

    spdlog::info("Framebuffer saved to {} ... done.", filename.string());

    printSuccess();
}
