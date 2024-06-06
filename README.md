# optix
* this is just me going through ingo wald's optix7 course here https://github.com/ingowald/optix7course/tree/master
* written using msvc 2022, optix 8, cuda 12.5
* updated to have native msvc cmake, dont need to generate a .sln file. just open the directory.
* slight changes here and there
* NOTE: the last example, example12 does not fully work. the output image goes black when you add any members to the per ray data struct "PRD"
in devicePrograms.cu. so its basically left off where example 11 is.
