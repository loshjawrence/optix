#pragma once

/*! simple 24-bit linear congruence generator */
template <unsigned N = 16>
struct LCG {

    __device__
    LCG() { /* intentionally empty so we can use it in device vars that
         don't allow dynamic initialization (ie, PRD) */
    }
    __device__ LCG(unsigned int val0, unsigned int val1) {
        init(val0, val1);
    }

    __device__ void init(unsigned int val0, unsigned int val1) {
        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;

        for (unsigned int n = 0; n < N; n++) {
            s0 += 0x9e3779b9;
            v0 +=
                ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 +=
                ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
        }
        state = v0;
    }

    // Generate random unsigned int in [0, 2^24)
    __device__ float operator()() {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return (state & 0x00FFFFFF) / (float)0x01000000;
    }

    uint32_t state;
};
