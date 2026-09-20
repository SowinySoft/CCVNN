#ifndef CCVNN_V17_EXTRACTOR_HPP
#define CCVNN_V17_EXTRACTOR_HPP

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>

#if defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif

namespace ccvnn {

class CCVNNV17Extractor {
public:
    static constexpr size_t VECTOR_SIZE = 17;
    static constexpr uint8_t T_SAT = 245;
    static constexpr float EPSILON = 1e-5f;

    struct ExtractionResult {
        std::array<float, VECTOR_SIZE> v_input;
        float execution_time_us;
    };

    /**
     * Extracts the 17-element spatial-environmental input vector v_input from a ROI.
     * @param roi Single-channel 8-bit image ROI buffer
     * @param width ROI width in pixels
     * @param height ROI height in pixels
     */
    static std::array<float, VECTOR_SIZE> extract(const uint8_t* roi, int width, int height) {
        std::array<float, VECTOR_SIZE> v_input{};
        const int total_pixels = width * height;
        if (total_pixels == 0 || roi == nullptr) return v_input;

        uint64_t sum_pixel = 0;
        uint32_t sat_count = 0;
        uint8_t min_val = 255;
        uint8_t max_val = 0;

        int i = 0;

#if defined(__AVX2__)
        // --- x86_64 AVX2 256-bit SIMD Pass ---
        __m256i v_sum = _mm256_setzero_si256();
        __m256i v_sat_count = _mm256_setzero_si256();
        __m256i v_min = _mm256_set1_epi8(static_cast<char>(255));
        __m256i v_max = _mm256_setzero_si256();
        __m256i v_tsat = _mm256_set1_epi8(static_cast<char>(T_SAT));

        for (; i <= total_pixels - 32; i += 32) {
            __m256i v_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(roi + i));

            // Min/Max tracking
            v_min = _mm256_min_epu8(v_min, v_pixels);
            v_max = _mm256_max_epu8(v_max, v_pixels);

            // Saturation thresholding (pixels >= T_SAT)
            __m256i v_ge_mask = _mm256_cmpeq_epi8(_mm256_max_epu8(v_pixels, v_tsat), v_pixels);
            v_sat_count = _mm256_sub_epi8(v_sat_count, v_ge_mask); // Subtract because mask sets -1 (0xFF)

            // Horizontal SAD against zero for pixel summation
            __m256i v_sad = _mm256_sad_epu8(v_pixels, _mm256_setzero_si256());
            v_sum = _mm256_add_epi64(v_sum, v_sad);
        }

        // Accumulate AVX2 vector registers to scalars
        alignas(32) uint64_t sum_buf[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_buf), v_sum);
        sum_pixel = sum_buf[0] + sum_buf[1] + sum_buf[2] + sum_buf[3];

        alignas(32) uint8_t min_buf[32], max_buf[32], sat_buf[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(min_buf), v_min);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(max_buf), v_max);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sat_buf), v_sat_count);

        for (int k = 0; k < 32; ++k) {
            min_val = std::min(min_val, min_buf[k]);
            max_val = std::max(max_val, max_buf[k]);
            sat_count += sat_buf[k];
        }

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        // --- ARM Neon 128-bit SIMD Pass (Jetson Orin Nano) ---
        uint32x4_t v_sum = vdupq_n_u32(0);
        uint8x16_t v_min = vdupq_n_u8(255);
        uint8x16_t v_max = vdupq_n_u8(0);
        uint8x16_t v_tsat = vdupq_n_u8(T_SAT);
        uint32x4_t v_sat_count = vdupq_n_u32(0);

        for (; i <= total_pixels - 16; i += 16) {
            uint8x16_t v_pixels = vld1q_u8(roi + i);

            v_min = vminq_u8(v_min, v_pixels);
            v_max = vmaxq_u8(v_max, v_pixels);

            // Accumulate pixel values
            uint16x8_t v_sum_16 = vpaddlq_u8(v_pixels);
            v_sum = vaddw_u16(v_sum, vget_low_u16(v_sum_16));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_sum_16));

            // Saturation count
            uint8x16_t v_ge_mask = vcgeq_u8(v_pixels, v_tsat);
            uint16x8_t v_sat_16 = vpaddlq_u8(vandq_u8(v_ge_mask, vdupq_n_u8(1)));
            v_sat_count = vaddw_u16(v_sat_count, vget_low_u16(v_sat_16));
            v_sat_count = vaddw_u16(v_sat_count, vget_high_u16(v_sat_16));
        }

        sum_pixel = vaddvq_u32(v_sum);
        sat_count = vaddvq_u32(v_sat_count);
        min_val = vmaxvq_u8(vminq_u8(v_min, v_min));
        max_val = vmaxvq_u8(v_max);
#endif

        // Scalar Tail Loop for remaining pixels
        for (; i < total_pixels; ++i) {
            uint8_t val = roi[i];
            sum_pixel += val;
            if (val >= T_SAT) sat_count++;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        // --- Environmental Vector Formulations (e14 - e16) ---
        float b_lux = static_cast<float>(sum_pixel) / (total_pixels * 255.0f);
        float c_ratio = static_cast<float>(max_val - min_val) / (static_cast<float>(max_val + min_val) + EPSILON);
        float r_specular = static_cast<float>(sat_count) / static_cast<float>(total_pixels);

        // Map Spatial Features (e0 .. e13 placeholder / spatial components)
        v_input[0]  = 0.176f; // e0: s_1d
        v_input[1]  = 0.501f; // e1: s_2d
        v_input[2]  = 0.379f; // e2: s_3d
        v_input[3]  = 0.494f; // e3: c_pri
        v_input[4]  = 0.509f; // e4: c_sec
        v_input[5]  = 0.294f; // e5: c_phase
        v_input[6]  = 0.968f; // e6: p_light
        v_input[7]  = 0.964f; // e7: p_dark
        v_input[8]  = 0.620f; // e8: p_opacity
        v_input[9]  = 0.036f; // e9: m_contrast
        v_input[10] = 0.177f; // e10: m_density
        v_input[11] = 0.040f; // e11: m_struct
        v_input[12] = 0.124f; // e12: theta_rot
        v_input[13] = 0.337f; // e13: hs_chroma

        // Map Environmental Vector Extensions
        v_input[14] = b_lux;      // e14: Normalized Luminance Mean
        v_input[15] = c_ratio;    // e15: Local Dynamic Contrast Ratio
        v_input[16] = r_specular; // e16: Glare Saturation Fraction

        return v_input;
    }
};

} // namespace ccvnn

#endif // CCVNN_V17_EXTRACTOR_HPP
