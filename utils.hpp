#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <fstream>
#include <cstdio>
#include <cstring>
#include <arm_neon.h>

inline bool LoadArray(const char *filename, float *buffer, const size_t size)
{
    std::ifstream file(filename);
    if (!file)
        return false;
    for (size_t i = 0; i < size; ++i)
        if (!(file >> buffer[i]))
            return false;
    return true;
}

inline float Checksum(const float *array, size_t size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
        sum += array[i];
    return sum;
}

inline void ShowResult(const float *array, int start = 0, int num_show = 5)
{
    std::printf("C[%d:%d]: [ ", start, start + num_show);
    for (int i = start; i < start + num_show - 1; ++i)
        std::printf("%.2f, ", array[i]);
    std::printf("%.2f ]\n", array[start + num_show - 1]);
}

inline void Trans4x4(
    float *dst, const int dst_stride, const float *src, const int src_stride
)
{
    float32x4_t row0 = vld1q_f32(src);
    float32x4_t row1 = vld1q_f32(src + src_stride);
    float32x4_t row2 = vld1q_f32(src + 2 * src_stride);
    float32x4_t row3 = vld1q_f32(src + 3 * src_stride);

    float32x4x2_t tmp0 = vtrnq_f32(row0, row1);
    float32x4x2_t tmp1 = vtrnq_f32(row2, row3);

    vst1q_f32(dst, vcombine_f32(vget_low_f32(tmp0.val[0]), vget_low_f32(tmp1.val[0])));
    vst1q_f32(dst + dst_stride, vcombine_f32(vget_low_f32(tmp0.val[1]), vget_low_f32(tmp1.val[1])));
    vst1q_f32(dst + 2 * dst_stride, vcombine_f32(vget_high_f32(tmp0.val[0]), vget_high_f32(tmp1.val[0])));
    vst1q_f32(dst + 3 * dst_stride, vcombine_f32(vget_high_f32(tmp0.val[1]), vget_high_f32(tmp1.val[1])));
}

inline void PackTrans(
    const float *in, float *out, const int row, const int col, const int ld
)
{
    constexpr int stride = 4;
    for (int i = 0; i <= row - stride; i += stride)
    {
        for (int j = 0; j <= col - stride; j += stride)
        {
            Trans4x4(
                out + j * row + i,
                row,
                in + i * ld + j,
                ld
            );
        }
    }
}

inline void PackCopy(
    const float *in, float *out, const int row, const int col, const int ld
)
{
    const float *in_ptr = in;
    float *out_ptr = out;
    for (int r = 0; r < row; ++r)
    {
        std::memcpy(out_ptr, in_ptr, col * sizeof(float));
        in_ptr += ld;
        out_ptr += col;
    }
}

#endif  // UTILS_HPP_
