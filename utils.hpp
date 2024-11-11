#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <fstream>
#include <cstdio>
#include <cstring>

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

inline void PackTrans(
    const float *in, float *out, const int row, const int col, const int ld
)
{
    float *ptr = out;
    for (int c = 0; c < col; ++c)
        for (int r = 0; r < row; ++r)
            *ptr++ = in[r * ld + c];
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
