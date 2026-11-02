#include <algorithm>
#include <cmath>
#include <jni.h>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <vector>

static float srgbToLinearLUT[256];
static std::once_flag lutsOnceFlag;

/* linearToSRGBExact sampled finely enough that it differs only at rounding
 * edges - pow per channel was most of a halving's time. */
constexpr int LINEAR_STEPS = 1 << 16;
static uint8_t linearToSrgbLUT[LINEAR_STEPS + 1];

static inline uint8_t linearToSRGBExact(float linearVal);

/* A bare bool flag races: two mipmap levels can resize concurrently on first
 * use. */
static void initLUTs() {
  std::call_once(lutsOnceFlag, [] {
    for (int i = 0; i < 256; i++) {
      float val01 = i / 255.0f;
      srgbToLinearLUT[i] = (val01 <= 0.04045f)
                               ? (val01 / 12.92f)
                               : std::pow((val01 + 0.055f) / 1.055f, 2.4f);
    }
    for (int i = 0; i <= LINEAR_STEPS; i++) {
      linearToSrgbLUT[i] = linearToSRGBExact(i / (float)LINEAR_STEPS);
    }
  });
}

static inline uint8_t linearToSRGBExact(float linearVal) {
  if (linearVal <= 0.0f)
    return 0;
  if (linearVal >= 1.0f)
    return 255;

  float srgb01 = (linearVal <= 0.0031308f)
                     ? (linearVal * 12.92f)
                     : (1.055f * std::pow(linearVal, 1.0f / 2.4f) - 0.055f);

  return static_cast<uint8_t>(std::roundf(srgb01 * 255.0f));
}

static inline uint8_t linearToSRGBFast(float linearVal) {
  // Negated so NaN lands here too, not on an index past the table.
  if (!(linearVal > 0.0f))
    return 0;
  if (linearVal >= 1.0f)
    return 255;
  return linearToSrgbLUT[(int)(linearVal * LINEAR_STEPS + 0.5f)];
}

static inline uint8_t exactLinearToAlpha(float linearAlpha) {
  if (linearAlpha <= 0.0f)
    return 0;
  if (linearAlpha >= 1.0f)
    return 255;
  return static_cast<uint8_t>(std::roundf(linearAlpha * 255.0f));
}

static void halveRows(const uint32_t *src, uint32_t *dst, int srcWidth,
                      int dstWidth, int yStart, int yEnd);

struct Spans {
  std::vector<int> start, count, offset;
  std::vector<float> weight;
};

static Spans makeSpans(int srcSize, int dstSize);
template <typename Sum>
static void areaRows(const uint32_t *src, uint32_t *dst, int srcWidth,
                     int dstWidth, const Spans &xs, const Spans &ys, int yStart,
                     int yEnd);

/* Splits [0, rows) into bands, one per core up to 4, run concurrently. */
template <typename F> static void parallelRows(int rows, F work) {
  int threads =
      (int)std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
  threads = std::min(threads, std::max(1, rows / 64));
  if (threads == 1) {
    work(0, rows);
    return;
  }
  std::vector<std::thread> pool;
  int band = (rows + threads - 1) / threads;
  for (int t = 1; t < threads; ++t) {
    int start = t * band;
    if (start >= rows)
      break;
    int end = std::min(rows, start + band);
    try {
      pool.emplace_back(work, start, end);
    } catch (const std::system_error &) {
      work(start, end);
    }
  }
  work(0, std::min(rows, band));
  for (auto &th : pool)
    th.join();
}

/* Linear-light box filter. Unchecked: sizes > 0 and buffers that fit them. */
static void resizePixels(const uint32_t *src, uint32_t *dst, int srcWidth,
                         int srcHeight, int dstWidth, int dstHeight) {
  initLUTs();

  if (srcWidth == dstWidth * 2 && srcHeight == dstHeight * 2) {
    parallelRows(dstHeight, [=](int yStart, int yEnd) {
      halveRows(src, dst, srcWidth, dstWidth, yStart, yEnd);
    });
    return;
  }

  Spans xs, ys;
  try {
    xs = makeSpans(srcWidth, dstWidth);
    ys = makeSpans(srcHeight, dstHeight);
  } catch (const std::bad_alloc &) {
    return;
  }
  // Float sums drift tens of levels over millions of pixels
  const bool hugeBoxes =
      (double)srcWidth * srcHeight > (double)dstWidth * dstHeight * 65536.0;
  // By reference, or each thread copies the tables.
  parallelRows(dstHeight, [&](int yStart, int yEnd) {
    if (hugeBoxes)
      areaRows<double>(src, dst, srcWidth, dstWidth, xs, ys, yStart, yEnd);
    else
      areaRows<float>(src, dst, srcWidth, dstWidth, xs, ys, yStart, yEnd);
  });
}

static void resizeArea(JNIEnv *env, jobject src_buffer, jobject dst_buffer,
                       jint srcWidth, jint srcHeight, jint dstWidth,
                       jint dstHeight) {
  if (!env || !src_buffer || !dst_buffer)
    return;
  if (srcWidth <= 0 || srcHeight <= 0)
    return;

  uint32_t *src = (uint32_t *)env->GetDirectBufferAddress(src_buffer);
  uint32_t *dst = (uint32_t *)env->GetDirectBufferAddress(dst_buffer);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return;
  }
  if (!src || !dst)
    return;

  /* 64-bit: a 32-bit product wraps for a large enough image. */
  const jlong srcCapacity = env->GetDirectBufferCapacity(src_buffer);
  const jlong srcNeeded =
      static_cast<jlong>(srcWidth) * static_cast<jlong>(srcHeight) * 4LL;
  if (srcCapacity < 0 || srcCapacity < srcNeeded)
    return;

  if (dstWidth <= 0 || dstHeight <= 0)
    return;

  const jlong dstCapacity = env->GetDirectBufferCapacity(dst_buffer);
  const jlong dstNeeded =
      static_cast<jlong>(dstWidth) * static_cast<jlong>(dstHeight) * 4LL;
  if (dstCapacity < 0 || dstCapacity < dstNeeded)
    return;

  resizePixels(src, dst, srcWidth, srcHeight, dstWidth, dstHeight);
}

/* Each 2x2 block in linear light, the common mipmap case without the general
 * filter's fractional weights. */
static void halveRows(const uint32_t *src, uint32_t *dst, int srcWidth,
                      int dstWidth, int yStart, int yEnd) {
  for (int y = yStart; y < yEnd; ++y) {
    const uint32_t *row0 = src + (size_t)(y * 2) * srcWidth;
    const uint32_t *row1 = row0 + srcWidth;
    for (int x = 0; x < dstWidth; ++x) {
      const uint32_t px[4] = {row0[x * 2], row0[x * 2 + 1], row1[x * 2],
                              row1[x * 2 + 1]};
      float sumA = 0.0f, sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
      for (uint32_t pixel : px) {
        float aVal = ((pixel >> 24) & 0xFF) / 255.0f;
        sumA += aVal;
        sumR += srgbToLinearLUT[(pixel >> 16) & 0xFF] * aVal;
        sumG += srgbToLinearLUT[(pixel >> 8) & 0xFF] * aVal;
        sumB += srgbToLinearLUT[pixel & 0xFF] * aVal;
      }
      uint32_t finalA = 0, finalR = 0, finalG = 0, finalB = 0;
      float a = sumA * 0.25f;
      if (a > 0.00001f) {
        float invAlpha = 1.0f / sumA;
        finalR = linearToSRGBFast(std::min(1.0f, sumR * invAlpha));
        finalG = linearToSRGBFast(std::min(1.0f, sumG * invAlpha));
        finalB = linearToSRGBFast(std::min(1.0f, sumB * invAlpha));
      }
      finalA = exactLinearToAlpha(a);
      dst[(size_t)y * dstWidth + x] =
          (finalA << 24) | (finalR << 16) | (finalG << 8) | finalB;
    }
  }
}

static Spans makeSpans(int srcSize, int dstSize) {
  Spans s;
  s.start.resize(dstSize);
  s.count.resize(dstSize);
  s.offset.resize(dstSize);
  s.weight.reserve((size_t)dstSize *
                   ((size_t)std::ceil((double)srcSize / dstSize) + 1));
  // Not float: box edges drift at large sizes and flip pixels.
  double scale = (double)srcSize / dstSize;
  for (int d = 0; d < dstSize; ++d) {
    double lo = d * scale;
    double hi = lo + scale;
    int first = std::min(srcSize - 1, std::max(0, (int)lo));
    // Not (int)hi, which adds a zero-weight pixel at whole-number ends.
    int last = std::min(srcSize - 1, (int)std::ceil(hi) - 1);
    s.start[d] = first;
    s.count[d] = std::max(0, last - first + 1);
    s.offset[d] = (int)s.weight.size();
    for (int i = first; i <= last; ++i) {
      double w = std::min((double)i + 1.0, hi) - std::max((double)i, lo);
      s.weight.push_back(w > 0 ? (float)w : 0.0f);
    }
  }
  return s;
}

template <typename Sum>
static void areaRows(const uint32_t *src, uint32_t *dst, int srcWidth,
                     int dstWidth, const Spans &xs, const Spans &ys, int yStart,
                     int yEnd) {
  for (int y = yStart; y < yEnd; ++y) {
    const float *yWeights = &ys.weight[ys.offset[y]];
    int y0 = ys.start[y], yCount = ys.count[y];
    for (int x = 0; x < dstWidth; ++x) {
      const float *xWeights = &xs.weight[xs.offset[x]];
      int x0 = xs.start[x], xCount = xs.count[x];
      Sum sumA = 0, sumR = 0, sumG = 0, sumB = 0;
      Sum totalWeight = 0;
      for (int j = 0; j < yCount; ++j) {
        const uint32_t *row = src + (size_t)(y0 + j) * srcWidth + x0;
        for (int i = 0; i < xCount; ++i) {
          float w = xWeights[i] * yWeights[j];
          uint32_t pixel = row[i];
          float aw = ((pixel >> 24) & 0xFF) / 255.0f * w;
          totalWeight += w;
          sumA += aw;
          sumR += srgbToLinearLUT[(pixel >> 16) & 0xFF] * aw;
          sumG += srgbToLinearLUT[(pixel >> 8) & 0xFF] * aw;
          sumB += srgbToLinearLUT[pixel & 0xFF] * aw;
        }
      }
      uint32_t finalA = 0, finalR = 0, finalG = 0, finalB = 0;
      if (totalWeight > 0) {
        float a = (float)(sumA / totalWeight);
        if (a > 0.00001f) {
          Sum invAlpha = 1 / sumA;
          finalR = linearToSRGBFast(std::min(1.0f, (float)(sumR * invAlpha)));
          finalG = linearToSRGBFast(std::min(1.0f, (float)(sumG * invAlpha)));
          finalB = linearToSRGBFast(std::min(1.0f, (float)(sumB * invAlpha)));
        }
        finalA = exactLinearToAlpha(a);
      }
      dst[(size_t)y * dstWidth + x] =
          (finalA << 24) | (finalR << 16) | (finalG << 8) | finalB;
    }
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_resizeLinearAreaNative(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint srcWidth, jint srcHeight) {
  resizeArea(env, src_buffer, dst_buffer, srcWidth, srcHeight, srcWidth / 2,
             srcHeight / 2);
}

extern "C" JNIEXPORT void JNICALL
Java_ca_mpreg_webgpuviewer_ImageUtil_resizeLinearAreaToNative(
    JNIEnv *env, jobject thiz, jobject src_buffer, jobject dst_buffer,
    jint srcWidth, jint srcHeight, jint dstWidth, jint dstHeight) {
  resizeArea(env, src_buffer, dst_buffer, srcWidth, srcHeight, dstWidth,
             dstHeight);
}
