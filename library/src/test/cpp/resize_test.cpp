// Tests for the linear-light box filter behind every mipmap and downscale.
//
// Its failures are silent: a dropped edge pixel, a weight off by a fraction or
// a sum that drifts shows up as a slightly wrong colour, never as a crash. Each
// result is checked against a reference that does the same filter in double
// with no lookup tables, allowing the one level the tables round by. The
// implementation is included rather than linked for the reason given in
// trim_test.cpp.

#include "resize.cpp" // NOLINT(bugprone-suspicious-include)

#include "check.h"

#include <random>
#include <vector>

namespace {

uint32_t argb(uint32_t a, uint32_t r, uint32_t g, uint32_t b) {
  return (a << 24) | (r << 16) | (g << 8) | b;
}

std::vector<uint32_t> resized(const std::vector<uint32_t> &src, int srcWidth,
                              int srcHeight, int dstWidth, int dstHeight) {
  std::vector<uint32_t> dst(static_cast<size_t>(dstWidth) * dstHeight,
                            0xDEADBEEFu);
  resizePixels(src.data(), dst.data(), srcWidth, srcHeight, dstWidth,
               dstHeight);
  return dst;
}

double decode(int v) {
  const double x = v / 255.0;
  return x <= 0.04045 ? x / 12.92 : std::pow((x + 0.055) / 1.055, 2.4);
}

uint32_t encode(double linear) {
  if (linear <= 0)
    return 0;
  if (linear >= 1)
    return 255;
  const double s = linear <= 0.0031308
                       ? linear * 12.92
                       : 1.055 * std::pow(linear, 1 / 2.4) - 0.055;
  return static_cast<uint32_t>(std::lround(s * 255));
}

/** The same box filter in double, edges in long double, no lookup tables. */
std::vector<uint32_t> reference(const std::vector<uint32_t> &src, int srcWidth,
                                int srcHeight, int dstWidth, int dstHeight) {
  std::vector<uint32_t> dst(static_cast<size_t>(dstWidth) * dstHeight);
  const long double sx = static_cast<long double>(srcWidth) / dstWidth;
  const long double sy = static_cast<long double>(srcHeight) / dstHeight;
  for (int y = 0; y < dstHeight; ++y) {
    const long double y0 = y * sy, y1 = y0 + sy;
    for (int x = 0; x < dstWidth; ++x) {
      const long double x0 = x * sx, x1 = x0 + sx;
      double sumW = 0, sumA = 0, sumR = 0, sumG = 0, sumB = 0;
      for (int j = static_cast<int>(y0); j < srcHeight && j < y1; ++j) {
        const double wy = static_cast<double>(std::min<long double>(j + 1, y1) -
                                              std::max<long double>(j, y0));
        for (int i = static_cast<int>(x0); i < srcWidth && i < x1; ++i) {
          const double wx = static_cast<double>(
              std::min<long double>(i + 1, x1) - std::max<long double>(i, x0));
          if (wx <= 0 || wy <= 0)
            continue;
          const uint32_t p = src[static_cast<size_t>(j) * srcWidth + i];
          const double w = wx * wy;
          const double a = (p >> 24) / 255.0 * w;
          sumW += w;
          sumA += a;
          sumR += decode((p >> 16) & 0xFF) * a;
          sumG += decode((p >> 8) & 0xFF) * a;
          sumB += decode(p & 0xFF) * a;
        }
      }
      const double a = sumA / sumW;
      uint32_t r = 0, g = 0, b = 0;
      if (a > 0.00001) {
        r = encode(sumR / sumA);
        g = encode(sumG / sumA);
        b = encode(sumB / sumA);
      }
      const uint32_t alpha = a <= 0 ? 0
                             : a >= 1
                                 ? 255
                                 : static_cast<uint32_t>(std::lround(a * 255));
      dst[static_cast<size_t>(y) * dstWidth + x] = argb(alpha, r, g, b);
    }
  }
  return dst;
}

/** Largest per-channel difference between two images of the same size. */
int maxChannelDiff(const std::vector<uint32_t> &a,
                   const std::vector<uint32_t> &b) {
  int worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    for (int shift = 0; shift < 32; shift += 8) {
      const int d = static_cast<int>((a[i] >> shift) & 0xFF) -
                    static_cast<int>((b[i] >> shift) & 0xFF);
      worst = std::max(worst, std::abs(d));
    }
  }
  return worst;
}

/** Noise with transparent, opaque and nearly transparent patches - the alphas
 * that stress the premultiplied sums. */
std::vector<uint32_t> noise(int width, int height, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> px(static_cast<size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int zone = (x / 7 + y / 5) % 4;
      const uint32_t a = zone == 0   ? 0
                         : zone == 1 ? 255
                         : zone == 2 ? (rng() & 0xFF)
                                     : (rng() & 7);
      px[static_cast<size_t>(y) * width + x] = (a << 24) | (rng() & 0xFFFFFF);
    }
  }
  return px;
}

void checkAgainstReference(int srcWidth, int srcHeight, int dstWidth,
                           int dstHeight, uint32_t seed) {
  const std::vector<uint32_t> src = noise(srcWidth, srcHeight, seed);
  const int diff =
      maxChannelDiff(resized(src, srcWidth, srcHeight, dstWidth, dstHeight),
                     reference(src, srcWidth, srcHeight, dstWidth, dstHeight));
  if (diff > 1) {
    check::fail(__FILE__, __LINE__,
                std::to_string(srcWidth) + "x" + std::to_string(srcHeight) +
                    " -> " + std::to_string(dstWidth) + "x" +
                    std::to_string(dstHeight) +
                    " differs from the reference by " + std::to_string(diff));
  }
}

} // namespace

// --- spans
// -------------------------------------------------------------------------------------

TEST(spansGiveEverySourcePixelAWholeWeight) {
  // The boxes tile the source, so each source pixel's weights across all boxes
  // sum to 1 - a dropped or doubled edge pixel shows here first.
  const int sizes[][2] = {{8, 4}, {9, 4}, {4096, 1365}, {7, 3},
                          {3, 7}, {5, 5}, {1, 9},       {9, 1}};
  for (const auto &s : sizes) {
    const Spans spans = makeSpans(s[0], s[1]);
    std::vector<double> perSource(s[0], 0.0);
    for (int d = 0; d < s[1]; ++d) {
      for (int i = 0; i < spans.count[d]; ++i) {
        perSource[spans.start[d] + i] += spans.weight[spans.offset[d] + i];
      }
    }
    for (int i = 0; i < s[0]; ++i)
      CHECK_NEAR(perSource[i], 1.0, 1e-5);
  }
}

TEST(spansEndOnTheirLastCoveredPixel) {
  // A box ending on a whole number must not reach into the next pixel with
  // weight 0.
  const int sizes[][2] = {{8, 4}, {4096, 1024}, {6, 3}, {9, 3}, {12, 4}};
  for (const auto &s : sizes) {
    const Spans spans = makeSpans(s[0], s[1]);
    for (int d = 0; d < s[1]; ++d) {
      CHECK(spans.count[d] >= 1);
      CHECK(spans.weight[spans.offset[d] + spans.count[d] - 1] > 0.0f);
      CHECK(spans.start[d] + spans.count[d] <= s[0]);
    }
  }
}

// --- resize
// ------------------------------------------------------------------------------------

TEST(solidColourSurvivesEverySize) {
  const uint32_t colour = argb(255, 200, 30, 90);
  const std::vector<uint32_t> src(64 * 48, colour);
  const int sizes[][2] = {{32, 24},  {16, 12}, {21, 16}, {64, 48},
                          {100, 75}, {1, 1},   {64, 1}};
  for (const auto &s : sizes) {
    for (uint32_t px : resized(src, 64, 48, s[0], s[1]))
      CHECK_EQ(px, colour);
  }
}

TEST(transparentPixelsDoNotTintTheirNeighbours) {
  // Two opaque red and two transparent green: premultiplied averaging keeps the
  // colour red and halves the alpha. A straight average would give a muddy
  // brown.
  const std::vector<uint32_t> src = {argb(255, 255, 0, 0), argb(0, 0, 255, 0),
                                     argb(0, 0, 255, 0), argb(255, 255, 0, 0)};
  const std::vector<uint32_t> halved = resized(src, 2, 2, 1, 1);
  CHECK_EQ(halved[0], argb(128, 255, 0, 0));
  const std::vector<uint32_t> area = resized(src, 2, 2, 1, 2);
  CHECK_EQ(area[0] & 0x00FFFFFFu, argb(0, 255, 0, 0));
}

TEST(fullyTransparentComesOutZero) {
  const std::vector<uint32_t> src(16 * 16, argb(0, 255, 255, 255));
  for (uint32_t px : resized(src, 16, 16, 8, 8))
    CHECK_EQ(px, 0u);
  for (uint32_t px : resized(src, 16, 16, 5, 7))
    CHECK_EQ(px, 0u);
}

TEST(upscaleRepeatsPixels) {
  const std::vector<uint32_t> src = {
      argb(255, 10, 20, 30), argb(255, 200, 100, 0), argb(128, 0, 0, 255),
      argb(255, 255, 255, 255)};
  const std::vector<uint32_t> big = resized(src, 2, 2, 6, 4);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x)
      CHECK_EQ(big[y * 6 + x], src[(y / 2) * 2 + x / 3]);
  }
}

TEST(halvingMatchesTheReference) {
  checkAgainstReference(2, 2, 1, 1, 1);
  checkAgainstReference(64, 64, 32, 32, 2);
  checkAgainstReference(250, 98, 125, 49, 3);
}

TEST(otherRatiosMatchTheReference) {
  const int cases[][4] = {
      {1, 1, 1, 1},        {1, 1, 7, 5},        {7, 5, 1, 1},
      {5, 7, 2, 3},        {3, 3, 2, 2},        {65, 33, 32, 16},
      {256, 256, 64, 64},  {300, 200, 100, 67}, {97, 89, 31, 211},
      {640, 3, 639, 2},    {1, 300, 4, 7},      {300, 1, 7, 4},
      {128, 128, 300, 300}};
  uint32_t seed = 10;
  for (const auto &c : cases)
    checkAgainstReference(c[0], c[1], c[2], c[3], seed++);

  std::mt19937 rng(99);
  for (int i = 0; i < 60; ++i) {
    checkAgainstReference(1 + rng() % 160, 1 + rng() % 160, 1 + rng() % 160,
                          1 + rng() % 160, seed++);
  }
}

TEST(hugeBoxesDoNotDrift) {
  // 8M equal terms into one sum round the same way every time, so float drifts
  // far off the one right answer; boxes this size take the double path.
  const uint32_t colour = argb(200, 90, 160, 220);
  const std::vector<uint32_t> solid(4096 * 2048, colour);
  CHECK_EQ(resized(solid, 4096, 2048, 1, 1)[0], colour);
  checkAgainstReference(4096, 2048, 3, 2, 8);
}

TEST(threadedBandsMatchOnePass) {
  const int w = 700, h = 520, dw = 233, dh = 300;
  const std::vector<uint32_t> src = noise(w, h, 5);
  const Spans xs = makeSpans(w, dw), ys = makeSpans(h, dh);
  std::vector<uint32_t> onePass(static_cast<size_t>(dw) * dh);
  areaRows<float>(src.data(), onePass.data(), w, dw, xs, ys, 0, dh);
  CHECK(resized(src, w, h, dw, dh) == onePass);
}

TEST(linearToSrgbFastSurvivesNaN) {
  initLUTs();
  CHECK_EQ(static_cast<int>(linearToSRGBFast(std::nanf(""))), 0);
  CHECK_EQ(static_cast<int>(linearToSRGBFast(-1.0f)), 0);
  CHECK_EQ(static_cast<int>(linearToSRGBFast(2.0f)), 255);
}

int main() { return check::main(); }
