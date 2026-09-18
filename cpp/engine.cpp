#include <cmath>
#include <vector>
#include <algorithm>
#include <emscripten/bind.h>

using namespace emscripten;

class PerlinNoise {
private:
    uint8_t p[512];

    double fade(double t) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    double lerp(double a, double b, double t) {
        return a + t * (b - a);
    }

    double grad(int hash, double x, double y) {
        int h = hash & 3;
        return ((h & 1) == 0 ? x : -x) + ((h & 2) == 0 ? y : -y);
    }

public:
    PerlinNoise() {
        init(42);
    }

    void init(int seed) {
        uint8_t base[256];
        for (int i = 0; i < 256; i++) {
            base[i] = i;
        }

        long long s = seed;
        for (int i = 255; i > 0; i--) {
            s = ((s * 16807) + 7) % 2147483647;
            int j = s % (i + 1);
            std::swap(base[i], base[j]);
        }

        for (int i = 0; i < 512; i++) {
            p[i] = base[i & 255];
        }
    }

    double noise(double x, double y) {
        int X = (int)std::floor(x) & 255;
        int Y = (int)std::floor(y) & 255;

        double xf = x - std::floor(x);
        double yf = y - std::floor(y);

        double u = fade(xf);
        double v = fade(yf);

        int A = p[X] + Y;
        int B = p[X + 1] + Y;

        return lerp(
            lerp(grad(p[A], xf, yf), grad(p[B], xf - 1.0, yf), u),
            lerp(grad(p[A + 1], xf, yf - 1.0), grad(p[B + 1], xf - 1.0, yf - 1.0), u),
            v
        );
    }

    double fbm(double x, double y, int octaves, double lac, double gain) {
        double sum = 0.0;
        double amp = 1.0;
        double freq = 1.0;
        double maxVal = 0.0;
        for (int i = 0; i < octaves; i++) {
            sum += noise(x * freq, y * freq) * amp;
            maxVal += amp;
            amp *= gain;
            freq *= lac;
        }
        return sum / maxVal;
    }

    double ridged(double x, double y, int octaves, double lac, double gain) {
        double sum = 0.0;
        double amp = 1.0;
        double freq = 1.0;
        double prev = 1.0;
        for (int i = 0; i < octaves; i++) {
            double n = noise(x * freq, y * freq);
            n = 1.0 - std::abs(n);
            n = n * n * prev;
            prev = n;
            sum += n * amp;
            freq *= lac;
            amp *= gain;
        }
        return sum;
    }

    double generateHeight(double x, double z, double maxHeight) {
        double nx = x / 2000.0;
        double nz = z / 2000.0;

        double valleyNoise = (noise(nx * 0.5, nz * 0.5) + 1.0) * 0.5;
        
        double t = std::max(0.0, std::min(1.0, (valleyNoise - 0.25) / 0.50));
        double envelope = t * t * (3.0 - 2.0 * t);

        double h = (fbm(nx * 3.5 + 10.0, nz * 3.5 + 10.0, 5, 2.0, 0.5) + 1.0) * 0.5;
        double ridge = ridged(nx * 3.0 + 5.0, nz * 3.0 + 5.0, 5, 2.2, 0.52);

        h = h * 0.3 + ridge * 0.7;
        h *= envelope;

        h += (fbm(nx * 12.0, nz * 12.0, 4, 2.0, 0.45) + 1.0) * 0.04;
        h += (fbm(nx * 28.0, nz * 28.0, 3, 2.0, 0.4) + 1.0) * 0.012;

        double finalHeight = h * maxHeight;
        return finalHeight < 15.0 ? 15.0 : finalHeight;
    }
};

EMSCRIPTEN_BINDINGS(engine_module) {
    class_<PerlinNoise>("PerlinNoise")
        .constructor<>()
        .function("init", &PerlinNoise::init)
        .function("noise", &PerlinNoise::noise)
        .function("fbm", &PerlinNoise::fbm)
        .function("ridged", &PerlinNoise::ridged)
        .function("generateHeight", &PerlinNoise::generateHeight);
}
