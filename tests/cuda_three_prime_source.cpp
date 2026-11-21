#include "cuda/three_prime.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("pass kernels/src/three-prime.cl");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot read kernel source");
        const std::string kernel(std::istreambuf_iterator<char>(input), {});
        using arithmetic = cuda_small_prime::transformGPUs<3, false>;
        const uint32_t bases[] = {1000000, 100000000, 499122172, 499122174,
                                  520000000, 620000000, 943718396, 943718398};
        unsigned cases = 0;
        for (unsigned n : {16, 17}) {
            for (unsigned bits : {32, 64}) {
                for (size_t workgroup : {32, 64}) {
                    const std::string reference = arithmetic::kernel_source(n, bits, workgroup, 1024, kernel);
                    for (uint32_t b : bases) {
                        if (!small_prime_bounds::supported(b)) throw std::runtime_error("test base rejected: " + std::to_string(b));
                        const cuda_small_prime::crt_parameters crt(b);
                        const int64_t pair = int64_t(small_prime_bounds::prime2) * small_prime_bounds::prime3;
                        if (crt.qpair*b + crt.rpair != pair
                            || crt.qthird*b + crt.rthird != small_prime_bounds::prime3)
                            throw std::runtime_error("CRT parameter mismatch");
                        const std::string source = arithmetic::kernel_source(n, bits, workgroup, 1024, kernel);
                        if (source != reference) throw std::runtime_error("source depends on the base");
                        for (const char *name : {"GFN_FIXED_", "CRT_QPAIR", "CRT_RPAIR", "CRT_QTHIRD", "CRT_RTHIRD", "GFN_PAIRED_DIGITS"})
                            if (source.find(name) != std::string::npos)
                                throw std::runtime_error("compiled-in base parameter remains");
                        ++cases;
                    }
                }
            }
        }
        std::cout << "PASS: " << cases << " base-independent three-prime sources and CRT parameters\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
