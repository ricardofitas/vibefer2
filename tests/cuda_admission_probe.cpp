#include "cuda/kernel.h"
#include "cuda/parameters.h"
#include <cuda/embedded_kernels.h>
#include <iostream>

// Stream selection results without generating megabytes of kernel source per base.
int main() {
    using namespace cuda_gfn;
    uint32_t base;
    unsigned length, pack, ratio, first, last;
    while (std::cin >> base >> length >> pack >> ratio >> first >> last) {
        const uint32_t *pool = length == 65536 ? embedded::primes_16 : embedded::primes_17;
        unsigned selected = 0, selected_ratio = 0, mask = 0;
        for (unsigned fields = first; fields <= last; ++fields) {
            const std::vector<uint32_t> primes(pool, pool + fields);
            for (unsigned candidate : {ratio, 2u}) {
                try {
                    (void)admit(base, length, pack, candidate, primes);
                    mask |= 1u << (2 * (fields - first) + (candidate == 2));
                } catch (const std::invalid_argument &) {
                }
            }
            if (!selected) {
                try {
                    const auto a = admit_profile(base, length, pack, ratio, primes);
                    selected = fields;
                    selected_ratio = a.ratio;
                } catch (const std::invalid_argument &) {
                }
            }
        }
        std::cout << selected << ' ' << selected_ratio << ' ' << mask << '\n';
    }
}
