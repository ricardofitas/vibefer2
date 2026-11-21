#include "file.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
static uint32_t reference(uint32_t initial, const char *p, size_t n) {
    uint32_t r = ~initial;
    for (size_t i = 0; i < n; ++i) {
        r ^= static_cast<unsigned char>(p[i]);
        for (int bit=0;bit<8;++bit) r = (r >> 1) ^ ((r & 1u) ? 0xedb88320u : 0u);
    }
    return ~r;
}

int main() {
    std::mt19937 generator(24017);
    std::vector<char> bytes(8 * 1024 * 1024 + 8);
    for (auto &c : bytes) c = static_cast<char>(generator());
    unsigned cases = 0;
    for (unsigned offset = 0; offset < 8; ++offset) {
        for (size_t len = 0; len < 256; ++len) {
            const auto *p = bytes.data() + offset;
            const auto initial = uint32_t(generator());
            const auto expected = reference(initial,p,len);
            const size_t split = len / 3;
            auto crc = file::rc_crc32(initial,p,split);
            crc = file::rc_crc32(crc,p+split,len-split);
            if (expected != crc) return 1;
            ++cases;
        }
    }
    if (file::rc_crc32(0,"123456789",9) != 0xcbf43926u) return 2;
    const auto start = std::chrono::steady_clock::now();
    uint32_t crc = 0;
    for (int i=0;i<32;++i) crc = file::rc_crc32(crc,bytes.data(),8*1024*1024);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start);
    std::cout << "{\"cases\":" << cases+1 << ",\"stream_crc\":" << crc
              << ",\"mib_s\":" << 256 / elapsed.count() << "}\n";
}
