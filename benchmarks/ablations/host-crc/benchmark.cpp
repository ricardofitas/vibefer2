// SPDX-License-Identifier: MIT
// Build with run.py, which extracts the two unmodified production functions.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <zlib.h>
#include "crc_methods.h"

using Method = uint32_t (*)(uint32_t, const char *, size_t);
constexpr size_t proof_bytes = 4194380;
constexpr unsigned sweeps = 12;
constexpr unsigned pairs = 20;
constexpr size_t page_bytes = 4096;

uint32_t oracle(uint32_t seed, const std::vector<char> &data, unsigned repeats) {
    for (unsigned i = 0; i < repeats; ++i)
        seed = static_cast<uint32_t>(crc32_z(seed, reinterpret_cast<const Bytef *>(data.data()), data.size()));
    return seed;
}

uint32_t stream(Method update, uint32_t seed, const std::vector<char> &data,
                size_t chunk, unsigned repeats) {
    for (unsigned i = 0; i < repeats; ++i) {
        for (size_t offset = 0; offset < data.size(); offset += chunk) {
            const size_t count = std::min(chunk, data.size() - offset);
            seed = update(seed, data.data() + offset, count);
        }
    }
    return seed;
}

long long cpu_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) != 0)
        throw std::runtime_error("clock_gettime failed");
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}

int main() {
    std::vector<char> data(proof_bytes);
    uint32_t state = 0x31415926u;
    for (char &byte : data) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        byte = static_cast<char>(state & 255u);
    }
    const std::array<Method, 2> methods{Historical::rc_crc32, Released::rc_crc32};
    const std::array<const char *, 2> names{"vibefer_bytewise", "vibefer2_slice8"};
    const std::array<size_t, 2> chunks{proof_bytes, page_bytes};

    for (auto update : methods) {
        if (update(0, "123456789", 9) != 0xcbf43926u)
            throw std::runtime_error("standard CRC check failed");
        for (size_t chunk : chunks) {
            for (unsigned warm = 0; warm < 2; ++warm) {
                if (stream(update, warm, data, chunk, sweeps) != oracle(warm, data, sweeps))
                    throw std::runtime_error("warm-up CRC failed");
            }
        }
    }
    std::cout << "pair,chunk_bytes,position,method,sweeps,bytes_per_sweep,wall_ns,thread_cpu_ns,crc32_hex\n";
    for (unsigned pair = 0; pair < pairs; ++pair) {
        const auto seed = 0x9e3779b9u * (pair + 1);
        const auto expected = oracle(seed, data, sweeps);
        for (unsigned block = 0; block < 2; ++block) {
            const unsigned condition = (block + pair) % 2;
            const size_t chunk = chunks[condition];
            for (unsigned order = 0; order < 2; ++order) {
                // Counterbalance method order independently of within-pair position of condition.
                const unsigned method = (pair + condition + order) % 2;
                const auto cpu_start = cpu_ns();
                const auto start = std::chrono::steady_clock::now();
                const auto crc = stream(methods[method], seed, data, chunk, sweeps);
                const auto stop = std::chrono::steady_clock::now();
                const auto cpu_stop = cpu_ns();
                if (crc != expected) throw std::runtime_error("measured CRC differs from zlib");
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                std::cout << pair + 1 << ',' << chunk << ',' << order + 1 << ',' << names[method]
                          << ',' << sweeps << ',' << proof_bytes << ',' << ns << ',' << cpu_stop - cpu_start
                          << ',' << std::hex << crc << std::dec << '\n';
            }
        }
    }
}
