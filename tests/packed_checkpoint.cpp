// Check packed checkpoint exchange with original arithmetic at both sides of
// an admission decision. Link with the normal GPU build's objects except main.
#include "ocl.h"
#include "transform.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <memory>

int main(int argc, char **argv) {
    try {
        if (argc != 2 && argc != 4)
            throw std::runtime_error("Usage: packed_checkpoint EMPTY_OUTPUT_DIRECTORY [n base]");
        const std::filesystem::path directory(argv[1]);
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("Output directory must not exist");
        const unsigned n = argc == 4 ? std::stoul(argv[2]) : 19;
        if (n < 16 || n > 23) throw std::runtime_error("Unsupported checkpoint size");
        const size_t size = size_t(1) << n;
        const uint32_t first_base = argc == 4 ? std::stoul(argv[3]) : 31296292;
        for (uint32_t base : std::array<uint32_t, 2>{first_base, first_base + 2}) {
            std::unique_ptr<transform> original(
                transform::create_original_gpu(base, n, false, 0, 3, nullptr, nullptr, false));
            std::unique_ptr<transform> optimized(
                transform::create_gpu(base, n, false, 0, 3, nullptr, nullptr, true));
            gint digits(size, base), expected(size, base), actual(size, base);
            for (size_t i = 0; i < size; ++i)
                digits.data()[i] = i % 3 == 0 ? -int(base / 2) : int((i * 31337u) % (base / 2));
            digits.balance();
            for (bool from_original : {true, false}) {
                transform *from = from_original ? original.get() : optimized.get();
                transform *to = from_original ? optimized.get() : original.get();
                from->setInt(digits);
                from->copy(1, 0);
                from->squareDup(true);
                from->copy(2, 0);
                from->squareDup(false);
                const auto path = directory / (std::to_string(base) + ".ctx");
                {
                    file context(path.string(), "wb", true);
                    from->saveContext(context, 3);
                    context.write_crc32();
                }
                {
                    file context(path.string(), "rb", true);
                    if (!to->readContext(context, 3) || !context.check_crc32())
                        throw std::runtime_error("Cross-backend checkpoint was rejected");
                }
                for (size_t reg : {size_t(0), size_t(1), size_t(2)}) {
                    if (reg) {
                        from->copy(0, reg);
                        to->copy(0, reg);
                    }
                    for (unsigned step = 0; step < 2; ++step) {
                        from->getInt(expected);
                        to->getInt(actual);
                        expected.unbalance();
                        actual.unbalance();
                        if (!std::equal(expected.data(), expected.data() + size, actual.data()))
                            throw std::runtime_error("Cross-backend register or resumed square mismatch");
                        from->squareDup(true);
                        to->squareDup(true);
                    }
                }
            }
        }
        std::cout << "PASS: packed/original checkpoint exchange, three registers and resumed arithmetic\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
