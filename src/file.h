/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <iomanip>
#include <array>

#include <gmp.h>

// #if defined(_WIN64)
// #include <IO.h>
// #else
// #include <unistd.h>
// #endif

#include "pio.h"

class file
{
private:
	const std::string _filename;
	FILE * const _cFile;
	const bool _fatal;
	// const bool _isSync;
	uint32_t _crc32 = 0;

public:
	file(const std::string & filename, const char * const mode, const bool fatal)
		: _filename(filename), _cFile(pio::open(filename.c_str(), mode)), _fatal(fatal), _crc32(0) //, _isSync(std::string(mode) == "wb")
	{
		if (_cFile == nullptr) error("cannot open file");
	}

	file(const std::string & filename)
		: _filename(filename), _cFile(pio::open(filename.c_str(), "rb")), _fatal(false), _crc32(0) //, _isSync(false)
	{
		// _cFile may be null
	}

	virtual ~file()
	{
		if (_cFile != nullptr)
		{
// 			if (_isSync)
// 			{
// #if defined(_WIN64)
// 				_commit(_fileno(_cFile));
// #else
// 				fsync(fileno(_cFile));
// #endif
// 			}
			if (std::fclose(_cFile) != 0) error("cannot close file");
		}
	}

	void error(const std::string & str) const
	{
		std::ostringstream ss; ss << _filename << ": " << str;
		pio::error(ss.str(), _fatal);
	}

	bool exists() const { return (_cFile != nullptr); }

	uint32_t crc32() const { return _crc32; }

    // IEEE CRC-32, slicing by eight. Keep the upstream initial/final complement
    // and streaming contract so existing proofs and checkpoints remain readable.
    static uint32_t rc_crc32(const uint32_t previous, const char *buf, size_t len)
    {
        using tables = std::array<std::array<uint32_t, 256>, 8>;
        static const tables table = [] {
            tables t{};
            for (unsigned i = 0; i < 256; ++i) {
                uint32_t r = i;
                for (unsigned bit = 0; bit < 8; ++bit)
                    r = (r >> 1) ^ ((r & 1u) ? 0xedb88320u : 0u);
                t[0][i] = r;
            }
            for (unsigned slice = 1; slice < 8; ++slice)
                for (unsigned i = 0; i < 256; ++i) {
                    const uint32_t r = t[slice - 1][i];
                    t[slice][i] = (r >> 8) ^ t[0][r & 255u];
                }
            return t;
        }();
        const auto *p = reinterpret_cast<const unsigned char *>(buf);
        uint32_t crc = ~previous;
        while (len >= 8) {
            crc = table[7][(crc ^ p[0]) & 255u]
                ^ table[6][((crc >> 8) ^ p[1]) & 255u]
                ^ table[5][((crc >> 16) ^ p[2]) & 255u]
                ^ table[4][(crc >> 24) ^ p[3]]
                ^ table[3][p[4]] ^ table[2][p[5]] ^ table[1][p[6]] ^ table[0][p[7]];
            p += 8;
            len -= 8;
        }
        for (size_t i = 0; i < len; ++i)
            crc = (crc >> 8) ^ table[0][(crc ^ p[i]) & 255u];
        return ~crc;
    }

	bool read(char * const ptr, const size_t size)
	{
		const size_t ret = std::fread(ptr , sizeof(char), size, _cFile);
		_crc32 = rc_crc32(_crc32, ptr, ret);
		if (ret == size * sizeof(char)) return true;
		error("failure of a read operation");
		return false;
	}

	bool write(const char * const ptr, const size_t size)
	{
		const size_t ret = std::fwrite(ptr , sizeof(char), size, _cFile);
		_crc32 = rc_crc32(_crc32, ptr, size);
		if (ret == size * sizeof(char)) return true;
		error("failure of a write operation");
		return false;
	}

	bool read(mpz_t & z)
	{
		if (mpz_inp_raw(z, _cFile) != 0) return true;
		_crc32 = _crc32 ^ static_cast<uint32_t>(mpz_fdiv_ui(z, 0xfedcba98));	// fake crc32
		error("failure of a read operation");
		return false;
	}

	bool write(const mpz_t & z)
	{
		if (mpz_out_raw(_cFile, z) != 0) return true;
		_crc32 = _crc32 ^ static_cast<uint32_t>(mpz_fdiv_ui(z, 0xfedcba98));	// fake crc32
		error("failure of a write operation");
		return false;
	}

	bool print(const char * const str)
	{
		const int ret = std::fprintf(_cFile, "%s", str);
		if (ret >= 0) return true;
		error("failure of a print operation");
		return false;
	}

	void write_crc32()
	{
		uint32_t crc32 = ~_crc32 ^ 0xa23777ac;
		write(reinterpret_cast<const char *>(&crc32), sizeof(crc32));
	}

	bool check_crc32()
	{
		uint32_t crc32 = 0, ocrc32 = ~_crc32 ^ 0xa23777ac;	// before the read operation
		if (!read(reinterpret_cast<char *>(&crc32), sizeof(crc32))) return false;
		const bool success = (crc32 == ocrc32);
		if (!success) error("bad file (crc32)");
		return success;
	}
};
