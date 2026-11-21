#pragma once
#include <stdexcept>
#include <string>
namespace cuda_gfn {
inline std::string validate_opencl_ptx(std::string binary) {
    while(!binary.empty() && binary.back()=='\0') binary.pop_back();
    if(binary.find('\0')!=std::string::npos || binary.find(".version ")==std::string::npos ||
       binary.find(".entry ")==std::string::npos)
        throw std::runtime_error("NVIDIA OpenCL compiler did not export PTX");
    size_t pos=0;
    while((pos=binary.find("%envreg3",pos))!=std::string::npos) {
        if(pos+8<binary.size() && binary[pos+8]>='0' && binary[pos+8]<='9') { pos+=8; continue; }
        binary.replace(pos,8,"0");
        ++pos;
    }
    if(binary.find("%envreg")!=std::string::npos)
        throw std::runtime_error("Unsupported OpenCL PTX environment register");
    return binary;
}
}
