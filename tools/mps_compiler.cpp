// Read OpenCL C on stdin, select exactly the assigned GPU and write raw PTX.
// Vibefer validates and normalizes the returned PTX before loading the module.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "../src/cuda/mps_compiler.h"
#include <iostream>
#include <iterator>
#include <vector>

static void check(cl_int code,const char *name) {
    if(code!=CL_SUCCESS) throw std::runtime_error(std::string(name)+": "+std::to_string(code));
}
static cl_device_id find_device(const std::string &uuid) {
    cl_uint n=0;
    check(clGetPlatformIDs(0,nullptr,&n),"enumerate platforms");
    std::vector<cl_platform_id> platforms(n);
    check(clGetPlatformIDs(n,platforms.data(),nullptr),"read platforms");
    cl_device_id match=nullptr;
    for(auto platform:platforms) {
        cl_int e=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,nullptr,&n);
        if(e==CL_DEVICE_NOT_FOUND) continue;
        check(e,"enumerate GPUs");
        std::vector<cl_device_id> devices(n);
        check(clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,n,devices.data(),nullptr),"read GPUs");
        for(auto d:devices) {
            cl_uint vendor=0;
            check(clGetDeviceInfo(d,CL_DEVICE_VENDOR_ID,sizeof(vendor),&vendor,nullptr),"read vendor");
            if(vendor!=0x10de || vibefer_mps::uuid_hex(d)!=uuid) continue;
            if(match) throw std::runtime_error("Ambiguous device UUID");
            match=d;
        }
    }
    if(!match) throw std::runtime_error("Assigned NVIDIA GPU UUID unavailable");
    return match;
}
int main(int argc,char **argv) {
    try {
        if(argc!=2) throw std::runtime_error("Pass one GPU UUID encoded as 32 hex digits");
        const auto device=find_device(argv[1]);
        const std::string source{std::istreambuf_iterator<char>(std::cin),{}};
        if(source.empty() || std::cin.bad()) throw std::runtime_error("Missing OpenCL source");
        const char *text=source.data(); size_t length=source.size(); cl_int e;
        const auto context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&e);
        check(e,"create compiler context");
        auto program=clCreateProgramWithSource(context,1,&text,&length,&e);
        check(e,"create program");
        e=clBuildProgram(program,1,&device,"",nullptr,nullptr);
        if(e!=CL_SUCCESS) {
            size_t n=0;
            clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&n);
            std::vector<char> log(n+1);
            clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,n,log.data(),nullptr);
            throw std::runtime_error(std::string("OpenCL compilation failed: ")+log.data());
        }
        check(clGetProgramInfo(program,CL_PROGRAM_BINARY_SIZES,sizeof(length),&length,nullptr),"read binary size");
        std::string binary(length,'\0'); char *out=binary.data();
        check(clGetProgramInfo(program,CL_PROGRAM_BINARIES,sizeof(out),&out,nullptr),"read binary");
        std::cout.write(binary.data(),binary.size());
        clReleaseProgram(program); clReleaseContext(context);
        return std::cout ? 0 : 1;
    } catch(const std::exception &e) {
        std::cerr << "Vibefer OpenCL compiler: " << e.what() << "\n";
        return 1;
    }
}
