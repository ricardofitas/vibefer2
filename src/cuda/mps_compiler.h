// Linux CUDA MPS clients can compile OpenCL kernels in an independent process.
// Both transfers are private anonymous files. Neither uses application state paths.
#pragma once
#ifdef __linux__
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace vibefer_mps {
inline std::string uuid_hex(cl_device_id device) {
    std::array<unsigned char,16> uuid{};
    if (clGetDeviceInfo(device,0x106a,uuid.size(),uuid.data(),nullptr)!=CL_SUCCESS)
        throw std::runtime_error("Cannot query the assigned OpenCL device UUID");
    static constexpr char hex[]="0123456789abcdef";
    std::string out;
    for(auto byte:uuid) { out+=hex[byte>>4]; out+=hex[byte&15]; }
    return out;
}
inline std::string compile(const char *helper,cl_device_id device,const std::string &source) {
    if (helper[0]!='/') throw std::runtime_error("VIBEFER_MPS_COMPILER requires an absolute executable path");
    struct close_file { void operator()(FILE *f) const { std::fclose(f); } };
    using temp=std::unique_ptr<FILE,close_file>;
    temp input(std::tmpfile()),output(std::tmpfile());
    if(!input || !output) throw std::runtime_error("Cannot create private CUDA compiler files");
    if(std::fwrite(source.data(),1,source.size(),input.get())!=source.size() ||
       std::fflush(input.get()) || std::fseek(input.get(),0,SEEK_SET))
        throw std::runtime_error("Cannot prepare OpenCL compiler input");
    // Give the redirections distinct descriptors >= 3 even when the embedding
    // process starts with stdin, stdout or stderr closed.
    struct descriptor {
        int fd;
        explicit descriptor(FILE *file):fd(fcntl(fileno(file),F_DUPFD_CLOEXEC,3)) {}
        ~descriptor() { if(fd>=0) close(fd); }
    };
    descriptor in(input.get()),out(output.get());
    if(in.fd<0 || out.fd<0) throw std::runtime_error("Cannot duplicate the compiler files");
    std::string uuid=uuid_hex(device);
    char *argv[]={const_cast<char*>(helper),uuid.data(),nullptr};
    std::vector<char*> env;
    for(char **e=environ;*e;++e) {
        if(std::string_view(*e).rfind("CUDA_MPS_PIPE_DIRECTORY=",0)!=0) env.push_back(*e);
    }
    // A private/nonexistent MPS pipe disables the child connection even when
    // another user's default MPS service exists. Keep the same GPU visibility.
    char pipe[]="CUDA_MPS_PIPE_DIRECTORY=/dev/null";
    env.push_back(pipe); env.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int err=posix_spawn_file_actions_init(&actions);
    if(err) throw std::runtime_error("Cannot initialize compiler process");
    pid_t pid{};
    err=posix_spawn_file_actions_adddup2(&actions,in.fd,STDIN_FILENO);
    if(!err) err=posix_spawn_file_actions_adddup2(&actions,out.fd,STDOUT_FILENO);
    if(!err) err=posix_spawn(&pid,helper,&actions,nullptr,argv,env.data());
    posix_spawn_file_actions_destroy(&actions);
    if(err) throw std::runtime_error("Cannot start the CUDA compiler helper: "+std::to_string(err));
    int status=0;
    pid_t done;
    do { done=waitpid(pid,&status,0); } while(done<0 && errno==EINTR);
    if(done!=pid || !WIFEXITED(status) || WEXITSTATUS(status)!=0)
        throw std::runtime_error("The CUDA compiler helper failed");
    if(std::fseek(output.get(),0,SEEK_END)) throw std::runtime_error("Cannot inspect compiler output");
    const long size=std::ftell(output.get());
    if(size<1 || size>64L*1024*1024 || std::fseek(output.get(),0,SEEK_SET))
        throw std::runtime_error("Invalid PTX output length from compiler helper");
    std::string ptx(size,'\0');
    if(std::fread(ptx.data(),1,ptx.size(),output.get())!=ptx.size())
        throw std::runtime_error("Could not read all generated PTX");
    return ptx;
}
}
#endif
