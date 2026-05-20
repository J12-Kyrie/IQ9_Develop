/*
 * IQ9 DMA-BUF channel protocol helpers.
 * Transport-agnostic sendAll/recvAll plus a UDS connect helper.
 * Packet types (IQ9_Task_Packet, IQ9_Release_Packet) are defined in
 * <dmabuf_consumer/types.h>.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <dmabuf_consumer/types.h>

namespace iq9 {
namespace channel {

inline bool recvAll(int fd, void* buf, size_t len)
{
    size_t total = 0;
    while (total < len)
    {
        ssize_t n = recv(fd, (char*)buf + total, len - total, 0);
        if (n <= 0) return false;
        total += (size_t)n;
    }
    return true;
}

inline bool sendAll(int fd, void const* buf, size_t len)
{
    size_t total = 0;
    while (total < len)
    {
        ssize_t n = send(fd, (const char*)buf + total, len - total, 0);
        if (n <= 0) return false;
        total += (size_t)n;
    }
    return true;
}

inline bool connectUds(const char* path, int* out_fd, std::string* out_error)
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        if (out_error) *out_error = "socket(AF_UNIX) failed";
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (out_error) *out_error = std::string("connect failed: ") + path;
        close(fd);
        return false;
    }
    *out_fd = fd;
    return true;
}

} // namespace channel
} // namespace iq9
