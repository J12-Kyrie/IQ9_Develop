#include "ipc_sender.h"

#include <cerrno>
#include <cstring>
#include <sstream>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace dmabuf_producer {

IpcSender::IpcSender() {}

IpcSender::~IpcSender() {
    Shutdown();
}

bool IpcSender::Init(const char* socket_path, std::string* out_error) {
    if (socket_path == nullptr) {
        if (out_error != nullptr) {
            *out_error = "IpcSender::Init: socket_path is null";
        }
        return false;
    }

    socket_path_ = socket_path;

    // Remove stale socket file
    unlink(socket_path);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "socket(AF_UNIX, SOCK_STREAM) failed: " << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }
    server_fd_ = DmaBufFd(fd);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (bind(server_fd_.get(), reinterpret_cast<struct sockaddr*>(&addr),
             sizeof(addr)) < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "bind('" << socket_path << "') failed: " << strerror(errno);
            *out_error = oss.str();
        }
        server_fd_.reset();
        return false;
    }

    if (listen(server_fd_.get(), 1) < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "listen() failed: " << strerror(errno);
            *out_error = oss.str();
        }
        server_fd_.reset();
        return false;
    }

    return true;
}

bool IpcSender::AcceptClient(std::string* out_error) {
    if (server_fd_.get() < 0) {
        if (out_error != nullptr) {
            *out_error = "IpcSender::AcceptClient: server not initialized";
        }
        return false;
    }

    int fd = accept(server_fd_.get(), nullptr, nullptr);
    if (fd < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "accept() failed: " << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }

    client_fd_ = DmaBufFd(fd);
    return true;
}

bool IpcSender::SendInit(const IQ9_Init_Packet& pkt, int shared_fd,
                         std::string* out_error) {
    if (client_fd_.get() < 0) {
        if (out_error != nullptr) {
            *out_error = "IpcSender::SendInit: no client connected";
        }
        return false;
    }

    // Step 1: send IQ9_Init_Packet via plain send
    ssize_t n = ::send(client_fd_.get(), &pkt, sizeof(pkt), 0);
    if (n < 0 || static_cast<size_t>(n) != sizeof(pkt)) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "SendInit: send init packet failed: " << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }

    if (shared_fd < 0) {
        if (out_error != nullptr) {
            *out_error = "SendInit: invalid shared fd";
        }
        return false;
    }

    // Step 2: send shared fd via single sendmsg + SCM_RIGHTS
    uint8_t marker = 0xFF;
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));

    struct iovec iov;
    iov.iov_base = &marker;
    iov.iov_len = sizeof(marker);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    char ctrl_buf[CMSG_SPACE(sizeof(int))];
    memset(ctrl_buf, 0, sizeof(ctrl_buf));
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &shared_fd, sizeof(int));

    ssize_t sent = sendmsg(client_fd_.get(), &msg, 0);
    if (sent < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "SendInit: sendmsg shared fd failed: " << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }
    return true;
}

bool IpcSender::SendFrame(const IQ9_Task_Packet& pkt,
                          std::string* out_error) {
    if (client_fd_.get() < 0) {
        if (out_error != nullptr) {
            *out_error = "IpcSender::SendFrame: no client connected";
        }
        return false;
    }

    ssize_t sent = ::send(client_fd_.get(), &pkt, sizeof(pkt), 0);
    if (sent < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "SendFrame: send failed: " << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }

    if (static_cast<size_t>(sent) != sizeof(pkt)) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "SendFrame short write: sent=" << sent
                << " expected=" << sizeof(pkt);
            *out_error = oss.str();
        }
        return false;
    }

    return true;
}

bool IpcSender::PollRelease(IQ9_Release_Packet* out_pkt) {
    if (client_fd_.get() < 0 || out_pkt == nullptr) {
        return false;
    }

    ssize_t n = recv(client_fd_.get(), out_pkt, sizeof(*out_pkt), MSG_DONTWAIT);
    if (n < 0) {
        // EAGAIN/EWOULDBLOCK means no data available (expected)
        return false;
    }

    if (static_cast<size_t>(n) != sizeof(*out_pkt)) {
        return false;
    }

    if (out_pkt->magic != IQ9_RELEASE_MAGIC) {
        return false;
    }

    return true;
}

void IpcSender::Shutdown() {
    client_fd_.reset();
    server_fd_.reset();

    if (!socket_path_.empty()) {
        unlink(socket_path_.c_str());
        socket_path_.clear();
    }
}

}  // namespace dmabuf_producer
