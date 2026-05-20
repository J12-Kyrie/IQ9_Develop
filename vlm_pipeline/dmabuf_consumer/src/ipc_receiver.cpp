#include "ipc_receiver.h"

#include <cerrno>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace dmabuf_consumer {

IpcReceiver::IpcReceiver() {}

IpcReceiver::~IpcReceiver() {
    Shutdown();
}

bool IpcReceiver::Connect(const char* socket_path, std::string* out_error) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        if (out_error != nullptr) {
            *out_error = std::string("socket() failed: ") + std::strerror(errno);
        }
        return false;
    }
    sock_fd_ = DmaBufFd(fd);

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    size_t path_len = std::strlen(socket_path);
    if (path_len >= sizeof(addr.sun_path)) {
        if (out_error != nullptr) {
            *out_error = "socket path too long";
        }
        sock_fd_.reset();
        return false;
    }
    std::memcpy(addr.sun_path, socket_path, path_len);

    if (::connect(sock_fd_.get(),
                  reinterpret_cast<struct sockaddr*>(&addr),
                  sizeof(addr)) < 0) {
        if (out_error != nullptr) {
            *out_error = std::string("connect() failed: ") + std::strerror(errno);
        }
        sock_fd_.reset();
        return false;
    }

    return true;
}

bool IpcReceiver::RecvInit(IQ9_Init_Packet* out_pkt, int* out_fd,
                           std::string* out_error) {
    if (sock_fd_.get() < 0) {
        if (out_error != nullptr) {
            *out_error = "IpcReceiver::RecvInit: not connected";
        }
        return false;
    }

    // Step 1: recv IQ9_Init_Packet via plain recv
    ssize_t n = ::recv(sock_fd_.get(), out_pkt, sizeof(*out_pkt), MSG_WAITALL);
    if (n <= 0) {
        if (out_error != nullptr) {
            if (n == 0) {
                *out_error = "RecvInit: EOF during init packet";
            } else {
                *out_error = std::string("RecvInit: recv failed: ") +
                             std::strerror(errno);
            }
        }
        return false;
    }

    if (static_cast<size_t>(n) != sizeof(*out_pkt)) {
        if (out_error != nullptr) {
            *out_error = "RecvInit: short read on init packet: got " +
                         std::to_string(n);
        }
        return false;
    }

    if (out_pkt->magic != IQ9_INIT_MAGIC) {
        if (out_error != nullptr) {
            *out_error = "RecvInit: invalid init magic: 0x" +
                         std::to_string(out_pkt->magic);
        }
        return false;
    }

    int slot_count = static_cast<int>(out_pkt->slot_count);
    if (slot_count <= 0) {
        if (out_error != nullptr) {
            *out_error = "RecvInit: invalid slot_count " + std::to_string(slot_count);
        }
        return false;
    }
    if (out_fd == nullptr) {
        if (out_error != nullptr) {
            *out_error = "RecvInit: out_fd is null";
        }
        return false;
    }

    // Step 2: recv shared fd via recvmsg + SCM_RIGHTS
    uint8_t marker = 0;
    struct msghdr msg;
    std::memset(&msg, 0, sizeof(msg));

    struct iovec iov;
    iov.iov_base = &marker;
    iov.iov_len = sizeof(marker);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    char ctrl_buf[CMSG_SPACE(sizeof(int))];
    std::memset(ctrl_buf, 0, sizeof(ctrl_buf));
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    ssize_t nr = ::recvmsg(sock_fd_.get(), &msg, 0);
    if (nr <= 0) {
        if (out_error != nullptr) {
            *out_error = std::string("RecvInit: recvmsg shared fd failed: ") +
                         std::strerror(errno);
        }
        return false;
    }

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    int received_fd = -1;
    if (cmsg && cmsg->cmsg_level == SOL_SOCKET &&
        cmsg->cmsg_type == SCM_RIGHTS) {
        std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
    }
    if (received_fd < 0) {
        if (out_error != nullptr) {
            *out_error = "RecvInit: no shared fd received";
        }
        return false;
    }

    *out_fd = received_fd;
    return true;
}

bool IpcReceiver::RecvFrame(IQ9_Task_Packet* out_pkt,
                            std::string* out_error) {
    ssize_t n = ::recv(sock_fd_.get(), out_pkt, sizeof(*out_pkt), MSG_WAITALL);

    if (n == 0) {
        if (out_error != nullptr) {
            out_error->clear();
        }
        return false;
    }

    if (n < 0) {
        if (out_error != nullptr) {
            *out_error = std::string("recv() failed: ") + std::strerror(errno);
        }
        return false;
    }

    if (static_cast<size_t>(n) != sizeof(IQ9_Task_Packet)) {
        if (out_error != nullptr) {
            *out_error = "short read: expected " +
                         std::to_string(sizeof(IQ9_Task_Packet)) +
                         " bytes, got " + std::to_string(n);
        }
        return false;
    }

    if (out_pkt->magic != IQ9_TASK_MAGIC) {
        if (out_error != nullptr) {
            *out_error = "invalid task packet magic: 0x" +
                         std::to_string(out_pkt->magic);
        }
        return false;
    }

    return true;
}

bool IpcReceiver::SendRelease(const IQ9_Release_Packet& pkt,
                              std::string* out_error) {
    ssize_t n = ::send(sock_fd_.get(), &pkt, sizeof(pkt), 0);
    if (n < 0) {
        if (out_error != nullptr) {
            *out_error = std::string("send() release failed: ") +
                         std::strerror(errno);
        }
        return false;
    }
    if (n != static_cast<ssize_t>(sizeof(pkt))) {
        if (out_error != nullptr) {
            *out_error = "short write on release packet: sent " +
                         std::to_string(n) + " of " +
                         std::to_string(sizeof(pkt));
        }
        return false;
    }
    return true;
}

void IpcReceiver::Shutdown() {
    sock_fd_.reset();
}

}  // namespace dmabuf_consumer
