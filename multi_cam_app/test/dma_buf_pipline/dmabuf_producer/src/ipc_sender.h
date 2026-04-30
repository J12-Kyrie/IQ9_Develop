#ifndef DMABUF_PRODUCER_IPC_SENDER_H
#define DMABUF_PRODUCER_IPC_SENDER_H

#include <string>

#include "dmabuf_producer/types.h"

namespace dmabuf_producer {

class IpcSender {
public:
    IpcSender();
    ~IpcSender();

    IpcSender(const IpcSender&) = delete;
    IpcSender& operator=(const IpcSender&) = delete;

    bool Init(const char* socket_path, std::string* out_error);
    bool AcceptClient(std::string* out_error);

    // Handshake: send init packet + shared dma-buf fd via SCM_RIGHTS
    bool SendInit(const IQ9_Init_Packet& pkt, int shared_fd,
                  std::string* out_error);

    // Send task packet only (plain send, no SCM_RIGHTS)
    bool SendFrame(const IQ9_Task_Packet& pkt, std::string* out_error);

    // Non-blocking recv for release packets. Returns true if packet received.
    bool PollRelease(IQ9_Release_Packet* out_pkt);

    void Shutdown();

private:
    DmaBufFd server_fd_;
    DmaBufFd client_fd_;
    std::string socket_path_;
};

}  // namespace dmabuf_producer

#endif  // DMABUF_PRODUCER_IPC_SENDER_H
