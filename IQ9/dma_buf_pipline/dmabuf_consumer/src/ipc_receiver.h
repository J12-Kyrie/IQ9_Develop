#ifndef DMABUF_CONSUMER_IPC_RECEIVER_H
#define DMABUF_CONSUMER_IPC_RECEIVER_H

#include <string>
#include <dmabuf_producer/types.h>

namespace dmabuf_consumer {

class IpcReceiver {
public:
    IpcReceiver();
    ~IpcReceiver();

    IpcReceiver(const IpcReceiver&) = delete;
    IpcReceiver& operator=(const IpcReceiver&) = delete;

    // Connect to the producer's UDS socket.
    bool Connect(const char* socket_path, std::string* out_error);

    // Handshake: receive init packet + shared dma-buf fd via SCM_RIGHTS.
    bool RecvInit(IQ9_Init_Packet* out_pkt, int* out_fd,
                  std::string* out_error);

    // Receive task packet only (plain recv, no SCM_RIGHTS).
    // Returns false with empty error on EOF (producer disconnected gracefully).
    bool RecvFrame(IQ9_Task_Packet* out_pkt, std::string* out_error);

    // Send a release packet back to the producer.
    bool SendRelease(const IQ9_Release_Packet& pkt, std::string* out_error);

    // Close the socket.
    void Shutdown();

private:
    DmaBufFd sock_fd_;
};

}  // namespace dmabuf_consumer

#endif  // DMABUF_CONSUMER_IPC_RECEIVER_H
