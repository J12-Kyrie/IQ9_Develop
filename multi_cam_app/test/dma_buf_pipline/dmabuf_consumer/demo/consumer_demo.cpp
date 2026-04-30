#include <dmabuf_consumer/consumer.h>
// IQ9_Task_Packet, IQ9_TASK_MAGIC pulled in transitively via consumer.h
//   -> dmabuf_producer/types.h

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

static bool WritePpm(const dmabuf_consumer::DmaBufConsumer::Frame& frame,
                     const char* output_dir, int frame_number) {
    char filename[512];
    std::snprintf(filename, sizeof(filename), "%s/frame_%04d.ppm",
                  output_dir, frame_number);

    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        std::fprintf(stderr, "[consumer] failed to open %s: %s\n",
                     filename, std::strerror(errno));
        return false;
    }

    // P6 binary PPM: "P6\n{width} {height}\n255\n" + raw RGB24 data
    std::fprintf(fp, "P6\n%u %u\n255\n", frame.width, frame.height);
    std::fwrite(frame.data, 1, frame.data_size, fp);
    std::fclose(fp);

    std::fprintf(stdout, "[consumer] wrote %s\n", filename);
    return true;
}

static void MkdirP(const char* path) {
    // Simple mkdir -p: create the directory (ignore EEXIST)
    ::mkdir(path, 0755);
}

// Receive exactly `len` bytes from `fd`, retrying on partial reads.
// Returns true on success, false on EOF or error (error string set).
static bool RecvAll(int fd, void* buf, size_t len, std::string* out_error) {
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = recv(fd, ptr, remaining, 0);
        if (n == 0) {
            if (out_error != nullptr) {
                *out_error = "";  // EOF: peer disconnected
            }
            return false;
        }
        if (n < 0) {
            if (out_error != nullptr) {
                *out_error = std::string("recv failed: ") + strerror(errno);
            }
            return false;
        }
        ptr += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Usage: consumer_demo <init_socket> [relay_socket] [num_frames] [output_dir] [max_ppm_dumps] [ppm_interval_ms]
    const char* socket_path      = "/tmp/dmabuf_ipc.sock";
    const char* relay_socket     = "/tmp/iq9_rule_relay.sock";
    int num_frames               = 100;
    const char* output_dir       = "./ppm_output/";
    int max_ppm_dumps            = 10;
    int ppm_interval_ms          = 1000;

    if (argc > 1) socket_path     = argv[1];
    if (argc > 2) relay_socket    = argv[2];
    if (argc > 3) num_frames      = std::atoi(argv[3]);
    if (argc > 4) output_dir      = argv[4];
    if (argc > 5) max_ppm_dumps   = std::atoi(argv[5]);
    if (argc > 6) ppm_interval_ms = std::atoi(argv[6]);

    uint64_t ppm_interval_ns = static_cast<uint64_t>(ppm_interval_ms) * 1000000ULL;

    std::fprintf(stdout,
                 "[consumer] init_socket=%s relay_socket=%s output=%s "
                 "num_frames=%d max_ppm=%d interval=%dms\n",
                 socket_path, relay_socket, output_dir,
                 num_frames, max_ppm_dumps, ppm_interval_ms);

    MkdirP(output_dir);

    // --- Step 1: Set up relay UDS listener for rule_process ---
    // Must happen BEFORE connecting to producer, because the producer
    // starts frame processing as soon as the consumer connects.
    // Rule_process needs to be subscribed to MQTT before that.
    unlink(relay_socket);

    int relay_server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (relay_server_fd < 0) {
        std::fprintf(stderr, "[consumer] socket(relay) failed: %s\n",
                     strerror(errno));
        return 1;
    }

    struct sockaddr_un relay_addr;
    memset(&relay_addr, 0, sizeof(relay_addr));
    relay_addr.sun_family = AF_UNIX;
    strncpy(relay_addr.sun_path, relay_socket, sizeof(relay_addr.sun_path) - 1);

    if (bind(relay_server_fd, reinterpret_cast<struct sockaddr*>(&relay_addr),
             sizeof(relay_addr)) < 0) {
        std::fprintf(stderr, "[consumer] bind(relay '%s') failed: %s\n",
                     relay_socket, strerror(errno));
        close(relay_server_fd);
        return 1;
    }

    if (listen(relay_server_fd, 1) < 0) {
        std::fprintf(stderr, "[consumer] listen(relay) failed: %s\n",
                     strerror(errno));
        close(relay_server_fd);
        unlink(relay_socket);
        return 1;
    }

    std::fprintf(stdout,
                 "[consumer] relay socket listening on %s, waiting for rule_process...\n",
                 relay_socket);

    int relay_fd = accept(relay_server_fd, nullptr, nullptr);
    if (relay_fd < 0) {
        std::fprintf(stderr, "[consumer] accept(relay) failed: %s\n",
                     strerror(errno));
        close(relay_server_fd);
        unlink(relay_socket);
        return 1;
    }

    std::fprintf(stdout, "[consumer] rule_process connected on relay socket\n");

    close(relay_server_fd);
    relay_server_fd = -1;

    // --- Step 2: Connect to producer via init UDS (DMA-BUF mapping) ---
    // This triggers the producer to start processing frames, so rule_process
    // must already be connected and subscribed to MQTT by this point.
    dmabuf_consumer::DmaBufConsumer consumer;
    std::string error;

    if (!consumer.Init(socket_path, &error)) {
        std::fprintf(stderr, "[consumer] init failed: %s\n", error.c_str());
        close(relay_fd);
        unlink(relay_socket);
        return 1;
    }

    std::fprintf(stdout, "[consumer] connected to producer via init socket\n");

    // --- Step 3: Receive IQ9_Task_Packets from relay and process frames ---
    int frame_count = 0;
    int ppm_count   = 0;
    uint64_t next_ppm_ts = 0;

    while (frame_count < num_frames) {
        IQ9_Task_Packet pkt;
        memset(&pkt, 0, sizeof(pkt));

        error.clear();
        bool got_pkt = RecvAll(relay_fd, &pkt, sizeof(pkt), &error);
        if (!got_pkt) {
            if (error.empty()) {
                std::fprintf(stdout, "[consumer] rule_process disconnected\n");
            } else {
                std::fprintf(stderr, "[consumer] relay recv error: %s\n",
                             error.c_str());
            }
            break;
        }

        if (pkt.magic != IQ9_TASK_MAGIC) {
            std::fprintf(stderr,
                         "[consumer] bad magic 0x%08X (expected 0x%08X), dropping\n",
                         pkt.magic, IQ9_TASK_MAGIC);
            continue;
        }

        dmabuf_consumer::DmaBufConsumer::Frame frame;
        memset(&frame, 0, sizeof(frame));
        frame.fd = -1;

        error.clear();
        if (!consumer.MapFrame(pkt, &frame, &error)) {
            std::fprintf(stderr, "[consumer] MapFrame error: %s\n", error.c_str());
            break;
        }

        // Dump PPM based on timestamp interval
        bool should_dump = false;
        if (ppm_count < max_ppm_dumps) {
            if (ppm_interval_ns == 0) {
                should_dump = true;
            } else if (frame.timestamp_ns >= next_ppm_ts) {
                should_dump = true;
            }
        }

        if (should_dump) {
            WritePpm(frame, output_dir, ppm_count);
            std::fprintf(stdout,
                         "[consumer] PPM #%d from frame %d ts=%.2fs\n",
                         ppm_count, frame_count,
                         static_cast<double>(frame.timestamp_ns) / 1e9);
            next_ppm_ts = frame.timestamp_ns + ppm_interval_ns;
            ++ppm_count;
        }

        std::fprintf(stdout,
                     "[consumer] frame=%d slot=%u stride=%u offset=%lu expected_offset=%lu offset_check=%s %ux%u ts=%lu size=%u\n",
                     frame_count, frame.slot_index,
                     frame.slot_stride,
                     static_cast<unsigned long>(frame.slot_offset),
                     static_cast<unsigned long>(
                         static_cast<uint64_t>(frame.slot_index) *
                         static_cast<uint64_t>(frame.slot_stride)),
                     (frame.slot_offset ==
                      static_cast<uint64_t>(frame.slot_index) *
                      static_cast<uint64_t>(frame.slot_stride)) ? "ok" : "mismatch",
                     frame.width, frame.height,
                     static_cast<unsigned long>(frame.timestamp_ns),
                     frame.data_size);

        error.clear();
        if (!consumer.ReleaseFrame(frame, &error)) {
            std::fprintf(stderr, "[consumer] release error: %s\n",
                         error.c_str());
            break;
        }

        ++frame_count;

        if (ppm_count >= max_ppm_dumps && ppm_interval_ns > 0) {
            std::fprintf(stdout,
                         "[consumer] collected %d PPM dumps, continuing...\n",
                         ppm_count);
        }
    }

    std::fprintf(stdout,
                 "[consumer] processed %d frames, saved %d PPMs, shutting down\n",
                 frame_count, ppm_count);

    // --- Cleanup ---
    close(relay_fd);
    unlink(relay_socket);
    consumer.Shutdown();
    return 0;
}
