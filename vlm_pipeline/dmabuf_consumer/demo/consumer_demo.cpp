#include <dmabuf_consumer/consumer.h>
// IQ9_Task_Packet, IQ9_TASK_MAGIC pulled in transitively via consumer.h
//   -> dmabuf_producer/types.h

#ifdef ENABLE_VLM
#include "vlm/thread_pool.h"
#include "vlm/jpeg_encoder.h"
#include "vlm/vlm_client.h"
#include "vlm/jsonl_logger.h"
#include <memory>
#endif

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <atomic>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>

static std::atomic<bool> g_shutdown{false};

static void signalHandler(int /*sig*/) {
    g_shutdown.store(true);
}

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

#ifdef ENABLE_VLM
// Build VLM prompt from metadata JSON and template.
// Replaces {rule} and {dets} placeholders.
static std::string buildPrompt(const std::string& metadata_json,
                               const std::string& tmpl) {
    std::string result = tmpl;
    if (metadata_json.empty()) {
        // No metadata available — strip placeholders with defaults
        size_t pos;
        while ((pos = result.find("{rule}")) != std::string::npos)
            result.replace(pos, 6, "unknown");
        while ((pos = result.find("{dets}")) != std::string::npos)
            result.replace(pos, 6, "none");
        return result;
    }
    // Parse metadata to extract rule and detections
    // Use simple string extraction to avoid requiring nlohmann/json here
    // (it is linked but we keep this lightweight)
    std::string rule = "unknown";
    std::string dets = "none";

    // Extract "rule": "..."
    size_t rp = metadata_json.find("\"rule\":");
    if (rp != std::string::npos) {
        rp = metadata_json.find('"', rp + 7);
        if (rp != std::string::npos) {
            size_t re = metadata_json.find('"', rp + 1);
            if (re != std::string::npos) {
                rule = metadata_json.substr(rp + 1, re - rp - 1);
            }
        }
    }

    // Extract "detections": [...] — use as-is for the prompt
    size_t dp = metadata_json.find("\"detections\":");
    if (dp != std::string::npos) {
        size_t da = metadata_json.find('[', dp);
        if (da != std::string::npos) {
            int depth = 0;
            size_t end = da;
            for (size_t i = da; i < metadata_json.size(); ++i) {
                if (metadata_json[i] == '[') ++depth;
                else if (metadata_json[i] == ']') {
                    --depth;
                    if (depth == 0) { end = i + 1; break; }
                }
            }
            dets = metadata_json.substr(da, end - da);
        }
    }

    size_t pos;
    while ((pos = result.find("{rule}")) != std::string::npos)
        result.replace(pos, 6, rule);
    while ((pos = result.find("{dets}")) != std::string::npos)
        result.replace(pos, 6, dets);
    return result;
}
#endif

int main(int argc, char* argv[]) {
    // Usage: consumer_demo <init_socket> [relay_socket] [num_frames] [output_dir]
    //                       [max_ppm_dumps] [ppm_interval_ms]
    //                       [--vlm-enabled] [--vlm-url URL] [--vlm-prompt TEMPLATE]
    //                       [--vlm-threads N] [--vlm-queue N] [--vlm-jpeg-quality N]
    //                       [--vlm-timeout N] [--vlm-jsonl PATH]
    //                       [--vlm-drop-policy drop-newest|drop-oldest]

    // Install signal handlers
    struct sigaction sa{};
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);

    const char* socket_path      = "/tmp/dmabuf_ipc.sock";
    const char* relay_socket     = "/tmp/iq9_rule_relay.sock";
    int num_frames               = 100;
    const char* output_dir       = "./ppm_output/";
    int max_ppm_dumps            = 10;
    int ppm_interval_ms          = 1000;

    // VLM defaults
    bool vlm_enabled             = false;
    const char* vlm_url          = "http://127.0.0.1:8000";
    const char* vlm_prompt_tmpl  = "Describe image";
    int vlm_threads              = 2;
    int vlm_queue                = 16;
    int vlm_jpeg_quality         = 80;
    int vlm_timeout              = 30;
    const char* vlm_jsonl        = "/tmp/vlm_responses.jsonl";
    const char* vlm_drop_policy  = "drop-newest";

    // First pass: positional arguments (up to 6)
    int positional_end = 1;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') break;
        positional_end = i + 1;
    }

    if (positional_end > 1) socket_path     = argv[1];
    if (positional_end > 2) relay_socket    = argv[2];
    if (positional_end > 3) num_frames      = std::atoi(argv[3]);
    if (positional_end > 4) output_dir      = argv[4];
    if (positional_end > 5) max_ppm_dumps   = std::atoi(argv[5]);
    if (positional_end > 6) ppm_interval_ms = std::atoi(argv[6]);

    // Second pass: VLM flags
    for (int i = positional_end; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vlm-enabled") {
            vlm_enabled = true;
        } else if (arg == "--vlm-url" && i + 1 < argc) {
            vlm_url = argv[++i];
        } else if (arg == "--vlm-prompt" && i + 1 < argc) {
            vlm_prompt_tmpl = argv[++i];
        } else if (arg == "--vlm-threads" && i + 1 < argc) {
            vlm_threads = std::atoi(argv[++i]);
        } else if (arg == "--vlm-queue" && i + 1 < argc) {
            vlm_queue = std::atoi(argv[++i]);
        } else if (arg == "--vlm-jpeg-quality" && i + 1 < argc) {
            vlm_jpeg_quality = std::atoi(argv[++i]);
        } else if (arg == "--vlm-timeout" && i + 1 < argc) {
            vlm_timeout = std::atoi(argv[++i]);
        } else if (arg == "--vlm-jsonl" && i + 1 < argc) {
            vlm_jsonl = argv[++i];
        } else if (arg == "--vlm-drop-policy" && i + 1 < argc) {
            vlm_drop_policy = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stdout,
                "Usage: %s <init_socket> [relay_socket] [num_frames] [output_dir]\n"
                "       [max_ppm_dumps] [ppm_interval_ms]\n"
                "       [--vlm-enabled] [--vlm-url URL] [--vlm-prompt TEMPLATE]\n"
                "       [--vlm-threads N] [--vlm-queue N] [--vlm-jpeg-quality N]\n"
                "       [--vlm-timeout N] [--vlm-jsonl PATH]\n"
                "       [--vlm-drop-policy drop-newest|drop-oldest]\n",
                argv[0]);
            return 0;
        }
    }

    uint64_t ppm_interval_ns = static_cast<uint64_t>(ppm_interval_ms) * 1000000ULL;

    std::fprintf(stdout,
                 "[consumer] init_socket=%s relay_socket=%s output=%s "
                 "num_frames=%d max_ppm=%d interval=%dms\n",
                 socket_path, relay_socket, output_dir,
                 num_frames, max_ppm_dumps, ppm_interval_ms);

#ifdef ENABLE_VLM
    if (vlm_enabled) {
        std::fprintf(stdout,
                     "[vlm] enabled url=%s threads=%d queue=%d quality=%d "
                     "timeout=%ds jsonl=%s drop_policy=%s\n",
                     vlm_url, vlm_threads, vlm_queue, vlm_jpeg_quality,
                     vlm_timeout, vlm_jsonl, vlm_drop_policy);
    }
#else
    if (vlm_enabled) {
        std::fprintf(stderr, "[vlm] WARNING: --vlm-enabled passed but binary built without ENABLE_VLM\n");
        vlm_enabled = false;
    }
#endif

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

#ifdef ENABLE_VLM
    // --- Initialize VLM components (heap-allocated, safe for lambda capture) ---
    std::shared_ptr<vlm::JpegEncoder> encoder;
    std::shared_ptr<vlm::VlmClient> client;
    std::shared_ptr<vlm::JsonlLogger> logger;
    std::unique_ptr<vlm::ThreadPool> pool;

    if (vlm_enabled) {
        vlm::JpegEncoder::Config enc_cfg;
        enc_cfg.quality = vlm_jpeg_quality;
        encoder = std::make_shared<vlm::JpegEncoder>(enc_cfg);

        vlm::VlmClient::Config cli_cfg;
        cli_cfg.server_url = vlm_url;
        cli_cfg.timeout_seconds = vlm_timeout;
        client = std::make_shared<vlm::VlmClient>(cli_cfg);

        logger = std::make_shared<vlm::JsonlLogger>(vlm_jsonl);

        pool.reset(new vlm::ThreadPool(static_cast<size_t>(vlm_threads),
                                        static_cast<size_t>(vlm_queue)));
    }
#endif

    // --- Step 3: Receive IQ9_Task_Packets from relay and process frames ---
    int frame_count = 0;
    int ppm_count   = 0;
    uint64_t next_ppm_ts = 0;

#ifdef ENABLE_VLM
    uint64_t vlm_frame_num = 0;
#endif

    while (frame_count < num_frames && !g_shutdown.load()) {
        IQ9_Task_Packet pkt;
        memset(&pkt, 0, sizeof(pkt));

        // MSG_PEEK-based framing dispatch (supports metadata header protocol)
        uint32_t peek;
        ssize_t n = recv(relay_fd, &peek, sizeof(peek), MSG_PEEK);
        if (n != sizeof(peek)) {
            if (n == 0) {
                std::fprintf(stdout, "[consumer] rule_process disconnected\n");
            } else {
                std::fprintf(stderr, "[consumer] relay peek error: %s\n",
                             strerror(errno));
            }
            break;
        }

        std::string metadata_json;

        if (peek == IQ9_TASK_MAGIC) {
            // Backward compatibility: no metadata header, old protocol
            error.clear();
            if (!RecvAll(relay_fd, &pkt, sizeof(pkt), &error)) {
                if (error.empty()) {
                    std::fprintf(stdout, "[consumer] rule_process disconnected\n");
                } else {
                    std::fprintf(stderr, "[consumer] relay recv error: %s\n",
                                 error.c_str());
                }
                break;
            }
            metadata_json = "";
        } else if (peek > 0 && peek < 65536) {
            // Metadata header present: [uint32_t meta_len][JSON][IQ9_Task_Packet]
            uint32_t meta_len;
            error.clear();
            if (!RecvAll(relay_fd, &meta_len, sizeof(meta_len), &error)) {
                std::fprintf(stderr, "[consumer] relay recv meta_len error: %s\n",
                             error.empty() ? "EOF" : error.c_str());
                break;
            }
            std::string meta_json(meta_len, '\0');
            if (!RecvAll(relay_fd, &meta_json[0], meta_len, &error)) {
                std::fprintf(stderr, "[consumer] relay recv meta_json error: %s\n",
                             error.empty() ? "EOF" : error.c_str());
                break;
            }
            error.clear();
            if (!RecvAll(relay_fd, &pkt, sizeof(pkt), &error)) {
                if (error.empty()) {
                    std::fprintf(stdout, "[consumer] rule_process disconnected\n");
                } else {
                    std::fprintf(stderr, "[consumer] relay recv pkt error: %s\n",
                                 error.c_str());
                }
                break;
            }
            metadata_json = meta_json;
        } else {
            std::fprintf(stderr, "bad framing: peek=0x%08x, skipping\n", peek);
            continue;
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

#ifdef ENABLE_VLM
        // VLM path (async, non-blocking)
        if (vlm_enabled && pkt.data_type == PIXEL_FMT_RGB24) {
            // Copy RGB24 data to owned buffer (frame.data is DMA-BUF mmap)
            auto rgb_copy = std::make_shared<std::vector<uint8_t>>(
                static_cast<uint8_t*>(frame.data),
                static_cast<uint8_t*>(frame.data) + frame.data_size);

            // Build prompt from metadata
            std::string prompt = buildPrompt(metadata_json, vlm_prompt_tmpl);
            int source_id = static_cast<int>(pkt.slot_index);
            std::string rule_name;
            // Extract rule name from metadata for logging
            if (!metadata_json.empty()) {
                size_t rp = metadata_json.find("\"rule\":");
                if (rp != std::string::npos) {
                    rp = metadata_json.find('"', rp + 7);
                    if (rp != std::string::npos) {
                        size_t re = metadata_json.find('"', rp + 1);
                        if (re != std::string::npos) {
                            rule_name = metadata_json.substr(rp + 1, re - rp - 1);
                        }
                    }
                }
            }
            uint64_t ts_ns = pkt.timestamp_ns;
            uint64_t fnum = vlm_frame_num++;
            uint32_t w = frame.width;
            uint32_t h = frame.height;
            uint32_t rb = frame.width * 3; // RGB24 row bytes

            bool submitted = pool->submit([=, rgb_copy]() {
                // 1. JPEG encode (with resize)
                std::vector<uint8_t> jpeg;
                std::string enc_err;
                struct timespec t0;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                bool ok = encoder->encode(rgb_copy->data(), w, h, rb, jpeg, &enc_err);
                struct timespec t1;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double enc_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                                (t1.tv_nsec - t0.tv_nsec) / 1e6;

                // 2. HTTP POST to VLM server
                vlm::VlmResponse resp;
                resp.success = false;
                resp.latency_ms = 0;
                if (ok) {
                    vlm::VlmRequest req;
                    req.jpeg_data = jpeg;
                    req.prompt = prompt;
                    req.source_id = source_id;
                    req.rule_name = rule_name;
                    req.timestamp_ns = ts_ns;
                    req.frame_number = fnum;
                    req.detections_json = metadata_json;
                    resp = client->infer(req);
                }
                struct timespec t2;
                clock_gettime(CLOCK_MONOTONIC, &t2);
                double total_ms = (t2.tv_sec - t0.tv_sec) * 1000.0 +
                                  (t2.tv_nsec - t0.tv_nsec) / 1e6;

                // 3. Log to JSONL
                vlm::VlmLogEntry entry;
                entry.timestamp_ns = ts_ns;
                entry.source_id = source_id;
                entry.rule_name = rule_name;
                entry.detections = metadata_json;
                entry.vlm_response = resp.content;
                entry.vlm_success = resp.success;
                entry.vlm_error = ok ? resp.error : enc_err;
                entry.encode_ms = enc_ms;
                entry.http_ms = resp.latency_ms;
                entry.total_ms = total_ms;
                entry.frame_number = fnum;
                entry.jpeg_size_bytes = jpeg.size();
                logger->log(entry);
            });

            if (!submitted) {
                std::string policy(vlm_drop_policy);
                if (policy == "drop-oldest") {
                    pool->evict_oldest();
                    // Re-submit after eviction
                    pool->submit([=, rgb_copy]() {
                        std::vector<uint8_t> jpeg;
                        std::string enc_err;
                        struct timespec t0;
                        clock_gettime(CLOCK_MONOTONIC, &t0);
                        bool ok = encoder->encode(rgb_copy->data(), w, h, rb, jpeg, &enc_err);
                        struct timespec t1;
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        double enc_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                                        (t1.tv_nsec - t0.tv_nsec) / 1e6;

                        vlm::VlmResponse resp;
                        resp.success = false;
                        resp.latency_ms = 0;
                        if (ok) {
                            vlm::VlmRequest req;
                            req.jpeg_data = jpeg;
                            req.prompt = prompt;
                            req.source_id = source_id;
                            req.rule_name = rule_name;
                            req.timestamp_ns = ts_ns;
                            req.frame_number = fnum;
                            req.detections_json = metadata_json;
                            resp = client->infer(req);
                        }
                        struct timespec t2;
                        clock_gettime(CLOCK_MONOTONIC, &t2);
                        double total_ms = (t2.tv_sec - t0.tv_sec) * 1000.0 +
                                          (t2.tv_nsec - t0.tv_nsec) / 1e6;

                        vlm::VlmLogEntry entry;
                        entry.timestamp_ns = ts_ns;
                        entry.source_id = source_id;
                        entry.rule_name = rule_name;
                        entry.detections = metadata_json;
                        entry.vlm_response = resp.content;
                        entry.vlm_success = resp.success;
                        entry.vlm_error = ok ? resp.error : enc_err;
                        entry.encode_ms = enc_ms;
                        entry.http_ms = resp.latency_ms;
                        entry.total_ms = total_ms;
                        entry.frame_number = fnum;
                        entry.jpeg_size_bytes = jpeg.size();
                        logger->log(entry);
                    });
                } else {
                    fprintf(stderr, "[vlm] queue full, dropping frame %d\n", frame_count);
                }
            }
        } else if (vlm_enabled && pkt.data_type != PIXEL_FMT_RGB24) {
            fprintf(stderr, "[vlm] skipping frame %d: unexpected format %d (expected RGB24)\n",
                    frame_count, pkt.data_type);
        }
#endif

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

#ifdef ENABLE_VLM
    // Drain VLM queue before exit (blocks until all in-flight tasks complete)
    if (vlm_enabled && pool) {
        std::fprintf(stdout, "[vlm] draining thread pool...\n");
        pool->drain();
        std::fprintf(stdout, "[vlm] pool drained, completed=%zu\n",
                     pool->completed());
    }
#endif

    std::fprintf(stdout,
                 "[consumer] processed %d frames, saved %d PPMs, shutting down\n",
                 frame_count, ppm_count);

    // --- Cleanup ---
    close(relay_fd);
    unlink(relay_socket);
    consumer.Shutdown();
    return 0;
}
