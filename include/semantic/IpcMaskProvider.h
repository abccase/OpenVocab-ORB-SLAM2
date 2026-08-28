#ifndef ORB_SLAM2_SEMANTIC_IPC_MASK_PROVIDER_H
#define ORB_SLAM2_SEMANTIC_IPC_MASK_PROVIDER_H

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "semantic/Telemetry.h"

namespace ORB_SLAM2 {
namespace semantic {

class ProtocolError : public std::runtime_error {
public:
    ProtocolError(const std::string& code, const std::string& detail = "");
    const std::string& code() const { return code_; }
private:
    std::string code_;
};

struct PacketExpectations {
    PacketExpectations();
    std::string run_id;
    std::string prompt_sha256;
    std::string model_manifest_sha256;
    int image_width;
    int image_height;
    std::uint64_t current_timestamp_ns;
    std::uint64_t max_age_ns;
};

struct OnlineInstance {
    std::uint32_t local_id;
    std::string label;
    double score;
    std::array<double, 4> box_xyxy;
    int mask_height;
    int mask_width;
    std::vector<std::uint32_t> mask_counts;
};

struct SemanticPacket {
    std::string run_id;
    std::uint64_t frame_id;
    std::uint64_t source_timestamp_ns;
    std::uint64_t produced_timestamp_ns;
    std::string prompt_sha256;
    std::string model_manifest_sha256;
    int image_width;
    int image_height;
    double inference_ms;
    std::uint64_t age_ns;
    std::vector<OnlineInstance> instances;
};

SemanticPacket decodeSemanticPacket(const std::vector<unsigned char>& payload,
                                    const PacketExpectations& expected);

class IpcClock {
public:
    virtual ~IpcClock() {}
    virtual std::uint64_t monotonicNanoseconds() const = 0;
};

class SystemIpcClock : public IpcClock {
public:
    std::uint64_t monotonicNanoseconds() const;
};

class IpcTransport {
public:
    virtual ~IpcTransport() {}
    virtual bool trySend(const std::vector<unsigned char>& payload) = 0;
    virtual bool tryReceive(std::vector<unsigned char>* payload) = 0;
};

std::unique_ptr<IpcTransport> makeZmqIpcTransport(
    const std::string& request_endpoint,
    const std::string& result_endpoint);

struct IpcProviderConfig {
    IpcProviderConfig();
    std::string run_id;
    std::string prompt_sha256;
    std::string model_manifest_sha256;
    double request_rate_cap_hz;
    std::uint64_t max_age_ns;
};

struct IpcPollResult {
    IpcPollResult();
    SemanticState state;
    std::string reason;
    bool request_attempted;
    bool request_sent;
    bool has_packet;
    double call_duration_ms;
    SemanticPacket packet;
};

class IpcMaskProvider {
public:
    IpcMaskProvider(const IpcProviderConfig& config,
                    std::unique_ptr<IpcTransport> transport,
                    const IpcClock& clock);
    IpcPollResult poll(std::uint64_t frame_id,
                       std::uint64_t source_timestamp_ns,
                       const std::vector<unsigned char>& jpeg_bytes,
                       int image_width, int image_height);
private:
    struct RequestRecord {
        std::uint64_t frame_id;
        std::uint64_t source_timestamp_ns;
        int image_width;
        int image_height;
    };
    IpcProviderConfig config_;
    std::unique_ptr<IpcTransport> transport_;
    const IpcClock& clock_;
    bool has_request_attempt_;
    std::uint64_t last_request_attempt_ns_;
    bool has_latest_payload_;
    std::vector<unsigned char> latest_payload_;
    std::string last_receive_error_;
    std::deque<RequestRecord> request_ledger_;
};

}  // namespace semantic
}  // namespace ORB_SLAM2

#endif
