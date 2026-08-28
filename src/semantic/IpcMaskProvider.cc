#include "semantic/IpcMaskProvider.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include <zmq.h>

namespace ORB_SLAM2 {
namespace semantic {
namespace {

struct Value {
    enum Type { NIL, BOOL, UINT, SINT, REAL, STRING, BINARY, ARRAY, MAP };
    Value() : type(NIL), boolean(false), uint_value(0), sint_value(0), real_value(0.0) {}
    Type type;
    bool boolean;
    std::uint64_t uint_value;
    std::int64_t sint_value;
    double real_value;
    std::string string_value;
    std::vector<unsigned char> binary_value;
    std::vector<Value> array_value;
    std::map<std::string, Value> map_value;
};

class Reader {
public:
    explicit Reader(const std::vector<unsigned char>& bytes) : bytes_(bytes), offset_(0) {}

    Value read() {
        const unsigned char marker = byte();
        if (marker <= 0x7f) return unsignedValue(marker);
        if (marker >= 0xe0) return signedValue(static_cast<std::int8_t>(marker));
        if ((marker & 0xf0) == 0x80) return map(marker & 0x0f);
        if ((marker & 0xf0) == 0x90) return array(marker & 0x0f);
        if ((marker & 0xe0) == 0xa0) return string(marker & 0x1f);
        switch (marker) {
            case 0xc0: return Value();
            case 0xc2: return boolValue(false);
            case 0xc3: return boolValue(true);
            case 0xc4: return binary(integer<std::uint8_t>());
            case 0xc5: return binary(integer<std::uint16_t>());
            case 0xc6: return binary(integer<std::uint32_t>());
            case 0xca: return real32();
            case 0xcb: return real64();
            case 0xcc: return unsignedValue(integer<std::uint8_t>());
            case 0xcd: return unsignedValue(integer<std::uint16_t>());
            case 0xce: return unsignedValue(integer<std::uint32_t>());
            case 0xcf: return unsignedValue(integer<std::uint64_t>());
            case 0xd0: return signedValue(static_cast<std::int8_t>(integer<std::uint8_t>()));
            case 0xd1: return signedValue(static_cast<std::int16_t>(integer<std::uint16_t>()));
            case 0xd2: return signedValue(static_cast<std::int32_t>(integer<std::uint32_t>()));
            case 0xd3: return signedValue(static_cast<std::int64_t>(integer<std::uint64_t>()));
            case 0xd9: return string(integer<std::uint8_t>());
            case 0xda: return string(integer<std::uint16_t>());
            case 0xdb: return string(integer<std::uint32_t>());
            case 0xdc: return array(integer<std::uint16_t>());
            case 0xdd: return array(integer<std::uint32_t>());
            case 0xde: return map(integer<std::uint16_t>());
            case 0xdf: return map(integer<std::uint32_t>());
            default: throw std::runtime_error("unsupported MessagePack marker");
        }
    }

    bool complete() const { return offset_ == bytes_.size(); }

private:
    unsigned char byte() {
        if (offset_ >= bytes_.size()) throw std::runtime_error("truncated MessagePack");
        return bytes_[offset_++];
    }

    template <typename T>
    T integer() {
        T value = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index)
            value = static_cast<T>((value << 8) | byte());
        return value;
    }

    Value boolValue(bool value) {
        Value out; out.type = Value::BOOL; out.boolean = value; return out;
    }
    Value unsignedValue(std::uint64_t value) {
        Value out; out.type = Value::UINT; out.uint_value = value; return out;
    }
    Value signedValue(std::int64_t value) {
        Value out; out.type = Value::SINT; out.sint_value = value; return out;
    }
    Value real32() {
        const std::uint32_t bits = integer<std::uint32_t>();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        Value out; out.type = Value::REAL; out.real_value = value; return out;
    }
    Value real64() {
        const std::uint64_t bits = integer<std::uint64_t>();
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        Value out; out.type = Value::REAL; out.real_value = value; return out;
    }
    Value string(std::size_t size) {
        if (size > bytes_.size() - offset_) throw std::runtime_error("truncated string");
        Value out; out.type = Value::STRING;
        out.string_value.assign(reinterpret_cast<const char*>(&bytes_[offset_]), size);
        offset_ += size;
        return out;
    }
    Value binary(std::size_t size) {
        if (size > bytes_.size() - offset_) throw std::runtime_error("truncated binary");
        Value out; out.type = Value::BINARY;
        out.binary_value.assign(bytes_.begin() + offset_, bytes_.begin() + offset_ + size);
        offset_ += size;
        return out;
    }
    Value array(std::size_t size) {
        Value out; out.type = Value::ARRAY; out.array_value.reserve(size);
        for (std::size_t index = 0; index < size; ++index) out.array_value.push_back(read());
        return out;
    }
    Value map(std::size_t size) {
        Value out; out.type = Value::MAP;
        for (std::size_t index = 0; index < size; ++index) {
            Value key = read();
            if (key.type != Value::STRING) throw std::runtime_error("map key is not string");
            if (!out.map_value.insert(std::make_pair(key.string_value, read())).second)
                throw std::runtime_error("duplicate map key");
        }
        return out;
    }

    const std::vector<unsigned char>& bytes_;
    std::size_t offset_;
};

const Value& field(const Value& value, const std::string& name, Value::Type type) {
    if (value.type != Value::MAP) throw std::runtime_error("expected map");
    std::map<std::string, Value>::const_iterator found = value.map_value.find(name);
    if (found == value.map_value.end() || found->second.type != type)
        throw std::runtime_error("missing or wrongly typed field: " + name);
    return found->second;
}

std::uint64_t number(const Value& value, const std::string& name) {
    const Value& item = value.map_value.at(name);
    if (item.type == Value::UINT) return item.uint_value;
    if (item.type == Value::SINT && item.sint_value >= 0)
        return static_cast<std::uint64_t>(item.sint_value);
    throw std::runtime_error("field is not nonnegative integer: " + name);
}

double real(const Value& value, const std::string& name) {
    const Value& item = value.map_value.at(name);
    if (item.type == Value::REAL) return item.real_value;
    if (item.type == Value::UINT) return static_cast<double>(item.uint_value);
    if (item.type == Value::SINT) return static_cast<double>(item.sint_value);
    throw std::runtime_error("field is not numeric: " + name);
}

void exactKeys(const Value& value, const std::set<std::string>& keys) {
    if (value.type != Value::MAP || value.map_value.size() != keys.size())
        throw ProtocolError("WRONG_FIELDS");
    for (std::set<std::string>::const_iterator key = keys.begin(); key != keys.end(); ++key)
        if (value.map_value.count(*key) != 1) throw ProtocolError("WRONG_FIELDS");
}

bool isSha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::find_if(value.begin(), value.end(), [](char c) {
        return !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }) == value.end();
}

class Writer {
public:
    void map(std::size_t size) {
        if (size <= 15) bytes_.push_back(static_cast<unsigned char>(0x80 | size));
        else throw std::runtime_error("request map too large");
    }
    void string(const std::string& value) {
        if (value.size() <= 31) bytes_.push_back(static_cast<unsigned char>(0xa0 | value.size()));
        else if (value.size() <= 255) {
            bytes_.push_back(0xd9);
            bytes_.push_back(static_cast<unsigned char>(value.size()));
        } else {
            throw std::runtime_error("request string too large");
        }
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    void integer(std::uint64_t value) {
        if (value <= 0x7f) bytes_.push_back(static_cast<unsigned char>(value));
        else if (value <= 0xff) {
            bytes_.push_back(0xcc); append(value, 1);
        } else if (value <= 0xffff) {
            bytes_.push_back(0xcd); append(value, 2);
        } else if (value <= 0xffffffffULL) {
            bytes_.push_back(0xce); append(value, 4);
        } else {
            bytes_.push_back(0xcf); append(value, 8);
        }
    }
    void binary(const std::vector<unsigned char>& value) {
        if (value.size() <= 0xff) {
            bytes_.push_back(0xc4); append(value.size(), 1);
        } else if (value.size() <= 0xffff) {
            bytes_.push_back(0xc5); append(value.size(), 2);
        } else {
            bytes_.push_back(0xc6); append(value.size(), 4);
        }
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    const std::vector<unsigned char>& bytes() const { return bytes_; }
private:
    void append(std::uint64_t value, std::size_t size) {
        for (std::size_t index = size; index > 0; --index)
            bytes_.push_back(static_cast<unsigned char>((value >> ((index - 1) * 8)) & 0xff));
    }
    std::vector<unsigned char> bytes_;
};

std::vector<unsigned char> encodeFrameRequest(
    const IpcProviderConfig& config, std::uint64_t frame_id,
    std::uint64_t source_timestamp_ns, const std::vector<unsigned char>& jpeg,
    int width, int height) {
    Writer writer;
    writer.map(9);
    writer.string("protocol_version"); writer.integer(1);
    writer.string("kind"); writer.string("frame_request");
    writer.string("run_id"); writer.string(config.run_id);
    writer.string("frame_id"); writer.integer(frame_id);
    writer.string("source_timestamp_ns"); writer.integer(source_timestamp_ns);
    writer.string("prompt_sha256"); writer.string(config.prompt_sha256);
    writer.string("image_width"); writer.integer(static_cast<std::uint64_t>(width));
    writer.string("image_height"); writer.integer(static_cast<std::uint64_t>(height));
    writer.string("jpeg_bytes"); writer.binary(jpeg);
    return writer.bytes();
}

class ZmqIpcTransport : public IpcTransport {
public:
    ZmqIpcTransport(const std::string& request_endpoint,
                    const std::string& result_endpoint)
        : context_(NULL), request_socket_(NULL), result_socket_(NULL) {
        context_ = zmq_ctx_new();
        if (!context_) throw std::runtime_error("zmq_ctx_new failed");
        request_socket_ = zmq_socket(context_, ZMQ_PUB);
        result_socket_ = zmq_socket(context_, ZMQ_SUB);
        if (!request_socket_ || !result_socket_) {
            cleanup();
            throw std::runtime_error("zmq_socket failed");
        }
        const int zero = 0;
        const int one = 1;
        if (zmq_setsockopt(request_socket_, ZMQ_LINGER, &zero, sizeof(zero)) != 0 ||
            zmq_setsockopt(request_socket_, ZMQ_SNDHWM, &one, sizeof(one)) != 0 ||
            zmq_setsockopt(request_socket_, ZMQ_CONFLATE, &one, sizeof(one)) != 0 ||
            zmq_setsockopt(result_socket_, ZMQ_LINGER, &zero, sizeof(zero)) != 0 ||
            zmq_setsockopt(result_socket_, ZMQ_RCVHWM, &one, sizeof(one)) != 0 ||
            zmq_setsockopt(result_socket_, ZMQ_CONFLATE, &one, sizeof(one)) != 0 ||
            zmq_setsockopt(result_socket_, ZMQ_SUBSCRIBE, "", 0) != 0 ||
            zmq_bind(request_socket_, request_endpoint.c_str()) != 0 ||
            zmq_connect(result_socket_, result_endpoint.c_str()) != 0) {
            const std::string detail = zmq_strerror(zmq_errno());
            cleanup();
            throw std::runtime_error("ZeroMQ transport setup failed: " + detail);
        }
    }

    ~ZmqIpcTransport() { cleanup(); }

    bool trySend(const std::vector<unsigned char>& payload) {
        const int result = zmq_send(request_socket_, payload.data(), payload.size(), ZMQ_DONTWAIT);
        if (result >= 0) return true;
        if (zmq_errno() == EAGAIN) return false;
        throw std::runtime_error(std::string("ZeroMQ send failed: ") + zmq_strerror(zmq_errno()));
    }

    bool tryReceive(std::vector<unsigned char>* payload) {
        zmq_msg_t message;
        if (zmq_msg_init(&message) != 0)
            throw std::runtime_error("ZeroMQ message initialization failed");
        const int result = zmq_msg_recv(&message, result_socket_, ZMQ_DONTWAIT);
        if (result < 0) {
            const int error = zmq_errno();
            zmq_msg_close(&message);
            if (error == EAGAIN) return false;
            throw std::runtime_error(std::string("ZeroMQ receive failed: ") + zmq_strerror(error));
        }
        const unsigned char* begin = static_cast<const unsigned char*>(zmq_msg_data(&message));
        payload->assign(begin, begin + zmq_msg_size(&message));
        zmq_msg_close(&message);
        return true;
    }

private:
    void cleanup() {
        if (request_socket_) { zmq_close(request_socket_); request_socket_ = NULL; }
        if (result_socket_) { zmq_close(result_socket_); result_socket_ = NULL; }
        if (context_) { zmq_ctx_term(context_); context_ = NULL; }
    }
    void* context_;
    void* request_socket_;
    void* result_socket_;
};

OnlineInstance decodeInstance(const Value& value, int width, int height,
                              std::size_t expected_id) {
    exactKeys(value, {"local_id", "label", "score", "box_xyxy", "mask_rle"});
    OnlineInstance instance;
    const std::uint64_t local_id = number(value, "local_id");
    if (local_id > std::numeric_limits<std::uint32_t>::max() ||
        local_id != expected_id)
        throw ProtocolError("INVALID_INSTANCE");
    instance.local_id = static_cast<std::uint32_t>(local_id);
    instance.label = field(value, "label", Value::STRING).string_value;
    instance.score = real(value, "score");
    if (instance.label.empty() || !std::isfinite(instance.score) ||
        instance.score < 0.0 || instance.score > 1.0)
        throw ProtocolError("INVALID_INSTANCE");
    const Value& box = field(value, "box_xyxy", Value::ARRAY);
    if (box.array_value.size() != 4) throw ProtocolError("INVALID_INSTANCE");
    for (std::size_t index = 0; index < 4; ++index) {
        const Value& item = box.array_value[index];
        if (item.type == Value::REAL) instance.box_xyxy[index] = item.real_value;
        else if (item.type == Value::UINT) instance.box_xyxy[index] = item.uint_value;
        else if (item.type == Value::SINT) instance.box_xyxy[index] = item.sint_value;
        else throw ProtocolError("INVALID_INSTANCE");
        if (!std::isfinite(instance.box_xyxy[index])) throw ProtocolError("INVALID_INSTANCE");
    }
    if (instance.box_xyxy[2] <= instance.box_xyxy[0] ||
        instance.box_xyxy[3] <= instance.box_xyxy[1])
        throw ProtocolError("INVALID_INSTANCE");
    const Value& rle = field(value, "mask_rle", Value::MAP);
    exactKeys(rle, {"size", "counts"});
    const Value& size = field(rle, "size", Value::ARRAY);
    const Value& counts = field(rle, "counts", Value::ARRAY);
    if (size.array_value.size() != 2 || counts.array_value.empty())
        throw ProtocolError("MALFORMED_RLE");
    if (size.array_value[0].type != Value::UINT ||
        size.array_value[1].type != Value::UINT ||
        size.array_value[0].uint_value > static_cast<std::uint64_t>(
            std::numeric_limits<int>::max()) ||
        size.array_value[1].uint_value > static_cast<std::uint64_t>(
            std::numeric_limits<int>::max()))
        throw ProtocolError("MALFORMED_RLE");
    instance.mask_height = static_cast<int>(size.array_value[0].uint_value);
    instance.mask_width = static_cast<int>(size.array_value[1].uint_value);
    if (instance.mask_width != width || instance.mask_height != height)
        throw ProtocolError("MALFORMED_RLE");
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < counts.array_value.size(); ++index) {
        if (counts.array_value[index].type != Value::UINT ||
            counts.array_value[index].uint_value > std::numeric_limits<std::uint32_t>::max())
            throw ProtocolError("MALFORMED_RLE");
        const std::uint32_t count = static_cast<std::uint32_t>(counts.array_value[index].uint_value);
        instance.mask_counts.push_back(count);
        if (total > std::numeric_limits<std::uint64_t>::max() - count)
            throw ProtocolError("MALFORMED_RLE");
        total += count;
    }
    if (total != static_cast<std::uint64_t>(width) * height)
        throw ProtocolError("MALFORMED_RLE");
    return instance;
}

}  // namespace

ProtocolError::ProtocolError(const std::string& code, const std::string& detail)
    : std::runtime_error(detail.empty() ? code : code + ": " + detail), code_(code) {}

PacketExpectations::PacketExpectations()
    : image_width(0), image_height(0), current_timestamp_ns(0), max_age_ns(0) {}

std::uint64_t SystemIpcClock::monotonicNanoseconds() const {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::unique_ptr<IpcTransport> makeZmqIpcTransport(
    const std::string& request_endpoint, const std::string& result_endpoint) {
    return std::unique_ptr<IpcTransport>(
        new ZmqIpcTransport(request_endpoint, result_endpoint));
}

IpcProviderConfig::IpcProviderConfig()
    : request_rate_cap_hz(5.0), max_age_ns(250000000ULL) {}

IpcPollResult::IpcPollResult()
    : state(SemanticState::DEGRADED_TO_BASELINE), request_attempted(false),
      request_sent(false), has_packet(false), call_duration_ms(0.0) {}

static SemanticPacket decodeSemanticPacketUnchecked(
    const std::vector<unsigned char>& payload,
    const PacketExpectations& expected) {
    Value root;
    try {
        Reader reader(payload);
        root = reader.read();
        if (!reader.complete()) throw std::runtime_error("trailing MessagePack bytes");
    } catch (const std::exception& error) {
        throw ProtocolError("CORRUPT_MESSAGEPACK", error.what());
    }
    exactKeys(root, {
        "protocol_version", "kind", "run_id", "frame_id", "source_timestamp_ns",
        "produced_timestamp_ns", "prompt_sha256", "model_manifest_sha256",
        "image_width", "image_height", "inference_ms", "instances"
    });
    if (number(root, "protocol_version") != 1) throw ProtocolError("WRONG_PROTOCOL_VERSION");
    if (field(root, "kind", Value::STRING).string_value != "semantic_packet")
        throw ProtocolError("WRONG_KIND");
    SemanticPacket packet;
    packet.run_id = field(root, "run_id", Value::STRING).string_value;
    if (packet.run_id != expected.run_id) throw ProtocolError("WRONG_RUN_ID");
    packet.prompt_sha256 = field(root, "prompt_sha256", Value::STRING).string_value;
    if (packet.prompt_sha256 != expected.prompt_sha256) throw ProtocolError("WRONG_PROMPT");
    packet.model_manifest_sha256 =
        field(root, "model_manifest_sha256", Value::STRING).string_value;
    if (packet.model_manifest_sha256 != expected.model_manifest_sha256)
        throw ProtocolError("WRONG_MODEL_MANIFEST");
    if (!isSha256(packet.prompt_sha256) || !isSha256(packet.model_manifest_sha256))
        throw ProtocolError("INVALID_IDENTITY");
    const std::uint64_t image_width = number(root, "image_width");
    const std::uint64_t image_height = number(root, "image_height");
    if (image_width == 0 || image_height == 0 ||
        image_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        image_height > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw ProtocolError("INVALID_PACKET");
    packet.image_width = static_cast<int>(image_width);
    packet.image_height = static_cast<int>(image_height);
    if (packet.image_width != expected.image_width || packet.image_height != expected.image_height)
        throw ProtocolError("WRONG_DIMENSIONS");
    packet.frame_id = number(root, "frame_id");
    packet.source_timestamp_ns = number(root, "source_timestamp_ns");
    packet.produced_timestamp_ns = number(root, "produced_timestamp_ns");
    if (packet.produced_timestamp_ns < packet.source_timestamp_ns)
        throw ProtocolError("INVALID_TIMESTAMP");
    if (packet.source_timestamp_ns > expected.current_timestamp_ns)
        throw ProtocolError("FUTURE_PACKET");
    packet.age_ns = expected.current_timestamp_ns - packet.source_timestamp_ns;
    if (packet.age_ns > expected.max_age_ns) throw ProtocolError("STALE_PACKET");
    packet.inference_ms = real(root, "inference_ms");
    if (!std::isfinite(packet.inference_ms) || packet.inference_ms < 0.0)
        throw ProtocolError("INVALID_INFERENCE_TIME");
    const Value& instances = field(root, "instances", Value::ARRAY);
    for (std::size_t index = 0; index < instances.array_value.size(); ++index)
        packet.instances.push_back(decodeInstance(
            instances.array_value[index], packet.image_width, packet.image_height, index));
    return packet;
}

SemanticPacket decodeSemanticPacket(const std::vector<unsigned char>& payload,
                                    const PacketExpectations& expected) {
    try {
        return decodeSemanticPacketUnchecked(payload, expected);
    } catch (const ProtocolError&) {
        throw;
    } catch (const std::exception& error) {
        throw ProtocolError("INVALID_PACKET", error.what());
    }
}

IpcMaskProvider::IpcMaskProvider(const IpcProviderConfig& config,
                                 std::unique_ptr<IpcTransport> transport,
                                 const IpcClock& clock)
    : config_(config), transport_(std::move(transport)), clock_(clock),
      has_request_attempt_(false), last_request_attempt_ns_(0),
      has_latest_payload_(false) {
    if (!transport_) throw std::invalid_argument("IPC transport must not be null");
    if (config_.run_id.empty() || !isSha256(config_.prompt_sha256) ||
        !isSha256(config_.model_manifest_sha256))
        throw std::invalid_argument("IPC provider identity is invalid");
    if (!std::isfinite(config_.request_rate_cap_hz) ||
        config_.request_rate_cap_hz <= 0.0 || config_.request_rate_cap_hz > 5.0)
        throw std::invalid_argument("IPC request rate cap must be in (0, 5]");
    if (config_.max_age_ns == 0)
        throw std::invalid_argument("IPC max age must be positive");
}

IpcPollResult IpcMaskProvider::poll(
    std::uint64_t frame_id, std::uint64_t source_timestamp_ns,
    const std::vector<unsigned char>& jpeg_bytes, int image_width, int image_height) {
    const std::uint64_t started_ns = clock_.monotonicNanoseconds();
    IpcPollResult result;
    if (image_width <= 0 || image_height <= 0 || jpeg_bytes.empty()) {
        result.reason = "INVALID_FRAME_REQUEST";
        return result;
    }
    const std::uint64_t interval_ns = static_cast<std::uint64_t>(
        1000000000.0 / config_.request_rate_cap_hz);
    const bool due = !has_request_attempt_ ||
        (started_ns >= last_request_attempt_ns_ &&
         started_ns - last_request_attempt_ns_ >= interval_ns);
    if (due) {
        result.request_attempted = true;
        try {
            result.request_sent = transport_->trySend(encodeFrameRequest(
                config_, frame_id, source_timestamp_ns, jpeg_bytes,
                image_width, image_height));
        } catch (const std::exception&) {
            last_receive_error_ = "TRANSPORT_SEND_ERROR";
            has_latest_payload_ = false;
        }
        has_request_attempt_ = true;
        last_request_attempt_ns_ = started_ns;
        if (result.request_sent) {
            RequestRecord record;
            record.frame_id = frame_id;
            record.source_timestamp_ns = source_timestamp_ns;
            record.image_width = image_width;
            record.image_height = image_height;
            request_ledger_.push_back(record);
            while (request_ledger_.size() > 64) request_ledger_.pop_front();
        }
    }

    PacketExpectations expected;
    expected.run_id = config_.run_id;
    expected.prompt_sha256 = config_.prompt_sha256;
    expected.model_manifest_sha256 = config_.model_manifest_sha256;
    expected.image_width = image_width;
    expected.image_height = image_height;
    expected.current_timestamp_ns = source_timestamp_ns;
    expected.max_age_ns = config_.max_age_ns;
    std::vector<unsigned char> received;
    bool receive_available = true;
    while (receive_available) {
        try {
            receive_available = transport_->tryReceive(&received);
        } catch (const std::exception&) {
            last_receive_error_ = "TRANSPORT_RECEIVE_ERROR";
            has_latest_payload_ = false;
            break;
        }
        if (!receive_available) break;
        try {
            const SemanticPacket packet = decodeSemanticPacket(received, expected);
            std::deque<RequestRecord>::const_iterator request = request_ledger_.end();
            for (std::deque<RequestRecord>::const_iterator item = request_ledger_.begin();
                 item != request_ledger_.end(); ++item) {
                if (item->frame_id == packet.frame_id) {
                    request = item;
                    break;
                }
            }
            if (request == request_ledger_.end())
                throw ProtocolError("UNREQUESTED_FRAME");
            if (request->source_timestamp_ns != packet.source_timestamp_ns)
                throw ProtocolError("SOURCE_TIMESTAMP_MISMATCH");
            if (request->image_width != packet.image_width ||
                request->image_height != packet.image_height)
                throw ProtocolError("REQUEST_DIMENSIONS_MISMATCH");
            latest_payload_ = received;
            has_latest_payload_ = true;
            last_receive_error_.clear();
        } catch (const ProtocolError& error) {
            last_receive_error_ = error.code();
            has_latest_payload_ = false;
            latest_payload_.clear();
        }
    }
    if (has_latest_payload_) {
        try {
            result.packet = decodeSemanticPacket(latest_payload_, expected);
            result.has_packet = true;
            result.state = SemanticState::ONLINE_VALID;
            result.reason = "ONLINE_PACKET_VALID";
        } catch (const ProtocolError& error) {
            result.reason = error.code();
        }
    } else {
        result.reason = last_receive_error_.empty() ? "NO_PACKET" : last_receive_error_;
    }
    const std::uint64_t ended_ns = clock_.monotonicNanoseconds();
    if (ended_ns >= started_ns)
        result.call_duration_ms = (ended_ns - started_ns) / 1000000.0;
    return result;
}

}  // namespace semantic
}  // namespace ORB_SLAM2
