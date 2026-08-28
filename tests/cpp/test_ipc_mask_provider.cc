#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <memory>
#include <deque>
#include <string>
#include <vector>

#include <unistd.h>

#include "semantic/IpcMaskProvider.h"

namespace {

std::vector<unsigned char> Fixture(const std::string& name) {
    std::ifstream stream(std::string(IPC_FIXTURE_DIR) + "/" + name,
                         std::ios::binary);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream),
                                      std::istreambuf_iterator<char>());
}

ORB_SLAM2::semantic::PacketExpectations Expectations() {
    ORB_SLAM2::semantic::PacketExpectations expected;
    expected.run_id = "p06-fixture-run";
    expected.prompt_sha256 = std::string(64, '1');
    expected.model_manifest_sha256 = std::string(64, '2');
    expected.image_width = 3;
    expected.image_height = 2;
    expected.current_timestamp_ns = 1200000000ULL;
    expected.max_age_ns = 250000000ULL;
    return expected;
}

class FakeClock : public ORB_SLAM2::semantic::IpcClock {
public:
    FakeClock() : now_ns(0) {}
    std::uint64_t monotonicNanoseconds() const { return now_ns; }
    std::uint64_t now_ns;
};

class FakeTransport : public ORB_SLAM2::semantic::IpcTransport {
public:
    bool trySend(const std::vector<unsigned char>& payload) {
        if (throw_on_send) throw std::runtime_error("send failed");
        sent.push_back(payload);
        return send_succeeds;
    }
    bool tryReceive(std::vector<unsigned char>* payload) {
        if (throw_on_receive) throw std::runtime_error("receive failed");
        if (received.empty()) return false;
        *payload = received.front();
        received.pop_front();
        return true;
    }
    bool send_succeeds = true;
    bool throw_on_send = false;
    bool throw_on_receive = false;
    std::vector<std::vector<unsigned char> > sent;
    std::deque<std::vector<unsigned char> > received;
};

ORB_SLAM2::semantic::IpcProviderConfig ProviderConfig() {
    ORB_SLAM2::semantic::IpcProviderConfig config;
    config.run_id = "p06-fixture-run";
    config.prompt_sha256 = std::string(64, '1');
    config.model_manifest_sha256 = std::string(64, '2');
    config.request_rate_cap_hz = 5.0;
    config.max_age_ns = 250000000ULL;
    return config;
}

struct ProviderFixture {
    ProviderFixture() : transport(new FakeTransport()), raw_transport(transport.get()),
        provider(ProviderConfig(), std::move(transport), clock) {}
    FakeClock clock;
    std::unique_ptr<FakeTransport> transport;
    FakeTransport* raw_transport;
    ORB_SLAM2::semantic::IpcMaskProvider provider;
};

TEST(IpcWireFixtures, ValidPacketDecodesSameIdentityAndRleAsPython) {
    const ORB_SLAM2::semantic::SemanticPacket packet =
        ORB_SLAM2::semantic::decodeSemanticPacket(
            Fixture("valid_semantic_packet.msgpack"), Expectations());

    ASSERT_EQ(7U, packet.frame_id);
    EXPECT_EQ(200000000ULL, packet.age_ns);
    ASSERT_EQ(1U, packet.instances.size());
    EXPECT_EQ("person", packet.instances[0].label);
    EXPECT_EQ((std::vector<std::uint32_t>{1, 2, 3}),
              packet.instances[0].mask_counts);
}

TEST(IpcWireFixtures, SharedInvalidPacketsFailWithStableReason) {
    const std::vector<std::pair<std::string, std::string> > cases = {
        {"wrong_version.msgpack", "WRONG_PROTOCOL_VERSION"},
        {"wrong_run_id.msgpack", "WRONG_RUN_ID"},
        {"stale_timestamp.msgpack", "STALE_PACKET"},
        {"malformed_rle.msgpack", "MALFORMED_RLE"},
        {"wrong_dimensions.msgpack", "WRONG_DIMENSIONS"},
        {"wrong_field_type.msgpack", "INVALID_PACKET"},
        {"oversized_dimensions.msgpack", "INVALID_PACKET"},
        {"oversized_local_id.msgpack", "INVALID_INSTANCE"},
        {"oversized_rle_size.msgpack", "MALFORMED_RLE"},
    };
    for (std::size_t index = 0; index < cases.size(); ++index) {
        try {
            ORB_SLAM2::semantic::decodeSemanticPacket(
                Fixture(cases[index].first), Expectations());
            FAIL() << "fixture unexpectedly passed: " << cases[index].first;
        } catch (const ORB_SLAM2::semantic::ProtocolError& error) {
            EXPECT_EQ(cases[index].second, error.code());
        }
    }
}

TEST(IpcWireFixtures, TruncatedMessagepackIsRejectedAsCorrupt) {
    std::vector<unsigned char> payload = Fixture("valid_semantic_packet.msgpack");
    payload.resize(payload.size() - 3);
    try {
        ORB_SLAM2::semantic::decodeSemanticPacket(payload, Expectations());
        FAIL() << "truncated packet unexpectedly passed";
    } catch (const ORB_SLAM2::semantic::ProtocolError& error) {
        EXPECT_EQ("CORRUPT_MESSAGEPACK", error.code());
    }
}

TEST(IpcMaskProvider, NoServiceDegradesImmediatelyWithoutBlocking) {
    ProviderFixture fixture;

    const ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
        7, 1000000000ULL, std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);

    EXPECT_EQ(ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE, result.state);
    EXPECT_EQ("NO_PACKET", result.reason);
    EXPECT_TRUE(result.request_attempted);
    EXPECT_LT(result.call_duration_ms, 5.0);
}

TEST(IpcMaskProvider, ValidLatestPacketTransitionsOnlineAndStalePacketDegrades) {
    ProviderFixture fixture;
    fixture.provider.poll(
        7, 1000000000ULL, std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
    fixture.raw_transport->received.push_back(Fixture("valid_semantic_packet.msgpack"));
    fixture.clock.now_ns = 100000000ULL;

    ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
        8, 1200000000ULL, std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
    EXPECT_EQ(ORB_SLAM2::semantic::SemanticState::ONLINE_VALID, result.state);
    ASSERT_TRUE(result.has_packet);
    EXPECT_EQ(7U, result.packet.frame_id);

    fixture.clock.now_ns = 300000000ULL;
    result = fixture.provider.poll(
        8, 1300000000ULL, std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
    EXPECT_EQ(ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE, result.state);
    EXPECT_EQ("STALE_PACKET", result.reason);
}

TEST(IpcMaskProvider, InvalidLatestPacketDoesNotBecomeOnline) {
    ProviderFixture fixture;
    fixture.raw_transport->received.push_back(Fixture("wrong_run_id.msgpack"));

    const ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
        7, 1200000000ULL, std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);

    EXPECT_EQ(ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE, result.state);
    EXPECT_EQ("WRONG_RUN_ID", result.reason);
    EXPECT_FALSE(result.has_packet);
}

TEST(IpcMaskProvider, RejectsUnrequestedFrameAndMismatchedSourceTimestamp) {
    for (const std::string name : {
             "mismatched_frame_id.msgpack", "mismatched_source_timestamp.msgpack"}) {
        ProviderFixture fixture;
        fixture.raw_transport->received.push_back(Fixture(name));

        const ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
            7, 1200000000ULL,
            std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);

        EXPECT_EQ(ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE,
                  result.state) << name;
        EXPECT_EQ(name == "mismatched_frame_id.msgpack"
                      ? "UNREQUESTED_FRAME" : "SOURCE_TIMESTAMP_MISMATCH",
                  result.reason) << name;
        EXPECT_FALSE(result.has_packet) << name;
    }
}

TEST(IpcMaskProvider, ContainsDecoderAndTransportExceptionsAsDegradation) {
    {
        ProviderFixture fixture;
        fixture.raw_transport->received.push_back(Fixture("wrong_field_type.msgpack"));
        EXPECT_NO_THROW({
            const ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
                7, 1200000000ULL,
                std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
            EXPECT_EQ("INVALID_PACKET", result.reason);
        });
    }
    {
        ProviderFixture fixture;
        fixture.raw_transport->throw_on_send = true;
        EXPECT_NO_THROW({
            const ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
                7, 1200000000ULL,
                std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
            EXPECT_EQ("TRANSPORT_SEND_ERROR", result.reason);
        });
    }
    {
        ProviderFixture fixture;
        fixture.raw_transport->throw_on_receive = true;
        EXPECT_NO_THROW({
            const ORB_SLAM2::semantic::IpcPollResult result = fixture.provider.poll(
                7, 1200000000ULL,
                std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
            EXPECT_EQ("TRANSPORT_RECEIVE_ERROR", result.reason);
        });
    }
}

TEST(IpcMaskProvider, RequestAttemptsNeverExceedFiveHertz) {
    ProviderFixture fixture;
    const std::vector<unsigned char> jpeg{0xff, 0xd8, 0xff, 0xd9};

    fixture.provider.poll(1, 1000000000ULL, jpeg, 3, 2);
    fixture.clock.now_ns = 199999999ULL;
    fixture.provider.poll(2, 1100000000ULL, jpeg, 3, 2);
    fixture.clock.now_ns = 200000000ULL;
    fixture.provider.poll(3, 1200000000ULL, jpeg, 3, 2);

    ASSERT_EQ(2U, fixture.raw_transport->sent.size());
    EXPECT_FALSE(fixture.raw_transport->sent[0].empty());
    EXPECT_FALSE(fixture.raw_transport->sent[1].empty());
}

TEST(IpcMaskProvider, RealZmqTransportWithoutServiceReturnsImmediately) {
    const std::string suffix = std::to_string(static_cast<long long>(getpid()));
    const std::string request_endpoint = "inproc://ovorb-p06-request-" + suffix;
    const std::string result_endpoint = "inproc://ovorb-p06-result-" + suffix;
    ORB_SLAM2::semantic::SystemIpcClock clock;
    {
        ORB_SLAM2::semantic::IpcMaskProvider provider(
            ProviderConfig(),
            ORB_SLAM2::semantic::makeZmqIpcTransport(request_endpoint, result_endpoint),
            clock);
        const ORB_SLAM2::semantic::IpcPollResult result = provider.poll(
            7, 1000000000ULL, std::vector<unsigned char>{0xff, 0xd8, 0xff, 0xd9}, 3, 2);
        EXPECT_EQ(ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE, result.state);
        EXPECT_EQ("NO_PACKET", result.reason);
        EXPECT_LT(result.call_duration_ms, 5.0);
    }
}

}  // namespace
