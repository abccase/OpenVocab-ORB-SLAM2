#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

#include "semantic/MaskProvider.h"
#include "semantic/FeatureMaskPolicy.h"

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        char pattern[] = "/tmp/ovorb-cache-test-XXXXXX";
        char* created = mkdtemp(pattern);
        if (created) path_ = created;
    }
    ~TemporaryDirectory() {
        if (!path_.empty()) {
            std::remove((path_ + "/score.npy").c_str());
            std::remove((path_ + "/cache_index.jsonl").c_str());
            std::remove((path_ + "/cache_manifest.json").c_str());
            std::remove((path_ + "/cache_complete.json").c_str());
            rmdir(path_.c_str());
        }
    }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

void Write(const std::string& path, const std::string& value) {
    std::ofstream stream(path.c_str(), std::ios::binary);
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string Npy2x3(const std::string& descr = "<f4", bool fortran = false,
                   bool version_two = false,
                   const std::vector<float>& values =
                       std::vector<float>{0.0f, 0.25f, 0.5f, 0.7f, 0.9f, 1.0f}) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': " +
        (fortran ? "True" : "False") + ", 'shape': (2, 3), }";
    const std::size_t preamble = version_two ? 12 : 10;
    const std::size_t padded = ((preamble + header.size() + 1 + 63) / 64) * 64 - preamble;
    header.resize(padded - 1, ' ');
    header.push_back('\n');
    std::string bytes(version_two ? "\x93NUMPY\x02\x00" : "\x93NUMPY\x01\x00", 8);
    const unsigned int size = static_cast<unsigned int>(header.size());
    bytes.push_back(static_cast<char>(size & 0xff));
    bytes.push_back(static_cast<char>((size >> 8) & 0xff));
    if (version_two) {
        bytes.push_back(static_cast<char>((size >> 16) & 0xff));
        bytes.push_back(static_cast<char>((size >> 24) & 0xff));
    }
    bytes += header;
    bytes.append(reinterpret_cast<const char*>(&values[0]),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return bytes;
}

struct CacheFixture {
    TemporaryDirectory directory;
    std::string manifest_sha;
    std::string index_sha;
    std::string completion_sha;
    CacheFixture() {
        const std::string manifest =
            "{\"expected_frame_count\":1,\"schema\":\"ovorb.dynamic-cache.v1\","
            "\"sequence_id\":\"tiny\"}\n";
        manifest_sha = ORB_SLAM2::semantic::sha256Bytes(manifest);
        Write(directory.path() + "/cache_manifest.json", manifest);
        rewriteScore(Npy2x3());
    }
    void rewriteScore(const std::string& npy,
                      const std::string& relative_path = "score.npy",
                      const std::string& timestamp_lexeme = "1.25",
                      int frame_id = 0,
                      const std::string& semantic_sha = std::string(64, 'a')) {
        if (relative_path == "score.npy")
            Write(directory.path() + "/score.npy", npy);
        const std::string score_sha = ORB_SLAM2::semantic::sha256Bytes(npy);
        const std::string index =
              "{\"dtype\":\"float32\",\"frame_id\":" + std::to_string(frame_id) + ",\"height\":2,"
              "\"path\":\"" + relative_path + "\",\"semantic_packet_sha256\":\"" + semantic_sha +
              "\",\"sha256\":\"" + score_sha +
              "\",\"timestamp\":" + timestamp_lexeme + ",\"width\":3}\n";
        Write(directory.path() + "/cache_index.jsonl", index);
        index_sha = ORB_SLAM2::semantic::sha256Bytes(index);
        const std::string completion =
            "{\"frame_count\":1,\"index_sha256\":\"" + index_sha +
            "\",\"manifest_sha256\":\"" + manifest_sha + "\"}\n";
        Write(directory.path() + "/cache_complete.json", completion);
        completion_sha = ORB_SLAM2::semantic::sha256Bytes(completion);
    }
};

ORB_SLAM2::semantic::CacheMaskProvider MakeProvider(const CacheFixture& fixture) {
    return ORB_SLAM2::semantic::CacheMaskProvider(
        fixture.directory.path(), "tiny", fixture.manifest_sha,
        fixture.completion_sha, fixture.index_sha);
}

TEST(CacheMaskProvider, Sha256ImplementationMatchesGoldenVector) {
    EXPECT_EQ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              ORB_SLAM2::semantic::sha256Bytes("abc"));
}

TEST(CacheMaskProvider, LoadsExactTrustedTimestampAndValidatedFloatMap) {
    CacheFixture fixture;
    ORB_SLAM2::semantic::CacheMaskProvider provider = MakeProvider(fixture);
    const ORB_SLAM2::semantic::DynamicScoreMap score_map = provider.load(1.25, 3, 2);
    EXPECT_EQ(CV_32FC1, score_map.scores_f32.type());
    EXPECT_FLOAT_EQ(0.7f, score_map.scores_f32.at<float>(1, 0));
    EXPECT_EQ(fixture.manifest_sha, score_map.manifest_sha256);
}

TEST(CacheMaskProvider, RejectsUntrustedManifestAndTimestampOrDimensionMismatch) {
    CacheFixture fixture;
    EXPECT_THROW(ORB_SLAM2::semantic::CacheMaskProvider(
                     fixture.directory.path(), "tiny", std::string(64, '0'),
                     fixture.completion_sha, fixture.index_sha),
                 ORB_SLAM2::semantic::CacheValidationError);
    ORB_SLAM2::semantic::CacheMaskProvider provider = MakeProvider(fixture);
    EXPECT_THROW(provider.load(1.250001, 3, 2), ORB_SLAM2::semantic::CacheValidationError);
    EXPECT_THROW(provider.load(1.25, 4, 2), ORB_SLAM2::semantic::CacheValidationError);
}

TEST(CacheMaskProvider, RejectsChangedScoreMapBeforeReturningIt) {
    CacheFixture fixture;
    ORB_SLAM2::semantic::CacheMaskProvider provider = MakeProvider(fixture);
    Write(fixture.directory.path() + "/score.npy", "corrupt");
    EXPECT_THROW(provider.load(1.25, 3, 2), ORB_SLAM2::semantic::CacheValidationError);
}

TEST(CacheMaskProvider, PreservesExactTimestampLexemeForFrameKey) {
    CacheFixture fixture;
    fixture.rewriteScore(Npy2x3(), "score.npy", "1.250000");
    ORB_SLAM2::semantic::CacheMaskProvider provider = MakeProvider(fixture);
    EXPECT_EQ(ORB_SLAM2::semantic::makeFrameKey("tiny", "1.250000"),
              provider.load(1.25, 3, 2).frame_key);
}

TEST(CacheMaskProvider, AcceptsNpyV2AndRejectsUnsupportedOrUnsafePayloads) {
    CacheFixture fixture;
    fixture.rewriteScore(Npy2x3("<f4", false, true));
    EXPECT_NO_THROW(MakeProvider(fixture).load(1.25, 3, 2));

    fixture.rewriteScore(Npy2x3("<f4", true));
    EXPECT_THROW(MakeProvider(fixture).load(1.25, 3, 2),
                 ORB_SLAM2::semantic::CacheValidationError);
    fixture.rewriteScore(Npy2x3(">f4"));
    EXPECT_THROW(MakeProvider(fixture).load(1.25, 3, 2),
                 ORB_SLAM2::semantic::CacheValidationError);
    std::string truncated = Npy2x3();
    truncated.resize(truncated.size() - 1);
    fixture.rewriteScore(truncated);
    EXPECT_THROW(MakeProvider(fixture).load(1.25, 3, 2),
                 ORB_SLAM2::semantic::CacheValidationError);
    fixture.rewriteScore(Npy2x3("<f4", false, false,
        std::vector<float>{0.0f, 0.25f, 0.5f, 0.7f, 0.9f, 1.1f}));
    EXPECT_THROW(MakeProvider(fixture).load(1.25, 3, 2),
                 ORB_SLAM2::semantic::CacheValidationError);
    fixture.rewriteScore(Npy2x3("<f4", false, false,
        std::vector<float>{0.0f, 0.25f, 0.5f, 0.7f, 0.9f,
                           std::numeric_limits<float>::quiet_NaN()}));
    EXPECT_THROW(MakeProvider(fixture).load(1.25, 3, 2),
                 ORB_SLAM2::semantic::CacheValidationError);
}

TEST(CacheMaskProvider, RejectsTraversalBadPacketIdentityAndIndexDiscontinuity) {
    CacheFixture fixture;
    fixture.rewriteScore(Npy2x3(), "../score.npy");
    EXPECT_THROW(MakeProvider(fixture), ORB_SLAM2::semantic::CacheValidationError);
    fixture.rewriteScore(Npy2x3(), "score.npy", "1.25", 0, "not-a-sha");
    EXPECT_THROW(MakeProvider(fixture), ORB_SLAM2::semantic::CacheValidationError);
    fixture.rewriteScore(Npy2x3(), "score.npy", "1.25", 1);
    EXPECT_THROW(MakeProvider(fixture), ORB_SLAM2::semantic::CacheValidationError);
}

}  // namespace
