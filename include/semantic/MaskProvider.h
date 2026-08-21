#ifndef ORB_SLAM2_SEMANTIC_MASK_PROVIDER_H
#define ORB_SLAM2_SEMANTIC_MASK_PROVIDER_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "semantic/DynamicScoreMap.h"

namespace ORB_SLAM2 {
namespace semantic {

class CacheValidationError : public std::runtime_error {
public:
    explicit CacheValidationError(const std::string& message)
        : std::runtime_error(message) {}
};

std::string sha256Bytes(const std::string& bytes);
std::string sha256File(const std::string& path);

class MaskProvider {
public:
    virtual ~MaskProvider() {}
    virtual DynamicScoreMap load(double source_timestamp,
                                 int image_width, int image_height) const = 0;
};

class CacheMaskProvider : public MaskProvider {
public:
    CacheMaskProvider(const std::string& cache_root,
                      const std::string& expected_sequence_id,
                      const std::string& expected_manifest_sha256,
                      const std::string& expected_completion_sha256,
                      const std::string& expected_index_sha256);

    DynamicScoreMap load(double source_timestamp,
                         int image_width, int image_height) const override;

    const std::string& sequenceId() const { return sequence_id_; }
    const std::string& manifestSha256() const { return manifest_sha256_; }
    const std::string& completionSha256() const { return completion_sha256_; }
    const std::string& indexSha256() const { return index_sha256_; }
    std::size_t frameCount() const { return entries_.size(); }

private:
    struct Entry {
        int frame_id;
        int width;
        int height;
        double timestamp;
        std::string timestamp_lexeme;
        std::string relative_path;
        std::string score_sha256;
        std::string semantic_packet_sha256;
    };

    std::string cache_root_;
    std::string sequence_id_;
    std::string manifest_sha256_;
    std::string completion_sha256_;
    std::string index_sha256_;
    std::vector<Entry> entries_;
};

}  // namespace semantic
}  // namespace ORB_SLAM2

#endif
