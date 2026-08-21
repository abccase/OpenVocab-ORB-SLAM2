#include "semantic/MaskProvider.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>

#include <limits.h>
#include <openssl/evp.h>
#include <opencv2/core.hpp>

#include "semantic/FeatureMaskPolicy.h"

namespace ORB_SLAM2 {
namespace semantic {
namespace {

std::string ReadBytes(const std::string& path) {
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream)
        throw CacheValidationError("cannot open cache file: " + path);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    if (!stream.good() && !stream.eof())
        throw CacheValidationError("cannot read cache file: " + path);
    return bytes.str();
}

bool IsSha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (std::size_t i = 0; i < value.size(); ++i)
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f')))
            return false;
    return true;
}

cv::FileStorage ParseJson(const std::string& bytes, const std::string& context) {
    cv::FileStorage storage(bytes,
        cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened())
        throw CacheValidationError("invalid JSON in " + context);
    return storage;
}

std::string RequiredString(const cv::FileNode& root, const char* name,
                           const std::string& context) {
    const cv::FileNode node = root[name];
    if (node.empty() || !node.isString())
        throw CacheValidationError(context + " missing string field " + name);
    return static_cast<std::string>(node);
}

int RequiredInt(const cv::FileNode& root, const char* name,
                const std::string& context) {
    const cv::FileNode node = root[name];
    if (node.empty() || !node.isInt())
        throw CacheValidationError(context + " missing integer field " + name);
    return static_cast<int>(node);
}

double RequiredReal(const cv::FileNode& root, const char* name,
                    const std::string& context) {
    const cv::FileNode node = root[name];
    if (node.empty() || (!node.isReal() && !node.isInt()))
        throw CacheValidationError(context + " missing numeric field " + name);
    const double value = static_cast<double>(node);
    if (!std::isfinite(value))
        throw CacheValidationError(context + " has non-finite field " + name);
    return value;
}

std::string TimestampLexeme(const std::string& line) {
    const std::regex expression("\\\"timestamp\\\"[[:space:]]*:[[:space:]]*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(line, match, expression) || match.size() != 2)
        throw CacheValidationError("cache index timestamp lexeme is missing");
    char* end = NULL;
    errno = 0;
    const std::string lexeme = match[1].str();
    const double value = std::strtod(lexeme.c_str(), &end);
    if (errno != 0 || end != lexeme.c_str() + lexeme.size() || !std::isfinite(value))
        throw CacheValidationError("cache index timestamp lexeme is invalid");
    return lexeme;
}

std::string CanonicalDirectory(const std::string& path) {
    char resolved[PATH_MAX];
    if (!realpath(path.c_str(), resolved))
        throw CacheValidationError("cache root cannot be resolved: " + path);
    return resolved;
}

bool SafeRelativePath(const std::string& path) {
    if (path.empty() || path[0] == '/' || path.find('\\') != std::string::npos ||
        path.find(':') != std::string::npos)
        return false;
    std::istringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/'))
        if (part.empty() || part == "." || part == "..") return false;
    return true;
}

std::string ResolveContained(const std::string& root, const std::string& relative) {
    if (!SafeRelativePath(relative))
        throw CacheValidationError("unsafe cache-relative path: " + relative);
    const std::string joined = root + "/" + relative;
    char resolved[PATH_MAX];
    if (!realpath(joined.c_str(), resolved))
        throw CacheValidationError("cache payload cannot be resolved: " + relative);
    const std::string canonical = resolved;
    if (canonical.compare(0, root.size() + 1, root + "/") != 0)
        throw CacheValidationError("cache payload escapes cache root: " + relative);
    return canonical;
}

std::uint32_t LittleU32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

cv::Mat LoadNpy(const std::string& bytes, int expected_width, int expected_height) {
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
        throw CacheValidationError("score payload is not an NPY file");
    const unsigned char major = static_cast<unsigned char>(bytes[6]);
    const unsigned char minor = static_cast<unsigned char>(bytes[7]);
    if ((major != 1 && major != 2) || minor != 0)
        throw CacheValidationError("only NPY v1.0 and v2.0 are supported");
    const std::size_t length_bytes = major == 1 ? 2 : 4;
    const std::size_t preamble = 8 + length_bytes;
    if (bytes.size() < preamble)
        throw CacheValidationError("truncated NPY preamble");
    const unsigned char* raw = reinterpret_cast<const unsigned char*>(bytes.data());
    std::uint32_t header_length = raw[8] | (static_cast<std::uint32_t>(raw[9]) << 8);
    if (major == 2)
        header_length |= (static_cast<std::uint32_t>(raw[10]) << 16) |
                         (static_cast<std::uint32_t>(raw[11]) << 24);
    if (header_length == 0 || preamble + header_length > bytes.size())
        throw CacheValidationError("truncated NPY header");
    const std::string header = bytes.substr(preamble, header_length);
    if (!std::regex_search(header, std::regex("'descr'[[:space:]]*:[[:space:]]*'<f4'")))
        throw CacheValidationError("NPY dtype must be little-endian float32");
    if (!std::regex_search(header, std::regex("'fortran_order'[[:space:]]*:[[:space:]]*False")))
        throw CacheValidationError("NPY array must be C-contiguous");
    const std::regex shape_expression(
        "'shape'[[:space:]]*:[[:space:]]*\\([[:space:]]*([0-9]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*,?[[:space:]]*\\)");
    std::smatch shape;
    if (!std::regex_search(header, shape, shape_expression) || shape.size() != 3)
        throw CacheValidationError("NPY shape must contain exactly height and width");
    const long rows = std::strtol(shape[1].str().c_str(), NULL, 10);
    const long cols = std::strtol(shape[2].str().c_str(), NULL, 10);
    if (rows != expected_height || cols != expected_width)
        throw CacheValidationError("NPY dimensions do not match cache index");
    const std::size_t elements = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t offset = preamble + header_length;
    if (elements > (std::numeric_limits<std::size_t>::max() - offset) / 4 ||
        bytes.size() != offset + elements * 4)
        throw CacheValidationError("NPY payload length does not match shape");
    cv::Mat scores(static_cast<int>(rows), static_cast<int>(cols), CV_32FC1);
    for (std::size_t i = 0; i < elements; ++i) {
        const std::uint32_t bits = LittleU32(raw + offset + i * 4);
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
            throw CacheValidationError("dynamic score is non-finite or outside [0,1]");
        scores.ptr<float>()[i] = value;
    }
    return scores;
}

}  // namespace

std::string sha256Bytes(const std::string& bytes) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) throw CacheValidationError("cannot allocate SHA256 context");
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), NULL) == 1 &&
        EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32)
        throw CacheValidationError("SHA256 computation failed");
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i)
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    return output.str();
}

std::string sha256File(const std::string& path) {
    return sha256Bytes(ReadBytes(path));
}

CacheMaskProvider::CacheMaskProvider(
    const std::string& cache_root, const std::string& expected_sequence_id,
    const std::string& expected_manifest_sha256,
    const std::string& expected_completion_sha256,
    const std::string& expected_index_sha256)
    : cache_root_(CanonicalDirectory(cache_root)), sequence_id_(expected_sequence_id),
      manifest_sha256_(expected_manifest_sha256),
      completion_sha256_(expected_completion_sha256), index_sha256_(expected_index_sha256) {
    if (sequence_id_.empty() || !IsSha256(manifest_sha256_) ||
        !IsSha256(completion_sha256_) || !IsSha256(index_sha256_))
        throw CacheValidationError("trusted cache identity is incomplete or malformed");

    const std::string manifest_path = cache_root_ + "/cache_manifest.json";
    const std::string completion_path = cache_root_ + "/cache_complete.json";
    const std::string index_path = cache_root_ + "/cache_index.jsonl";
    const std::string manifest_bytes = ReadBytes(manifest_path);
    const std::string completion_bytes = ReadBytes(completion_path);
    const std::string index_bytes = ReadBytes(index_path);
    if (sha256Bytes(manifest_bytes) != manifest_sha256_)
        throw CacheValidationError("cache manifest SHA256 does not match trusted identity");
    if (sha256Bytes(completion_bytes) != completion_sha256_)
        throw CacheValidationError("cache completion SHA256 does not match trusted identity");
    if (sha256Bytes(index_bytes) != index_sha256_)
        throw CacheValidationError("cache index SHA256 does not match trusted identity");

    cv::FileStorage manifest = ParseJson(manifest_bytes, "cache manifest");
    const cv::FileNode manifest_root = manifest.root();
    if (RequiredString(manifest_root, "schema", "cache manifest") !=
        "ovorb.dynamic-cache.v1")
        throw CacheValidationError("cache manifest schema mismatch");
    if (RequiredString(manifest_root, "sequence_id", "cache manifest") != sequence_id_)
        throw CacheValidationError("cache manifest sequence mismatch");
    const int expected_frames = RequiredInt(manifest_root, "expected_frame_count",
                                            "cache manifest");
    if (expected_frames <= 0)
        throw CacheValidationError("cache manifest expected frame count is invalid");

    cv::FileStorage completion = ParseJson(completion_bytes, "cache completion");
    const cv::FileNode completion_root = completion.root();
    if (RequiredString(completion_root, "manifest_sha256", "cache completion") != manifest_sha256_ ||
        RequiredString(completion_root, "index_sha256", "cache completion") != index_sha256_)
        throw CacheValidationError("cache completion identity chain is broken");
    const int completed_frames = RequiredInt(completion_root, "frame_count", "cache completion");
    if (completed_frames != expected_frames)
        throw CacheValidationError("cache completion frame count mismatch");

    std::istringstream index_stream(index_bytes);
    std::string line;
    double previous_timestamp = -std::numeric_limits<double>::infinity();
    while (std::getline(index_stream, line)) {
        if (line.empty())
            throw CacheValidationError("cache index contains an empty row");
        cv::FileStorage row_storage = ParseJson(line, "cache index row");
        const cv::FileNode row = row_storage.root();
        Entry entry;
        entry.frame_id = RequiredInt(row, "frame_id", "cache index row");
        entry.width = RequiredInt(row, "width", "cache index row");
        entry.height = RequiredInt(row, "height", "cache index row");
        entry.timestamp = RequiredReal(row, "timestamp", "cache index row");
        entry.timestamp_lexeme = TimestampLexeme(line);
        entry.relative_path = RequiredString(row, "path", "cache index row");
        entry.score_sha256 = RequiredString(row, "sha256", "cache index row");
        entry.semantic_packet_sha256 = RequiredString(
            row, "semantic_packet_sha256", "cache index row");
        if (RequiredString(row, "dtype", "cache index row") != "float32" ||
            entry.width <= 0 || entry.height <= 0 ||
            !SafeRelativePath(entry.relative_path) || !IsSha256(entry.score_sha256) ||
            !IsSha256(entry.semantic_packet_sha256) ||
            entry.frame_id != static_cast<int>(entries_.size()) ||
            entry.timestamp <= previous_timestamp)
            throw CacheValidationError("cache index row violates identity/order constraints");
        previous_timestamp = entry.timestamp;
        entries_.push_back(entry);
    }
    if (entries_.size() != static_cast<std::size_t>(expected_frames))
        throw CacheValidationError("cache index frame count mismatch");
}

DynamicScoreMap CacheMaskProvider::load(double source_timestamp,
                                        int image_width, int image_height) const {
    if (!std::isfinite(source_timestamp) || image_width <= 0 || image_height <= 0)
        throw CacheValidationError("requested frame identity is invalid");
    const Entry* found = NULL;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].timestamp == source_timestamp) {
            if (found) throw CacheValidationError("ambiguous exact cache timestamp");
            found = &entries_[i];
        }
    }
    if (!found)
        throw CacheValidationError("exact source timestamp is absent from cache index");
    if (found->width != image_width || found->height != image_height)
        throw CacheValidationError("requested image dimensions do not match cache index");
    const std::string payload_path = ResolveContained(cache_root_, found->relative_path);
    const std::string payload = ReadBytes(payload_path);
    if (sha256Bytes(payload) != found->score_sha256)
        throw CacheValidationError("dynamic score payload SHA256 mismatch");

    DynamicScoreMap result;
    result.source_timestamp = found->timestamp;
    result.frame_key = makeFrameKey(sequence_id_, found->timestamp_lexeme);
    result.scores_f32 = LoadNpy(payload, found->width, found->height);
    result.manifest_sha256 = manifest_sha256_;
    result.semantic_packet_sha256 = found->semantic_packet_sha256;
    return result;
}

}  // namespace semantic
}  // namespace ORB_SLAM2
