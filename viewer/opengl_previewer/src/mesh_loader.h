#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace faithc::viewer {

struct MeshData {
    static constexpr size_t kVertexStride = 8;

    // Interleaved vertex buffer: position (xyz) + normal (xyz) + uv (uv)
    std::vector<float> vertices;
    std::vector<uint32_t> indices;

    bool has_uv = false;
    bool has_base_color_texture = false;
    int texture_width = 0;
    int texture_height = 0;
    int texture_channels = 0;
    std::vector<uint8_t> texture_pixels;

    float min_bound[3] = {0.0f, 0.0f, 0.0f};
    float max_bound[3] = {0.0f, 0.0f, 0.0f};

    [[nodiscard]] bool empty() const { return vertices.empty() || indices.empty(); }
    [[nodiscard]] size_t vertex_count() const { return vertices.size() / kVertexStride; }
    [[nodiscard]] size_t face_count() const { return indices.size() / 3; }
};

bool LoadMeshFile(const std::filesystem::path &path, MeshData &out, std::string &error);

}  // namespace faithc::viewer
