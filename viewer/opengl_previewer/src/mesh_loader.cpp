#include "mesh_loader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include <tinyply.h>

namespace faithc::viewer {
namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

void ComputeBounds(MeshData &mesh) {
    if (mesh.vertices.empty()) {
        return;
    }

    mesh.min_bound[0] = mesh.min_bound[1] = mesh.min_bound[2] = std::numeric_limits<float>::max();
    mesh.max_bound[0] = mesh.max_bound[1] = mesh.max_bound[2] = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        const float x = mesh.vertices[i * MeshData::kVertexStride + 0];
        const float y = mesh.vertices[i * MeshData::kVertexStride + 1];
        const float z = mesh.vertices[i * MeshData::kVertexStride + 2];

        mesh.min_bound[0] = std::min(mesh.min_bound[0], x);
        mesh.min_bound[1] = std::min(mesh.min_bound[1], y);
        mesh.min_bound[2] = std::min(mesh.min_bound[2], z);

        mesh.max_bound[0] = std::max(mesh.max_bound[0], x);
        mesh.max_bound[1] = std::max(mesh.max_bound[1], y);
        mesh.max_bound[2] = std::max(mesh.max_bound[2], z);
    }
}

bool HasValidNormals(const MeshData &mesh) {
    const float eps = 1e-12f;
    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        const float nx = mesh.vertices[i * MeshData::kVertexStride + 3];
        const float ny = mesh.vertices[i * MeshData::kVertexStride + 4];
        const float nz = mesh.vertices[i * MeshData::kVertexStride + 5];
        const float len2 = nx * nx + ny * ny + nz * nz;
        if (len2 > eps) {
            return true;
        }
    }
    return false;
}

void RecomputeNormals(MeshData &mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        return;
    }

    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        mesh.vertices[i * MeshData::kVertexStride + 3] = 0.0f;
        mesh.vertices[i * MeshData::kVertexStride + 4] = 0.0f;
        mesh.vertices[i * MeshData::kVertexStride + 5] = 0.0f;
    }

    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t i0 = mesh.indices[t + 0];
        const uint32_t i1 = mesh.indices[t + 1];
        const uint32_t i2 = mesh.indices[t + 2];

        const float *p0 = &mesh.vertices[static_cast<size_t>(i0) * MeshData::kVertexStride];
        const float *p1 = &mesh.vertices[static_cast<size_t>(i1) * MeshData::kVertexStride];
        const float *p2 = &mesh.vertices[static_cast<size_t>(i2) * MeshData::kVertexStride];

        const float ax = p1[0] - p0[0];
        const float ay = p1[1] - p0[1];
        const float az = p1[2] - p0[2];

        const float bx = p2[0] - p0[0];
        const float by = p2[1] - p0[1];
        const float bz = p2[2] - p0[2];

        const float nx = ay * bz - az * by;
        const float ny = az * bx - ax * bz;
        const float nz = ax * by - ay * bx;

        mesh.vertices[static_cast<size_t>(i0) * MeshData::kVertexStride + 3] += nx;
        mesh.vertices[static_cast<size_t>(i0) * MeshData::kVertexStride + 4] += ny;
        mesh.vertices[static_cast<size_t>(i0) * MeshData::kVertexStride + 5] += nz;

        mesh.vertices[static_cast<size_t>(i1) * MeshData::kVertexStride + 3] += nx;
        mesh.vertices[static_cast<size_t>(i1) * MeshData::kVertexStride + 4] += ny;
        mesh.vertices[static_cast<size_t>(i1) * MeshData::kVertexStride + 5] += nz;

        mesh.vertices[static_cast<size_t>(i2) * MeshData::kVertexStride + 3] += nx;
        mesh.vertices[static_cast<size_t>(i2) * MeshData::kVertexStride + 4] += ny;
        mesh.vertices[static_cast<size_t>(i2) * MeshData::kVertexStride + 5] += nz;
    }

    for (size_t i = 0; i < mesh.vertex_count(); ++i) {
        float &nx = mesh.vertices[i * MeshData::kVertexStride + 3];
        float &ny = mesh.vertices[i * MeshData::kVertexStride + 4];
        float &nz = mesh.vertices[i * MeshData::kVertexStride + 5];
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-12f) {
            nx /= len;
            ny /= len;
            nz /= len;
        } else {
            nx = 0.0f;
            ny = 1.0f;
            nz = 0.0f;
        }
    }
}

bool FinalizeMesh(MeshData &mesh, std::string &error) {
    if (mesh.vertices.empty()) {
        error = "No vertices parsed";
        return false;
    }
    if (mesh.indices.empty()) {
        error = "No triangle indices parsed";
        return false;
    }
    if (mesh.indices.size() % 3 != 0) {
        error = "Triangle index buffer is not divisible by 3";
        return false;
    }

    const size_t vcount = mesh.vertex_count();
    for (const uint32_t idx : mesh.indices) {
        if (idx >= vcount) {
            std::ostringstream oss;
            oss << "Index out of range: " << idx << " >= " << vcount;
            error = oss.str();
            return false;
        }
    }

    if (!HasValidNormals(mesh)) {
        RecomputeNormals(mesh);
    }

    ComputeBounds(mesh);
    return true;
}

struct VertexKey {
    int v = -1;
    int n = -1;
    int t = -1;

    bool operator==(const VertexKey &other) const { return v == other.v && n == other.n && t == other.t; }
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey &key) const noexcept {
        const std::size_t h1 = std::hash<int>{}(key.v);
        const std::size_t h2 = std::hash<int>{}(key.n);
        const std::size_t h3 = std::hash<int>{}(key.t);
        return h1 ^ (h2 << 1U) ^ (h3 << 2U);
    }
};

bool LoadObj(const std::filesystem::path &path, MeshData &out, std::string &error) {
    tinyobj::ObjReaderConfig cfg;
    cfg.triangulate = true;
    cfg.vertex_color = false;

    if (path.has_parent_path()) {
        cfg.mtl_search_path = path.parent_path().string();
    }

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string(), cfg)) {
        std::ostringstream oss;
        if (!reader.Warning().empty()) {
            oss << reader.Warning() << "\n";
        }
        oss << reader.Error();
        error = oss.str();
        return false;
    }

    const tinyobj::attrib_t &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();

    if (attrib.vertices.empty()) {
        error = "OBJ has no vertices";
        return false;
    }

    out.vertices.clear();
    out.indices.clear();
    out.has_uv = false;
    out.has_base_color_texture = false;
    out.texture_width = 0;
    out.texture_height = 0;
    out.texture_channels = 0;
    out.texture_pixels.clear();

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> map;
    map.reserve(100000);

    for (const auto &shape : shapes) {
        for (const auto &idx : shape.mesh.indices) {
            if (idx.vertex_index < 0) {
                error = "OBJ contains invalid vertex index";
                return false;
            }
            VertexKey key{idx.vertex_index, idx.normal_index, idx.texcoord_index};
            auto iter = map.find(key);
            if (iter == map.end()) {
                const uint32_t out_index = static_cast<uint32_t>(out.vertex_count());
                map.emplace(key, out_index);

                const size_t vp = static_cast<size_t>(idx.vertex_index) * 3;
                out.vertices.push_back(attrib.vertices[vp + 0]);
                out.vertices.push_back(attrib.vertices[vp + 1]);
                out.vertices.push_back(attrib.vertices[vp + 2]);

                if (idx.normal_index >= 0 && static_cast<size_t>(idx.normal_index * 3 + 2) < attrib.normals.size()) {
                    const size_t np = static_cast<size_t>(idx.normal_index) * 3;
                    out.vertices.push_back(attrib.normals[np + 0]);
                    out.vertices.push_back(attrib.normals[np + 1]);
                    out.vertices.push_back(attrib.normals[np + 2]);
                } else {
                    out.vertices.push_back(0.0f);
                    out.vertices.push_back(0.0f);
                    out.vertices.push_back(0.0f);
                }

                if (idx.texcoord_index >= 0 && static_cast<size_t>(idx.texcoord_index * 2 + 1) < attrib.texcoords.size()) {
                    const size_t tp = static_cast<size_t>(idx.texcoord_index) * 2;
                    out.vertices.push_back(attrib.texcoords[tp + 0]);
                    out.vertices.push_back(attrib.texcoords[tp + 1]);
                    out.has_uv = true;
                } else {
                    out.vertices.push_back(0.0f);
                    out.vertices.push_back(0.0f);
                }

                out.indices.push_back(out_index);
            } else {
                out.indices.push_back(iter->second);
            }
        }
    }

    return FinalizeMesh(out, error);
}

float ReadFloatComponent(const tinyply::Type type, const uint8_t *ptr) {
    switch (type) {
    case tinyply::Type::FLOAT32:
        return *reinterpret_cast<const float *>(ptr);
    case tinyply::Type::FLOAT64:
        return static_cast<float>(*reinterpret_cast<const double *>(ptr));
    case tinyply::Type::INT8:
        return static_cast<float>(*reinterpret_cast<const int8_t *>(ptr));
    case tinyply::Type::UINT8:
        return static_cast<float>(*reinterpret_cast<const uint8_t *>(ptr));
    case tinyply::Type::INT16:
        return static_cast<float>(*reinterpret_cast<const int16_t *>(ptr));
    case tinyply::Type::UINT16:
        return static_cast<float>(*reinterpret_cast<const uint16_t *>(ptr));
    case tinyply::Type::INT32:
        return static_cast<float>(*reinterpret_cast<const int32_t *>(ptr));
    case tinyply::Type::UINT32:
        return static_cast<float>(*reinterpret_cast<const uint32_t *>(ptr));
    default:
        return 0.0f;
    }
}

template <typename T>
void AppendIndexDataAsUInt32(const uint8_t *data, size_t count, std::vector<uint32_t> &out) {
    const T *ptr = reinterpret_cast<const T *>(data);
    out.reserve(out.size() + count);
    for (size_t i = 0; i < count; ++i) {
        out.push_back(static_cast<uint32_t>(ptr[i]));
    }
}

bool LoadPly(const std::filesystem::path &path, MeshData &out, std::string &error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot open PLY file";
        return false;
    }

    tinyply::PlyFile file;
    std::shared_ptr<tinyply::PlyData> verts;
    std::shared_ptr<tinyply::PlyData> normals;
    std::shared_ptr<tinyply::PlyData> faces;

    try {
        file.parse_header(stream);
        verts = file.request_properties_from_element("vertex", {"x", "y", "z"});
        try {
            normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
        } catch (...) {
            normals = nullptr;
        }

        try {
            faces = file.request_properties_from_element("face", {"vertex_indices"}, 3);
        } catch (...) {
            faces = nullptr;
        }

        file.read(stream);
    } catch (const std::exception &e) {
        error = std::string("Failed to parse PLY: ") + e.what();
        return false;
    }

    if (!verts || verts->count == 0) {
        error = "PLY has no vertex data";
        return false;
    }

    out.vertices.clear();
    out.indices.clear();
    out.has_uv = false;
    out.has_base_color_texture = false;
    out.texture_width = 0;
    out.texture_height = 0;
    out.texture_channels = 0;
    out.texture_pixels.clear();
    out.vertices.resize(verts->count * MeshData::kVertexStride, 0.0f);

    const size_t scalar_size = tinyply::PropertyTable[verts->t].stride;
    for (size_t i = 0; i < verts->count; ++i) {
        const uint8_t *ptr = verts->buffer.get() + i * 3 * scalar_size;
        out.vertices[i * MeshData::kVertexStride + 0] = ReadFloatComponent(verts->t, ptr + scalar_size * 0);
        out.vertices[i * MeshData::kVertexStride + 1] = ReadFloatComponent(verts->t, ptr + scalar_size * 1);
        out.vertices[i * MeshData::kVertexStride + 2] = ReadFloatComponent(verts->t, ptr + scalar_size * 2);
    }

    if (normals && normals->count == verts->count) {
        const size_t n_scalar_size = tinyply::PropertyTable[normals->t].stride;
        for (size_t i = 0; i < normals->count; ++i) {
            const uint8_t *ptr = normals->buffer.get() + i * 3 * n_scalar_size;
            out.vertices[i * MeshData::kVertexStride + 3] = ReadFloatComponent(normals->t, ptr + n_scalar_size * 0);
            out.vertices[i * MeshData::kVertexStride + 4] = ReadFloatComponent(normals->t, ptr + n_scalar_size * 1);
            out.vertices[i * MeshData::kVertexStride + 5] = ReadFloatComponent(normals->t, ptr + n_scalar_size * 2);
        }
    }

    if (!faces || faces->count == 0) {
        error = "PLY has no face data";
        return false;
    }

    const size_t face_index_count = faces->count * 3;
    switch (faces->t) {
    case tinyply::Type::UINT8:
        AppendIndexDataAsUInt32<uint8_t>(faces->buffer.get(), face_index_count, out.indices);
        break;
    case tinyply::Type::INT8:
        AppendIndexDataAsUInt32<int8_t>(faces->buffer.get(), face_index_count, out.indices);
        break;
    case tinyply::Type::UINT16:
        AppendIndexDataAsUInt32<uint16_t>(faces->buffer.get(), face_index_count, out.indices);
        break;
    case tinyply::Type::INT16:
        AppendIndexDataAsUInt32<int16_t>(faces->buffer.get(), face_index_count, out.indices);
        break;
    case tinyply::Type::UINT32:
        AppendIndexDataAsUInt32<uint32_t>(faces->buffer.get(), face_index_count, out.indices);
        break;
    case tinyply::Type::INT32:
        AppendIndexDataAsUInt32<int32_t>(faces->buffer.get(), face_index_count, out.indices);
        break;
    default:
        error = "Unsupported PLY face index type";
        return false;
    }

    return FinalizeMesh(out, error);
}

size_t ComponentByteSize(int component_type) {
    switch (component_type) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return 4;
    default:
        return 0;
    }
}

float ReadAccessorComponentAsFloat(const uint8_t *ptr, int component_type, bool normalized) {
    switch (component_type) {
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
        int8_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        if (!normalized) {
            return static_cast<float>(value);
        }
        return std::max(-1.0f, static_cast<float>(value) / 127.0f);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        uint8_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        if (!normalized) {
            return static_cast<float>(value);
        }
        return static_cast<float>(value) / 255.0f;
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
        int16_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        if (!normalized) {
            return static_cast<float>(value);
        }
        return std::max(-1.0f, static_cast<float>(value) / 32767.0f);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        if (!normalized) {
            return static_cast<float>(value);
        }
        return static_cast<float>(value) / 65535.0f;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        uint32_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        if (!normalized) {
            return static_cast<float>(value);
        }
        return static_cast<float>(value) / 4294967295.0f;
    }
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        float value = 0.0f;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }
    default:
        return 0.0f;
    }
}

bool ResolveAccessorLayout(const tinygltf::Model &model, int accessor_index, int expected_type, size_t component_count,
                           int expected_component_type, const uint8_t *&base, size_t &stride, bool &normalized,
                           std::string &error) {
    if (accessor_index < 0 || accessor_index >= static_cast<int>(model.accessors.size())) {
        error = "Invalid accessor index";
        return false;
    }

    const tinygltf::Accessor &acc = model.accessors[accessor_index];
    if (acc.bufferView < 0 || acc.bufferView >= static_cast<int>(model.bufferViews.size())) {
        error = "Accessor has invalid buffer view";
        return false;
    }
    if (acc.type != expected_type) {
        error = "Accessor has unexpected type";
        return false;
    }
    if (expected_component_type != -1 && acc.componentType != expected_component_type) {
        error = "Accessor has unexpected component type";
        return false;
    }

    const tinygltf::BufferView &view = model.bufferViews[acc.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size())) {
        error = "Accessor buffer view points to invalid buffer";
        return false;
    }
    const tinygltf::Buffer &buffer = model.buffers[view.buffer];
    const size_t component_size = ComponentByteSize(acc.componentType);
    if (component_size == 0) {
        error = "Unsupported accessor component type";
        return false;
    }

    stride = acc.ByteStride(view) > 0 ? static_cast<size_t>(acc.ByteStride(view)) : component_size * component_count;
    if (stride < component_size * component_count) {
        error = "Accessor stride is smaller than element size";
        return false;
    }

    const size_t base_offset = view.byteOffset + acc.byteOffset;
    if (base_offset >= buffer.data.size()) {
        error = "Accessor byte offset out of buffer range";
        return false;
    }

    if (acc.count > 0) {
        const size_t last_offset = base_offset + (acc.count - 1) * stride + component_size * component_count;
        if (last_offset > buffer.data.size()) {
            error = "Accessor data exceeds buffer size";
            return false;
        }
    }

    base = buffer.data.data() + base_offset;
    normalized = acc.normalized;
    return true;
}

bool ReadAccessorVec3(const tinygltf::Model &model, int accessor_index, std::vector<std::array<float, 3>> &out,
                      std::string &error) {
    const uint8_t *base = nullptr;
    size_t stride = 0;
    bool normalized = false;
    if (!ResolveAccessorLayout(model, accessor_index, TINYGLTF_TYPE_VEC3, 3, TINYGLTF_COMPONENT_TYPE_FLOAT, base, stride,
                               normalized, error)) {
        return false;
    }
    const tinygltf::Accessor &acc = model.accessors[accessor_index];

    out.resize(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        const float *p = reinterpret_cast<const float *>(base + i * stride);
        out[i] = {p[0], p[1], p[2]};
    }

    return true;
}

bool ReadAccessorVec2(const tinygltf::Model &model, int accessor_index, std::vector<std::array<float, 2>> &out,
                      std::string &error) {
    const uint8_t *base = nullptr;
    size_t stride = 0;
    bool normalized = false;
    if (!ResolveAccessorLayout(model, accessor_index, TINYGLTF_TYPE_VEC2, 2, -1, base, stride, normalized, error)) {
        return false;
    }
    const tinygltf::Accessor &acc = model.accessors[accessor_index];

    const size_t component_size = ComponentByteSize(acc.componentType);
    out.resize(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        const uint8_t *ptr = base + i * stride;
        out[i][0] = ReadAccessorComponentAsFloat(ptr + component_size * 0, acc.componentType, normalized);
        out[i][1] = ReadAccessorComponentAsFloat(ptr + component_size * 1, acc.componentType, normalized);
    }

    return true;
}

void TryExtractBaseColorTexture(const tinygltf::Model &model, int material_index, MeshData &out) {
    if (out.has_base_color_texture) {
        return;
    }
    if (material_index < 0 || material_index >= static_cast<int>(model.materials.size())) {
        return;
    }

    const tinygltf::Material &material = model.materials[material_index];
    const int texture_index = material.pbrMetallicRoughness.baseColorTexture.index;
    if (texture_index < 0 || texture_index >= static_cast<int>(model.textures.size())) {
        return;
    }

    const tinygltf::Texture &texture = model.textures[texture_index];
    if (texture.source < 0 || texture.source >= static_cast<int>(model.images.size())) {
        return;
    }

    const tinygltf::Image &image = model.images[texture.source];
    if (image.image.empty() || image.width <= 0 || image.height <= 0) {
        return;
    }
    if (image.bits != 8 || image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        return;
    }
    if (image.component < 1 || image.component > 4) {
        return;
    }

    out.has_base_color_texture = true;
    out.texture_width = image.width;
    out.texture_height = image.height;
    out.texture_channels = image.component;
    out.texture_pixels = image.image;
}

bool ReadAccessorIndices(const tinygltf::Model &model, int accessor_index, std::vector<uint32_t> &out,
                         std::string &error) {
    if (accessor_index < 0 || accessor_index >= static_cast<int>(model.accessors.size())) {
        error = "Invalid index accessor";
        return false;
    }

    const tinygltf::Accessor &acc = model.accessors[accessor_index];
    if (acc.bufferView < 0 || acc.bufferView >= static_cast<int>(model.bufferViews.size())) {
        error = "Index accessor has invalid buffer view";
        return false;
    }
    if (acc.type != TINYGLTF_TYPE_SCALAR) {
        error = "Index accessor is not scalar";
        return false;
    }

    const tinygltf::BufferView &view = model.bufferViews[acc.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size())) {
        error = "Index buffer view points to invalid buffer";
        return false;
    }
    const tinygltf::Buffer &buffer = model.buffers[view.buffer];

    size_t scalar_size = 0;
    switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        scalar_size = sizeof(uint8_t);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        scalar_size = sizeof(uint16_t);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        scalar_size = sizeof(uint32_t);
        break;
    default:
        error = "Unsupported index component type in glTF";
        return false;
    }

    const size_t stride = acc.ByteStride(view) > 0 ? static_cast<size_t>(acc.ByteStride(view)) : scalar_size;
    const size_t base_offset = view.byteOffset + acc.byteOffset;
    if (base_offset >= buffer.data.size()) {
        error = "Index accessor byte offset out of range";
        return false;
    }
    if (acc.count > 0) {
        const size_t last_offset = base_offset + (acc.count - 1) * stride + scalar_size;
        if (last_offset > buffer.data.size()) {
            error = "Index accessor data exceeds buffer size";
            return false;
        }
    }
    const uint8_t *base = buffer.data.data() + base_offset;

    out.reserve(out.size() + acc.count);

    for (size_t i = 0; i < acc.count; ++i) {
        const uint8_t *ptr = base + i * stride;
        uint32_t value = 0;
        switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            value = static_cast<uint32_t>(*reinterpret_cast<const uint8_t *>(ptr));
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t tmp = 0;
            std::memcpy(&tmp, ptr, sizeof(tmp));
            value = static_cast<uint32_t>(tmp);
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t tmp = 0;
            std::memcpy(&tmp, ptr, sizeof(tmp));
            value = tmp;
            break;
        }
        default:
            break;
        }
        out.push_back(value);
    }

    return true;
}

bool LoadGltf(const std::filesystem::path &path, MeshData &out, std::string &error) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;

    std::string warn;
    std::string err;

    bool ok = false;
    const auto ext = ToLower(path.extension().string());
    if (ext == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    }

    if (!ok) {
        error = "Failed to load glTF/GLB: " + warn + " " + err;
        return false;
    }

    out.vertices.clear();
    out.indices.clear();
    out.has_uv = false;
    out.has_base_color_texture = false;
    out.texture_width = 0;
    out.texture_height = 0;
    out.texture_channels = 0;
    out.texture_pixels.clear();

    bool parsed_any = false;
    for (const auto &mesh : model.meshes) {
        for (const auto &prim : mesh.primitives) {
            if (prim.mode != -1 && prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            auto it_pos = prim.attributes.find("POSITION");
            if (it_pos == prim.attributes.end()) {
                continue;
            }

            std::vector<std::array<float, 3>> positions;
            if (!ReadAccessorVec3(model, it_pos->second, positions, error)) {
                return false;
            }

            std::vector<std::array<float, 3>> normals;
            auto it_norm = prim.attributes.find("NORMAL");
            if (it_norm != prim.attributes.end()) {
                if (!ReadAccessorVec3(model, it_norm->second, normals, error)) {
                    return false;
                }
                if (normals.size() != positions.size()) {
                    normals.clear();
                }
            }

            std::vector<std::array<float, 2>> texcoords;
            auto it_uv = prim.attributes.find("TEXCOORD_0");
            if (it_uv != prim.attributes.end()) {
                if (!ReadAccessorVec2(model, it_uv->second, texcoords, error)) {
                    return false;
                }
                if (texcoords.size() != positions.size()) {
                    texcoords.clear();
                } else {
                    out.has_uv = true;
                }
            }

            if (positions.empty()) {
                continue;
            }

            TryExtractBaseColorTexture(model, prim.material, out);

            const uint32_t base_vertex = static_cast<uint32_t>(out.vertex_count());
            out.vertices.reserve(out.vertices.size() + positions.size() * MeshData::kVertexStride);
            for (size_t i = 0; i < positions.size(); ++i) {
                out.vertices.push_back(positions[i][0]);
                out.vertices.push_back(positions[i][1]);
                out.vertices.push_back(positions[i][2]);

                if (!normals.empty()) {
                    out.vertices.push_back(normals[i][0]);
                    out.vertices.push_back(normals[i][1]);
                    out.vertices.push_back(normals[i][2]);
                } else {
                    out.vertices.push_back(0.0f);
                    out.vertices.push_back(0.0f);
                    out.vertices.push_back(0.0f);
                }

                if (!texcoords.empty()) {
                    out.vertices.push_back(texcoords[i][0]);
                    out.vertices.push_back(texcoords[i][1]);
                } else {
                    out.vertices.push_back(0.0f);
                    out.vertices.push_back(0.0f);
                }
            }

            if (prim.indices >= 0) {
                std::vector<uint32_t> local_indices;
                if (!ReadAccessorIndices(model, prim.indices, local_indices, error)) {
                    return false;
                }

                out.indices.reserve(out.indices.size() + local_indices.size());
                for (uint32_t idx : local_indices) {
                    out.indices.push_back(base_vertex + idx);
                }
            } else {
                out.indices.reserve(out.indices.size() + positions.size());
                for (uint32_t i = 0; i < static_cast<uint32_t>(positions.size()); ++i) {
                    out.indices.push_back(base_vertex + i);
                }
            }

            parsed_any = true;
        }
    }

    if (!parsed_any) {
        error = "No triangle primitives found in glTF/GLB";
        return false;
    }

    return FinalizeMesh(out, error);
}

}  // namespace

bool LoadMeshFile(const std::filesystem::path &path, MeshData &out, std::string &error) {
    if (!std::filesystem::exists(path)) {
        error = "File not found: " + path.string();
        return false;
    }

    const std::string ext = ToLower(path.extension().string());
    if (ext == ".obj") {
        return LoadObj(path, out, error);
    }
    if (ext == ".ply") {
        return LoadPly(path, out, error);
    }
    if (ext == ".gltf" || ext == ".glb") {
        return LoadGltf(path, out, error);
    }

    error = "Unsupported file format: " + ext;
    return false;
}

}  // namespace faithc::viewer
