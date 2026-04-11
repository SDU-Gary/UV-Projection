#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "job_output_view.h"
#include "job_result_schema.h"
#include "mesh_loader.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std::chrono_literals;
using FaithCJobResult = faithc::viewer::FaithCJobResult;

namespace {

constexpr int kWindowWidth = 1600;
constexpr int kWindowHeight = 950;
constexpr size_t kSeamOverlayFaceLimit = 800000;

const char *g_crash_stage = "startup";
volatile sig_atomic_t g_crash_handling = 0;

void SetCrashStage(const char *stage) {
    g_crash_stage = (stage != nullptr) ? stage : "(null)";
}

void SafeWrite(int fd, const void *buf, size_t len) {
    const ssize_t written = write(fd, buf, len);
    (void)written;
}

void CrashSignalHandler(int sig) {
    if (g_crash_handling != 0) {
        _exit(128 + sig);
    }
    g_crash_handling = 1;

    char header[512];
    const int n = std::snprintf(
        header,
        sizeof(header),
        "\n==== faithc_viewer crash ====\nsignal=%d\nstage=%s\n",
        sig,
        g_crash_stage != nullptr ? g_crash_stage : "(null)"
    );
    if (n > 0) {
        SafeWrite(STDERR_FILENO, header, static_cast<size_t>(n));
    }

    void *frames[128];
    const int frame_count = backtrace(frames, 128);
    backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);

    int fd = ::open("/tmp/faithc_viewer_crash.log", O_CREAT | O_WRONLY | O_APPEND, 0644);
    if (fd >= 0) {
        if (n > 0) {
            SafeWrite(fd, header, static_cast<size_t>(n));
        }
        backtrace_symbols_fd(frames, frame_count, fd);
        static const char newline = '\n';
        SafeWrite(fd, &newline, 1);
        close(fd);
    }

    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void InstallCrashHandlers() {
    std::signal(SIGSEGV, CrashSignalHandler);
    std::signal(SIGABRT, CrashSignalHandler);
    std::signal(SIGBUS, CrashSignalHandler);
    std::signal(SIGILL, CrashSignalHandler);
    std::signal(SIGFPE, CrashSignalHandler);
}

std::string ShellQuote(const std::string &value) {
    std::string out;
    out.reserve(value.size() + 8);
    out.push_back('\'');
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

uint64_t NowMillis() {
    const auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

struct QuantizedPosKey {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    bool operator==(const QuantizedPosKey &other) const { return x == other.x && y == other.y && z == other.z; }
};

struct QuantizedPosKeyHash {
    std::size_t operator()(const QuantizedPosKey &k) const noexcept {
        const std::size_t h1 = std::hash<int64_t>{}(k.x);
        const std::size_t h2 = std::hash<int64_t>{}(k.y);
        const std::size_t h3 = std::hash<int64_t>{}(k.z);
        return h1 ^ (h2 << 1U) ^ (h3 << 2U);
    }
};

struct WeldEdgeKey {
    int64_t a = 0;
    int64_t b = 0;

    bool operator==(const WeldEdgeKey &other) const { return a == other.a && b == other.b; }
};

struct WeldEdgeKeyHash {
    std::size_t operator()(const WeldEdgeKey &k) const noexcept {
        const std::size_t h1 = std::hash<int64_t>{}(k.a);
        const std::size_t h2 = std::hash<int64_t>{}(k.b);
        return h1 ^ (h2 << 1U);
    }
};

struct TopoEdgeKey {
    uint32_t a = 0;
    uint32_t b = 0;

    bool operator==(const TopoEdgeKey &other) const { return a == other.a && b == other.b; }
};

struct TopoEdgeKeyHash {
    std::size_t operator()(const TopoEdgeKey &k) const noexcept {
        const std::size_t h1 = std::hash<uint32_t>{}(k.a);
        const std::size_t h2 = std::hash<uint32_t>{}(k.b);
        return h1 ^ (h2 << 1U);
    }
};

struct UVSeamOverlay {
    std::vector<glm::vec3> line_points;
    bool has_uv = false;
    int seam_edges = 0;
    int boundary_edges = 0;
    int nonmanifold_edges = 0;
    int interior_seam_edges = 0;
};

struct ClosureValidationSummary {
    bool available = false;
    bool partition_has_leakage = false;
    bool seam_topology_valid = false;
    int partition_mixed_components = -1;
    int partition_label_split_count = -1;
    int seam_components = -1;
    int seam_loops_closed = -1;
    int seam_components_open = -1;
    int low_island_count = -1;
    int high_island_count = -1;
    int semantic_unknown_faces = -1;
    double uv_bbox_iou_mean = -1.0;
    double uv_overlap_ratio = -1.0;
    double uv_stretch_p95 = -1.0;
    double uv_stretch_p99 = -1.0;
    std::string uv_png_path;
};

struct EdgeRecord {
    int face_id = -1;
    glm::vec2 uv_a = glm::vec2(0.0f);
    glm::vec2 uv_b = glm::vec2(0.0f);
    glm::vec3 pos_a = glm::vec3(0.0f);
    glm::vec3 pos_b = glm::vec3(0.0f);
};

glm::vec3 ReadVertexPosition(const faithc::viewer::MeshData &mesh, size_t vid) {
    const size_t base = vid * faithc::viewer::MeshData::kVertexStride;
    return glm::vec3(mesh.vertices[base + 0], mesh.vertices[base + 1], mesh.vertices[base + 2]);
}

glm::vec2 ReadVertexUV(const faithc::viewer::MeshData &mesh, size_t vid) {
    const size_t base = vid * faithc::viewer::MeshData::kVertexStride;
    return glm::vec2(mesh.vertices[base + 6], mesh.vertices[base + 7]);
}

UVSeamOverlay BuildUVSeamOverlay(const faithc::viewer::MeshData &mesh, double position_eps = 1e-6, double uv_eps = 1e-5) {
    UVSeamOverlay out;
    out.has_uv = mesh.has_uv;
    if (!mesh.has_uv || mesh.empty()) {
        return out;
    }

    const size_t vertex_count = mesh.vertex_count();
    const size_t face_count = mesh.face_count();
    if (vertex_count == 0 || face_count == 0) {
        return out;
    }

    const double pos_eps = std::max(position_eps, 1e-12);
    const float uv_eps2 = static_cast<float>(std::max(uv_eps, 0.0) * std::max(uv_eps, 0.0));

    std::vector<int64_t> weld_id(vertex_count, -1);
    std::unordered_map<QuantizedPosKey, int64_t, QuantizedPosKeyHash> weld_map;
    weld_map.reserve(vertex_count * 2);
    int64_t next_weld = 0;
    for (size_t vid = 0; vid < vertex_count; ++vid) {
        const glm::vec3 p = ReadVertexPosition(mesh, vid);
        const QuantizedPosKey key{
            static_cast<int64_t>(std::llround(static_cast<double>(p.x) / pos_eps)),
            static_cast<int64_t>(std::llround(static_cast<double>(p.y) / pos_eps)),
            static_cast<int64_t>(std::llround(static_cast<double>(p.z) / pos_eps)),
        };
        auto it = weld_map.find(key);
        if (it == weld_map.end()) {
            const auto inserted = weld_map.emplace(key, next_weld);
            it = inserted.first;
            ++next_weld;
        }
        weld_id[vid] = it->second;
    }

    std::unordered_map<WeldEdgeKey, std::vector<EdgeRecord>, WeldEdgeKeyHash> grouped_edges;
    grouped_edges.reserve(face_count * 3);
    out.line_points.reserve(face_count);

    for (size_t f = 0; f < face_count; ++f) {
        const size_t base = f * 3;
        const uint32_t tri[3] = {
            mesh.indices[base + 0],
            mesh.indices[base + 1],
            mesh.indices[base + 2],
        };
        for (int c = 0; c < 3; ++c) {
            const uint32_t va = tri[c];
            const uint32_t vb = tri[(c + 1) % 3];
            if (va >= vertex_count || vb >= vertex_count) {
                continue;
            }

            int64_t ga = weld_id[va];
            int64_t gb = weld_id[vb];
            glm::vec2 uv_a = ReadVertexUV(mesh, va);
            glm::vec2 uv_b = ReadVertexUV(mesh, vb);
            glm::vec3 pos_a = ReadVertexPosition(mesh, va);
            glm::vec3 pos_b = ReadVertexPosition(mesh, vb);
            if (ga > gb) {
                std::swap(ga, gb);
                std::swap(uv_a, uv_b);
                std::swap(pos_a, pos_b);
            }

            const WeldEdgeKey key{ga, gb};
            grouped_edges[key].push_back(EdgeRecord{
                static_cast<int>(f),
                uv_a,
                uv_b,
                pos_a,
                pos_b,
            });
        }
    }

    auto add_seam = [&](const EdgeRecord &rec) {
        out.line_points.push_back(rec.pos_a);
        out.line_points.push_back(rec.pos_b);
        out.seam_edges += 1;
    };

    for (const auto &kv : grouped_edges) {
        const std::vector<EdgeRecord> &records = kv.second;
        if (records.empty()) {
            continue;
        }
        if (records.size() == 1) {
            out.boundary_edges += 1;
            add_seam(records[0]);
            continue;
        }
        if (records.size() > 2) {
            out.nonmanifold_edges += 1;
            add_seam(records[0]);
            continue;
        }

        const EdgeRecord &r0 = records[0];
        const EdgeRecord &r1 = records[1];
        const glm::vec2 d_a = r0.uv_a - r1.uv_a;
        const glm::vec2 d_b = r0.uv_b - r1.uv_b;
        const bool same_a = glm::dot(d_a, d_a) <= uv_eps2;
        const bool same_b = glm::dot(d_b, d_b) <= uv_eps2;
        const bool can_link = (r0.face_id != r1.face_id) && same_a && same_b;
        if (!can_link) {
            out.interior_seam_edges += 1;
            add_seam(r0);
        }
    }

    return out;
}

UVSeamOverlay BuildUVBoundaryAuditOverlay(
    const faithc::viewer::MeshData &mesh,
    double position_eps = 1e-6,
    double uv_eps = 1e-5
) {
    UVSeamOverlay out;
    out.has_uv = mesh.has_uv;
    if (!mesh.has_uv || mesh.empty()) {
        return out;
    }

    const size_t vertex_count = mesh.vertex_count();
    const size_t face_count = mesh.face_count();
    if (vertex_count == 0 || face_count == 0) {
        return out;
    }

    const double pos_eps = std::max(position_eps, 1e-12);
    const float uv_eps2 = static_cast<float>(std::max(uv_eps, 0.0) * std::max(uv_eps, 0.0));

    std::vector<int64_t> weld_id(vertex_count, -1);
    std::unordered_map<QuantizedPosKey, int64_t, QuantizedPosKeyHash> weld_map;
    weld_map.reserve(vertex_count * 2);
    int64_t next_weld = 0;
    for (size_t vid = 0; vid < vertex_count; ++vid) {
        const glm::vec3 p = ReadVertexPosition(mesh, vid);
        const QuantizedPosKey key{
            static_cast<int64_t>(std::llround(static_cast<double>(p.x) / pos_eps)),
            static_cast<int64_t>(std::llround(static_cast<double>(p.y) / pos_eps)),
            static_cast<int64_t>(std::llround(static_cast<double>(p.z) / pos_eps)),
        };
        auto it = weld_map.find(key);
        if (it == weld_map.end()) {
            const auto inserted = weld_map.emplace(key, next_weld);
            it = inserted.first;
            ++next_weld;
        }
        weld_id[vid] = it->second;
    }

    std::unordered_map<TopoEdgeKey, int, TopoEdgeKeyHash> topo_edge_count;
    topo_edge_count.reserve(face_count * 3);
    for (size_t f = 0; f < face_count; ++f) {
        const size_t base = f * 3;
        const uint32_t tri[3] = {mesh.indices[base + 0], mesh.indices[base + 1], mesh.indices[base + 2]};
        for (int c = 0; c < 3; ++c) {
            uint32_t va = tri[c];
            uint32_t vb = tri[(c + 1) % 3];
            if (va >= vertex_count || vb >= vertex_count) {
                continue;
            }
            if (va > vb) {
                std::swap(va, vb);
            }
            const TopoEdgeKey tk{va, vb};
            topo_edge_count[tk] += 1;
        }
    }

    std::unordered_map<WeldEdgeKey, std::vector<EdgeRecord>, WeldEdgeKeyHash> grouped_boundary_edges;
    grouped_boundary_edges.reserve(face_count);
    for (size_t f = 0; f < face_count; ++f) {
        const size_t base = f * 3;
        const uint32_t tri[3] = {mesh.indices[base + 0], mesh.indices[base + 1], mesh.indices[base + 2]};
        for (int c = 0; c < 3; ++c) {
            const uint32_t va = tri[c];
            const uint32_t vb = tri[(c + 1) % 3];
            if (va >= vertex_count || vb >= vertex_count) {
                continue;
            }

            uint32_t ta = va;
            uint32_t tb = vb;
            if (ta > tb) {
                std::swap(ta, tb);
            }
            const TopoEdgeKey tk{ta, tb};
            const auto it_topo = topo_edge_count.find(tk);
            if (it_topo == topo_edge_count.end() || it_topo->second != 1) {
                continue;
            }

            int64_t ga = weld_id[va];
            int64_t gb = weld_id[vb];
            glm::vec2 uv_a = ReadVertexUV(mesh, va);
            glm::vec2 uv_b = ReadVertexUV(mesh, vb);
            glm::vec3 pos_a = ReadVertexPosition(mesh, va);
            glm::vec3 pos_b = ReadVertexPosition(mesh, vb);
            if (ga > gb) {
                std::swap(ga, gb);
                std::swap(uv_a, uv_b);
                std::swap(pos_a, pos_b);
            }

            const WeldEdgeKey wk{ga, gb};
            grouped_boundary_edges[wk].push_back(EdgeRecord{
                static_cast<int>(f),
                uv_a,
                uv_b,
                pos_a,
                pos_b,
            });
        }
    }

    auto add_seam = [&](const EdgeRecord &rec) {
        out.line_points.push_back(rec.pos_a);
        out.line_points.push_back(rec.pos_b);
        out.seam_edges += 1;
        out.interior_seam_edges += 1;
    };

    for (const auto &kv : grouped_boundary_edges) {
        const std::vector<EdgeRecord> &records = kv.second;
        if (records.empty()) {
            continue;
        }
        if (records.size() == 1) {
            // True mesh boundary: not an internal UV seam.
            out.boundary_edges += 1;
            continue;
        }
        if (records.size() > 2) {
            out.nonmanifold_edges += 1;
        }

        bool seam = false;
        const EdgeRecord &ref = records[0];
        for (size_t i = 1; i < records.size(); ++i) {
            const EdgeRecord &cur = records[i];
            const glm::vec2 d_a = ref.uv_a - cur.uv_a;
            const glm::vec2 d_b = ref.uv_b - cur.uv_b;
            const bool same_a = glm::dot(d_a, d_a) <= uv_eps2;
            const bool same_b = glm::dot(d_b, d_b) <= uv_eps2;
            if (!(same_a && same_b)) {
                seam = true;
                break;
            }
        }
        if (seam) {
            add_seam(ref);
        }
    }

    return out;
}

glm::vec3 EdgeHeatmapColor(float t) {
    const float x = std::clamp(t, 0.0f, 1.0f);
    if (x < 0.5f) {
        const float k = x / 0.5f;
        return glm::vec3(0.08f + 0.92f * k, 0.90f + 0.08f * (1.0f - k), 0.10f);
    }
    const float k = (x - 0.5f) / 0.5f;
    return glm::vec3(1.0f, 0.95f - 0.80f * k, 0.10f);
}

float LabelToDeterministicScalar(int label) {
    if (label < 0) {
        return 0.0f;
    }
    uint32_t x = static_cast<uint32_t>(label + 1);
    x *= 2654435761u;
    x ^= (x >> 16u);
    const float v = static_cast<float>(x & 0x00FFFFFFu) / static_cast<float>(0x00FFFFFFu);
    return std::clamp(v, 0.0f, 1.0f);
}

struct ShaderProgram {
    GLuint program = 0;

    ShaderProgram() = default;
    ShaderProgram(const ShaderProgram &) = delete;
    ShaderProgram &operator=(const ShaderProgram &) = delete;
    ShaderProgram(ShaderProgram &&other) noexcept : program(other.program) { other.program = 0; }
    ShaderProgram &operator=(ShaderProgram &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        Destroy();
        program = other.program;
        other.program = 0;
        return *this;
    }

    void Destroy() {
        if (program != 0) {
            glDeleteProgram(program);
            program = 0;
        }
    }

    ~ShaderProgram() { Destroy(); }

    static std::optional<ShaderProgram> Create(const char *vs_source, const char *fs_source, std::string &error) {
        auto compile = [&](GLenum type, const char *src) -> GLuint {
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint ok = GL_FALSE;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (ok == GL_TRUE) {
                return shader;
            }

            GLint len = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
            std::string log(static_cast<size_t>(std::max(1, len)), '\0');
            glGetShaderInfoLog(shader, len, nullptr, log.data());
            error = log;
            glDeleteShader(shader);
            return 0;
        };

        const GLuint vs = compile(GL_VERTEX_SHADER, vs_source);
        if (vs == 0) {
            return std::nullopt;
        }

        const GLuint fs = compile(GL_FRAGMENT_SHADER, fs_source);
        if (fs == 0) {
            glDeleteShader(vs);
            return std::nullopt;
        }

        const GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);

        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint ok = GL_FALSE;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (ok != GL_TRUE) {
            GLint len = 0;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
            std::string log(static_cast<size_t>(std::max(1, len)), '\0');
            glGetProgramInfoLog(prog, len, nullptr, log.data());
            error = log;
            glDeleteProgram(prog);
            return std::nullopt;
        }

        ShaderProgram shader;
        shader.program = prog;
        return shader;
    }
};

struct MeshGPU {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLuint base_color_tex = 0;
    bool has_base_color_tex = false;
    GLsizei index_count = 0;

    void Destroy() {
        if (base_color_tex) {
            glDeleteTextures(1, &base_color_tex);
            base_color_tex = 0;
        }
        has_base_color_tex = false;
        if (ebo) {
            glDeleteBuffers(1, &ebo);
            ebo = 0;
        }
        if (vbo) {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        if (vao) {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
        index_count = 0;
    }

    ~MeshGPU() { Destroy(); }

    bool Upload(const faithc::viewer::MeshData &mesh, std::string &error) {
        Destroy();

        if (mesh.empty()) {
            error = "Cannot upload empty mesh";
            return false;
        }

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(float)), mesh.vertices.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(uint32_t)),
                     mesh.indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, faithc::viewer::MeshData::kVertexStride * sizeof(float),
                              reinterpret_cast<void *>(0));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, faithc::viewer::MeshData::kVertexStride * sizeof(float),
                              reinterpret_cast<void *>(3 * sizeof(float)));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, faithc::viewer::MeshData::kVertexStride * sizeof(float),
                              reinterpret_cast<void *>(6 * sizeof(float)));

        glBindVertexArray(0);

        if (mesh.has_base_color_texture && !mesh.texture_pixels.empty() && mesh.texture_width > 0 && mesh.texture_height > 0 &&
            mesh.texture_channels >= 1 && mesh.texture_channels <= 4) {
            GLenum data_format = GL_RGBA;
            GLenum internal_format = GL_RGBA8;
            if (mesh.texture_channels == 1) {
                data_format = GL_RED;
                internal_format = GL_R8;
            } else if (mesh.texture_channels == 2) {
                data_format = GL_RG;
                internal_format = GL_RG8;
            } else if (mesh.texture_channels == 3) {
                data_format = GL_RGB;
                internal_format = GL_RGB8;
            }

            glGenTextures(1, &base_color_tex);
            glBindTexture(GL_TEXTURE_2D, base_color_tex);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internal_format), mesh.texture_width, mesh.texture_height, 0,
                         data_format, GL_UNSIGNED_BYTE, mesh.texture_pixels.data());
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
            has_base_color_tex = true;
        }

        index_count = static_cast<GLsizei>(mesh.indices.size());
        return true;
    }

    [[nodiscard]] bool HasBaseColorTexture() const { return has_base_color_tex && base_color_tex != 0; }

    void Draw() const {
        if (vao == 0 || index_count <= 0) {
            return;
        }
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }
};

struct LineRenderer {
    GLuint vao = 0;
    GLuint vbo = 0;

    void Init() {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void *>(0));
        glBindVertexArray(0);
    }

    void Destroy() {
        if (vbo) {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        if (vao) {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
    }

    ~LineRenderer() { Destroy(); }

    void DrawLines(const std::vector<glm::vec3> &pts) {
        if (pts.empty()) {
            return;
        }

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(pts.size() * sizeof(glm::vec3)), pts.data(),
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(pts.size()));
        glBindVertexArray(0);
    }

    void DrawPoints(const std::vector<glm::vec3> &pts) {
        if (pts.empty()) {
            return;
        }

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(pts.size() * sizeof(glm::vec3)), pts.data(),
                     GL_DYNAMIC_DRAW);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pts.size()));
        glBindVertexArray(0);
    }
};

struct FaceHeatmapRenderer {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei vertex_count = 0;

    void Destroy() {
        if (vbo) {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        if (vao) {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
        vertex_count = 0;
    }

    ~FaceHeatmapRenderer() { Destroy(); }

    bool Upload(const faithc::viewer::MeshData &mesh, const std::vector<float> &face_scalar, std::string &error) {
        Destroy();
        if (mesh.empty()) {
            error = "Heatmap upload failed: mesh is empty";
            return false;
        }
        const size_t face_count = mesh.face_count();
        if (face_scalar.size() != face_count) {
            error = "Heatmap upload failed: scalar count does not match face count";
            return false;
        }

        std::vector<float> data;
        data.reserve(face_count * 3 * 4);
        for (size_t fi = 0; fi < face_count; ++fi) {
            const float s = std::clamp(face_scalar[fi], 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                const uint32_t vid = mesh.indices[fi * 3 + static_cast<size_t>(c)];
                const size_t base = static_cast<size_t>(vid) * faithc::viewer::MeshData::kVertexStride;
                data.push_back(mesh.vertices[base + 0]);
                data.push_back(mesh.vertices[base + 1]);
                data.push_back(mesh.vertices[base + 2]);
                data.push_back(s);
            }
        }

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(3 * sizeof(float)));
        glBindVertexArray(0);

        vertex_count = static_cast<GLsizei>(face_count * 3);
        return true;
    }

    void Draw() const {
        if (vao == 0 || vertex_count <= 0) {
            return;
        }
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertex_count);
        glBindVertexArray(0);
    }
};

struct OrbitCamera {
    glm::vec3 target = glm::vec3(0.0f);
    float yaw_deg = 45.0f;
    float pitch_deg = 22.0f;
    float distance = 3.0f;
    float scene_radius = 1.0f;

    [[nodiscard]] glm::vec3 Forward() const {
        const float yaw = glm::radians(yaw_deg);
        const float pitch = glm::radians(pitch_deg);
        glm::vec3 v;
        v.x = std::cos(pitch) * std::cos(yaw);
        v.y = std::sin(pitch);
        v.z = std::cos(pitch) * std::sin(yaw);
        return glm::normalize(v);
    }

    [[nodiscard]] glm::vec3 Position() const { return target - Forward() * distance; }

    [[nodiscard]] glm::mat4 ViewMatrix() const { return glm::lookAt(Position(), target, glm::vec3(0.0f, 1.0f, 0.0f)); }

    [[nodiscard]] glm::mat4 ProjMatrix(float aspect) const {
        // Keep depth range tight around the object to avoid depth precision loss
        // ("light leaking"/semi-transparent look on dense meshes).
        const float r = std::max(scene_radius, 1e-3f);
        float near_plane = distance - r * 1.35f;
        near_plane = std::clamp(near_plane, 0.01f, 50.0f);

        float far_plane = distance + r * 1.65f;
        far_plane = std::max(far_plane, near_plane + std::max(10.0f, r * 2.0f));
        far_plane = std::min(far_plane, 20000.0f);
        return glm::perspective(glm::radians(45.0f), std::max(aspect, 0.1f), near_plane, far_plane);
    }

    void FitToBounds(const float min_bound[3], const float max_bound[3]) {
        const glm::vec3 min_v(min_bound[0], min_bound[1], min_bound[2]);
        const glm::vec3 max_v(max_bound[0], max_bound[1], max_bound[2]);

        target = 0.5f * (min_v + max_v);
        scene_radius = std::max(glm::length(max_v - min_v) * 0.5f, 1e-3f);
        distance = std::max(scene_radius * 2.4f, 0.25f);
    }
};

struct LaunchOptions {
    fs::path mesh;
    fs::path repo_root;
    std::string python_bin;
    fs::path bridge_script;
    fs::path work_dir;
};

void PrintUsage() {
    std::cout << "faithc_viewer [--mesh <path>] [--repo-root <path>] [--python-bin <path>]"
              << " [--bridge-script <path>] [--work-dir <path>]\n";
}

bool ParseArgs(int argc, char **argv, LaunchOptions &opt, std::string &error) {
    opt.repo_root = fs::current_path();
    opt.python_bin = "python";
    opt.bridge_script = opt.repo_root / "tools/preview/run_faithc_preview.py";
    opt.work_dir = opt.repo_root / "experiments/runs/preview_tmp";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char *name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                error = std::string("Missing value for ") + name;
                return std::nullopt;
            }
            ++i;
            return std::string(argv[i]);
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage();
            std::exit(0);
        }
        if (arg == "--mesh") {
            auto val = next_value("--mesh");
            if (!val) {
                return false;
            }
            opt.mesh = fs::path(*val);
            continue;
        }
        if (arg == "--repo-root") {
            auto val = next_value("--repo-root");
            if (!val) {
                return false;
            }
            opt.repo_root = fs::path(*val);
            continue;
        }
        if (arg == "--python-bin") {
            auto val = next_value("--python-bin");
            if (!val) {
                return false;
            }
            opt.python_bin = *val;
            continue;
        }
        if (arg == "--bridge-script") {
            auto val = next_value("--bridge-script");
            if (!val) {
                return false;
            }
            opt.bridge_script = fs::path(*val);
            continue;
        }
        if (arg == "--work-dir") {
            auto val = next_value("--work-dir");
            if (!val) {
                return false;
            }
            opt.work_dir = fs::path(*val);
            continue;
        }

        error = "Unknown argument: " + arg;
        return false;
    }

    if (opt.repo_root.empty()) {
        opt.repo_root = fs::current_path();
    }
    if (opt.work_dir.empty()) {
        opt.work_dir = opt.repo_root / "experiments/runs/preview_tmp";
    }
    if (opt.bridge_script.empty()) {
        opt.bridge_script = opt.repo_root / "tools/preview/run_faithc_preview.py";
    }

    return true;
}

struct FaithCJobConfig {
    fs::path input_mesh;
    fs::path output_mesh;
    fs::path status_json;
    int resolution = 128;
    int min_level = -1;
    std::string tri_mode = "auto";
    std::string uv_mode = "method2";
    std::string uv_seam_strategy = "legacy";
    std::string uv_solve_backend = "auto";
    bool uv_island_guard = true;
    std::string uv_island_guard_mode = "soft";
    float uv_island_guard_confidence_min = 0.55f;
    bool uv_island_guard_allow_unknown = false;
    int uv_batch_size = 200000;
    float uv_m2_outlier_sigma = 4.0f;
    float uv_m2_outlier_quantile = 0.95f;
    int uv_m2_min_samples_per_face = 2;
    float uv_m2_face_weight_floor = 1e-6f;
    std::string uv_m2_anchor_mode = "component_minimal";
    int uv_m2_anchor_points_per_component = 4;
    int uv_m2_irls_iters = 2;
    float uv_m2_huber_delta = 3.0f;
    std::string uv_m2_laplacian_mode = "cotan";
    std::string uv_m2_system_cond_estimate = "diag_ratio";
    float margin = 0.05f;
    std::string python_bin = "python";
    fs::path bridge_script;
};

faithc::viewer::FaithCJobResult RunFaithCJob(const FaithCJobConfig &cfg) {

    std::error_code ec;
    fs::create_directories(cfg.output_mesh.parent_path(), ec);
    fs::create_directories(cfg.status_json.parent_path(), ec);

    std::ostringstream cmd;
    cmd << ShellQuote(cfg.python_bin) << " " << ShellQuote(cfg.bridge_script.string()) << " --input "
        << ShellQuote(cfg.input_mesh.string()) << " --output " << ShellQuote(cfg.output_mesh.string()) << " --status "
        << ShellQuote(cfg.status_json.string()) << " --resolution " << cfg.resolution << " --tri-mode "
        << ShellQuote(cfg.tri_mode) << " --margin " << cfg.margin << " --min-level " << cfg.min_level
        << " --project-uv --uv-mode " << ShellQuote(cfg.uv_mode) << " --uv-seam-strategy "
        << ShellQuote(cfg.uv_seam_strategy) << " --uv-solve-backend " << ShellQuote(cfg.uv_solve_backend) << " "
        << (cfg.uv_island_guard ? "--uv-island-guard" : "--no-uv-island-guard") << " --uv-island-guard-mode "
        << ShellQuote(cfg.uv_island_guard_mode) << " --uv-island-guard-confidence-min "
        << cfg.uv_island_guard_confidence_min << " "
        << (cfg.uv_island_guard_allow_unknown ? "--uv-island-guard-allow-unknown"
                                              : "--no-uv-island-guard-allow-unknown")
        << " --uv-batch-size " << cfg.uv_batch_size << " --uv-m2-outlier-sigma " << cfg.uv_m2_outlier_sigma
        << " --uv-m2-outlier-quantile " << cfg.uv_m2_outlier_quantile << " --uv-m2-min-samples-per-face "
        << cfg.uv_m2_min_samples_per_face << " --uv-m2-face-weight-floor " << cfg.uv_m2_face_weight_floor
        << " --uv-m2-anchor-mode " << ShellQuote(cfg.uv_m2_anchor_mode) << " --uv-m2-anchor-points-per-component "
        << cfg.uv_m2_anchor_points_per_component << " --uv-m2-irls-iters " << cfg.uv_m2_irls_iters
        << " --uv-m2-huber-delta " << cfg.uv_m2_huber_delta << " --uv-m2-laplacian-mode "
        << ShellQuote(cfg.uv_m2_laplacian_mode) << " --uv-m2-system-cond-estimate "
        << ShellQuote(cfg.uv_m2_system_cond_estimate);

    const int rc = std::system(cmd.str().c_str());
    return faithc::viewer::ParseJobResultFromStatusJson(cfg.status_json, cfg.output_mesh, rc);
}

class ViewerApp {
public:
    explicit ViewerApp(LaunchOptions options) : options_(std::move(options)) {
        if (!options_.mesh.empty()) {
            initial_mesh_ = fs::absolute(options_.mesh);
            current_source_mesh_ = initial_mesh_;
        }
    }

    bool Init(std::string &error) {
        if (!glfwInit()) {
            error = "Failed to initialize GLFW";
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "FaithC OpenGL Previewer", nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            error = "Failed to create GLFW window";
            return false;
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            error = "Failed to initialize GLAD";
            return false;
        }

        glfwSetWindowUserPointer(window_, this);
        glfwSetCursorPosCallback(window_, &ViewerApp::CursorPosCallback);
        glfwSetMouseButtonCallback(window_, &ViewerApp::MouseButtonCallback);
        glfwSetScrollCallback(window_, &ViewerApp::ScrollCallback);
        glfwSetFramebufferSizeCallback(window_, &ViewerApp::FramebufferSizeCallback);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        const char *mesh_vs = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aUV;
            uniform mat4 u_mvp;
            uniform mat4 u_model;
            out vec3 vNormal;
            out vec3 vWorldPos;
            out vec2 vUV;
            void main() {
                vec4 world = u_model * vec4(aPos, 1.0);
                vWorldPos = world.xyz;
                vNormal = mat3(transpose(inverse(u_model))) * aNormal;
                vUV = aUV;
                gl_Position = u_mvp * vec4(aPos, 1.0);
            }
        )";

        const char *mesh_fs = R"(
            #version 330 core
            in vec3 vNormal;
            in vec3 vWorldPos;
            in vec2 vUV;
            uniform vec3 u_camera_pos;
            uniform vec3 u_base_color;
            uniform sampler2D u_base_tex;
            uniform int u_use_texture;
            out vec4 FragColor;
            void main() {
                vec3 base_col = u_base_color;
                if (u_use_texture == 1) {
                    base_col = texture(u_base_tex, vUV).rgb;
                }
                vec3 n = normalize(vNormal);
                vec3 l = normalize(vec3(0.35, 0.8, 0.22));
                float diff = max(dot(n, l), 0.0);
                vec3 v = normalize(u_camera_pos - vWorldPos);
                vec3 h = normalize(l + v);
                float spec = pow(max(dot(n, h), 0.0), 48.0);
                vec3 ambient = 0.18 * base_col;
                vec3 col = ambient + (0.82 * diff) * base_col + 0.18 * spec * vec3(1.0);
                FragColor = vec4(col, 1.0);
            }
        )";

        const char *line_vs = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * vec4(aPos, 1.0);
            }
        )";

        const char *line_fs = R"(
            #version 330 core
            uniform vec3 u_color;
            out vec4 FragColor;
            void main() {
                FragColor = vec4(u_color, 1.0);
            }
        )";

        const char *heatmap_vs = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in float aScalar;
            uniform mat4 u_mvp;
            out float vScalar;
            void main() {
                vScalar = clamp(aScalar, 0.0, 1.0);
                gl_Position = u_mvp * vec4(aPos, 1.0);
            }
        )";

        const char *heatmap_fs = R"(
            #version 330 core
            in float vScalar;
            uniform float u_alpha;
            out vec4 FragColor;

            vec3 heatmap(float t) {
                t = clamp(t, 0.0, 1.0);
                if (t < 0.25) {
                    return mix(vec3(0.0, 0.0, 0.0), vec3(0.05, 0.20, 1.0), t / 0.25);
                }
                if (t < 0.50) {
                    return mix(vec3(0.05, 0.20, 1.0), vec3(0.0, 0.95, 1.0), (t - 0.25) / 0.25);
                }
                if (t < 0.75) {
                    return mix(vec3(0.0, 0.95, 1.0), vec3(1.0, 0.95, 0.0), (t - 0.50) / 0.25);
                }
                return mix(vec3(1.0, 0.95, 0.0), vec3(1.0, 0.12, 0.0), (t - 0.75) / 0.25);
            }

            void main() {
                vec3 c = heatmap(vScalar);
                FragColor = vec4(c, clamp(u_alpha, 0.0, 1.0));
            }
        )";

        auto mesh_shader = ShaderProgram::Create(mesh_vs, mesh_fs, error);
        if (!mesh_shader) {
            return false;
        }
        mesh_shader_ = std::move(*mesh_shader);

        auto line_shader = ShaderProgram::Create(line_vs, line_fs, error);
        if (!line_shader) {
            return false;
        }
        line_shader_ = std::move(*line_shader);

        auto heatmap_shader = ShaderProgram::Create(heatmap_vs, heatmap_fs, error);
        if (!heatmap_shader) {
            return false;
        }
        heatmap_shader_ = std::move(*heatmap_shader);

        line_renderer_.Init();

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);

        if (!initial_mesh_.empty()) {
            std::string load_error;
            if (!LoadMesh(initial_mesh_, load_error)) {
                status_text_ = "Initial mesh load failed: " + load_error;
            }
        }

        const std::string init_mesh = initial_mesh_.empty() ? std::string() : initial_mesh_.string();
        SetPathInput(init_mesh);
        return true;
    }

    ~ViewerApp() { Shutdown(); }

    int Run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            PollFaithCJob();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            DrawUI();

            int width = 1;
            int height = 1;
            glfwGetFramebufferSize(window_, &width, &height);
            const float aspect = static_cast<float>(width) / static_cast<float>(std::max(1, height));

            const glm::mat4 view = camera_.ViewMatrix();
            const glm::mat4 proj = camera_.ProjMatrix(aspect);
            const glm::mat4 model = glm::mat4(1.0f);
            const glm::mat4 mvp = proj * view * model;

            glViewport(0, 0, width, height);
            glClearColor(background_color_.x, background_color_.y, background_color_.z, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            DrawScene(mvp, model);

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window_);
        }

        return 0;
    }

private:
    [[nodiscard]] bool IsSamePath(const fs::path &a, const fs::path &b) const {
        std::error_code ec_eq;
        if (fs::equivalent(a, b, ec_eq) && !ec_eq) {
            return true;
        }
        std::error_code ec_a;
        std::error_code ec_b;
        const fs::path wa = fs::weakly_canonical(a, ec_a);
        const fs::path wb = fs::weakly_canonical(b, ec_b);
        if (!ec_a && !ec_b) {
            return wa == wb;
        }
        return fs::absolute(a).lexically_normal() == fs::absolute(b).lexically_normal();
    }

    [[nodiscard]] bool IsFaithCOutputPath(const fs::path &path) const {
        std::error_code ec;
        const fs::path abs = fs::absolute(path, ec);
        const fs::path work = fs::absolute(options_.work_dir, ec);
        if (ec) {
            return false;
        }
        if (abs.parent_path() != work) {
            return false;
        }
        const std::string name = abs.filename().string();
        return name.rfind("preview_low_", 0) == 0;
    }

    void CacheSourceTexture(const faithc::viewer::MeshData &mesh) {
        source_tex_valid_ = mesh.has_base_color_texture && !mesh.texture_pixels.empty() && mesh.texture_width > 0 &&
                            mesh.texture_height > 0 && mesh.texture_channels >= 1 && mesh.texture_channels <= 4;
        if (!source_tex_valid_) {
            source_tex_width_ = 0;
            source_tex_height_ = 0;
            source_tex_channels_ = 0;
            source_tex_pixels_.clear();
            return;
        }
        source_tex_width_ = mesh.texture_width;
        source_tex_height_ = mesh.texture_height;
        source_tex_channels_ = mesh.texture_channels;
        source_tex_pixels_ = mesh.texture_pixels;
    }

    static void CursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
        auto *self = static_cast<ViewerApp *>(glfwGetWindowUserPointer(window));
        if (!self) {
            return;
        }

        if (self->first_mouse_) {
            self->last_x_ = xpos;
            self->last_y_ = ypos;
            self->first_mouse_ = false;
            return;
        }

        const double dx = xpos - self->last_x_;
        const double dy = ypos - self->last_y_;
        self->last_x_ = xpos;
        self->last_y_ = ypos;

        if (ImGui::GetIO().WantCaptureMouse) {
            return;
        }

        if (self->mouse_rotating_) {
            self->camera_.yaw_deg += static_cast<float>(dx) * 0.25f;
            self->camera_.pitch_deg -= static_cast<float>(dy) * 0.25f;
            self->camera_.pitch_deg = std::max(-89.0f, std::min(89.0f, self->camera_.pitch_deg));
        }

        if (self->mouse_panning_) {
            const glm::vec3 fwd = self->camera_.Forward();
            glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)));
            glm::vec3 up = glm::normalize(glm::cross(right, fwd));
            const float scale = 0.0022f * std::max(0.25f, self->camera_.distance);
            self->camera_.target += right * static_cast<float>(-dx * scale);
            self->camera_.target += up * static_cast<float>(dy * scale);
        }
    }

    static void MouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/) {
        auto *self = static_cast<ViewerApp *>(glfwGetWindowUserPointer(window));
        if (!self) {
            return;
        }

        if (ImGui::GetIO().WantCaptureMouse) {
            return;
        }

        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            self->mouse_rotating_ = (action == GLFW_PRESS);
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            self->mouse_panning_ = (action == GLFW_PRESS);
        }
    }

    static void ScrollCallback(GLFWwindow *window, double /*xoffset*/, double yoffset) {
        auto *self = static_cast<ViewerApp *>(glfwGetWindowUserPointer(window));
        if (!self) {
            return;
        }
        if (ImGui::GetIO().WantCaptureMouse) {
            return;
        }

        const float scale = std::exp(static_cast<float>(-yoffset * 0.11));
        self->camera_.distance = std::max(0.05f, self->camera_.distance * scale);
    }

    static void FramebufferSizeCallback(GLFWwindow * /*window*/, int w, int h) { glViewport(0, 0, w, h); }

    void Shutdown() {
        if (window_ != nullptr) {
            SetCrashStage("Shutdown");
            // Release GPU resources while a valid OpenGL context still exists.
            glfwMakeContextCurrent(window_);
            face_heatmap_renderer_.Destroy();
            semantic_heatmap_renderer_.Destroy();
            line_renderer_.Destroy();
            mesh_gpu_.Destroy();
            mesh_shader_.Destroy();
            line_shader_.Destroy();
            heatmap_shader_.Destroy();

            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            glfwDestroyWindow(window_);
            window_ = nullptr;
            glfwTerminate();
        }
    }

    void SetPathInput(const std::string &path) {
        std::fill(path_input_.begin(), path_input_.end(), 0);
        const size_t n = std::min(path.size(), path_input_.size() - 1);
        std::memcpy(path_input_.data(), path.data(), n);
    }

    std::string GetPathInput() const { return std::string(path_input_.data()); }

    bool LoadMesh(const fs::path &path, std::string &error) {
        SetCrashStage("LoadMesh:begin");
        faithc::viewer::MeshData mesh;
        if (!faithc::viewer::LoadMeshFile(path, mesh, error)) {
            return false;
        }

        const fs::path abs_path = fs::absolute(path);
        const bool is_source_mesh = !current_source_mesh_.empty() && IsSamePath(abs_path, current_source_mesh_);
        used_source_texture_fallback_ = false;

        if (is_source_mesh) {
            CacheSourceTexture(mesh);
        } else if (IsFaithCOutputPath(abs_path) && mesh.has_uv && source_tex_valid_) {
            const bool missing_texture = !mesh.has_base_color_texture || mesh.texture_pixels.empty();
            const bool tiny_placeholder = mesh.texture_width <= 4 || mesh.texture_height <= 4;
            if (missing_texture || tiny_placeholder) {
                mesh.has_base_color_texture = true;
                mesh.texture_width = source_tex_width_;
                mesh.texture_height = source_tex_height_;
                mesh.texture_channels = source_tex_channels_;
                mesh.texture_pixels = source_tex_pixels_;
                used_source_texture_fallback_ = true;
            }
        }

        UVSeamOverlay seam_overlay;
        const bool is_generated_output = IsFaithCOutputPath(abs_path);
        if (mesh.face_count() <= kSeamOverlayFaceLimit) {
            SetCrashStage("LoadMesh:build_uv_seams");
            if (is_generated_output) {
                // For generated low mesh, use UV-boundary audit from final UVs instead of
                // topology-route intermediates to avoid index drift visualization artifacts.
                seam_overlay = BuildUVBoundaryAuditOverlay(
                    mesh,
                    static_cast<double>(uv_seam_position_eps_),
                    static_cast<double>(uv_seam_uv_eps_)
                );
            } else {
                seam_overlay = BuildUVSeamOverlay(
                    mesh,
                    static_cast<double>(uv_seam_position_eps_),
                    static_cast<double>(uv_seam_uv_eps_)
                );
            }
        } else {
            seam_overlay = UVSeamOverlay{};
            seam_overlay.has_uv = false;
        }
        ClearAcceptedSampleHeatmap();
        ClearSemanticHeatmap();
        ClearClosureValidationSummary();

        SetCrashStage("LoadMesh:upload_gpu");
        if (!mesh_gpu_.Upload(mesh, error)) {
            return false;
        }

        mesh_data_ = std::move(mesh);
        display_uv_seams_ = seam_overlay;
        if (is_source_mesh) {
            source_uv_seams_ = seam_overlay;
        } else if (is_generated_output) {
            low_generated_uv_seams_ = seam_overlay;
        }
        mesh_loaded_ = true;
        display_mesh_path_ = abs_path;
        use_basecolor_texture_ = mesh_data_.has_uv && mesh_data_.has_base_color_texture;

        if (current_source_mesh_.empty()) {
            current_source_mesh_ = display_mesh_path_;
            source_uv_seams_ = display_uv_seams_;
        }

        camera_.FitToBounds(mesh_data_.min_bound, mesh_data_.max_bound);
        PushRecent(path);
        SetCrashStage("LoadMesh:done");
        return true;
    }

    void PushRecent(const fs::path &path) {
        const std::string p = fs::absolute(path).string();
        recent_files_.erase(std::remove(recent_files_.begin(), recent_files_.end(), p), recent_files_.end());
        recent_files_.insert(recent_files_.begin(), p);
        if (recent_files_.size() > 8) {
            recent_files_.resize(8);
        }
    }

    void ClearAcceptedSampleHeatmap() {
        face_heatmap_renderer_.Destroy();
        accepted_sample_heatmap_available_ = false;
        accepted_sample_counts_raw_.clear();
        accepted_sample_face_count_ = 0;
        accepted_sample_nonzero_faces_ = 0;
        accepted_sample_max_count_ = 0;
        accepted_sample_heatmap_source_.clear();
        accepted_sample_heatmap_status_.clear();
    }

    void ClearSemanticHeatmap() {
        semantic_heatmap_renderer_.Destroy();
        semantic_heatmap_available_ = false;
        semantic_labels_raw_.clear();
        semantic_face_count_ = 0;
        semantic_unknown_face_count_ = 0;
        semantic_unique_label_count_ = 0;
        semantic_heatmap_source_.clear();
        semantic_heatmap_status_.clear();
    }

    void ClearClosureValidationSummary() {
        closure_summary_ = ClosureValidationSummary{};
        closure_sidecar_source_.clear();
        closure_validation_status_.clear();
        low_algorithm_uv_seams_ = UVSeamOverlay{};
        low_algorithm_uv_seams_.has_uv = false;
    }

    bool BuildAcceptedSampleHeatmapFromCounts(const std::vector<int> &accepted_counts, std::string &error) {
        if (!mesh_loaded_) {
            error = "Heatmap build failed: mesh not loaded";
            return false;
        }
        const size_t face_count = mesh_data_.face_count();
        if (accepted_counts.size() != face_count) {
            std::ostringstream oss;
            oss << "Heatmap build failed: count length mismatch (" << accepted_counts.size() << " vs face_count " << face_count
                << ")";
            error = oss.str();
            return false;
        }
        int max_count = 0;
        int nonzero = 0;
        for (int v : accepted_counts) {
            max_count = std::max(max_count, std::max(0, v));
            if (v > 0) {
                ++nonzero;
            }
        }
        const float denom = accepted_sample_heatmap_log_scale_ ? std::log1p(static_cast<float>(max_count))
                                                               : static_cast<float>(std::max(1, max_count));
        std::vector<float> scalar(face_count, 0.0f);
        for (size_t i = 0; i < face_count; ++i) {
            const float c = static_cast<float>(std::max(0, accepted_counts[i]));
            float t = 0.0f;
            if (max_count > 0) {
                if (accepted_sample_heatmap_log_scale_) {
                    t = std::log1p(c) / std::max(denom, 1e-6f);
                } else {
                    t = c / std::max(denom, 1e-6f);
                }
            }
            scalar[i] = std::clamp(t, 0.0f, 1.0f);
        }

        if (!face_heatmap_renderer_.Upload(mesh_data_, scalar, error)) {
            return false;
        }
        accepted_sample_counts_raw_ = accepted_counts;
        accepted_sample_heatmap_available_ = true;
        accepted_sample_face_count_ = static_cast<int>(face_count);
        accepted_sample_nonzero_faces_ = nonzero;
        accepted_sample_max_count_ = max_count;
        return true;
    }

    bool LoadAcceptedSampleHeatmapFromResult(const FaithCJobResult &result, std::string &error) {
        ClearAcceptedSampleHeatmap();
        if (result.uv_m2_face_sample_counts_path.empty()) {
            error = "No accepted-sample sidecar path in status JSON";
            return false;
        }
        fs::path sidecar = fs::path(result.uv_m2_face_sample_counts_path);
        if (sidecar.is_relative()) {
            sidecar = result.status_json.parent_path() / sidecar;
        }
        std::error_code ec;
        sidecar = fs::absolute(sidecar, ec);
        if (ec || !fs::exists(sidecar)) {
            error = "Accepted-sample sidecar not found: " + sidecar.string();
            return false;
        }

        json payload;
        try {
            std::ifstream in(sidecar);
            in >> payload;
        } catch (const std::exception &e) {
            error = std::string("Failed to parse accepted-sample sidecar: ") + e.what();
            return false;
        }
        const auto it = payload.find("accepted");
        if (it == payload.end() || !it->is_array()) {
            error = "Accepted-sample sidecar missing 'accepted' array";
            return false;
        }

        std::vector<int> accepted;
        accepted.reserve(it->size());
        for (const auto &v : *it) {
            if (v.is_number_integer()) {
                accepted.push_back(v.get<int>());
            } else if (v.is_number_float()) {
                accepted.push_back(static_cast<int>(std::llround(v.get<double>())));
            } else {
                accepted.push_back(0);
            }
        }

        if (!BuildAcceptedSampleHeatmapFromCounts(accepted, error)) {
            return false;
        }
        accepted_sample_heatmap_source_ = sidecar.string();
        std::ostringstream oss;
        oss << "Accepted-sample heatmap loaded: faces=" << accepted_sample_face_count_
            << ", nonzero=" << accepted_sample_nonzero_faces_ << ", max=" << accepted_sample_max_count_;
        accepted_sample_heatmap_status_ = oss.str();
        return true;
    }

    bool BuildSemanticHeatmapFromLabels(const std::vector<int> &labels, std::string &error) {
        if (!mesh_loaded_) {
            error = "Semantic heatmap build failed: mesh not loaded";
            return false;
        }
        const size_t face_count = mesh_data_.face_count();
        if (labels.size() != face_count) {
            std::ostringstream oss;
            oss << "Semantic heatmap build failed: label length mismatch (" << labels.size() << " vs face_count "
                << face_count << ")";
            error = oss.str();
            return false;
        }
        std::vector<float> scalar(face_count, 0.0f);
        int unknown = 0;
        std::unordered_set<int> uniq;
        uniq.reserve(face_count / 4 + 8);
        for (size_t i = 0; i < face_count; ++i) {
            const int label = labels[i];
            if (label < 0) {
                scalar[i] = 0.0f;
                unknown += 1;
                continue;
            }
            scalar[i] = LabelToDeterministicScalar(label);
            uniq.insert(label);
        }
        if (!semantic_heatmap_renderer_.Upload(mesh_data_, scalar, error)) {
            return false;
        }
        semantic_labels_raw_ = labels;
        semantic_heatmap_available_ = true;
        semantic_face_count_ = static_cast<int>(face_count);
        semantic_unknown_face_count_ = unknown;
        semantic_unique_label_count_ = static_cast<int>(uniq.size());
        return true;
    }

    bool LoadClosureValidationFromResult(const FaithCJobResult &result, std::string &error) {
        ClearSemanticHeatmap();
        ClearClosureValidationSummary();

        fs::path sidecar = result.status_json.parent_path() / (result.status_json.stem().string() + ".uv_closure_validation.json");
        std::error_code ec;
        sidecar = fs::absolute(sidecar, ec);
        if (ec || !fs::exists(sidecar)) {
            error = "Closure-validation sidecar not found: " + sidecar.string();
            return false;
        }

        json payload;
        try {
            std::ifstream in(sidecar);
            in >> payload;
        } catch (const std::exception &e) {
            error = std::string("Failed to parse closure-validation sidecar: ") + e.what();
            return false;
        }

        if (!payload.contains("semantic_labels") || !payload["semantic_labels"].is_array()) {
            error = "Closure-validation sidecar missing 'semantic_labels' array";
            return false;
        }
        std::vector<int> labels;
        labels.reserve(payload["semantic_labels"].size());
        for (const auto &v : payload["semantic_labels"]) {
            if (v.is_number_integer()) {
                labels.push_back(v.get<int>());
            } else if (v.is_number_float()) {
                labels.push_back(static_cast<int>(std::llround(v.get<double>())));
            } else {
                labels.push_back(-1);
            }
        }
        if (!BuildSemanticHeatmapFromLabels(labels, error)) {
            return false;
        }

        low_algorithm_uv_seams_.has_uv = mesh_data_.has_uv;
        if (payload.contains("seam_edges") && payload["seam_edges"].is_array()) {
            for (const auto &e : payload["seam_edges"]) {
                if (!e.is_array() || e.size() != 2 || !e[0].is_number_integer() || !e[1].is_number_integer()) {
                    continue;
                }
                const int a = e[0].get<int>();
                const int b = e[1].get<int>();
                if (a < 0 || b < 0 || static_cast<size_t>(a) >= mesh_data_.vertex_count() ||
                    static_cast<size_t>(b) >= mesh_data_.vertex_count()) {
                    continue;
                }
                low_algorithm_uv_seams_.line_points.push_back(ReadVertexPosition(mesh_data_, static_cast<size_t>(a)));
                low_algorithm_uv_seams_.line_points.push_back(ReadVertexPosition(mesh_data_, static_cast<size_t>(b)));
                low_algorithm_uv_seams_.seam_edges += 1;
                low_algorithm_uv_seams_.interior_seam_edges += 1;
            }
        }

        auto parse_bool = [](const json &obj, const char *key, bool fallback) -> bool {
            if (!obj.contains(key) || obj[key].is_null()) {
                return fallback;
            }
            const auto &v = obj[key];
            if (v.is_boolean()) {
                return v.get<bool>();
            }
            if (v.is_number()) {
                return v.get<double>() != 0.0;
            }
            return fallback;
        };
        auto parse_int = [](const json &obj, const char *key, int fallback) -> int {
            if (!obj.contains(key) || obj[key].is_null()) {
                return fallback;
            }
            const auto &v = obj[key];
            if (v.is_number_integer()) {
                return v.get<int>();
            }
            if (v.is_number_float()) {
                return static_cast<int>(std::llround(v.get<double>()));
            }
            return fallback;
        };
        auto parse_double = [](const json &obj, const char *key, double fallback) -> double {
            if (!obj.contains(key) || obj[key].is_null()) {
                return fallback;
            }
            const auto &v = obj[key];
            if (v.is_number()) {
                return v.get<double>();
            }
            return fallback;
        };
        auto parse_string = [](const json &obj, const char *key, const std::string &fallback) -> std::string {
            if (!obj.contains(key) || obj[key].is_null()) {
                return fallback;
            }
            const auto &v = obj[key];
            if (v.is_string()) {
                return v.get<std::string>();
            }
            return fallback;
        };

        const json *summary = nullptr;
        if (payload.contains("summary") && payload["summary"].is_object()) {
            summary = &payload["summary"];
        } else if (payload.is_object()) {
            summary = &payload;
        }
        if (summary != nullptr) {
            closure_summary_.available = true;
            closure_summary_.partition_has_leakage =
                parse_bool(*summary, "partition_has_leakage", false);
            closure_summary_.seam_topology_valid = parse_bool(*summary, "seam_topology_valid", false);
            closure_summary_.partition_mixed_components =
                parse_int(*summary, "partition_mixed_components", -1);
            closure_summary_.partition_label_split_count =
                parse_int(*summary, "partition_label_split_count", -1);
            closure_summary_.seam_components = parse_int(*summary, "seam_components", -1);
            closure_summary_.seam_loops_closed = parse_int(*summary, "seam_loops_closed", -1);
            closure_summary_.seam_components_open = parse_int(*summary, "seam_components_open", -1);
            closure_summary_.low_island_count = parse_int(*summary, "low_island_count", -1);
            closure_summary_.high_island_count = parse_int(*summary, "high_island_count", -1);
            closure_summary_.semantic_unknown_faces = parse_int(*summary, "semantic_unknown_faces", -1);
            closure_summary_.uv_bbox_iou_mean = parse_double(*summary, "uv_bbox_iou_mean", -1.0);
            closure_summary_.uv_overlap_ratio = parse_double(*summary, "uv_overlap_ratio", -1.0);
            closure_summary_.uv_stretch_p95 = parse_double(*summary, "uv_stretch_p95", -1.0);
            closure_summary_.uv_stretch_p99 = parse_double(*summary, "uv_stretch_p99", -1.0);
            closure_summary_.uv_png_path = parse_string(*summary, "uv_validation_png", "");
        }
        if (closure_summary_.uv_png_path.empty()) {
            closure_summary_.uv_png_path = parse_string(payload, "uv_validation_png", "");
        }

        semantic_heatmap_source_ = sidecar.string();
        closure_sidecar_source_ = sidecar.string();
        std::ostringstream oss;
        oss << "Closure validation loaded: semantic_faces=" << semantic_face_count_
            << ", unknown=" << semantic_unknown_face_count_ << ", seam_edges="
            << low_algorithm_uv_seams_.seam_edges;
        semantic_heatmap_status_ = oss.str();
        closure_validation_status_ = oss.str();
        return true;
    }

    const UVSeamOverlay *GetActiveUVSeamOverlay(const char **label = nullptr) const {
        if (label) {
            *label = "none";
        }
        if (!show_uv_island_seams_ || !mesh_loaded_) {
            return nullptr;
        }
        if (!current_source_mesh_.empty() && IsSamePath(display_mesh_path_, current_source_mesh_)) {
            if (label) {
                *label = "high/source mesh seams";
            }
            return source_uv_seams_.has_uv ? &source_uv_seams_ : nullptr;
        }
        if (IsFaithCOutputPath(display_mesh_path_)) {
            if (label) {
                *label = "low/generated seams";
            }
            if (low_algorithm_uv_seams_.has_uv && !low_algorithm_uv_seams_.line_points.empty()) {
                if (label) {
                    *label = "low/algorithm seams";
                }
                return &low_algorithm_uv_seams_;
            }
            if (low_generated_uv_seams_.has_uv) {
                return &low_generated_uv_seams_;
            }
        }
        if (label) {
            *label = "display mesh seams";
        }
        return display_uv_seams_.has_uv ? &display_uv_seams_ : nullptr;
    }

    void DrawScene(const glm::mat4 &mvp, const glm::mat4 &model) {
        SetCrashStage("DrawScene");
        if (mesh_loaded_) {
            if (enable_backface_culling_) {
                glEnable(GL_CULL_FACE);
            } else {
                glDisable(GL_CULL_FACE);
            }
            glUseProgram(mesh_shader_.program);
            glUniformMatrix4fv(glGetUniformLocation(mesh_shader_.program, "u_mvp"), 1, GL_FALSE, &mvp[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(mesh_shader_.program, "u_model"), 1, GL_FALSE, &model[0][0]);
            const glm::vec3 cam = camera_.Position();
            glUniform3f(glGetUniformLocation(mesh_shader_.program, "u_camera_pos"), cam.x, cam.y, cam.z);
            glUniform3f(glGetUniformLocation(mesh_shader_.program, "u_base_color"), mesh_color_.x, mesh_color_.y, mesh_color_.z);
            const bool use_texture =
                use_basecolor_texture_ && mesh_data_.has_uv && mesh_gpu_.HasBaseColorTexture() && mesh_data_.has_base_color_texture;
            glUniform1i(glGetUniformLocation(mesh_shader_.program, "u_use_texture"), use_texture ? 1 : 0);
            if (use_texture) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mesh_gpu_.base_color_tex);
                glUniform1i(glGetUniformLocation(mesh_shader_.program, "u_base_tex"), 0);
            }

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            mesh_gpu_.Draw();

            if (show_wireframe_) {
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(-1.0f, -1.0f);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glUniform1i(glGetUniformLocation(mesh_shader_.program, "u_use_texture"), 0);
                glUniform3f(glGetUniformLocation(mesh_shader_.program, "u_base_color"), 0.08f, 0.1f, 0.12f);
                mesh_gpu_.Draw();
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }

            if (use_texture) {
                glBindTexture(GL_TEXTURE_2D, 0);
            }
        }

        if (show_semantic_heatmap_ && semantic_heatmap_available_) {
            glUseProgram(heatmap_shader_.program);
            glUniformMatrix4fv(glGetUniformLocation(heatmap_shader_.program, "u_mvp"), 1, GL_FALSE, &mvp[0][0]);
            glUniform1f(glGetUniformLocation(heatmap_shader_.program, "u_alpha"), semantic_heatmap_alpha_);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(-0.8f, -0.8f);
            semantic_heatmap_renderer_.Draw();
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDisable(GL_BLEND);
        }

        if (show_accepted_sample_heatmap_ && accepted_sample_heatmap_available_) {
            glUseProgram(heatmap_shader_.program);
            glUniformMatrix4fv(glGetUniformLocation(heatmap_shader_.program, "u_mvp"), 1, GL_FALSE, &mvp[0][0]);
            glUniform1f(glGetUniformLocation(heatmap_shader_.program, "u_alpha"), accepted_sample_heatmap_alpha_);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(-1.0f, -1.0f);
            face_heatmap_renderer_.Draw();
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDisable(GL_BLEND);
        }

        glUseProgram(line_shader_.program);
        glUniformMatrix4fv(glGetUniformLocation(line_shader_.program, "u_mvp"), 1, GL_FALSE, &mvp[0][0]);

        if (show_axes_) {
            std::vector<glm::vec3> x_axis = {glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};
            std::vector<glm::vec3> y_axis = {glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)};
            std::vector<glm::vec3> z_axis = {glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};

            glUniform3f(glGetUniformLocation(line_shader_.program, "u_color"), 1.0f, 0.15f, 0.15f);
            line_renderer_.DrawLines(x_axis);
            glUniform3f(glGetUniformLocation(line_shader_.program, "u_color"), 0.2f, 1.0f, 0.2f);
            line_renderer_.DrawLines(y_axis);
            glUniform3f(glGetUniformLocation(line_shader_.program, "u_color"), 0.15f, 0.35f, 1.0f);
            line_renderer_.DrawLines(z_axis);
        }

        if (show_bbox_ && mesh_loaded_) {
            const glm::vec3 mn(mesh_data_.min_bound[0], mesh_data_.min_bound[1], mesh_data_.min_bound[2]);
            const glm::vec3 mx(mesh_data_.max_bound[0], mesh_data_.max_bound[1], mesh_data_.max_bound[2]);

            const std::array<glm::vec3, 8> c = {
                glm::vec3(mn.x, mn.y, mn.z), glm::vec3(mx.x, mn.y, mn.z), glm::vec3(mx.x, mx.y, mn.z),
                glm::vec3(mn.x, mx.y, mn.z), glm::vec3(mn.x, mn.y, mx.z), glm::vec3(mx.x, mn.y, mx.z),
                glm::vec3(mx.x, mx.y, mx.z), glm::vec3(mn.x, mx.y, mx.z),
            };

            const int edge_pairs[24] = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6,
                                        6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};
            std::vector<glm::vec3> lines;
            lines.reserve(24);
            for (int i = 0; i < 24; i += 2) {
                lines.push_back(c[edge_pairs[i + 0]]);
                lines.push_back(c[edge_pairs[i + 1]]);
            }

            glUniform3f(glGetUniformLocation(line_shader_.program, "u_color"), 1.0f, 0.85f, 0.2f);
            line_renderer_.DrawLines(lines);
        }

        auto draw_overlay_lines = [&](const std::vector<glm::vec3> &pts, const glm::vec3 &color, float width) {
            if (pts.empty()) {
                return;
            }
            const GLboolean depth_enabled = glIsEnabled(GL_DEPTH_TEST);
            glDisable(GL_DEPTH_TEST);
            glLineWidth(std::max(1.0f, width));
            glUniform3f(glGetUniformLocation(line_shader_.program, "u_color"), color.x, color.y, color.z);
            line_renderer_.DrawLines(pts);
            glLineWidth(1.0f);
            if (depth_enabled == GL_TRUE) {
                glEnable(GL_DEPTH_TEST);
            }
        };

        const bool showing_generated_output = IsFaithCOutputPath(display_mesh_path_);
        if (show_uv_island_seams_) {
            if (show_compare_uv_seams_ && showing_generated_output && source_uv_seams_.has_uv) {
                draw_overlay_lines(source_uv_seams_.line_points, compare_high_seam_color_, uv_seam_line_width_);
                const UVSeamOverlay *low_overlay = nullptr;
                if (low_algorithm_uv_seams_.has_uv && !low_algorithm_uv_seams_.line_points.empty()) {
                    low_overlay = &low_algorithm_uv_seams_;
                } else if (low_generated_uv_seams_.has_uv && !low_generated_uv_seams_.line_points.empty()) {
                    low_overlay = &low_generated_uv_seams_;
                }
                if (low_overlay != nullptr) {
                    draw_overlay_lines(low_overlay->line_points, compare_low_seam_color_, uv_seam_line_width_);
                }
            } else {
                const UVSeamOverlay *active_seams = GetActiveUVSeamOverlay(nullptr);
                if (active_seams != nullptr && !active_seams->line_points.empty()) {
                    draw_overlay_lines(active_seams->line_points, uv_seam_color_, uv_seam_line_width_);
                }
            }
        }

    }

    void StartFaithCJob() {
        SetCrashStage("StartFaithCJob");
        if (!mesh_loaded_ || current_source_mesh_.empty() || job_running_) {
            return;
        }

        const uint64_t tag = NowMillis();
        const fs::path output_mesh = options_.work_dir / ("preview_low_" + std::to_string(tag) + ".glb");
        const fs::path status_json = options_.work_dir / ("preview_status_" + std::to_string(tag) + ".json");

        FaithCJobConfig cfg;
        cfg.input_mesh = current_source_mesh_;
        cfg.output_mesh = output_mesh;
        cfg.status_json = status_json;
        cfg.resolution = resolution_values_[resolution_idx_];
        cfg.min_level = min_level_;
        cfg.tri_mode = tri_mode_values_[tri_mode_idx_];
        cfg.uv_mode = uv_mode_values_[uv_mode_idx_];
        cfg.uv_seam_strategy = uv_seam_strategy_values_[uv_seam_strategy_idx_];
        cfg.uv_solve_backend = uv_solve_backend_values_[uv_solve_backend_idx_];
        cfg.uv_island_guard = uv_island_guard_;
        cfg.uv_island_guard_mode = uv_island_guard_mode_values_[uv_island_guard_mode_idx_];
        cfg.uv_island_guard_confidence_min = uv_island_guard_confidence_min_;
        cfg.uv_island_guard_allow_unknown = uv_island_guard_allow_unknown_;
        cfg.uv_batch_size = uv_batch_size_;
        cfg.uv_m2_outlier_sigma = uv_m2_outlier_sigma_;
        cfg.uv_m2_outlier_quantile = uv_m2_outlier_quantile_;
        cfg.uv_m2_min_samples_per_face = uv_m2_min_samples_per_face_;
        cfg.uv_m2_face_weight_floor = uv_m2_face_weight_floor_;
        cfg.uv_m2_anchor_mode = uv_m2_anchor_mode_values_[uv_m2_anchor_mode_idx_];
        cfg.uv_m2_anchor_points_per_component = uv_m2_anchor_points_per_component_;
        cfg.uv_m2_irls_iters = uv_m2_irls_iters_;
        cfg.uv_m2_huber_delta = uv_m2_huber_delta_;
        cfg.uv_m2_laplacian_mode = uv_m2_laplacian_mode_values_[uv_m2_laplacian_mode_idx_];
        cfg.uv_m2_system_cond_estimate = uv_m2_system_cond_estimate_values_[uv_m2_system_cond_estimate_idx_];
        cfg.margin = margin_;
        cfg.python_bin = options_.python_bin;
        cfg.bridge_script = options_.bridge_script;

        if (!fs::exists(cfg.bridge_script)) {
            status_text_ = "Bridge script not found: " + cfg.bridge_script.string();
            return;
        }

        job_running_ = true;
        job_ignore_result_ = false;
        job_started_at_ = std::chrono::steady_clock::now();
        status_text_ = "FaithC job running...";

        job_future_ = std::async(std::launch::async, [cfg]() { return RunFaithCJob(cfg); });
    }

    void PollFaithCJob() {
        SetCrashStage("PollFaithCJob:begin");
        if (!job_running_ || !job_future_.valid()) {
            return;
        }

        if (job_future_.wait_for(0ms) != std::future_status::ready) {
            return;
        }

        SetCrashStage("PollFaithCJob:get_future");
        FaithCJobResult result = job_future_.get();
        job_running_ = false;

        if (job_ignore_result_) {
            status_text_ = "FaithC job finished but was canceled (result ignored).";
            return;
        }

        if (!result.success) {
            status_text_ = "FaithC failed: " + result.message;
            return;
        }

        if (result.output_faces <= 0 || result.active_voxels <= 0) {
            status_text_ = "FaithC produced empty output. Increase resolution (>=128/256) and retry.";
            return;
        }

        std::string error;
        SetCrashStage("PollFaithCJob:load_output_mesh");
        if (!LoadMesh(result.output_mesh, error)) {
            status_text_ = "FaithC output load failed: " + error;
            return;
        }

        std::string heatmap_error;
        if (!LoadAcceptedSampleHeatmapFromResult(result, heatmap_error)) {
            accepted_sample_heatmap_status_ = heatmap_error;
        }
        std::string closure_error;
        if (!LoadClosureValidationFromResult(result, closure_error)) {
            closure_validation_status_ = closure_error;
        }
        last_job_result_ = result;
        status_text_ = faithc::viewer::BuildFaithCJobSummary(result);
        SetCrashStage("PollFaithCJob:done");
    }

    void DrawUI() {
        SetCrashStage("DrawUI");
        DrawControlsUI();
        DrawExperimentsUI();
        DrawOutputUI();
    }

    void DrawControlsUI() {
        ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(460, 860), ImGuiCond_FirstUseEver);
        ImGui::Begin("FaithC Controls");

        ImGui::TextUnformatted("Model Import");
        ImGui::InputText("Path", path_input_.data(), path_input_.size());
        if (ImGui::Button("Load")) {
            const std::string p = GetPathInput();
            if (!p.empty()) {
                current_source_mesh_ = fs::absolute(fs::path(p));
                std::string error;
                if (!LoadMesh(current_source_mesh_, error)) {
                    status_text_ = "Load failed: " + error;
                } else {
                    status_text_ = "Loaded source mesh: " + current_source_mesh_.string();
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Source") && !current_source_mesh_.empty()) {
            std::string error;
            if (!LoadMesh(current_source_mesh_, error)) {
                status_text_ = "Reload failed: " + error;
            } else {
                status_text_ = "Reloaded: " + current_source_mesh_.string();
            }
        }

        if (!recent_files_.empty()) {
            ImGui::SeparatorText("Recent Files");
            for (const auto &p : recent_files_) {
                if (ImGui::Selectable(p.c_str())) {
                    SetPathInput(p);
                }
            }
        }

        ImGui::SeparatorText("Display");
        ImGui::Checkbox("Wireframe", &show_wireframe_);
        ImGui::Checkbox("Backface Culling", &enable_backface_culling_);
        ImGui::Checkbox("Show Axes", &show_axes_);
        ImGui::Checkbox("Show Bounding Box", &show_bbox_);
        if (mesh_data_.has_uv && mesh_data_.has_base_color_texture && mesh_gpu_.HasBaseColorTexture()) {
            ImGui::Checkbox("Use BaseColor Texture", &use_basecolor_texture_);
        } else {
            ImGui::BeginDisabled();
            bool disabled = false;
            ImGui::Checkbox("Use BaseColor Texture", &disabled);
            ImGui::EndDisabled();
        }
        ImGui::ColorEdit3("Mesh Color", &mesh_color_.x);
        ImGui::ColorEdit3("Background", &background_color_.x);

        ImGui::SeparatorText("FaithC Decimation");
        ImGui::Text("Bridge: %s", options_.bridge_script.string().c_str());
        ImGui::Text("Python: %s", options_.python_bin.c_str());
        ImGui::Text("Work dir: %s", options_.work_dir.string().c_str());

        ImGui::Combo("Resolution", &resolution_idx_, resolution_labels_, IM_ARRAYSIZE(resolution_labels_));
        ImGui::Combo("Tri Mode", &tri_mode_idx_, tri_mode_labels_, IM_ARRAYSIZE(tri_mode_labels_));
        ImGui::Combo("UV Mode", &uv_mode_idx_, uv_mode_labels_, IM_ARRAYSIZE(uv_mode_labels_));
        ImGui::Combo("Seam Strategy", &uv_seam_strategy_idx_, uv_seam_strategy_labels_, IM_ARRAYSIZE(uv_seam_strategy_labels_));
        ImGui::Combo("UV Solve Backend", &uv_solve_backend_idx_, uv_solve_backend_labels_, IM_ARRAYSIZE(uv_solve_backend_labels_));
        ImGui::Checkbox("Island Guard", &uv_island_guard_);
        ImGui::Combo("Island Guard Mode", &uv_island_guard_mode_idx_, uv_island_guard_mode_labels_,
                     IM_ARRAYSIZE(uv_island_guard_mode_labels_));
        ImGui::SliderFloat("Island Guard Min Confidence", &uv_island_guard_confidence_min_, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Island Guard Allow Unknown", &uv_island_guard_allow_unknown_);
        ImGui::InputInt("UV Batch Size", &uv_batch_size_, 10000, 100000);
        uv_batch_size_ = std::max(1, uv_batch_size_);

        const bool show_m2 = (uv_mode_idx_ == 1 || uv_mode_idx_ == 2);
        if (show_m2) {
            ImGui::SeparatorText("Method2 Options");
            ImGui::SliderFloat("M2 Outlier Sigma", &uv_m2_outlier_sigma_, 0.0f, 6.0f, "%.2f");
            ImGui::SliderFloat("M2 Outlier Quantile", &uv_m2_outlier_quantile_, 0.50f, 1.00f, "%.2f");
            ImGui::InputInt("M2 Min Samples/Face", &uv_m2_min_samples_per_face_, 1, 4);
            uv_m2_min_samples_per_face_ = std::max(1, uv_m2_min_samples_per_face_);
            ImGui::InputFloat("M2 Face Weight Floor", &uv_m2_face_weight_floor_, 1e-6f, 1e-5f, "%.6f");
            uv_m2_face_weight_floor_ = std::max(1e-12f, uv_m2_face_weight_floor_);
            ImGui::Combo("M2 Anchor Mode", &uv_m2_anchor_mode_idx_, uv_m2_anchor_mode_labels_,
                         IM_ARRAYSIZE(uv_m2_anchor_mode_labels_));
            ImGui::InputInt("M2 Anchor Points/Comp", &uv_m2_anchor_points_per_component_, 1, 4);
            uv_m2_anchor_points_per_component_ = std::max(1, uv_m2_anchor_points_per_component_);
            ImGui::InputInt("M2 IRLS Iters", &uv_m2_irls_iters_, 1, 4);
            uv_m2_irls_iters_ = std::max(1, uv_m2_irls_iters_);
            ImGui::SliderFloat("M2 Huber Delta", &uv_m2_huber_delta_, 0.1f, 5.0f, "%.2f");
            ImGui::Combo("M2 Laplacian", &uv_m2_laplacian_mode_idx_, uv_m2_laplacian_mode_labels_,
                         IM_ARRAYSIZE(uv_m2_laplacian_mode_labels_));
            ImGui::Combo("M2 Cond Estimate", &uv_m2_system_cond_estimate_idx_, uv_m2_system_cond_estimate_labels_,
                         IM_ARRAYSIZE(uv_m2_system_cond_estimate_labels_));
        }

        ImGui::SliderFloat("Margin", &margin_, 0.0f, 0.2f, "%.3f");
        const int max_level = static_cast<int>(std::log2(static_cast<float>(resolution_values_[resolution_idx_])));
        ImGui::SliderInt("Min Level", &min_level_, -1, max_level, min_level_ < 0 ? "auto" : "%d");

        const bool can_apply = mesh_loaded_ && !job_running_;
        if (!can_apply) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Apply")) {
            StartFaithCJob();
        }
        if (!can_apply) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();
        if (job_running_) {
            if (ImGui::Button("Cancel")) {
                job_ignore_result_ = true;
                status_text_ = "Cancel requested. Current job continues in background; result will be ignored.";
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Cancel");
            ImGui::EndDisabled();
        }

        ImGui::End();
    }

    void DrawOutputUI() {
        ImGui::SetNextWindowPos(ImVec2(476, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(560, 860), ImGuiCond_FirstUseEver);
        ImGui::Begin("FaithC Output");
        faithc::viewer::OutputPanelContext ctx;
        ctx.source_mesh = current_source_mesh_;
        ctx.displayed_mesh = display_mesh_path_;
        ctx.mesh = mesh_loaded_ ? &mesh_data_ : nullptr;
        ctx.used_source_texture_fallback = used_source_texture_fallback_;
        ctx.job_running = job_running_;
        if (job_running_) {
            ctx.job_elapsed_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - job_started_at_).count();
        }
        ctx.last_job_result = last_job_result_.has_value() ? &(*last_job_result_) : nullptr;
        ctx.status_text = status_text_;
        faithc::viewer::RenderFaithCOutputPanel(ctx);

        ImGui::End();
    }

    void DrawExperimentsUI() {
        ImGui::SetNextWindowPos(ImVec2(1048, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(540, 860), ImGuiCond_FirstUseEver);
        ImGui::Begin("FaithC Experiments");

        ImGui::SeparatorText("Overlay Toggles");
        ImGui::Checkbox("Show UV Island Seams", &show_uv_island_seams_);
        ImGui::Checkbox("Compare High vs Low Seams", &show_compare_uv_seams_);
        ImGui::ColorEdit3("Compare High Seam Color", &compare_high_seam_color_.x);
        ImGui::ColorEdit3("Compare Low Seam Color", &compare_low_seam_color_.x);
        ImGui::ColorEdit3("UV Seam Color", &uv_seam_color_.x);
        ImGui::SliderFloat("UV Seam Width", &uv_seam_line_width_, 1.0f, 4.0f, "%.1f");

        ImGui::Checkbox("Semantic-ID Heatmap", &show_semantic_heatmap_);
        ImGui::SliderFloat("Semantic Heatmap Alpha", &semantic_heatmap_alpha_, 0.05f, 1.0f, "%.2f");

        ImGui::Checkbox("Accepted-Samples Heatmap", &show_accepted_sample_heatmap_);
        ImGui::SliderFloat("Accepted Heatmap Alpha", &accepted_sample_heatmap_alpha_, 0.05f, 1.0f, "%.2f");
        bool changed_log_scale = ImGui::Checkbox("Accepted Heatmap Log Scale", &accepted_sample_heatmap_log_scale_);
        if (changed_log_scale && !accepted_sample_counts_raw_.empty()) {
            std::string rebuild_error;
            if (!BuildAcceptedSampleHeatmapFromCounts(accepted_sample_counts_raw_, rebuild_error)) {
                accepted_sample_heatmap_status_ = rebuild_error;
            }
        }

        ImGui::SeparatorText("Diagnostics");
        const char *seam_label = nullptr;
        const UVSeamOverlay *seam_overlay = GetActiveUVSeamOverlay(&seam_label);
        if (seam_overlay != nullptr) {
            ImGui::Text("UV Seam Source: %s", seam_label);
            ImGui::Text("Seam edges: %d (boundary=%d, interior=%d, nonmanifold=%d)", seam_overlay->seam_edges,
                        seam_overlay->boundary_edges, seam_overlay->interior_seam_edges, seam_overlay->nonmanifold_edges);
        } else {
            ImGui::TextDisabled("UV seam overlay unavailable for current mesh.");
        }

        if (semantic_heatmap_available_) {
            ImGui::Text("Semantic labels: faces=%d, unknown=%d, unique=%d", semantic_face_count_,
                        semantic_unknown_face_count_, semantic_unique_label_count_);
            if (!semantic_heatmap_source_.empty()) {
                ImGui::TextWrapped("Semantic sidecar: %s", semantic_heatmap_source_.c_str());
            }
        } else if (!semantic_heatmap_status_.empty()) {
            ImGui::TextDisabled("Semantic: %s", semantic_heatmap_status_.c_str());
        } else {
            ImGui::TextDisabled("Semantic: unavailable");
        }

        if (accepted_sample_heatmap_available_) {
            ImGui::Text("Accepted: faces=%d, nonzero=%d, max=%d", accepted_sample_face_count_,
                        accepted_sample_nonzero_faces_, accepted_sample_max_count_);
            if (!accepted_sample_heatmap_source_.empty()) {
                ImGui::TextWrapped("Accepted sidecar: %s", accepted_sample_heatmap_source_.c_str());
            }
        } else if (!accepted_sample_heatmap_status_.empty()) {
            ImGui::TextDisabled("Accepted: %s", accepted_sample_heatmap_status_.c_str());
        } else {
            ImGui::TextDisabled("Accepted: unavailable");
        }

        if (closure_summary_.available) {
            ImGui::SeparatorText("Closure Loop");
            ImGui::Text("Seam topology valid: %s", closure_summary_.seam_topology_valid ? "yes" : "no");
            ImGui::Text("Seam components/open/closed: %d / %d / %d", closure_summary_.seam_components,
                        closure_summary_.seam_components_open, closure_summary_.seam_loops_closed);
            ImGui::Text("Partition leakage: %s (mixed=%d, split=%d)",
                        closure_summary_.partition_has_leakage ? "yes" : "no",
                        closure_summary_.partition_mixed_components, closure_summary_.partition_label_split_count);
            ImGui::Text("Islands high/low: %d / %d", closure_summary_.high_island_count, closure_summary_.low_island_count);
            ImGui::Text("UV bbox IoU mean: %.4f", closure_summary_.uv_bbox_iou_mean);
            ImGui::Text("UV overlap ratio: %.6f", closure_summary_.uv_overlap_ratio);
            ImGui::Text("UV stretch p95/p99: %.4f / %.4f", closure_summary_.uv_stretch_p95, closure_summary_.uv_stretch_p99);
            if (!closure_summary_.uv_png_path.empty()) {
                ImGui::TextWrapped("UV validation PNG: %s", closure_summary_.uv_png_path.c_str());
            }
            if (!closure_sidecar_source_.empty()) {
                ImGui::TextWrapped("Closure sidecar: %s", closure_sidecar_source_.c_str());
            }
        } else if (!closure_validation_status_.empty()) {
            ImGui::TextDisabled("Closure: %s", closure_validation_status_.c_str());
        }

        ImGui::End();
    }

private:
    LaunchOptions options_;
    GLFWwindow *window_ = nullptr;

    ShaderProgram mesh_shader_;
    ShaderProgram line_shader_;
    ShaderProgram heatmap_shader_;
    MeshGPU mesh_gpu_;
    LineRenderer line_renderer_;
    FaceHeatmapRenderer face_heatmap_renderer_;
    FaceHeatmapRenderer semantic_heatmap_renderer_;

    faithc::viewer::MeshData mesh_data_;
    bool mesh_loaded_ = false;

    OrbitCamera camera_;

    bool show_wireframe_ = true;
    bool enable_backface_culling_ = true;
    bool show_axes_ = true;
    bool show_bbox_ = true;
    bool show_uv_island_seams_ = false;
    bool show_compare_uv_seams_ = true;
    bool show_semantic_heatmap_ = true;
    bool show_accepted_sample_heatmap_ = false;
    bool use_basecolor_texture_ = true;

    glm::vec3 mesh_color_ = glm::vec3(0.80f, 0.79f, 0.74f);
    glm::vec3 background_color_ = glm::vec3(0.09f, 0.10f, 0.12f);
    glm::vec3 uv_seam_color_ = glm::vec3(1.0f, 0.2f, 0.1f);
    glm::vec3 compare_high_seam_color_ = glm::vec3(0.15f, 0.95f, 1.0f);
    glm::vec3 compare_low_seam_color_ = glm::vec3(1.0f, 0.2f, 0.1f);
    float uv_seam_line_width_ = 1.0f;
    float uv_seam_position_eps_ = 1e-6f;
    float uv_seam_uv_eps_ = 1e-5f;
    float semantic_heatmap_alpha_ = 0.58f;
    float accepted_sample_heatmap_alpha_ = 0.72f;
    bool accepted_sample_heatmap_log_scale_ = true;

    UVSeamOverlay source_uv_seams_;
    UVSeamOverlay low_generated_uv_seams_;
    UVSeamOverlay low_algorithm_uv_seams_;
    UVSeamOverlay display_uv_seams_;
    ClosureValidationSummary closure_summary_;
    std::vector<int> accepted_sample_counts_raw_;
    std::vector<int> semantic_labels_raw_;
    bool semantic_heatmap_available_ = false;
    bool accepted_sample_heatmap_available_ = false;
    int semantic_face_count_ = 0;
    int semantic_unknown_face_count_ = 0;
    int semantic_unique_label_count_ = 0;
    int accepted_sample_face_count_ = 0;
    int accepted_sample_nonzero_faces_ = 0;
    int accepted_sample_max_count_ = 0;
    std::string semantic_heatmap_source_;
    std::string closure_sidecar_source_;
    std::string semantic_heatmap_status_;
    std::string closure_validation_status_;
    std::string accepted_sample_heatmap_source_;
    std::string accepted_sample_heatmap_status_;

    bool mouse_rotating_ = false;
    bool mouse_panning_ = false;
    bool first_mouse_ = true;
    double last_x_ = 0.0;
    double last_y_ = 0.0;

    fs::path initial_mesh_;
    fs::path current_source_mesh_;
    fs::path display_mesh_path_;

    std::array<char, 2048> path_input_{};
    std::vector<std::string> recent_files_;

    int resolution_idx_ = 1;
    float margin_ = 0.05f;
    int min_level_ = -1;
    int tri_mode_idx_ = 0;
    int uv_mode_idx_ = 1;
    int uv_seam_strategy_idx_ = 0;
    int uv_solve_backend_idx_ = 0;
    bool uv_island_guard_ = true;
    int uv_island_guard_mode_idx_ = 0;
    float uv_island_guard_confidence_min_ = 0.55f;
    bool uv_island_guard_allow_unknown_ = false;
    int uv_batch_size_ = 200000;
    float uv_m2_outlier_sigma_ = 4.0f;
    float uv_m2_outlier_quantile_ = 0.95f;
    int uv_m2_min_samples_per_face_ = 2;
    float uv_m2_face_weight_floor_ = 1e-6f;
    int uv_m2_anchor_mode_idx_ = 0;
    int uv_m2_anchor_points_per_component_ = 4;
    int uv_m2_irls_iters_ = 2;
    float uv_m2_huber_delta_ = 3.0f;
    int uv_m2_laplacian_mode_idx_ = 1;
    int uv_m2_system_cond_estimate_idx_ = 0;

    const int resolution_values_[8] = {4, 8, 16, 32, 64, 128, 256, 512};
    const char *resolution_labels_[8] = {"4", "8", "16", "32", "64", "128", "256", "512"};

    const char *tri_mode_labels_[7] = {
        "auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs",
    };
    const std::string tri_mode_values_[7] = {
        "auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs",
    };

    const char *uv_mode_labels_[6] = {
        "hybrid global opt (legacy)",
        "method2 gradient-poisson (recommended)",
        "method4 jacobian-injective",
        "barycentric closest-point",
        "nearest vertex",
        "auto (method2)",
    };
    const std::string uv_mode_values_[6] = {
        "hybrid",
        "method2",
        "method4",
        "barycentric",
        "nearest",
        "auto",
    };

    const char *uv_seam_strategy_labels_[2] = {
        "legacy (sample-span heuristic)",
        "halfedge UV-island split",
    };
    const std::string uv_seam_strategy_values_[2] = {
        "legacy",
        "halfedge_island",
    };

    const char *uv_solve_backend_labels_[3] = {
        "auto (prefer CUDA PCG)",
        "cuda_pcg",
        "cpu_scipy",
    };
    const std::string uv_solve_backend_values_[3] = {
        "auto",
        "cuda_pcg",
        "cpu_scipy",
    };

    const char *uv_island_guard_mode_labels_[2] = {
        "soft (recommended)",
        "strict (no fallback)",
    };
    const std::string uv_island_guard_mode_values_[2] = {
        "soft",
        "strict",
    };

    const char *uv_m2_anchor_mode_labels_[3] = {
        "component_minimal",
        "boundary",
        "none",
    };
    const std::string uv_m2_anchor_mode_values_[3] = {
        "component_minimal",
        "boundary",
        "none",
    };

    const char *uv_m2_laplacian_mode_labels_[2] = {
        "uniform",
        "cotan (recommended)",
    };
    const std::string uv_m2_laplacian_mode_values_[2] = {
        "uniform",
        "cotan",
    };

    const char *uv_m2_system_cond_estimate_labels_[2] = {
        "diag_ratio (recommended)",
        "eigsh (slow)",
    };
    const std::string uv_m2_system_cond_estimate_values_[2] = {
        "diag_ratio",
        "eigsh",
    };

    std::future<FaithCJobResult> job_future_;
    bool job_running_ = false;
    bool job_ignore_result_ = false;
    std::chrono::steady_clock::time_point job_started_at_{};

    std::optional<FaithCJobResult> last_job_result_;
    std::string status_text_;

    bool source_tex_valid_ = false;
    int source_tex_width_ = 0;
    int source_tex_height_ = 0;
    int source_tex_channels_ = 0;
    std::vector<uint8_t> source_tex_pixels_;
    bool used_source_texture_fallback_ = false;
};

}  // namespace

int main(int argc, char **argv) {
    InstallCrashHandlers();
    SetCrashStage("main:startup");
    LaunchOptions options;
    std::string error;
    if (!ParseArgs(argc, argv, options, error)) {
        std::cerr << "Argument error: " << error << "\n";
        PrintUsage();
        return 2;
    }

    std::error_code ec;
    fs::create_directories(options.work_dir, ec);

    ViewerApp app(std::move(options));
    if (!app.Init(error)) {
        std::cerr << "Initialization failed: " << error << "\n";
        return 1;
    }

    return app.Run();
}
