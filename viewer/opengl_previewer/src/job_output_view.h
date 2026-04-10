#pragma once

#include <filesystem>
#include <string>

#include "job_result_schema.h"
#include "mesh_loader.h"

namespace faithc::viewer {

struct OutputPanelContext {
    std::filesystem::path source_mesh;
    std::filesystem::path displayed_mesh;
    const MeshData *mesh = nullptr;
    bool used_source_texture_fallback = false;
    bool job_running = false;
    double job_elapsed_seconds = 0.0;
    const FaithCJobResult *last_job_result = nullptr;
    std::string status_text;
};

void RenderFaithCOutputPanel(const OutputPanelContext &ctx);
std::string BuildFaithCJobSummary(const FaithCJobResult &result);

}  // namespace faithc::viewer
