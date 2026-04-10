cmake_minimum_required(VERSION 3.20)
project(faithc_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(glfw)

# GLAD
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

# Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(imgui_lib PUBLIC glfw)

# GLM
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

# tinygltf
FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG v2.9.5
)
FetchContent_MakeAvailable(tinygltf)

# tinyobjloader
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG v2.0.0rc13
)
FetchContent_MakeAvailable(tinyobjloader)

# tinyply
FetchContent_Declare(
  tinyply
  GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
  GIT_TAG 2.3.4
)
FetchContent_MakeAvailable(tinyply)

# nlohmann/json
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenGL REQUIRED)

add_executable(faithc_viewer
  src/main.cpp
  src/mesh_loader.cpp
  src/job_result_schema.cpp
  src/job_output_view.cpp
  ${tinyply_SOURCE_DIR}/source/tinyply.cpp
)

target_include_directories(faithc_viewer PRIVATE
  ${tinygltf_SOURCE_DIR}
  ${tinyobjloader_SOURCE_DIR}
  ${tinyply_SOURCE_DIR}/source
  ${glad_SOURCE_DIR}/include
  ${glfw_SOURCE_DIR}/include
  ${glm_SOURCE_DIR}
)

target_compile_definitions(faithc_viewer PRIVATE
  IMGUI_IMPL_OPENGL_LOADER_GLAD
)

target_link_libraries(faithc_viewer PRIVATE
  OpenGL::GL
  glad
  glfw
  imgui_lib
  nlohmann_json::nlohmann_json
)
