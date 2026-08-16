-- xmake.lua — RoboMaster Autoaim System (Scenario C: Outpost)
-- C++23, OpenCV, Eigen, FFmpeg

set_project("rm_autoaim")
set_version("1.0.0")
set_languages("c++23")

add_rules("mode.debug", "mode.release", "mode.asan")

-- Compiler flags
if is_mode("debug") then
  set_optimize("none")
  add_cxflags("-g", "-O0")
elseif is_mode("release") then
  set_optimize("fastest")
  add_cxflags("-march=native", "-DNDEBUG")
elseif is_mode("asan") then
  set_optimize("none")
  add_cxflags("-g", "-O0", "-fsanitize=address", "-fno-omit-frame-pointer")
  add_ldflags("-fsanitize=address")
end

add_cxflags("-Wall", "-Wextra", "-Wpedantic")

-- xmake-repo packages (auto-fetch)
add_requires("spdlog", "fmt")

target("rm_autoaim")
  set_kind("binary")
  add_files("src/**.cpp")
  add_includedirs("include")
  add_packages("spdlog", "fmt")

  -- System OpenCV (via pkg-config)
  add_cxflags("$(shell pkg-config --cflags opencv4)")
  add_ldflags("$(shell pkg-config --libs opencv4)")

  -- System Eigen (via pkg-config)
  add_cxflags("$(shell pkg-config --cflags eigen3)")

  -- System FFmpeg (direct link)
  add_links("avcodec", "avformat", "avutil", "swscale")
  add_syslinks("pthread")

includes("tests/reader")
