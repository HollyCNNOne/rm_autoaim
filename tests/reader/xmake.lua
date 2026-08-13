-- xmake.lua — Module 1 Reader standalone test

target("test_reader")
  set_kind("binary")
  add_files("test_reader.cpp")
  add_files("../../src/Reader.cpp")
  add_includedirs("../../include")
  add_packages("spdlog", "fmt")

  add_cxflags("$(shell pkg-config --cflags opencv4)")
  add_ldflags("$(shell pkg-config --libs opencv4)")
  add_cxflags("$(shell pkg-config --cflags eigen3)")
  add_links("avcodec", "avformat", "avutil", "swscale")
  add_syslinks("pthread")