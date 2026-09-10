cmake_minimum_required(VERSION 3.16)
project(rl_client_sdk LANGUAGES CXX)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(RL_SDK_GRPC REQUIRED IMPORTED_TARGET grpc++)
add_library(rl_sdk INTERFACE)
add_library(rl_sdk::sdk ALIAS rl_sdk)
target_compile_features(rl_sdk INTERFACE cxx_std_17)
target_include_directories(rl_sdk INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rl_sdk INTERFACE protobuf::libprotobuf PkgConfig::RL_SDK_GRPC pthread)

# protocol_root contains the common, communication and selected task bindings.
# Consumers link the returned target without listing gRPC or generated sources.
function(rl_sdk_add_task target protocol_root task_stem)
    add_library(${target} STATIC
        ${protocol_root}/proto/common/identity.pb.cc
        ${protocol_root}/proto/communication/session.pb.cc
        ${protocol_root}/${task_stem}.pb.cc
        ${protocol_root}/${task_stem}.grpc.pb.cc)
    target_include_directories(${target} PUBLIC ${protocol_root})
    target_link_libraries(${target} PUBLIC rl_sdk::sdk)
endfunction()
