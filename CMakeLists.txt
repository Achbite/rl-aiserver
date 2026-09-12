# ============================================================================
# AIServer - AI 决策服务（gRPC 服务端）
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(rl_aiserver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依赖查找 ---
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
add_subdirectory(proto/rl_sdk)

include("${CMAKE_CURRENT_SOURCE_DIR}/src/maze/sources.cmake")

# --- 源文件 ---
set(SOURCES
    main/main.cpp
    ${AISERVER_MAZE_SOURCES}
    src/task/inference/onnx_inferencer.cpp
    src/task/model/model_manifest.cpp
    src/task/model/model_distributor_client.cpp
    src/task/metrics/episode_metrics.cpp
    src/task/sample/sample_sender.cpp
    src/task/sample/training_transition_builder.cpp
    src/task/sample/rollout_transition_builder.cpp
    src/task/policy/episode_action_policy.cpp
)

set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_SOURCES
    ${PROTO_GEN_DIR}/common/identity.pb.cc
    ${PROTO_GEN_DIR}/training/model_identity.pb.cc
    ${PROTO_GEN_DIR}/training/training.pb.cc
    ${PROTO_GEN_DIR}/training/training.grpc.pb.cc
    ${PROTO_GEN_DIR}/metrics/registry.pb.cc
    ${PROTO_GEN_DIR}/metrics/catalog.pb.cc
    ${PROTO_GEN_DIR}/metrics/catalog.grpc.pb.cc
    ${PROTO_GEN_DIR}/metrics/transport.pb.cc
    ${PROTO_GEN_DIR}/metrics/transport.grpc.pb.cc
    ${PROTO_GEN_DIR}/communication/session.pb.cc
    ${AISERVER_MAZE_PROTO_SOURCES}
)

# --- 可执行文件 ---
rl_sdk_generate_task(aiserver_task_protocol
    PROTO proto/maze/maze.proto IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
set(PRODUCTION_PROTO_SOURCES ${PROTO_SOURCES})
list(REMOVE_ITEM PRODUCTION_PROTO_SOURCES
    "${PROTO_GEN_DIR}/common/identity.pb.cc"
    "${PROTO_GEN_DIR}/communication/session.pb.cc"
    "proto/maze/maze.pb.cc"
    "proto/maze/maze.grpc.pb.cc")
add_executable(${PROJECT_NAME} ${SOURCES} ${PRODUCTION_PROTO_SOURCES})

# --- 头文件搜索路径 ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTO_GEN_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
    /usr/local/include/onnxruntime      # ONNX Runtime 头文件
)
get_target_property(TASK_GENERATED_INCLUDE_DIRS aiserver_task_protocol INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME} BEFORE PRIVATE ${TASK_GENERATED_INCLUDE_DIRS})

# --- 链接库 ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    aiserver_task_protocol
    rl_sdk::sdk
    protobuf::libprotobuf
    ${GRPC_LIBRARIES}
    pthread
    onnxruntime                          # ONNX Runtime 推理库
    crypto
)

include(CTest)
if(BUILD_TESTING)
    set(RL_CLIENT_SOURCE_DIR "" CACHE PATH "Client sources used only for the real SDK/action test")
    if(NOT EXISTS "${RL_CLIENT_SOURCE_DIR}/src/maze/environment/maze_env.cpp")
        message(FATAL_ERROR "Tests require RL_CLIENT_SOURCE_DIR pointing to the Client source repository")
    endif()
    set(TEST_PRODUCTION_SOURCES ${SOURCES})
    list(REMOVE_ITEM TEST_PRODUCTION_SOURCES main/main.cpp)
    add_library(aiserver_test_production STATIC ${TEST_PRODUCTION_SOURCES} ${PROTO_SOURCES})
    target_include_directories(aiserver_test_production PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTO_GEN_DIR} ${Protobuf_INCLUDE_DIRS} ${GRPC_INCLUDE_DIRS}
        /usr/local/include/onnxruntime)
    target_link_libraries(aiserver_test_production PUBLIC
        rl_sdk::sdk protobuf::libprotobuf ${GRPC_LIBRARIES} pthread onnxruntime crypto)
    add_executable(aiserver_model_update_development_test tests/model_update_development_test.cpp)
    add_executable(aiserver_gae_sample_delivery_development_test tests/gae_sample_delivery_development_test.cpp)
    add_executable(aiserver_sdk_rpc_test tests/sdk_aiserver_rpc_test.cpp
        ${RL_CLIENT_SOURCE_DIR}/src/maze/environment/maze_env.cpp)
    target_include_directories(aiserver_sdk_rpc_test PRIVATE ${RL_CLIENT_SOURCE_DIR}/src)
    foreach(target aiserver_model_update_development_test aiserver_gae_sample_delivery_development_test aiserver_sdk_rpc_test)
        target_link_libraries(${target} PRIVATE aiserver_test_production)
        target_compile_options(${target} PRIVATE -UNDEBUG)
    endforeach()
    add_test(NAME aiserver_model_update_data_path COMMAND aiserver_model_update_development_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/fixed_policy.onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/runtime_error.onnx)
    add_test(NAME aiserver_gae_sample_delivery_data_path COMMAND aiserver_gae_sample_delivery_development_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/fixed_policy.onnx)
    add_test(NAME aiserver_sdk_action_data_path COMMAND aiserver_sdk_rpc_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/fixed_policy.onnx)
    set_tests_properties(aiserver_model_update_data_path aiserver_gae_sample_delivery_data_path
        aiserver_sdk_action_data_path PROPERTIES TIMEOUT 30)
endif()
