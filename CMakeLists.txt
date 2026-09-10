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
add_executable(${PROJECT_NAME} ${SOURCES} ${PROTO_SOURCES})

# --- 头文件搜索路径 ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTO_GEN_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
    /usr/local/include/onnxruntime      # ONNX Runtime 头文件
)

# --- 链接库 ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    rl_sdk::sdk
    protobuf::libprotobuf
    ${GRPC_LIBRARIES}
    pthread
    onnxruntime                          # ONNX Runtime 推理库
    crypto
)

include(CTest)
if(BUILD_TESTING)
    add_executable(aiserver_model_update_development_test
        tests/model_update_development_test.cpp
        src/task/inference/onnx_inferencer.cpp
        src/task/model/model_manifest.cpp
        src/task/model/model_distributor_client.cpp
        ${PROTO_GEN_DIR}/common/identity.pb.cc
        ${PROTO_GEN_DIR}/training/model_identity.pb.cc
        ${PROTO_GEN_DIR}/training/training.pb.cc
        ${PROTO_GEN_DIR}/training/training.grpc.pb.cc
        ${PROTO_GEN_DIR}/metrics/registry.pb.cc
        ${PROTO_GEN_DIR}/metrics/catalog.pb.cc
        ${PROTO_GEN_DIR}/metrics/catalog.grpc.pb.cc
        ${PROTO_GEN_DIR}/metrics/transport.pb.cc
        ${PROTO_GEN_DIR}/metrics/transport.grpc.pb.cc
    )
    target_include_directories(aiserver_model_update_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
        /usr/local/include/onnxruntime
    )
    target_link_libraries(aiserver_model_update_development_test PRIVATE
        rl_sdk::sdk
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
        onnxruntime
        crypto
    )
    add_test(
        NAME aiserver_model_update_data_path
        COMMAND aiserver_model_update_development_test
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/maze_policy.onnx
    )

    add_executable(aiserver_gae_sample_delivery_development_test
        tests/gae_sample_delivery_development_test.cpp
        ${AISERVER_MAZE_SOURCES}
        src/task/inference/onnx_inferencer.cpp
        src/task/model/model_manifest.cpp
        src/task/model/model_distributor_client.cpp
        src/task/metrics/episode_metrics.cpp
        src/task/sample/sample_sender.cpp
        src/task/sample/training_transition_builder.cpp
        src/task/sample/rollout_transition_builder.cpp
        src/task/policy/episode_action_policy.cpp
        ${PROTO_SOURCES}
    )
    target_include_directories(
        aiserver_gae_sample_delivery_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
        /usr/local/include/onnxruntime
    )
    target_link_libraries(
        aiserver_gae_sample_delivery_development_test PRIVATE
        rl_sdk::sdk
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
        onnxruntime
        crypto
    )
    target_compile_options(
        aiserver_gae_sample_delivery_development_test PRIVATE -UNDEBUG)
    add_test(
        NAME aiserver_gae_sample_delivery_data_path
        COMMAND aiserver_gae_sample_delivery_development_test
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/maze_policy.onnx
    )

endif()
