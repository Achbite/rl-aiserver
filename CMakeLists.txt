# ============================================================================
# AIServer - AI 决策服务（gRPC 服务端）
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(maze_aiserver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依赖查找 ---
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# --- 源文件 ---
set(SOURCES
    main/main.cpp
    src/grpc/maze_service.cpp
    src/grpc/maze_service_rpc.cpp
    src/ai/maze_reward.cpp
    src/ai/maze_observation.cpp
    src/ai/astar_solver.cpp
    src/config/config_loader.cpp
    src/session/session_manager.cpp
    src/ai/onnx_inferencer.cpp
    src/model/model_manifest.cpp
    src/model/model_distributor_client.cpp
    src/metrics/episode_metrics.cpp
    src/sample/sample_sender.cpp
    src/sample/training_transition_builder.cpp
    src/task/maze_map_contract.cpp
    src/task/episode_action_policy.cpp
    src/task/single_map_task_controller.cpp
)

set(CONTRACT_CPP_DIR "" CACHE PATH "Generated C++ contract directory")
if(CONTRACT_CPP_DIR)
    set(PROTO_GEN_DIR ${CONTRACT_CPP_DIR})
else()
    set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
endif()
set(PROTO_SOURCES
    ${PROTO_GEN_DIR}/common.pb.cc
    ${PROTO_GEN_DIR}/training.pb.cc
    ${PROTO_GEN_DIR}/training.grpc.pb.cc
    ${PROTO_GEN_DIR}/maze_task.pb.cc
    ${PROTO_GEN_DIR}/maze_task.grpc.pb.cc
)

# --- 可执行文件 ---
add_executable(${PROJECT_NAME} ${SOURCES} ${PROTO_SOURCES})

# --- 头文件搜索路径 ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROTO_GEN_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
    /usr/local/include/onnxruntime      # ONNX Runtime 头文件
)

# --- 链接库 ---
target_link_libraries(${PROJECT_NAME} PRIVATE
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
        src/ai/onnx_inferencer.cpp
        src/model/model_manifest.cpp
        src/model/model_distributor_client.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
        ${PROTO_GEN_DIR}/training.grpc.pb.cc
    )
    target_include_directories(aiserver_model_update_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
        /usr/local/include/onnxruntime
    )
    target_link_libraries(aiserver_model_update_development_test PRIVATE
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
        onnxruntime
        crypto
    )
    add_test(
        NAME aiserver_model_update_development_contract
        COMMAND aiserver_model_update_development_test
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/a3_arch_policy_wire_v1.onnx
    )

    add_executable(aiserver_gae_sample_delivery_development_test
        tests/gae_sample_delivery_development_test.cpp
        src/grpc/maze_service.cpp
        src/grpc/maze_service_rpc.cpp
        src/ai/maze_reward.cpp
        src/ai/maze_observation.cpp
        src/ai/astar_solver.cpp
        src/config/config_loader.cpp
        src/session/session_manager.cpp
        src/ai/onnx_inferencer.cpp
        src/model/model_manifest.cpp
        src/model/model_distributor_client.cpp
        src/metrics/episode_metrics.cpp
        src/sample/sample_sender.cpp
        src/sample/training_transition_builder.cpp
        src/task/maze_map_contract.cpp
        src/task/episode_action_policy.cpp
        src/task/single_map_task_controller.cpp
        ${PROTO_SOURCES}
    )
    target_include_directories(
        aiserver_gae_sample_delivery_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
        /usr/local/include/onnxruntime
    )
    target_link_libraries(
        aiserver_gae_sample_delivery_development_test PRIVATE
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
        onnxruntime
        crypto
    )
    target_compile_options(
        aiserver_gae_sample_delivery_development_test PRIVATE -UNDEBUG)
    add_test(
        NAME aiserver_gae_sample_delivery_development_contract
        COMMAND aiserver_gae_sample_delivery_development_test
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/a3_arch_policy_wire_v1.onnx
    )

    add_executable(aiserver_evaluation_lifecycle_development_test
        tests/evaluation_lifecycle_development_test.cpp
        src/grpc/maze_service.cpp
        src/grpc/maze_service_rpc.cpp
        src/ai/maze_reward.cpp
        src/ai/maze_observation.cpp
        src/ai/astar_solver.cpp
        src/config/config_loader.cpp
        src/session/session_manager.cpp
        src/ai/onnx_inferencer.cpp
        src/model/model_manifest.cpp
        src/model/model_distributor_client.cpp
        src/metrics/episode_metrics.cpp
        src/sample/sample_sender.cpp
        src/sample/training_transition_builder.cpp
        src/task/maze_map_contract.cpp
        src/task/episode_action_policy.cpp
        src/task/single_map_task_controller.cpp
        ${PROTO_SOURCES}
    )
    target_include_directories(
        aiserver_evaluation_lifecycle_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
        /usr/local/include/onnxruntime
    )
    target_link_libraries(
        aiserver_evaluation_lifecycle_development_test PRIVATE
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
        onnxruntime
        crypto
    )
    target_compile_options(
        aiserver_evaluation_lifecycle_development_test PRIVATE -UNDEBUG)
    add_test(
        NAME aiserver_evaluation_lifecycle_development_contract
        COMMAND aiserver_evaluation_lifecycle_development_test
    )
endif()
