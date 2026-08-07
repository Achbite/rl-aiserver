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
    add_executable(config_loader_test
        tests/config_loader_test.cpp
        src/config/config_loader.cpp
    )
    target_include_directories(config_loader_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(config_loader_test PRIVATE
        protobuf::libprotobuf
    )
    add_test(NAME config_loader_contract
             COMMAND config_loader_test
                     ${CMAKE_CURRENT_SOURCE_DIR}/configs/server_config.yaml)

    add_executable(run_mode_test
        tests/run_mode_test.cpp
    )
    target_include_directories(run_mode_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME run_mode_contract COMMAND run_mode_test)

    add_executable(model_evaluation_config_test
        tests/model_evaluation_config_test.cpp
    )
    target_include_directories(model_evaluation_config_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    add_test(NAME model_evaluation_config_contract
             COMMAND model_evaluation_config_test)

    add_test(
        NAME run_launcher_mode_contract
        COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_sh_mode_test.sh
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_executable(model_manifest_test
        tests/model_manifest_test.cpp
        src/model/model_manifest.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
        ${PROTO_GEN_DIR}/maze_task.pb.cc
    )
    target_include_directories(model_manifest_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(model_manifest_test PRIVATE
        protobuf::libprotobuf
        crypto
    )
    add_test(NAME model_manifest_contract COMMAND model_manifest_test)

    add_executable(model_distributor_client_test
        tests/model_distributor_client_test.cpp
        src/model/model_distributor_client.cpp
        src/model/model_manifest.cpp
        ${PROTO_SOURCES}
    )
    target_include_directories(model_distributor_client_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
    )
    target_link_libraries(model_distributor_client_test PRIVATE
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
        crypto
    )
    add_test(NAME model_distributor_client_contract
             COMMAND model_distributor_client_test)

    add_executable(fragment_boundary_test
        tests/fragment_boundary_test.cpp
    )
    target_include_directories(fragment_boundary_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME fragment_boundary_contract
             COMMAND fragment_boundary_test)

    add_executable(sample_producer_identity_test
        tests/sample_producer_identity_test.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
    )
    target_include_directories(sample_producer_identity_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(sample_producer_identity_test PRIVATE
        protobuf::libprotobuf
    )
    add_test(NAME sample_producer_identity_contract
             COMMAND sample_producer_identity_test)

    add_executable(training_sample_budget_test
        tests/training_sample_budget_test.cpp
    )
    target_include_directories(training_sample_budget_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME training_sample_budget_contract
             COMMAND training_sample_budget_test)

    add_executable(single_map_task_controller_test
        tests/single_map_task_controller_test.cpp
        src/task/single_map_task_controller.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/maze_task.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
    )
    target_include_directories(single_map_task_controller_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(single_map_task_controller_test PRIVATE
        protobuf::libprotobuf
    )
    target_compile_options(single_map_task_controller_test PRIVATE -UNDEBUG)
    add_test(NAME single_map_task_controller_contract
             COMMAND single_map_task_controller_test)

    add_executable(episode_action_policy_test
        tests/episode_action_policy_test.cpp
        src/task/episode_action_policy.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/maze_task.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
    )
    target_include_directories(episode_action_policy_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(episode_action_policy_test PRIVATE
        protobuf::libprotobuf
    )
    target_compile_options(episode_action_policy_test PRIVATE -UNDEBUG)
    add_test(NAME episode_action_policy_contract
             COMMAND episode_action_policy_test)

    add_executable(episode_metrics_test
        tests/episode_metrics_test.cpp
        src/metrics/episode_metrics.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
    )
    target_include_directories(episode_metrics_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(episode_metrics_test PRIVATE
        protobuf::libprotobuf
    )
    add_test(NAME episode_metrics_contract COMMAND episode_metrics_test)

    add_executable(reward_contract_test
        tests/reward_contract_test.cpp
        src/ai/maze_reward.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
        ${PROTO_GEN_DIR}/maze_task.pb.cc
    )
    target_include_directories(reward_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(reward_contract_test PRIVATE
        protobuf::libprotobuf
    )
    add_test(NAME reward_contract COMMAND reward_contract_test)
    add_test(
        NAME reward_identity_contract
        COMMAND python3
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/reward_identity_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/server_config.yaml
    )

    add_executable(maze_map_contract_test
        tests/maze_map_contract_test.cpp
        src/task/maze_map_contract.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/maze_task.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
    )
    target_include_directories(maze_map_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(maze_map_contract_test PRIVATE
        protobuf::libprotobuf
        crypto
    )
    add_test(NAME maze_map_contract COMMAND maze_map_contract_test)

    add_executable(astar_solver_test
        tests/astar_solver_test.cpp
        src/ai/astar_solver.cpp
    )
    target_include_directories(astar_solver_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME astar_solver_contract COMMAND astar_solver_test)

    add_executable(maze_observation_test
        tests/maze_observation_test.cpp
        src/ai/maze_observation.cpp
        src/ai/astar_solver.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/maze_task.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
    )
    target_include_directories(maze_observation_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(maze_observation_test PRIVATE
        protobuf::libprotobuf
    )
    add_test(NAME maze_observation_contract COMMAND maze_observation_test)
endif()
