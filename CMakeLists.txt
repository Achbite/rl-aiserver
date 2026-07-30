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
    src/ai/maze_reward.cpp
    src/ai/astar_solver.cpp
    src/config/config_loader.cpp
    src/session/session_manager.cpp
    src/ai/onnx_inferencer.cpp
    src/model/model_manifest.cpp
    src/model/model_distributor_client.cpp
    src/sample/sample_sender.cpp
)

set(CONTRACT_CPP_DIR "" CACHE PATH "Generated C++ contract directory")
if(CONTRACT_CPP_DIR)
    set(PROTO_GEN_DIR ${CONTRACT_CPP_DIR})
else()
    set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
endif()
set(PROTO_SOURCES
    ${PROTO_GEN_DIR}/maze.pb.cc
    ${PROTO_GEN_DIR}/maze.grpc.pb.cc
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
    add_executable(run_mode_test
        tests/run_mode_test.cpp
    )
    target_include_directories(run_mode_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME run_mode_contract COMMAND run_mode_test)
    add_test(
        NAME run_launcher_mode_contract
        COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_sh_mode_test.sh
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_executable(model_manifest_test
        tests/model_manifest_test.cpp
        src/model/model_manifest.cpp
    )
    target_include_directories(model_manifest_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
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
endif()
