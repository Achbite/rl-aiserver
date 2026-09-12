cmake_minimum_required(VERSION 3.16)
project(rl_sdk LANGUAGES CXX)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(RL_SDK_GRPC REQUIRED IMPORTED_TARGET grpc++)
add_library(rl_sdk INTERFACE)
add_library(rl_sdk::sdk ALIAS rl_sdk)
target_compile_features(rl_sdk INTERFACE cxx_std_17)
target_include_directories(rl_sdk INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rl_sdk INTERFACE protobuf::libprotobuf PkgConfig::RL_SDK_GRPC pthread)
set_property(TARGET rl_sdk PROPERTY RL_SDK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

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

# Generate a task and its imported messages in the consuming project's build
# directory. IMPORT_DIRS preserve the import names used in the agreed Proto.
function(rl_sdk_generate_task target)
    cmake_parse_arguments(TASK "" "PROTO" "IMPORT_DIRS" ${ARGN})
    if(NOT TASK_PROTO OR TASK_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "rl_sdk_generate_task requires PROTO <file> and IMPORT_DIRS <roots...>")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(RL_SDK_GRPC_CPP_PLUGIN grpc_cpp_plugin)
    if(NOT RL_SDK_GRPC_CPP_PLUGIN)
        message(FATAL_ERROR "RL-SDK task generation requires grpc_cpp_plugin")
    endif()
    get_target_property(sdk_root rl_sdk RL_SDK_SOURCE_DIR)
    set(generator "${sdk_root}/tools/generate-task")
    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/${target}-generated")
    set(arguments --task-proto "${TASK_PROTO}" --sdk-root "${sdk_root}"
        --protoc "${Protobuf_PROTOC_EXECUTABLE}"
        --grpc-plugin "${RL_SDK_GRPC_CPP_PLUGIN}" --output-dir "${output_dir}")
    foreach(import_dir IN LISTS TASK_IMPORT_DIRS)
        get_filename_component(import_dir "${import_dir}" ABSOLUTE)
        list(APPEND arguments --import-dir "${import_dir}")
    endforeach()
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${generator}" ${arguments} --describe
        RESULT_VARIABLE described OUTPUT_VARIABLE description ERROR_VARIABLE describe_error)
    if(NOT described EQUAL 0)
        message(FATAL_ERROR "RL-SDK task Proto discovery failed: ${describe_error}")
    endif()
    string(REPLACE "\n" ";" description "${description}")
    set(sources)
    set(outputs)
    set(inputs)
    foreach(entry IN LISTS description)
        if(entry MATCHES "^input:(.+)$")
            list(APPEND inputs "${CMAKE_MATCH_1}")
        elseif(entry MATCHES "^cpp:(.+)$")
            list(APPEND sources "${output_dir}/${CMAKE_MATCH_1}.pb.cc")
            list(APPEND outputs "${output_dir}/${CMAKE_MATCH_1}.pb.cc"
                "${output_dir}/${CMAKE_MATCH_1}.pb.h")
        elseif(entry MATCHES "^service:(.+)$")
            list(APPEND sources "${output_dir}/${CMAKE_MATCH_1}.grpc.pb.cc")
            list(APPEND outputs "${output_dir}/${CMAKE_MATCH_1}.grpc.pb.cc"
                "${output_dir}/${CMAKE_MATCH_1}.grpc.pb.h"
                "${output_dir}/${CMAKE_MATCH_1}.sdk.pb.h")
        endif()
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${inputs})
    add_custom_command(OUTPUT ${outputs}
        COMMAND "${Python3_EXECUTABLE}" "${generator}" ${arguments}
        DEPENDS ${inputs} "${generator}" "${sdk_root}/tools/protoc-gen-rl-sdk"
        COMMENT "RL-SDK: compile ${TASK_PROTO} and its imports" VERBATIM)
    add_library(${target} STATIC ${sources})
    target_include_directories(${target} BEFORE PUBLIC "${output_dir}")
    target_link_libraries(${target} PUBLIC rl_sdk::sdk)
endfunction()
