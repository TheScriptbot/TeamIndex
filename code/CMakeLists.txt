cmake_minimum_required(VERSION 3.15)
project(TeamIndex LANGUAGES CXX)

set(PYPROJECT_NAME "_${PROJECT_NAME}")


if(SKBUILD)
    message(STATUS "The project is built using scikit-build")
endif()

if (NOT PROJECT_VERSION)
    if (EXISTS "${CMAKE_SOURCE_DIR}/VERSION")
        file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION)
        string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
        if (NOT PROJECT_VERSION)
            set(PROJECT_VERSION "0.0.0")
        endif()
    else()
        set(PROJECT_VERSION "0.0.0")
    endif()
endif()
message(STATUS "Version: ${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fno-omit-frame-pointer")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O0 -fno-omit-frame-pointer")
endif()
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

find_program(PYTHON_EXECUTABLE python3)

####################################################################################################
### CMake configuration options:

option(ENABLE_FASTPFOR "Build with support for compression codecs from the FastPFOR library." ON)
option(ENABLE_ROARING "Build with support for roaring bitmaps." ON)
option(GENERATE_COMPILE_COMMANDS "Enable generation of compile_commands.json" OFF)


if(NOT DEFINED ZSTD_MORE_LEVEL)
    add_definitions(-DZSTD_MORE_LEVEL=20)
else()
    add_definitions(-DZSTD_MORE_LEVEL=${ZSTD_MORE_LEVEL})
endif()

# Conditionally enable compile_commands.json generation, useful for IDEs, such as VS Code
if(GENERATE_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

####################################################################################################
#### Main target, but usually embedded into the python module:
add_library(${PROJECT_NAME} STATIC 
    ${PROJECT_SOURCE_DIR}/cpp/src/create/quantizer.cpp
    ${PROJECT_SOURCE_DIR}/cpp/src/create/inverter.cpp
    ${PROJECT_SOURCE_DIR}/cpp/src/runtime/runtime.cpp
    ${PROJECT_SOURCE_DIR}/cpp/src/runtime/io/Storage.cpp
    ${PROJECT_SOURCE_DIR}/cpp/src/runtime/io/liburingBackend.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 1)
target_compile_definitions(${PROJECT_NAME} PUBLIC VERSION_INFO=${PROJECT_VERSION})

target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)

target_include_directories(${PROJECT_NAME} PRIVATE /usr/include)
include_directories(cpp/include)


####################################################################################################
#### Backend dependencies:
include(FetchContent)

# # Taskflow (necessary for runtime lib)
# FetchContent_Declare(taskflow 
#     GIT_REPOSITORY https://github.com/taskflow/taskflow.git 
#     GIT_TAG v3.9.0 
#     CMAKE_ARGS
#         -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#         -DCMAKE_MESSAGE_LOG_LEVEL=WARNING
#         # -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern/zstd/
#     EXCLUDE_FROM_ALL
# )
# if(NOT taskflow_POPULATED)
#     message("Fetching external dependency \"taskflow\"...")
#     FetchContent_MakeAvailable(taskflow)
# endif()
# 
# target_include_directories(${PROJECT_NAME} PUBLIC ${taskflow_SOURCE_DIR})

# Liburing (necessary for runtime lib)
find_package(PkgConfig REQUIRED)
pkg_check_modules(uring REQUIRED IMPORTED_TARGET liburing)

target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::uring)


####################################################################################################
#### Frontend/Python bindings:

# pybind11 (for interface, not necessary for runtime lib)
find_package(Python COMPONENTS Interpreter Development)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_CMAKEDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKEDIR}")
    find_package(pybind11 CONFIG REQUIRED)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${pybind11_INCLUDE_DIRS})

## Frontend target, using pybind11
pybind11_add_module(${PYPROJECT_NAME} 
    ${PROJECT_SOURCE_DIR}/cpp/src/interface/interface_main.cpp 
    ${PROJECT_SOURCE_DIR}/cpp/src/interface/creation_interface.cpp 
    ${PROJECT_SOURCE_DIR}/cpp/src/interface/runtime_interface.cpp
)


target_include_directories(${PYPROJECT_NAME} PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(${PYPROJECT_NAME} PRIVATE ${PROJECT_NAME})

####################################################################################################
##### Additional dependencies:
### Compressions

## Roaring is mandatory, as all set operations are implemented using it
include(cmake/CPM.cmake)
CPMAddPackage(
    NAME roaring
        GITHUB_REPOSITORY "RoaringBitmap/CRoaring"
        GIT_TAG v2.0.4
        OPTIONS "BUILD_TESTING OFF"
)

target_link_libraries(${PROJECT_NAME} PRIVATE roaring::roaring)
target_link_libraries(${PYPROJECT_NAME} PRIVATE roaring::roaring)


## ZSTD is mandatory
find_package(zstd QUIET)

if(NOT ZSTD_FOUND)
    message("ZSTD not found on system, using FetchContent..")
    set(ZSTD_BUILD_STATIC ON)
    set(ZSTD_BUILD_SHARED OFF)

    set(ZSTD_BUILD_TESTS OFF)
    set(ZSTD_LEGACY_SUPPORT OFF)
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_CONTRIB OFF)
    add_definitions(-DZSTD_MORE_LEVEL=20) # compression level, when using ZSTDMORE compression option for lists
    add_definitions(-DZSTD_DISABLE_DEPRECATE_WARNINGS=OFF)
    
    # Specify custom directories for the external project
    set(ZSTD_SOURCE_DIR ${CMAKE_BINARY_DIR}/external/compressions/zstd)
    set(ZSTD_BINARY_DIR ${CMAKE_BINARY_DIR}/external/compressions/zstd_build)
    
    FetchContent_Declare(
        zstd
        URL "https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR ${CMAKE_SOURCE_DIR}/external/compressions/zstd
        BINARY_SUBDIR ${CMAKE_BINARY_DIR}/external/compressions/zstd
    )
    
    set(ZSTD_BUILD_TESTS OFF)
    set(ZSTD_LEGACY_SUPPORT OFF)
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_CONTRIB OFF)
    add_definitions(-DZSTD_DISABLE_DEPRECATE_WARNINGS=OFF)
    if(NOT zstd_POPULATED)
        message("Fetching external dependency \"zstd\"...")
        FetchContent_MakeAvailable(zstd)
    endif()

    install(TARGETS zstd LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
get_property(zstd DIRECTORY PROPERTY IMPORTED_TARGETS)
message(STATUS "zstd imported targets: ${IMPORTED_TARGETS}")

target_link_libraries(${PROJECT_NAME} PRIVATE zstd)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_link_libraries(${PYPROJECT_NAME} PRIVATE zstd)

## optional:
if (ENABLE_FASTPFOR)
    message(STATUS "FastPFOR compression enabled.")

    add_definitions(-DENABLE_FASTPFOR=ON)
    
    find_package(FastPFOR QUIET)

    if(NOT FastPFOR_FOUND)

        FetchContent_Declare(FastPFOR
            GIT_REPOSITORY https://github.com/fast-pack/FastPFor.git
            GIT_TAG        v0.3.0  # or the desired commit/branch/tag
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DWITH_TEST=OFF
            EXCLUDE_FROM_ALL
        )
        set(WITH_TEST OFF CACHE BOOL "" FORCE)
        
        if(NOT FastPFOR_POPULATED)
            message("Fetching external dependency \"FastPFOR\"...")
            FetchContent_MakeAvailable(FastPFOR)
        endif()
        
        add_custom_target(FastPFOR_install ALL
            COMMAND ${CMAKE_COMMAND} 
                --install ${fastpfor_BINARY_DIR}
                --prefix ${fastpfor_BINARY_DIR}/../fastpfor-install
            DEPENDS FastPFOR
        )
         
        add_dependencies(${PROJECT_NAME} FastPFOR_install) # FastPFOR
        set(FastPFOR_INCLUDE_DIRS ${fastpfor_BINARY_DIR}/../fastpfor-install/include)
    endif()

    target_include_directories(${PROJECT_NAME} PUBLIC ${FastPFOR_INCLUDE_DIRS})
    # target_include_directories(${PYPROJECT_NAME} PRIVATE ${FastPFOR_INCLUDE_DIRS})

    target_link_libraries(${PROJECT_NAME} PUBLIC FastPFOR)
    # target_link_libraries(${PYPROJECT_NAME} PRIVATE FastPFOR)
endif()




####################################################################################################

add_executable(standalone_runtime ${PROJECT_SOURCE_DIR}/cpp/src/runtime/standalone_runtime.cpp)
target_link_libraries(standalone_runtime PRIVATE ${PROJECT_NAME})
target_include_directories(standalone_runtime PUBLIC ${PROJECT_SOURCE_DIR}/extern)


#### Testing and benchmarking:
## benchmark target, without python
add_executable(throughput_benchmark EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/cpp/src/benchmark/throughput_benchmark.cpp)
target_link_libraries(throughput_benchmark PRIVATE ${PROJECT_NAME} pthread)

add_executable(test_query_importer EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/tests/test_query_importer.cpp)
target_include_directories(test_query_importer PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_link_libraries(test_query_importer PRIVATE ${PROJECT_NAME})

add_executable(test_liburingBackend EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/tests/io/test_liburingBackend.cpp)
target_link_libraries(test_liburingBackend PRIVATE ${PROJECT_NAME})

add_executable(test_dramBackend EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/tests/io/test_dramBackend.cpp)
target_link_libraries(test_dramBackend PRIVATE ${PROJECT_NAME})

install(TARGETS ${PYPROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})