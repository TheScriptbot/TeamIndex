# This file is used to startup the python virtual environment when using VS-Code.
# To use this, add 
# {
#     "cmake.configureArgs": [
#         "-DCMAKE_TOOLCHAIN_FILE=./cmake/toolchain_init.cmake"
#     ]
# }
# to your settings.json file in the .vscode folder.
# Of course, you first need to setup your virtual environment (virtualenv --python=3.12 build_env)
# message(STATUS "The path of the current CMake file is: ${CMAKE_CURRENT_LIST_DIR}/../build_env/bin/activate")
execute_process(
    COMMAND bash -c "source ${CMAKE_CURRENT_LIST_DIR}/../build_env/bin/activate && env"
    OUTPUT_VARIABLE ENV_VARS
)
string(REPLACE "\n" ";" ENV_VARS_LIST "${ENV_VARS}")
foreach(ENV_VAR ${ENV_VARS_LIST})
    string(REGEX MATCH "([^=]+)=(.*)" _ ${ENV_VAR})
    if(CMAKE_MATCH_1 AND CMAKE_MATCH_2)
        set(ENV{${CMAKE_MATCH_1}} "${CMAKE_MATCH_2}")
    endif()
endforeach()

