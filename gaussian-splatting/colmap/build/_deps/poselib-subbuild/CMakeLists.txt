# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(poselib-populate NONE)



include(ExternalProject)
ExternalProject_Add(poselib-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/PoseLib/PoseLib/archive/f119951fca625133112acde48daffa5f20eba451.zip" "URL_HASH" "SHA256=3faa27d9d1f6ae7cc47602a66c10ac6f64457bc65a959d1880947d3b3bfe1ed1"
                    SOURCE_DIR          "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/poselib-src"
                    BINARY_DIR          "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/poselib-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


