# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(faiss-populate NONE)



include(ExternalProject)
ExternalProject_Add(faiss-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/ahojnnes/faiss/archive/84a2fd7235be00add67d755e0300314cac669547.zip" "URL_HASH" "SHA256=919ae53ff365fd85ab2a0a51b0dd58aff03718b025817b411a95f1afeb43649f"
                    SOURCE_DIR          "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src"
                    BINARY_DIR          "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


