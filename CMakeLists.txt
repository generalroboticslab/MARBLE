cmake_minimum_required(VERSION 3.15...3.27)


# Pin the host compiler to gcc-12 on both supported architectures.
# This runs BEFORE project() so the compiler is set before CMake's
# auto-detection takes over (auto-detection happens inside project()).
# Done in CMakeLists (not in the preset) so a single preset works for
# both arm64 and x86_64 — CMake 3.22's presets v3 schema doesn't support
# ${hostSystemProcessor} in conditions, so we can't filter presets by
# arch. Honors user overrides via $env{CC} or -DCMAKE_C_COMPILER=...
# on the command line (those are stored in the initial cache and
# take precedence over the FORCE-set value below).
#
# CMAKE_HOST_SYSTEM_PROCESSOR is empty before project(), so we shell out
# to `uname -m` to detect the arch. (uname -m returns "aarch64" on
# aarch64, "x86_64" on x86_64.)
execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE _bb_host_arch
  OUTPUT_STRIP_TRAILING_WHITESPACE)
get_property(_bb_user_c_compiler CACHE CMAKE_C_COMPILER PROPERTY VALUE)
if(NOT _bb_user_c_compiler AND NOT DEFINED ENV{CC})
  if(_bb_host_arch STREQUAL "aarch64")
    set(_bb_default_c_compiler "/usr/bin/aarch64-linux-gnu-gcc-12")
  elseif(_bb_host_arch STREQUAL "x86_64")
    set(_bb_default_c_compiler "/usr/bin/gcc-12")
  endif()
  if(EXISTS "${_bb_default_c_compiler}")
    set(CMAKE_C_COMPILER "${_bb_default_c_compiler}"
        CACHE FILEPATH "C compiler (auto-pinned to gcc-12 by default)" FORCE)
  elseif(_bb_default_c_compiler)
    message(WARNING
      "MARBLE: no gcc-12 found at ${_bb_default_c_compiler}. "
      "Pass -DCMAKE_C_COMPILER=<path> to override.")
  else()
    message(WARNING
      "MARBLE: uname -m returned '${_bb_host_arch}', which is "
      "not in the supported set (aarch64, x86_64). Pass "
      "-DCMAKE_C_COMPILER=<path> to override.")
  endif()
  unset(_bb_default_c_compiler)
endif()
unset(_bb_user_c_compiler)
unset(_bb_host_arch)


project(nanobind_example
  LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Find Python and required components
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# FindPython/Support.cmake bug: on the first configure of a fresh build
# dir, _PYTHON_CHECK_DEVELOPMENT_SIGNATURE() runs *before* the include
# dir is discovered, so the cached `_Python_DEVELOPMENT_<MODULE>_SIGNATURE`
# is the MD5 of an empty INCLUDE_DIR. On the second configure, INCLUDE_DIR
# is populated (read from the cache), the recomputed signature no longer
# matches, the function UNSETS `_Python_INCLUDE_DIR` and forces a re-search
# — and the re-search then fails with "missing: Development.Module" even
# though the Python environment is intact.
#
# Invalidate the include dir, library, and signatures together so the
# second-run signature is computed against the same empty-INCLUDE_DIR
# state as the first run. (Both end up being re-discovered by the
# Python config script, which is the only way to get a stable signature.)
unset(_Python_INCLUDE_DIR     CACHE)
unset(_Python_LIBRARY_RELEASE CACHE)
unset(_Python_LIBRARY_DEBUG   CACHE)
unset(_Python_DEVELOPMENT_MODULE_SIGNATURE CACHE)
unset(_Python_DEVELOPMENT_EMBED_SIGNATURE   CACHE)
unset(_Python_INTERPRETER_SIGNATURE         CACHE)

find_package(Python 3.12
  COMPONENTS Interpreter ${DEV_MODULE} REQUIRED
  OPTIONAL_COMPONENTS Development.SABIModule)

# nanobind gates STABLE_ABI on whether TARGET Python::SABIModule
# exists. CMake's FindPython only creates that target when Python
# was built with --with-stable-abi AND a compiler probe succeeds.
# Some distributions (e.g. conda) ship a Python that fully supports
# Py_LIMITED_API but fail cmake's probe, so the target is never
# created and STABLE_ABI silently falls back to off. The empty
# placeholder below is enough — nanobind sets Py_LIMITED_API itself.
if(NOT TARGET Python::SABIModule AND Python_VERSION VERSION_GREATER_EQUAL 3.12)
  add_library(Python::SABIModule INTERFACE IMPORTED)
endif()

# Set default build type to Release
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

message(STATUS "nanobind_ROOT: ${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)
find_package(CSerialPort CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py") # Creates an empty file

# Hardware bindings (imu only) — vendored directory at hardware_bindings/.
# Built via add_subdirectory; imu_nanobind .so lands at hardware_bindings/imu/.
# Python wrapper + C++ sources both live in hardware_bindings/imu/.
add_subdirectory(hardware_bindings)