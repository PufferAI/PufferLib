cmake_minimum_required(VERSION 3.20)
project(fight_caves_viewer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

get_filename_component(PUFFERLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
                       ABSOLUTE)
set(FC_ENV_ROOT "${PUFFERLIB_ROOT}/ocean/fight_caves")
set(FC_CORE_SOURCE_LIST "${FC_ENV_ROOT}/sources.txt")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${FC_CORE_SOURCE_LIST}")
file(STRINGS "${FC_CORE_SOURCE_LIST}" FC_CORE_SOURCE_RELATIVE)
set(FC_CORE_SOURCES "")
foreach(relative_source IN LISTS FC_CORE_SOURCE_RELATIVE)
    if(NOT relative_source STREQUAL "" AND
       NOT relative_source MATCHES "^[ \\t]*#")
        list(APPEND FC_CORE_SOURCES "${FC_ENV_ROOT}/${relative_source}")
    endif()
endforeach()

add_library(fc_viewer_core STATIC ${FC_CORE_SOURCES})
target_include_directories(fc_viewer_core PUBLIC "${FC_ENV_ROOT}/include")
target_link_libraries(fc_viewer_core PUBLIC m)
set_target_properties(fc_viewer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(FC_ACTIVE_LOADOUT "" CACHE STRING
    "Optional Fight Caves active loadout override")
if(FC_ACTIVE_LOADOUT)
    target_compile_definitions(fc_viewer_core
                               PUBLIC FC_ACTIVE_LOADOUT=${FC_ACTIVE_LOADOUT})
endif()

if(NOT RAYLIB_ROOT)
    if(APPLE)
        set(RAYLIB_ROOT "${PUFFERLIB_ROOT}/raylib-5.5_macos")
    else()
        set(RAYLIB_ROOT "${PUFFERLIB_ROOT}/raylib-5.5_linux_amd64")
    endif()
endif()
set(RAYLIB_ROOT "${RAYLIB_ROOT}" CACHE PATH "Raylib distribution root")

if(NOT EXISTS "${RAYLIB_ROOT}/include/raylib.h" OR
   NOT EXISTS "${RAYLIB_ROOT}/lib/libraylib.a")
    message(FATAL_ERROR
        "Raylib is unavailable at ${RAYLIB_ROOT}. Run "
        "'./build.sh fight_caves --viewer' from the PufferLib root.")
endif()

set(FC_VIEWER_SOURCES
    src/viewer.c
    src/fc_anim_loader.c
    src/fc_animated_atlas.c
    src/fc_asset_raylib.c
    src/fc_assets.c
    src/fc_actor_visual.c
    src/fc_actor_animation.c
    src/fc_click_feedback.c
    src/fc_combat_presentation.c
    src/fc_debug_overlay.c
    src/fc_minimap.c
    src/fc_model_animation.c
    src/fc_models.c
    src/fc_npc_models.c
    src/fc_objects_loader.c
    src/fc_osrs_text.c
    src/fc_projectile_visual.c
    src/fc_spotanims.c
    src/fc_terrain_loader.c
    src/ui.c
    src/ui_assets.c
)

add_executable(fc_viewer ${FC_VIEWER_SOURCES})
target_include_directories(fc_viewer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${FC_ENV_ROOT}/include"
    "${RAYLIB_ROOT}/include"
)
target_link_libraries(fc_viewer PRIVATE
    fc_viewer_core
    "${RAYLIB_ROOT}/lib/libraylib.a"
    m
    pthread
)

if(APPLE)
    target_link_libraries(fc_viewer PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
        "-framework OpenGL"
    )
else()
    target_link_libraries(fc_viewer PRIVATE dl)
    find_package(X11 REQUIRED)
    target_link_libraries(fc_viewer PRIVATE ${X11_LIBRARIES} GL)
endif()
