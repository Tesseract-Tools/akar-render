
#####################################################################
# akar-render (Tesseract Studio): Akar's photoreal renderer.
# Appended to Cycles' src/app/CMakeLists.txt by the build; AKAR_RENDER_DIR is this repo's host/.
#####################################################################

if(WITH_CYCLES_STANDALONE AND DEFINED AKAR_RENDER_DIR)
  add_executable(akar_render ${AKAR_RENDER_DIR}/akar_render.cpp)
  target_link_libraries(akar_render PRIVATE ${LIB})
  set_target_properties(akar_render PROPERTIES OUTPUT_NAME "akar-render")
  if(CYCLES_STANDALONE_REPOSITORY)
    cycles_install_libraries(akar_render)
  endif()
  install(TARGETS akar_render DESTINATION ${app_install_dir})
endif()
