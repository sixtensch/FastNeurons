# Add executable

add_executable(FastNeuronsTest test.cpp)
 
set_target_properties(FastNeuronsTest PROPERTIES 
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(FastNeuronsTest PRIVATE FastNeurons)
 