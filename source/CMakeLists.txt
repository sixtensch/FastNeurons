# Add library

add_library(FastNeurons STATIC cnn.cu cnn.h)

set_target_properties(FastNeurons PROPERTIES 
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${NEURONS_BIN_DIR}"
)

target_compile_features(FastNeurons PUBLIC cxx_std_17)