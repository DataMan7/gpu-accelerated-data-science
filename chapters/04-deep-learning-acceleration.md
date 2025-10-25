# Chapter 4: Deep Learning Acceleration with cuDNN and TensorRT

## NVIDIA cuDNN: Deep Neural Network Library

cuDNN is NVIDIA's GPU-accelerated library of primitives for deep neural networks, providing highly tuned implementations for standard routines such as forward and backward convolution, attention mechanisms, pooling, and normalization.

### cuDNN Architecture

#### Backend API
The cuDNN backend API provides a graph-based execution model that automatically optimizes neural network operations:

```cpp
// Create cuDNN handle
cudnnHandle_t handle;
cudnnCreate(&handle);

// Define tensor descriptors
cudnnTensorDescriptor_t input_desc, output_desc;
cudnnCreateTensorDescriptor(&input_desc);
cudnnCreateTensorDescriptor(&output_desc);

// Set tensor properties
cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                          batch_size, channels, height, width);
```

#### Operation Descriptors
cuDNN provides descriptors for various neural network operations:

```cpp
// Convolution descriptor
cudnnConvolutionDescriptor_t conv_desc;
cudnnCreateConvolutionDescriptor(&conv_desc);
cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                               dilation_h, dilation_w, CUDNN_CONVOLUTION,
                               CUDNN_DATA_FLOAT);

// Pooling descriptor
cudnnPoolingDescriptor_t pool_desc;
cudnnCreatePoolingDescriptor(&pool_desc);
cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                           kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
```

### Key Features and Optimizations

#### Runtime Fusion
cuDNN automatically fuses compatible operations to reduce memory bandwidth and improve performance:

- **Conv-Bias-Activation fusion**: Combines convolution, bias addition, and activation
- **Batch normalization fusion**: Fuses normalization with adjacent operations
- **Element-wise operation fusion**: Combines multiple pointwise operations

#### Multi-Head Attention
cuDNN provides optimized implementations for transformer attention mechanisms:

```cpp
// Create attention descriptor
cudnnAttnDescriptor_t attn_desc;
cudnnCreateAttnDescriptor(&attn_desc);

// Configure attention parameters
cudnnSetAttnDescriptor(attn_desc, attn_mode, num_heads, sm_scaler, data_type,
                      compute_type, attn_dropout, post_dropout, q_size, k_size,
                      v_size, q_proj_size, k_proj_size, v_proj_size,
                      o_proj_size, qo_max_seq_len, kv_max_seq_len,
                      max_batch_size, max_beam_size);
```

#### Performance Optimizations
- **Tensor Core utilization**: Leverages NVIDIA Tensor Cores for mixed-precision computation
- **Memory layout optimization**: Automatic selection of optimal memory layouts
- **Algorithm selection**: Runtime algorithm selection based on problem size and hardware

### cuDNN Performance Benchmarks

#### Convolution Performance
- **Forward convolution**: Up to 6x speedup on ResNet-50 compared to CPU
- **Backward convolution**: Up to 8x speedup for gradient computation
- **Grouped convolution**: Optimized for depthwise separable convolutions

#### RNN Performance
- **LSTM/GRU**: 10-20x faster than CPU implementations
- **Bidirectional RNNs**: Efficient handling of bidirectional architectures
- **Multi-layer RNNs**: Optimized for deep recurrent networks

## NVIDIA TensorRT: High-Performance Inference

TensorRT is NVIDIA's SDK for high-performance deep learning inference, designed to optimize and deploy AI models from major deep learning frameworks.

### TensorRT Workflow

#### 1. Model Import and Parsing
TensorRT supports multiple model formats and frameworks:

```cpp
// Create builder and network
IBuilder* builder = createInferBuilder(gLogger);
INetworkDefinition* network = builder->createNetworkV2(0);

// Parse ONNX model
nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
parser->parseFromFile(model_file, static_cast<int>(ILogger::Severity::kWARNING));
```

#### 2. Optimization
TensorRT applies multiple optimization techniques:

```cpp
// Create optimization profile for dynamic shapes
IOptimizationProfile* profile = builder->createOptimizationProfile();
profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{1, 3, 224, 224});
profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{8, 3, 224, 224});
profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{16, 3, 224, 224});

// Build engine with optimizations
IBuilderConfig* config = builder->createBuilderConfig();
config->addOptimizationProfile(profile);
config->setMaxWorkspaceSize(1 << 30); // 1GB
config->setFlag(BuilderFlag::kFP16); // Enable FP16 precision

IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
```

#### 3. Engine Serialization and Deployment
```cpp
// Deserialize engine
IRuntime* runtime = createInferRuntime(gLogger);
ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize);

// Create execution context
IExecutionContext* context = engine->createExecutionContext();

// Allocate GPU memory
void* buffers[2];
cudaMalloc(&buffers[0], input_size);
cudaMalloc(&buffers[1], output_size);

// Execute inference
context->executeV2(buffers);
```

### TensorRT Optimization Techniques

#### Layer Fusion
TensorRT automatically fuses compatible layers to reduce computation and memory overhead:

- **Conv-BN-ReLU fusion**: Combines convolution, batch normalization, and ReLU
- **Element-wise operation fusion**: Merges consecutive element-wise operations
- **Pointwise fusion**: Combines multiple pointwise operations

#### Precision Optimization
TensorRT supports multiple precision modes for optimal performance:

```cpp
// Enable mixed precision
config->setFlag(BuilderFlag::kFP16);
config->setFlag(BuilderFlag::kINT8);

// Set precision for specific layers
ILayer* layer = network->getLayer(layerIndex);
layer->setPrecision(DataType::kHALF);
layer->setOutputType(0, DataType::kHALF);
```

#### Dynamic Shape Support
TensorRT handles models with variable input sizes:

```cpp
// Define dynamic input shapes
Dims min_dims = Dims4{1, 3, 224, 224};
Dims opt_dims = Dims4{8, 3, 224, 224};
Dims max_dims = Dims4{16, 3, 224, 224};

profile->setDimensions("input", OptProfileSelector::kMIN, min_dims);
profile->setDimensions("input", OptProfileSelector::kOPT, opt_dims);
profile->setDimensions("input", OptProfileSelector::kMAX, max_dims);
```

### TensorRT Performance Benchmarks

#### Inference Throughput
- **ResNet-50**: Up to 8x higher throughput than unoptimized inference
- **BERT**: 3-5x speedup for transformer-based models
- **YOLOv5**: Real-time object detection with sub-10ms latency

#### Memory Efficiency
- **Model size reduction**: Up to 50% smaller model footprints
- **Memory bandwidth optimization**: Reduced memory transfers
- **Workspace optimization**: Efficient memory allocation for operations

## Integration with Deep Learning Frameworks

### PyTorch Integration

#### Torch-TensorRT
```python
import torch
import torch_tensorrt

# Compile PyTorch model with TensorRT
model = torch.jit.trace(model, example_input)
trt_model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(example_input.shape)])

# Run optimized inference
output = trt_model(input_tensor)
```

#### cuDNN with PyTorch
```python
import torch
import torch.nn as nn

# cuDNN is automatically used by PyTorch for GPU operations
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1))
).cuda()

# cuDNN optimizations are applied automatically
output = model(input_tensor)
```

### TensorFlow Integration

#### TensorFlow-TensorRT
```python
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Convert SavedModel to TensorRT
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_model_dir,
    precision_mode=trt.TrtPrecisionMode.FP16
)
converter.convert()
converter.save(output_dir)
```

## Advanced Features

### Multi-Stream Execution
cuDNN and TensorRT support concurrent execution on multiple CUDA streams:

```cpp
// Create multiple streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Execute operations on different streams
cudnnSetStream(handle, stream1);
cudnnConvolutionForward(handle, ...);

cudnnSetStream(handle, stream2);
cudnnPoolingForward(handle, ...);
```

### Multi-GPU Support
Both libraries support multi-GPU configurations:

```cpp
// Set device for cuDNN operations
cudaSetDevice(device_id);
cudnnSetStream(handle, stream);

// TensorRT multi-GPU inference
for (int device = 0; device < num_devices; ++device) {
    cudaSetDevice(device);
    // Create engine and context for each device
}
```

## Performance Tuning and Profiling

### cuDNN Algorithm Selection
cuDNN provides multiple algorithms for the same operation:

```cpp
// Get available algorithms
int num_algos;
cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &num_algos);

cudnnConvolutionFwdAlgoPerf_t* perf_results =
    (cudnnConvolutionFwdAlgoPerf_t*)malloc(num_algos * sizeof(cudnnConvolutionFwdAlgoPerf_t));

cudnnFindConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc,
                                   output_desc, num_algos, &returned_algos, perf_results);

// Choose best algorithm
cudnnConvolutionFwdAlgo_t best_algo = perf_results[0].algo;
```

### TensorRT Profiling
TensorRT provides detailed profiling information:

```cpp
// Enable profiling
config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);

// Create profiler
IProfiler* profiler = createProfiler();
config->setProfiler(profiler);

// Execute and get profiling data
context->executeV2(buffers);
// Access profiling data through profiler callbacks
```

## Best Practices

### Memory Management
- Use pinned memory for host-device transfers
- Minimize memory allocations during inference
- Reuse memory buffers when possible

### Precision Selection
- Use FP16 for inference when accuracy allows
- Consider INT8 quantization for maximum performance
- Profile different precision modes for optimal balance

### Batch Size Optimization
- Experiment with different batch sizes for optimal throughput
- Consider dynamic batching for variable workloads
- Balance latency vs throughput requirements

### Model Optimization
- Remove unnecessary operations during conversion
- Use appropriate precision for different layers
- Consider model pruning and quantization techniques

## Real-World Applications

### Computer Vision
- **Image classification**: ResNet, EfficientNet optimization
- **Object detection**: YOLO, SSD acceleration
- **Image segmentation**: U-Net, DeepLab optimization

### Natural Language Processing
- **Text classification**: BERT, RoBERTa inference
- **Language generation**: GPT, T5 optimization
- **Question answering**: SQuAD, GLUE benchmark acceleration

### Recommendation Systems
- **Collaborative filtering**: Matrix factorization acceleration
- **Deep learning recommenders**: DLRM, DCN optimization
- **Real-time personalization**: Low-latency inference

This chapter provides the foundation for understanding how cuDNN and TensorRT accelerate deep learning workloads, enabling data scientists to deploy high-performance AI applications across various domains.