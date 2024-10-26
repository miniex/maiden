<div align="center">
    <h1>Maiden CUDA</h1>
    <p>Rust-based CUDA library designed for learning purposes and building my AI engines named Maiden Engine</p>
    <strong>🚧 This project is for personal learning and testing purposes, so it may not function properly. 🚧</strong>
</div>

## Current Status

### Implemented ✅
- Basic tensor operations with CUDA support
- Partial PyTorch-like tensor implementation
- Core CUDA infrastructure

### In Progress 🚧
- CPU operations
- Autograd functionality
- Basic neural network layers
- Unit test suite

## Roadmap

### Phase 1: Core Operations (High Priority)
- [ ] Matrix multiplication (matmul)
- [ ] Activation functions (ReLU, Sigmoid, Tanh)
- [ ] Backpropagation and autograd system
- [ ] Convolution operations (primarily conv2d)
- [ ] Batch normalization

### Phase 2: Performance Optimization
- [ ] cuBLAS integration
- [ ] cuDNN integration
- [ ] Stream-based asynchronous operations
- [ ] Memory pool management
- [ ] Memory optimization strategies

### Phase 3: Training Components
- [ ] Optimizers (SGD, Adam, AdamW)
- [ ] Loss functions
- [ ] Gradient clipping
- [ ] Learning rate schedulers

### Phase 4: Scalability
- [ ] Multi-GPU support
- [ ] Distributed training foundations
- [ ] Mixed precision training (FP16)

### Phase 5: Development Tools
- [ ] Model serialization
- [ ] Training progress monitoring
- [ ] Example implementations
- [ ] Comprehensive documentation

### Optional Features
- [ ] Python bindings (PyO3)

## Getting Started

### Prerequisites

- CUDA Toolkit

### Example

```rust
use maiden_cuda::prelude::Tensor;

fn main() {
    let tensor1 = Tensor::new(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]).expect("Failed to create tensor1");

    let tensor2 = Tensor::new(vec![
        vec![7.0, 8.0, 9.0],
        vec![10.0, 11.0, 12.0],
    ]).expect("Failed to create tensor2");

    let result = tensor1.add(&tensor2).expect("Failed to add tensors");
    
    println!("Shape: {:?}", result.shape());
    println!("Result: {:?}", result.to_vec().expect("Failed to get result data"));
}
```

For more examples, see [`examples`](examples/).

## Development Setup

### Prerequisites

- Rust
- CUDA Toolkit
- Cmake
- clangd


### LSP Setup

For IDE support with CUDA files, create `.clangd` file in `crates/cuda-kernels/`:

```yaml
CompileFlags:
  Remove: 
    - "-forward-unknown-to-host-compiler"
    - "-rdc=*"
    - "-Xcompiler*"
    - "--options-file"
    - "--generate-code*"
  Add: 
    - "-xcuda"
    - "-std=c++14"
    - "-I/YOUR/CUDA/PATH/include"    # Update this path
    - "-I../../cuda-headers"
    - "--cuda-gpu-arch=sm_75"
  Compiler: clang

Index:
  Background: Build

Diagnostics:
  UnusedIncludes: None
```

Find your CUDA include path:

```bash
# Linux
which nvcc | sed 's/\/bin\/nvcc//'

# Windows (PowerShell)
(Get-Command nvcc).Path -replace '\\bin\\nvcc.exe',''
```
