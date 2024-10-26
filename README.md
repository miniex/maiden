<div align="center">
    <h1>Maiden CUDA</h1>
    <p>Rust-based CUDA library designed for learning purposes and building my AI engines named Maiden Engine</p>
    <strong>🚧 This project is for personal learning and testing purposes, so it may not function properly. 🚧</strong>
</div>

## TODOS

- [x] Implement PyTorch tensor (partial implementation)
- [ ] Add CPU operations
- [ ] Implement basic autograd functionality
- [ ] Add common neural network layers (Linear, ReLU, etc.)
- [ ] Set up unit tests for core functionalities

## Getting Started

```rust
use maiden_cuda::prelude::Tensor;

fn main() {
    let tensor1 = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    ).expect("Failed to create tensor1");

    let tensor2 = Tensor::from_vec(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 3]
    ).expect("Failed to create tensor2");

    // Add the tensors
    let result = tensor1.add(&tensor2).expect("Failed to add tensors");
    
    // Print the result
    println!("Shape: {:?}", result.shape());
    println!("Result: {:?}", result.to_vec().expect("Failed to get result data"));
    // Output:
    // Shape: [2, 3]
    // Result: [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
}
```

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
