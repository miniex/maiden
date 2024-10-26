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
    let shape = vec![2, 3, 4];
    let mut data = Vec::new();
    for i in 0..24 {
        data.push(i as f32);
    }

    let tensor = Tensor::from_vec(data, shape).expect("Failed to create tensor");

    println!("Tensor shape: {:?}", tensor.shape());
    println!("Tensor strides: {:?}", tensor.strides());
    println!(
        "Tensor data: {:?}",
        tensor.to_vec().expect("Failed to get tensor data")
    );
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
