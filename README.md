<div align="center">
    <h1>Maiden CUDA</h1>
    <p>Rust-based CUDA library designed for learning purposes and building my AI engines named Maiden Engine</p>
    <strong>🚧 This project is for personal learning and testing purposes, so it may not function properly. 🚧</strong>
    <h3>
        <a href="TODOS.md">TODOS</a>
    </h3>
</div>

## Getting Started

### Prerequisites

- CUDA Toolkit

### Example

How to use Tensor:

```rust
use maiden_cuda::prelude::*;

fn main() -> CudaResult<()> {
    let tensor1 = Tensor::new(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ])?;

    let tensor2 = Tensor::new(vec![
        vec![7.0, 8.0, 9.0],
        vec![10.0, 11.0, 12.0],
    ])?;

    let result = tensor1.add(&tensor2)?;
    
    println!("Shape: {:?}", result.shape());
    println!("Result:\n{}", result);

    Ok(())
}
```

How to use linear module:
```rust
use maiden_cuda::prelude::*;

fn main() -> CudaResult<()> {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let linear = Linear::new_with_bias(2, 3, true)?;

    let linear_output = linear.forward(&input)?;

    let relu = ReLU::new();
    let relu_output = relu.forward(&linear_output)?;

    let sigmoid = Sigmoid::new();
    let sigmoid_output = sigmoid.forward(&relu_output)?;

    let tanh = Tanh::new();
    let tanh_output = tanh.forward(&sigmoid_output)?;

    println!("Input:\n{}", input);
    println!("Linear output:\n{}", linear_output);
    println!("ReLU output:\n{}", relu_output);
    println!("Sigmoid output:\n{}", sigmoid_output);
    println!("Tanh output:\n{}", tanh_output);

    Ok(())
}
```

For more examples, see [`examples`](examples/).

## Development Setup

### Prerequisites

- Rust
- CUDA Toolkit
- CMake
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
