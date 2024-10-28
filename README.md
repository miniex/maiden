<div align="center">
    <h1>MaidenX</h1>
    <p>Rust-based CUDA library designed for learning purposes and building my AI engines named Maiden Engine</p>
    
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/miniex/maidenx#license)
[![Crates.io](https://img.shields.io/crates/v/maidenx.svg)](https://crates.io/crates/maidenx)
    <h3>
        <a href="TODOS.md">TODOS</a>
    </h3>
</div>

> [!NOTE]  
> This project is structured to resemble the PyTorch framework where possible, to aid in familiarization and learning.

> [!WARNING]
> 🚧 This project is for personal learning and testing purposes, so it may not function properly. 🚧

## Getting Started

### Prerequisites

- CUDA Toolkit
- CMake

### How to use

|            | Using PyTorch                            | Using MaidenX                                                         |
|------------|------------------------------------------|-----------------------------------------------------------------------|
| Creation   | `torch.Tensor([[1, 2], [3, 4]])`         | `Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])`                   |
| Creation   | `torch.Tensor([[1, 2], [3, 4]])`         | `Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])`                 |
| Creation   | `nn.Sequential(A, B, C)`                 | `nn::ModuleBuilder::new().add_layer(A).add_layer(B).add_layer(C)`     |

### Example

How to use Tensor:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let linear = nn::Linear::new_with_bias(2, 3, true)?;

    let linear_output = linear.forward(&input)?;

    let relu = nn::ReLU::new();
    let relu_output = relu.forward(&linear_output)?;

    let sigmoid = nn::Sigmoid::new();
    let sigmoid_output = sigmoid.forward(&relu_output)?;

    let tanh = nn::Tanh::new();
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

For IDE support with CUDA files, create `.clangd` file in `crates/maidenx_cuda_kernels/`:

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
