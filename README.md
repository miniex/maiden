<div align="center">
    <h1>MaidenX</h1>
    <p>Rust-based CUDA library designed for learning purposes and building my AI engines named Maiden Engine</p>
    <strong>🚧 This project is for personal learning and testing purposes, so it may not function properly. 🚧</strong>
    <h3>
        <a href="TODOS.md">TODOS</a>
    </h3>
</div>

## Getting Started

### Prerequisites

- CUDA Toolkit

### Details

#### Tensor

<details>
  <summary>tensor</summary>
</details>

#### Module

<details>
  <summary>nn::layer</summary>

<!-- LAYERS -->
- **Linear** - A module that applies a linear transformation to the incoming data: y = xW^T + b. The parameters are: weight of shape (out_features, in_features) and an optional bias of shape (out_features)
- **Bilinear** - A module that applies a bilinear transformation to the incoming data: y = x1 * A * x2 + b. The parameters are: weight of shape (out_features, in1_features, in2_features) and an optional bias of shape (out_features)
<!-- /LAYERS -->

</details>

<details>
  <summary>nn::activation</summary>

<!-- ACTIVATIONS -->
- **ReLU** - A module that applies the rectified linear unit function element-wise: ReLU(x) = max(0, x). Can be done in-place to save memory.
- **Sigmoid** - A module that applies the sigmoid function element-wise: Sigmoid(x) = 1 / (1 + exp(-x)). Can be done in-place to save memory.
- **Tanh** - A module that applies the hyperbolic tangent function element-wise: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). Can be done in-place to save memory.
<!-- /ACTIVATIONS -->

</details>

### Example

How to use Tensor:

```rust
use maidenx::prelude::*;

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
use maidenx::prelude::*;

fn main() -> CudaResult<()> {
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
