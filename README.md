<div align="center">
    <h1>MaidenX</h1>
    <p>Rust ML Framework designed for learning purposes and building my AI engines named Maiden Engine</p>
    
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/miniex/maidenx#license)
[![Crates.io](https://img.shields.io/crates/v/maidenx.svg)](https://crates.io/crates/maidenx)
    <h3>
        <a href="TODOS.md">TODOS</a>
    </h3>
</div>

> This project is structured to resemble the PyTorch framework where possible,
> to aid in familiarization and learning.

> 🚧 This project is for personal learning and testing purposes,
> so it may not function properly. 🚧

## Getting Started

### Prerequisites

- if you want to use CUDA
    - CUDA Toolkit
    - CMake

### Example

```toml
[dependencies]
maidenx = { version = "0.0.5", features = ["full"] }
# only cpu
# maidenx = { version = "0.0.5" }
# only cuda, but cpu is default
# maidenx = { version = "0.0.5", features = ["cuda"] }
```

How to use:

```rust
use maidenx::prelude::*;
use std::f32;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_current_device(Device::cpu())?;

    let x = Tensor::linspace(-f32::consts::PI, f32::consts::PI, 2000)?;
    let y = sin(&x)?;

    let mut a = Tensor::randn(&[1])?;
    let mut b = Tensor::randn(&[1])?;
    let mut c = Tensor::randn(&[1])?;
    let mut d = Tensor::randn(&[1])?;

    let start = Instant::now();

    let learning_rate = 1e-6;
    for t in 1..=5000 {
        let y_pred = &a + &b * &x + &c * x.pow(2.0)? + &d * x.pow(3.0)?;
        let loss = (&y_pred - &y).pow(2.0)?.sum()?;

        if t % 100 == 0 {
            println!("t: {}, loss: {}", t, loss.item()?);
        }

        let grad_y_pred = 2.0 * (&y_pred - &y);
        let grad_a = grad_y_pred.sum()?;
        let grad_b = (&grad_y_pred * &x).sum()?;
        let grad_c = (&grad_y_pred * x.pow(2.0)?).sum()?;
        let grad_d = (&grad_y_pred * x.pow(3.0)?).sum()?;

        a = &a - learning_rate * grad_a;
        b = &b - learning_rate * grad_b;
        c = &c - learning_rate * grad_c;
        d = &d - learning_rate * grad_d;
    }

    let elapsed = start.elapsed();

    println!(
        "Result: y = {} + {}x + {}x^2 + {}x^3",
        a.item()?,
        b.item()?,
        c.item()?,
        d.item()?
    );
    println!("Time: {:?}", elapsed);

    Ok(())
}
```

How to select device:
*default: cpu*

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set CUDA if present on startup
    if let Ok(cuda_device) = Device::cuda(0) {
        set_current_device(cuda_device)?;
    }
    // all subsequent operations will use the set device
    ...
    Ok(())
}
```

How to use `DeviceGuard`:

```rust
use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
   // CPU
   {
       let _guard = DeviceGuard::new(Device::cpu())?;
       
       let a = Tensor::new(vec![1.0, 2.0, 3.0])?;
       let b = Tensor::new(vec![4.0, 5.0, 6.0])?;
       let c = a.add(&b)?;
       println!("CPU Result: {}", c);
   } // CPU guard drops here

   // CUDA 
   if let Ok(cuda_device) = Device::cuda(0) {
       let _guard = DeviceGuard::new(cuda_device)?;
       
       let x = Tensor::new(vec![1.0, 2.0, 3.0])?;
       let y = Tensor::new(vec![4.0, 5.0, 6.0])?;
       let z = x.add(&y)?;
       println!("CUDA Result: {}", z);
   } // CUDA guard drops here

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
