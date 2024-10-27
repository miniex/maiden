# TODOS

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
- [x] Matrix multiplication (matmul)
- [x] Activation functions (ReLU, Sigmoid, Tanh)
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
