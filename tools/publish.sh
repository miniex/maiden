#!/bin/bash

crates=(
    maiden_cuda-sys
    maiden_cuda-core
    maiden_cuda-kernels
    maiden_macro_utils
    maiden_nn/macros
    maiden_tensor
    maiden_nn
    maiden_internal
    maiden_cuda
)

if [ -n "$(git status --porcelain)" ]; then
    echo "You have local changes!"
    exit 1
fi

pushd crates

for crate in "${crates[@]}"
do
  echo "Publishing ${crate}"
  cp ../LICENSE-MIT "$crate"
  cp ../LICENSE-APACHE "$crate"
  pushd "$crate"
  git add LICENSE-MIT LICENSE-APACHE
  cargo publish --no-verify --allow-dirty
  popd
  sleep 20
done

popd

echo "Publishing root crate"
cargo publish --allow-dirty

echo "Cleaning local state"
git reset HEAD --hard
