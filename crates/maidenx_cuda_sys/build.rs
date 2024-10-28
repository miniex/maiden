use std::path::PathBuf;
use std::process::Command;

fn find_cuda_path() -> String {
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if let Ok(path) = String::from_utf8(output.stdout) {
            if let Some(cuda_path) = path.trim().strip_suffix("/bin/nvcc") {
                return cuda_path.to_string();
            }
        }
    }

    for path in &[
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "C:/CUDA",
    ] {
        if PathBuf::from(path).exists() {
            return path.to_string();
        }
    }

    "/usr/local/cuda".to_string()
}

fn main() {
    let cuda_path = find_cuda_path();

    println!("cargo:rustc-link-search={}/lib64", cuda_path);
    println!("cargo:rustc-link-search={}/lib", cuda_path);

    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed={}/include", cuda_path);
}
