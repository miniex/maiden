fn main() {
    println!("cargo:rerun-if-changed=cuda/");

    let dst = cmake::build("cuda");
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tensor_ops");
    println!("cargo:rustc-link-lib=cudart");
}
