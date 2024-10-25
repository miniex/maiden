#[link(name = "tensor_ops")]
extern "C" {
    pub fn tensor_add(a: *const f32, b: *const f32, c: *mut f32, n: i32) -> i32;
    pub fn tensor_multiply(a: *const f32, b: *const f32, c: *mut f32, n: i32) -> i32;
}
