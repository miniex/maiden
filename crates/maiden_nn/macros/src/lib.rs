extern crate proc_macro;

use maiden_macro_utils::MaidenManifest;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Module)]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let manifest = MaidenManifest::default();
    let maiden_nn_path = manifest.get_path("maiden_nn");
    let maiden_tensor_path = manifest.get_path("maiden_tensor");
    let maiden_cuda_core_path = manifest.get_path("maiden_cuda-core");

    let expanded = quote! {
        impl #impl_generics #maiden_nn_path::module::Module for #name #ty_generics #where_clause {
            fn forward(&self, input: &#maiden_tensor_path::Tensor)
                -> #maiden_cuda_core_path::error::CudaResult<#maiden_tensor_path::Tensor> {
                unimplemented!();
            }

            fn parameters(&self) -> Vec<#maiden_tensor_path::Tensor> {
                vec![]
            }
        }
    };

    TokenStream::from(expanded)
}
