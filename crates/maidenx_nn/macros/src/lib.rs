extern crate proc_macro;

use maidenx_macro_utils::MaidenXManifest;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Module)]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let manifest = MaidenXManifest::default();
    let maidenx_nn_path = manifest.get_path("maidenx_nn");
    let maidenx_tensor_path = manifest.get_path("maidenx_tensor");
    let maidenx_core_path = manifest.get_path("maidenx_core");

    let expanded = quote! {
        impl #impl_generics #maidenx_nn_path::module::Module for #name #ty_generics #where_clause {
            fn forward(&self, input: &#maidenx_tensor_path::Tensor)
                -> #maidenx_core_path::error::Result<#maidenx_tensor_path::Tensor> {
                self.forward(input)
            }

            fn parameters(&self) -> Vec<#maidenx_tensor_path::Tensor> {
                self.parameters()
            }
        }
    };

    TokenStream::from(expanded)
}
