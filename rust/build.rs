fn main() {
    // A Python extension leaves the interpreter's symbols unresolved; on
    // macOS the linker has to be told (maturin passes the same flags).
    #[cfg(feature = "python")]
    pyo3_build_config::add_extension_module_link_args();
}
