/// Attention kernels (flash attention, position-sensitive attention).
pub mod attention;
/// Kernel that decodes raw detection head output into boxes/scores/classes.
pub mod detect_decode;
/// Loss-related kernels (CIoU, classification loss).
pub mod loss;
