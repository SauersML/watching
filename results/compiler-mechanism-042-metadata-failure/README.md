# First CUDA integration failure

The pinned base model compiled and repeated unobserved inference successfully. The observer stopped at the selected query kernel because ordinary PyTorch 2.10 launchers omit `launcher.bin` unless `store_cubin` is enabled. No capture-parity or mechanism result was produced. The fix reads the existing compiled binary using both the selected configuration and its cache hash. It changes metadata access only.

The failed worker was deleted. This retry incurred $0.0535 including rejected incompatible community hosts; cumulative experiment startup cost was $0.0730.
