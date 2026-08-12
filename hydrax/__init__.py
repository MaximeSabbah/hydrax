import os
from pathlib import Path

import jax

# package root
ROOT = str(Path(__file__).parent.absolute())

# Set XLA flags for better performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

# Enable persistent compilation cache. Defaults beside the checkout, never
# /tmp: /tmp is wiped on reboot, which cost a 52 s cold compile on every first
# solve after a restart. An embedder (the ROS bridge) can still redirect it --
# without the env var its setting would be overwritten right here, on import.
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", str(Path(ROOT).parent / ".jax_cache")
    ),
)
