"""Keep host-specific libraries out of the bundles.

PyInstaller resolves shared libraries through the build machine's search
path, so a build host with its own PyTorch, Intel MKL or MPI installation
leaks those copies into the bundle next to the wheels' own libraries (seen
on Arch: /usr/lib/libtorch_cpu.so and /opt/intel/oneapi MKL). The bundles
must contain only what the Python wheels ship, plus the usual small C/C++
runtime libraries, so they run on machines without those host packages.
"""

import os
import re
import sys

_FOREIGN = re.compile(
    r"^lib(torch|c10|shm|caffe2|mkl|iomp|gomp|omp|mpi|open-pal|open-rte|fabric|pmix|"
    r"uc[mpst]|hwloc|ze_loader|vulkan|protobuf|glog|gflags|absl|event|udev|uv|numa|"
    r"ibverbs|rdmacm|cuda|cudart|nccl|nvrtc|amdhip|hiprtc|roctx|tbb|dnnl|zstd|"
    r"scipy_openblas)",
)
_HOST_ROOTS = ("/usr/", "/opt/", "/lib/", "/lib64/")


def _host(source: str) -> bool:
    if not source:
        return False
    real = os.path.realpath(source)
    inside_python = real.startswith(
        (os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix))
    )
    return real.startswith(_HOST_ROOTS) and not inside_python


def filter_binaries(binaries):
    kept, dropped = [], []
    for entry in binaries:
        dest, source = entry[0], entry[1]
        name = os.path.basename(dest)
        if _host(source) and _FOREIGN.match(name):
            dropped.append((dest, source))
        else:
            kept.append(entry)
    for dest, source in dropped:
        print(f"[litenotebook] not bundling host library {dest} ({source})")
    return kept
