from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kernel_cuda_extension',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='kernel_cuda_extension',
            sources=['src/kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)