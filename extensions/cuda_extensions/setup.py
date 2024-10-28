from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_extensions',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='cuda_extensions.imp',
            sources=['extensions/cuda_extensions/imp/imp.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        ),
        CUDAExtension(
            name='cuda_extensions.linear',
            sources=['extensions/cuda_extensions/linear/linear.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
