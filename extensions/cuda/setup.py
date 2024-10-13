from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='imp_cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='imp_cuda_extension',
            sources=['imp.cu'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
