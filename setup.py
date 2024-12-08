from setuptools import setup 

setup( 
    name='torchqc', 
    version='0.2', 
    description='Library for quantum system simulations based on Pytorch', 
    author='Dimitris Koutromanos', 
    author_email='dkoutromanos@upatras.gr', 
    packages=['torchqc'], 
    install_requires=[ 
        'numpy', 
        'torch',
        'matplotlib'
    ], 
)