from setuptools import setup, find_packages


setup(
    name='Transformer Transducer',
    version='latest',
    packages=find_packages(),
    description='Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss',
    author='Sangchun Ha',
    author_email='seomk9896@naver.com',
    url='https://github.com/hasangchun/Transformer-Transducer',
    install_requires=[
        'torch>=1.4.0',
        'numpy',
    ],
    python_requires='>=3.6',
)