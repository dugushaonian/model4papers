#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()


setup(
    name = 'm4p',
    packages = find_packages(exclude=['examples']),
    version = '0.0.1',
    license='MIT',
    description = 'model for papers, implementation by pytorch',
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    author = 'Dugufei',
    author_email = 'dugufei@bjtu.edu.cn',
    url = 'https://github.com/dugushaonian/model4papers',
    keywords = [
        'artificial intelligence',
        'attention mechanism',
        'image recognition'
    ],
    install_requires=[
        'tqdm>=4.66.4',
        'accelerate>=0.33.0',
        'pytorch>=2.3.0',
        'einops>=0.8.0',
        'scikit-learn>=1.5.1',
        'torchvision>=0.18.1',
        'pillow>=10.4.0'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
        'tqdm>=4.66.4',
        'accelerate>=0.33.0',
        'pytorch>=2.3.0',
        'einops>=0.8.0',
        'scikit-learn>=1.5.1',
        'torchvision>=0.18.1',
        'pillow>=10.4.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
    ],
)