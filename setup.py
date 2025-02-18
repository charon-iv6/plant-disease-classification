from setuptools import setup, find_packages

setup(
    name="plant-disease-classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'albumentations>=1.4.18',
        'opencv-python>=4.5.0',
        'pandas>=1.2.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'tqdm>=4.64.0',
        'pillow>=8.0.0',
        'scikit-learn>=0.24.0',
        'plotly>=5.0.0',
        'ipywidgets>=7.6.0',
        'nbformat>=5.1.0'
    ],
    author="Danial Jabbari",
    author_email="danial.jabbari@maxion.ir",
    description="Plant Disease Classification Challenge Solution",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plant-disease-classification",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 