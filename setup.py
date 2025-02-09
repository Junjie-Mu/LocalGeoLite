from setuptools import setup, find_packages

setup(
    name="LocalGeoLite",
    version="1.0.0",
    author="JunjieMu",
    description="**LocalGeoLite** is a GIS AI assistant powered by a local large language model, designed to generate GIS-related code and answer GIS-related questions! ",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Junjie-Mu/LocalGeoLite",
    packages=find_packages(),
    install_requires=[
        #!!IMPORTANT: Torch need to be manually installed according to the CUDA version.
        #"torch"
        "transformers",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "xformers",
        "numpy",
        "IPython"
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'localgeo=LocalGeoLite.cli:main',
        ],
    },
)
