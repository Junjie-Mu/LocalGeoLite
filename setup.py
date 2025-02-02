from setuptools import setup, find_packages

setup(
    name="LocalGeoLite",
    version="1.0.0",
    author="JunjieMu",
    description="",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Junjie-Mu/LocalGeoLite",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "unsloth",
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
    python_requires=">=3.8",
)
