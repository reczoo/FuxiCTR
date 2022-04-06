import setuptools

with open("README.md", "r") as fd:
    long_description = fd.read()

setuptools.setup(
    name="fuxictr",
    version="1.2.0",
    author="zhujiem",
    author_email="zhujiem@users.noreply.github.com",
    description="A configurable, tunable, and reproducible library for CTR prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xue-pai/FuxiCTR",
    download_url='https://github.com/xue-pai/FuxiCTR/tags',
    packages=setuptools.find_packages(
        exclude=["tests", "data", "docs", "demo"]),
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=["pandas", "numpy", "h5py", "PyYAML>=5.1", "scikit-learn", "tqdm"],
    extras_require={
        "cpu": ["pytorch==1.7.*"],
        "gpu": ["pytorch==1.7.*"],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="Apache-2.0 License",
    keywords=['ctr prediction', 'recommender systems',
              'ctr', 'cvr', 'pytorch'],
)
