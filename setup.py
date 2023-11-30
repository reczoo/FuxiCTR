import setuptools

with open("README.md", "r", encoding="utf-8") as fd:
    long_description = fd.read()

setuptools.setup(
    name="fuxictr",
    version="2.1.3",
    author="RECZOO",
    author_email="reczoo@users.noreply.github.com",
    description="A configurable, tunable, and reproducible library for CTR prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reczoo/FuxiCTR",
    download_url='https://github.com/reczoo/FuxiCTR/tags',
    packages=setuptools.find_packages(
        exclude=["model_zoo", "tests", "data", "docs", "demo"]),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=["pandas", "numpy", "h5py", "PyYAML>=5.1", "scikit-learn", "tqdm"],
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0 License",
    keywords=['ctr prediction', 'recommender systems',
              'ctr', 'cvr', 'pytorch'],
)
