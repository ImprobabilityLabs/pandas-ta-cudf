# -*- coding: utf-8 -*-
from distutils.core import setup

long_description = "An easy to use Python 3 CuDF Extension with 130+ Technical Analysis Indicators. Can be called from a CuDF DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib."

setup(
    name="cudf_ta",
    packages=[
        "cudf_ta",
        "cudf_ta.candles",
        "cudf_ta.cycles",
        "cudf_ta.momentum",
        "cudf_ta.overlap",
        "cudf_ta.performance",
        "cudf_ta.statistics",
        "cudf_ta.trend",
        "cudf_ta.utils",
        "cudf_ta.utils.data",
        "cudf_ta.volatility",
        "cudf_ta.volume"
    ],
    version=".".join(("0", "3", "14b")),
    description=long_description,
    long_description=long_description,
    author="Kevin Johnson",
    author_email="appliedmathkj@gmail.com",
    url="https://github.com/twopirllc/cudf-ta",
    maintainer="Kevin Johnson",
    maintainer_email="appliedmathkj@gmail.com",
    download_url="https://github.com/twopirllc/cudf-ta.git",
    keywords=["technical analysis", "trading", "python3", "cudf"],
    license="The MIT License (MIT)",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_data={
        "data": ["data/*.csv"],
    },
    install_requires=["cudf", "cupy"],
    # List additional groups of dependencies here (e.g. development dependencies).
    # You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": [
            "alphaVantage-api", "matplotlib", "mplfinance", "scipy",
            "sklearn", "statsmodels", "stochastic",
            "talib", "tqdm", "vectorbt", "yfinance",
        ],
        "test": ["ta-lib"],
    },
)