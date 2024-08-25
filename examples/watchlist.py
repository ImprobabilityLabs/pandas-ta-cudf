```python
# -*- coding: utf-8 -*-
import datetime as dt

from pathlib import Path
from random import random
from typing import Tuple

import cudf as pd
from cudf import DataFrame as pdDataFrame
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

from numpy import arange as npArange
from numpy import append as npAppend
from numpy import array as npArray

import alphaVantageAPI as AV
import pandas_ta as ta


def colors(colors: str = None, default: str = "GrRd"):
    aliases = {
        # Pairs
        "BkGy": ["black", "gray"],
        "BkSv": ["black", "silver"],
        "BkPr": ["black", "purple"],
        "BkBl": ["black", "blue"],
        "FcLi": ["fuchsia", "lime"],
        "GrRd": ["green", "red"],
        "GyBk": ["gray", "black"],
        "GyBl": ["gray", "blue"],
        "GyOr": ["gray", "orange"],
        "GyPr": ["gray", "purple"],
        "GySv": ["gray", "silver"],
        "RdGr": ["red", "green"],
        "SvGy": ["silver", "gray"],
        # Triples
        "BkGrRd": ["black", "green", "red"],
        "BkBlPr": ["black", "blue", "purple"],
        "GrOrRd": ["green", "orange", "red"],
        "RdOrGr": ["red", "orange", "green"],
        # Quads
        "BkGrOrRd": ["black", "green", "orange", "red"],
        # Quints
        "BkGrOrRdMr": ["black", "green", "orange", "red", "maroon"],
        # Indicators
        "bbands": ["blue", "navy", "blue"],
        "kc": ["purple", "fuchsia", "purple"],
    }
    aliases["default"] = aliases[default]
    if colors in aliases.keys():
        return aliases[colors]
    return aliases["default"]


class Watchlist(object):
    def __init__(self,
                 tickers: list, tf: str = None, name: str = None,
                 strategy: ta.Strategy = None, ds_name: str = "av", **kwargs,
                 ):
        self.verbose = kwargs.pop("verbose", False)
        self.debug = kwargs.pop("debug", False)
        self.timed = kwargs.pop("timed", False)

        self.tickers = tickers
        self.tf = tf
        self.name = name if isinstance(name, str) else f"Watch: {', '.join(tickers)}"
        self.data = None
        self.kwargs = kwargs
        self.strategy = strategy

        self._init_data_source(ds_name)

    def _init_data_source(self, ds: str) -> None:
        self.ds_name = ds.lower() if isinstance(ds, str) else "av"

        AVkwargs = {"api_key": "YOUR API KEY", "clean": True, "export": True, "output_size": "full", "premium": False}
        self.av_kwargs = self.kwargs.pop("av_kwargs", AVkwargs)
        self.ds = AV.AlphaVantage(**self.av_kwargs)
        self.file_path = self.ds.export_path

        if self.ds_name == "yahoo":
            self.ds = yf

    def _drop_columns(self, df: pdDataFrame, cols: list = None) -> pdDataFrame:
        if cols is None or not isinstance(cols, list):
            cols = ["Unnamed: 0", "date", "split", "split_coefficient", "dividend", "dividends"]
        df_columns = list(df.columns)
        if any(_ in df_columns for _ in cols):
            if self.debug:
                print(f"[i] Possible columns dropped: {', '.join(cols)}")
            df = df.drop(cols, axis=1, errors="ignore")
        return df

    def _load_all(self, **kwargs) -> dict:
        if (self.tickers is not None and isinstance(self.tickers, list) and len(self.tickers)):
            self.data = {ticker: self.load(ticker, **kwargs) for ticker in self.tickers}
            return self.data

    def load(self,
             ticker: str = None, tf: str = None, index: str = "date",
             drop: list = [], plot: bool = False, **kwargs
             ) -> pdDataFrame:

        tf = self.tf if tf is None else tf.upper()
        if ticker is not None and isinstance(ticker, str):
            ticker = str(ticker).upper()
        else:
            print(f"[!] Loading All: {', '.join(self.tickers)}")
            self._load_all(**kwargs)
            return

        filename_ = f"{ticker}_{tf}.csv"
        current_file = Path(self.file_path) / filename_

        if current_file.exists():
            file_loaded = f"[i] Loaded {ticker}[{tf}]: {filename_}"
            if self.ds_name in ["av", "yahoo"]:
                df = pd.read_csv(current_file, index_col=0)
                if not df.ta.datetime_ordered:
                    df = df.set_index(pd.to_datetime(df.index, unit='ns'))
                print(file_loaded)
            else:
                print(f"[X] {filename_} not found in {Path(self.file_path)}")
                return
        else:
            print(f"[+] Downloading[{self.ds_name}]: {ticker}[{tf}]")
            if self.ds_name == "av":
                df = self.ds.data(ticker, tf)
                if not df.ta.datetime_ordered:
                    df = df.set_index(pd.to_datetime(df[index], unit='ns'))
            if self.ds_name == "yahoo":
                yf_data = self.ds.Ticker(ticker)
                df = pd.DataFrame(yf_data.history(period="max"))
                to_save = f"{self.file_path}/{ticker}_{tf}.csv"
                print(f"[+] Saving: {to_save}")
                df.to_csv(to_save)

        df = self._drop_columns(df, drop)
        if kwargs.pop("analyze", True):
            if self.debug: print(f"[+] TA[{len(self.strategy.ta)}]: {self.strategy.name}")
            df.ta.strategy(self.strategy, timed=self.timed, **kwargs)

        df.ticker = ticker
        df.tf = tf

        if plot: self._plot(df, **kwargs)
        return df

    @property
    def data(self) -> dict:
        return self._data

    @data.setter
    def data(self, value: dict) -> None:
        if value is not None and isinstance(value, dict):
            if self.verbose:
                print(f"[+] New data")
            self._data = value
        else:
            self._data = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if isinstance(value, str):
            self._name = str(value)
        else:
            self._name = f"Watchlist: {', '.join(self.tickers)}"

    @property
    def strategy(self) -> ta.Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: ta.Strategy) -> None:
        if value is not None and isinstance(value, ta.Strategy):
            self._strategy = value
        else:
            self._strategy = ta.CommonStrategy

    @property
    def tf(self) -> str:
        return self._tf

    @tf.setter
    def tf(self, value: str) -> None:
        if isinstance(value, str):
            value = str(value)
            self._tf = value
        else:
            self._tf = "D"

    @property
    def tickers(self) -> list:
        return self._tickers

    @tickers.setter
    def tickers(self, value: Tuple[list, str]) -> None:
        if value is None:
            print(f"[X] {value} is not a value in Watchlist ticker.")
            return
        elif isinstance(value, list) and [isinstance(_, str) for _ in value]:
            self._tickers = list(map(str.upper, value))
        elif isinstance(value, str):
            self._tickers = [value.upper()]
        self.name = self._tickers

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        if isinstance(value, bool):
            self._verbose = bool(value)
        else:
            self._verbose = False

    def indicators(self, *args, **kwargs) -> any:
        pd.DataFrame().ta.indicators(*args, **kwargs)

    def __repr__(self) -> str:
        s = f"Watch(name='{self.name}', ds_name='{self.ds_name}', tickers[{len(self.tickers)}]='{', '.join(self.tickers)}', tf='{self.tf}', strategy[{self.strategy.total_ta()}]='{self.strategy.name}'"
        if self.data is not None:
            s += f", data[{len(self.data.keys())}])"
            return s
        return s + ")"
```