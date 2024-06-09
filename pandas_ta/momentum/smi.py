# -*- coding: utf-8 -*-
import cudf
from cuml.tsa.tsi import tsi as cuda_tsi
from cuml.ta.overlap import ema as cuda_ema
from cuml.ta.utils import get_offset, verify_series as verify_cuda_series

def smi(close, fast=None, slow=None, signal=None, scalar=None, offset=None, **kwargs):
    """Indicator: SMI Ergodic Indicator (SMIIO)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 5
    slow = int(slow) if slow and slow > 0 else 20
    signal = int(signal) if signal and signal > 0 else 5
    if slow < fast:
        fast, slow = slow, fast
    scalar = float(scalar) if scalar else 1
    close = verify_cuda_series(close, max(fast, slow, signal))
    offset = get_offset(offset)

    if close is None: return

    # Calculate Result
    tsi_df = cuda_tsi(close, fast=fast, slow=slow, signal=signal, scalar=scalar)
    smi = tsi_df.iloc[:, 0]
    signalma = tsi_df.iloc[:, 1]
    osc = smi - signalma

    # Offset
    if offset != 0:
        smi = smi.shift(offset)
        signalma = signalma.shift(offset)
        osc = osc.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        smi.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
        osc.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        smi.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)
        osc.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _scalar = f"_{scalar}" if scalar != 1 else ""
    _props = f"_{fast}_{slow}_{signal}{_scalar}"
    smi.name = f"SMI{_props}"
    signalma.name = f"SMIs{_props}"
    osc.name = f"SMIo{_props}"
    smi.category = signalma.category = osc.category = "momentum"

    # Prepare DataFrame to return
    data = {smi.name: smi, signalma.name: signalma, osc.name: osc}
    df = cudf.DataFrame(data)
    df.name = f"SMI{_props}"
    df.category = smi.category

    return df