import cudf
from numba import cuda
import numpy as np
import pandas as pd

# Define CUDA kernel
@cuda.jit
def rsx_kernel(close, length, drift, offset, result):
    idx = cuda.grid(1)
    if idx < close.size:
        if idx < length:
            result[idx] = np.nan
        else:
            # Perform calculations here
            vC, v1C = 0, 0
            v4, v8, v10, v14, v18, v20 = 0, 0, 0, 0, 0, 0
            f0, f8, f10, f18, f20, f28, f30, f38 = 0, 0, 0, 0, 0, 0, 0, 0
            f40, f48, f50, f58, f60, f68, f70, f78 = 0, 0, 0, 0, 0, 0, 0, 0
            f80, f88, f90 = 0, 0, 0
            if f90 == 0:
                f90 = 1.0
                f0 = 0.0
                if length - 1.0 >= 5:
                    f88 = length - 1.0
                else:
                    f88 = 5.0
                f8 = 100.0 * close[idx]
                f18 = 3.0 / (length + 2.0)
                f20 = 1.0 - f18
            else:
                if f88 <= f90:
                    f90 = f88 + 1
                else:
                    f90 = f90 + 1
                f10 = f8
                f8 = 100 * close[idx]
                v8 = f8 - f10
                f28 = f20 * f28 + f18 * v8
                f30 = f18 * f28 + f20 * f30
                vC = 1.5 * f28 - 0.5 * f30
                f38 = f20 * f38 + f18 * vC
                f40 = f18 * f38 + f20 * f40
                v10 = 1.5 * f38 - 0.5 * f40
                f48 = f20 * f48 + f18 * v10
                f50 = f18 * f48 + f20 * f50
                v14 = 1.5 * f48 - 0.5 * f50
                f58 = f20 * f58 + f18 * abs(v8)
                f60 = f18 * f58 + f20 * f60
                v18 = 1.5 * f58 - 0.5 * f60
                f68 = f20 * f68 + f18 * v18
                f70 = f18 * f68 + f20 * f70
                v1C = 1.5 * f68 - 0.5 * f70
                f78 = f20 * f78 + f18 * v1C
                f80 = f18 * f78 + f20 * f80
                v20 = 1.5 * f78 - 0.5 * f80

                if f88 >= f90 and f8 != f10:
                    f0 = 1.0
                if f88 == f90 and f0 == 0.0:
                    f90 = 0.0

            if f88 < f90 and v20 > 0.0000000001:
                v4 = (v14 / v20 + 1.0) * 50.0
                if v4 > 100.0:
                    v4 = 100.0
                if v4 < 0.0:
                    v4 = 0.0
            else:
                v4 = 50.0
            result[idx] = v4


def rsx(close, length=None, drift=None, offset=None, **kwargs):
    length = int(length) if length and length > 0 else 14
    close = cudf.Series(close).reset_index(drop=True)
    drift = get_drift(drift)
    offset = get_offset(offset)

    result = cudf.Series([np.nan for _ in range(0, length - 1)] + [0])

    # Allocate device memory
    close_gpu = cuda.to_device(close.values)
    result_gpu = cuda.device_array((len(close),), dtype=np.float64)

    # Launch CUDA kernel
    threadsperblock = 256
    blockspergrid = (len(close) + threadsperblock - 1) // threadsperblock
    rsx_kernel[blockspergrid, threadsperblock](close_gpu, length, drift, offset, result_gpu)

    # Copy result back to host
    result.values = result_gpu.copy_to_host()

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        result.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    result.name = f"RSX_{length}"
    result.category = "momentum"

    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        signalsdf = pd.concat(
            [
                pd.DataFrame({result.name: result}),
                signals(
                    indicator=result,
                    xa=kwargs.pop("xa", 80),
                    xb=kwargs.pop("xb", 20),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1
        )

        return signalsdf
    else:
        return result