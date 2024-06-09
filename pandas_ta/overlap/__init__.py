import cudf
from .alma import alma
from .dema import dema
from .ema import ema
from .fwma import fwma
from .hilo import hilo
from .hl2 import hl2
from .hlc3 import hlc3
from .hma import hma
from .hwma import hwma
from .ichimoku import ichimoku
from .jma import jma
from .kama import kama
from .linreg import linreg
from .ma import ma
from .mcgd import mcgd
from .midpoint import midpoint
from .midprice import midprice
from .ohlc4 import ohlc4
from .pwma import pwma
from .rma import rma
from .sinwma import sinwma
from .sma import sma
from .ssf import ssf
from .supertrend import supertrend
from .swma import swma
from .t3 import t3
from .tema import tema
from .trima import trima
from .vidya import vidya
from .vwap import vwap
from .vwma import vwma
from .wcp import wcp
from .wma import wma
from .zlma import zlma

def calculate_indicators(data):
    data = cudf.DataFrame.from_pandas(data)
    data['alma'] = alma(data['close'])
    data['dema'] = dema(data['close'])
    data['ema'] = ema(data['close'])
    data['fwma'] = fwma(data['close'])
    data['hilo'] = hilo(data['high'], data['low'])
    data['hl2'] = hl2(data['high'], data['low'])
    data['hlc3'] = hlc3(data['high'], data['low'], data['close'])
    data['hma'] = hma(data['close'])
    data['hwma'] = hwma(data['close'])
    data['ichimoku'] = ichimoku(data['high'], data['low'], data['close'])
    data['jma'] = jma(data['close'])
    data['kama'] = kama(data['close'])
    data['linreg'] = linreg(data['close'])
    data['ma'] = ma(data['close'])
    data['mcgd'] = mcgd(data['close'])
    data['midpoint'] = midpoint(data['high'], data['low'])
    data['midprice'] = midprice(data['high'], data['low'])
    data['ohlc4'] = ohlc4(data['open'], data['high'], data['low'], data['close'])
    data['pwma'] = pwma(data['close'])
    data['rma'] = rma(data['close'])
    data['sinwma'] = sinwma(data['close'])
    data['sma'] = sma(data['close'])
    data['ssf'] = ssf(data['close'])
    data['supertrend'] = supertrend(data['high'], data['low'], data['close'])
    data['swma'] = swma(data['close'])
    data['t3'] = t3(data['close'])
    data['tema'] = tema(data['close'])
    data['trima'] = trima(data['close'])
    data['vidya'] = vidya(data['close'])
    data['vwap'] = vwap(data['close'])
    data['vwma'] = vwma(data['close'])
    data['wcp'] = wcp(data['close'])
    data['wma'] = wma(data['close'])
    data['zlma'] = zlma(data['close'])
    return data