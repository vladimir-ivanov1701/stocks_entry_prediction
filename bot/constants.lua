-- constants used in script

ACCOUNT = '410GP12'
CLIENT_CODE = '547983'
DEPOSIT_AMT = 160000
RISK_PERCENT = 0.01
SLIP = 0
REAL_TRADING = false

POSITIONS_TABLE = 'futures_client_holding'
MOEX_CLASS = 'SPBFUT'

OPEN_HOUR = 10

WAIT_TIME = 1000

--[[
    FUTURES_LIST - main array with all stats by all futures used in trading.
    It consists of arrays with futures data:
    [1] Name
    [2] Code
    [3] Name of 5M graph
    [4] Name of 1D graph
    [5] Name of 1D ATR
    [6] Color of previous day daily candle
    [7] Stop loss size in pips
    [8] Take profit in pips
    [9] Stop loss in rubles
    [10] Take profit in rubles
    [11] Position size
    [12] Hour of stock open
    [13] Hour for position to be closed (default 23:00)
    [14] Price step
    [15] Price step cost
]]

FUTURES_LIST = {
    {
        [1] = 'Si-12.23 [FORTS]',
        [2] = 'SiZ3',
        [3] = 'Si_M5',
        [4] = 'Si_1D',
        [5] = 'Si_1D_ATR',
        [6] = 'No color',
        [7] = 0,
        [8] = 0,
        [9] = 0,
        [10] = 0,
        [11] = 0,
        [12] = 10,
        [13] = 23,
        [14] = 0,
        [15] = 0
    }
}

PARAM_TABLE_X = 0
PARAM_TABLE_Y = 570
PARAM_TABLE_DX = 1420
PARAM_TABLE_DY = 330

COLOR_DARK_GREY = RGB(120, 120, 120)
COLOR_WHITE = RGB(255, 255, 255)
COLOR_BLACK = RGB(0, 0, 0)
COLOR_BLUE = RGB(0, 220, 220)
COLOR_LIGHT_GREY = RGB(235, 235, 235)
COLOR_GREEN = RGB(50, 250, 50)
COLOR_RED = RGB(250, 50, 50)
