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
    [7] ATR value
    [8] Stop loss size in pips
    [9] Take profit in pips
    [10] Stop loss in rubles
    [11] Take profit in rubles
    [12] Position size
    [13] Hour of stock open
    [14] Hour for position to be closed (default 23:00)
    [15] Price step
    [16] Price step cost
    [17] Had open deal this day
]]

TEST = {
    ["test1"] = "test"
}

FUTURES_LIST = {
    {
        ["atr_d1_name"] = 'Si_1D_ATR',
        ["atr_value"] = 0,
        ["close_hour"] = 23,
        ["d1_name"] = 'Si_1D',
        ["futures_code"] = 'SiZ3',
        ["futures_name"] = 'Si-12.23 [FORTS]',
        ["m5_name"] = 'Si_M5',
        ["open_hour"] = 10,
        ["pos_size"] = 0,
        ["prev_candle_color"] = 'No color',
        ["price_step"] = 0,
        ["price_step_cost"] = 0,
        ["sl_pips"] = 0,
        ["sl_rub"] = 0,
        ["tp_pips"] = 0,
        ["tp_rub"] = 0,
        ["was_open"] = false
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
