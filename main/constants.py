# параметры чтения данных
DATA_PATH = "src/stock_data.xlsx"
FUTURES_NAME = "Si"

# параметры расчёта фичей
MAX_SHIFT = 8
MSAD_PERIODS = [3, 5, 10, 25, 50]

# стоп лосс и тейк профит
SL_RUB = 200
TP_RUB = 400

# параметры расчёта стоимости пункта
PIPS_COSTS = {
    "Si": {
        "PRICE_STEP": 1,
        "PRICE_STEP_COST": 1
    },
    "Eu": {
        "PRICE_STEP": 1,
        "PRICE_STEP_COST": 1
    },
    "CNY": {
        "PRICE_STEP": 0.001,
        "PRICE_STEP_COST": 1
    },
    "ED": {
        "PRICE_STEP": 0.0001,
        "PRICE_STEP_COST": 8.51646
    },
    "BR": {
        "PRICE_STEP": 0.01,
        "PRICE_STEP_COST": 8.51646
    },
    "NG": {
        "PRICE_STEP": 0.001,
        "PRICE_STEP_COST": 8.51646
    },
    "SBRF": {
        "PRICE_STEP": 1,
        "PRICE_STEP_COST": 1
    },
    "GAZR": {
        "PRICE_STEP": 1,
        "PRICE_STEP_COST": 1
    },
    "MOEX": {
        "PRICE_STEP": 0.05,
        "PRICE_STEP_COST": 0.5
    },
    "RTS": {
        "PRICE_STEP": 0.5,
        "PRICE_STEP_COST": 8.51646
    },
    "GOLD": {
        "PRICE_STEP": 0.1,
        "PRICE_STEP_COST": 8.51646
    },
    "SILV": {
        "PRICE_STEP": 0.01,
        "PRICE_STEP_COST": 8.51646
    },
    "NASD": {
        "PRICE_STEP": 1,
        "PRICE_STEP_COST": 0.85165
    },
    "SPYF": {
        "PRICE_STEP": 0.01,
        "PRICE_STEP_COST": 0.85165
    }
}
