-- functions for bot

--[[
    Запуск бота
    [ ] Посчитать показатели
        [x] цвет предыдущей свечи
        [x] размер стопа в пунктах
        [x] размер тейка в пунктах
        [x] размер стопа в рублях
        [x] размер тейка в рублях
        [x] размер позиции
    [X] Отрисовать таблицу настроек
    [ ] Проверить время
    [ ] Если время 10:00 и нет открытых позиций - открыть лонг/шорт по рынку
    [ ] Рассчитать стоп лосс и тейк профит
    [ ] Записать стоп лосс и тейк профит в таблицу параметров
    [ ] Отобразить стоп и тейк в таблице параметров
    [ ] Если цена пробила стоп лосс или тейк профит - закрыть позицию по рынку
    [ ] Если время 23:00 и позиция открыта - закрыть по рынку
]]

-- importing constants
dofile(getScriptPath()..'\\constants.lua')

function robot_body()
    -- main calculations

    local dt = {}
    dt.hour, dt.min, dt.sec = string.match(getInfoParam('SERVERTIME'), '(%d*):(%d*):(%d*)')

    local server_time = getInfoParam('SERVERTIME')
    if server_time == nil or server_time == '' then
        logging('ERROR', 'Server time not received!')
    end

    if IsWindowClosed(table_id) then
        fill_params_table(table_id)
    end

    for i = 1, n_instruments do
        --здесь вызов функции со всей математикой
        get_trading_signal(i)
    end

    sleep(WAIT_TIME)
end

function logging(msg_type, message_text)
    --[[
        Logging bot's actions.
        Logs are written in the floder "logs" near the bot script.

        :param: msg_type     - level of logging (ex. "INFO")
        :param: message text - text of message that is written to log file
    ]]

    local server_date = getInfoParam('TRADEDATE')
    local server_time = getInfoParam('SERVERTIME')
    local output_message = server_date..' '..server_time..' '..msg_type..' '..message_text..'\n'
    local log_file_name = getScriptPath()..'\\logs\\log_'..tostring(server_date)..'_'..'txt'
    local f = io.open(log_file_name, 'r+')
    if f == nil then
        f = io.open(log_file_name, 'w')
    end
    if f ~= nil then
        f:seek('end', 0)
        f:write(output_message)
        f:flush()
        f:close()
    end
end


function curr_balance()

    res = getItem('futures_client_limits', 0).cbplimit
    if res == nil then
         logging('ERROR', 'Could not find info about limits!')
        return 0
    end
    return res
end

function curr_position(account, futures_code)
    --[[
        Function for getting current nett position from futures table.
        :param: account      - account ID
        :param: futures_code - futures code from FUTURES_LIST[i][2]
    ]]

    local table_size = getNumberOf(POSITIONS_TABLE)
    if table_size ~= nil then
        for i = 0, table_size - 1 do
            local row = getItem(POSITIONS_TABLE, i)
            if (
                row ~= nil and
                row.sec_code == futures_code and
                row.trdaccid == account
            ) then
                return row.totalnet
            end
        end
    end
    return 0
end

function calc_futures_data(futures_index)
    --[[
        Function calculates missing params in FUTURES_LIST:
        - color of previous daily candle
        - stop loss size in rubles
        - take profit size in rubles
        - position size

        :param: futures_index - futures number in FUTURES_LIST
    ]]

    local dt = {}
    dt.hour, dt.min, dt.sec = string.match(getInfoParam('SERVERTIME'), '(%d*):(%d*):(%d*)')
    local n_candles = getNumCandles(FUTURES_LIST[futures_index].d1_name)
    local ps = price_step(FUTURES_LIST[futures_index].futures_code, MOEX_CLASS)
    local psc = price_step_cost(FUTURES_LIST[futures_index].futures_code, MOEX_CLASS)
    local money_on_acc = get_curr_balance()

    local candle_color = prev_candle_color(
        futures_index,
        FUTURES_LIST[futures_index].d1_name,
        n_candles,
        dt.hour,
        FUTURES_LIST[futures_index].open_hour
    )
    local atr, sl_pips, tp_pips, sl_rub, tp_rub = sl_tp_size(
        FUTURES_LIST[futures_index].atr_d1_name,
        dt.hour,
        FUTURES_LIST[futures_index].open_hour,
        ps,
        psc
    )

    local pos = pos_size(money_on_acc, sl_rub)

    -- saving params to table
    FUTURES_LIST[futures_index].atr_value = atr
    FUTURES_LIST[futures_index].sl_pips = sl_pips
    FUTURES_LIST[futures_index].tp_pips = tp_pips
    FUTURES_LIST[futures_index].sl_rub = sl_rub
    FUTURES_LIST[futures_index].tp_rub = tp_rub
    FUTURES_LIST[futures_index].pos_size = pos
    FUTURES_LIST[futures_index].price_step = ps
    FUTURES_LIST[futures_index].price_step_cost = psc
end

function prev_candle_color(futures_index, price_graph_name, n_candles, curr_hour, stock_open_hour)
    --[[
        Function returns color of previous daily candle.

        :param: futures_index    - futures number in FUTURES_LIST
        :param: price_graph_name - name of 1D graph
        :param: n_candles        - total number of candles on graph from the very beginning
        :param: curr_hour        - current hour from getInfoParam('SERVERTIME') table
        :param: stock_open_hour  - hour when trades start for exact futures
    ]]
    
    local price_table, _, _ = getCandlesByIndex(
        price_graph_name,
        0,
        n_candles - 1,
        n_candles
    )

    if curr_hour < stock_open_hour then
        price_open = price_table[1].open
        price_close = price_table[1].close
    else
        price_open = price_table[0].open
        price_close = price_table[0].close
    end

    if price_close > price_open then
        candle_color = "green"
    elseif price_close < price_open then
        candle_color = "red"
    end
    return candle_color
end

function sl_tp_size(atr_name, curr_hour, stock_open_hour, price_step, price_step_cost)
    --[[
        Function calculates SL and TP size in rubles.

        :param: atr_name        - name of 1D ATR graph
        :param: curr_hour       - current hour from getInfoParam('SERVERTIME') table
        :param: stock_open_hour - hour when trades start for exact futures
        :param: price_step      - price step for exact futures
        :param: price_step_cost - price step cost
    ]]

    local atr_value = 0
    local atr_table, _, _ = getCandlesByIndex(
        atr_name,
        0,
        n_candles - 1,
        n_candles
    )

    if curr_hour < stock_open_hour then
        atr_value = atr_table[1]
    else
        atr_value = atr_table[0]
    end

    sl_pips = atr_value / 3
    tp_pips = atr_value * 2 / 3

    sl_rub = sl_pips / price_step * price_step_cost
    tp_rub = tp_pips / price_step * price_step_cost

    return atr_value, sl_pips, tp_pips, sl_rub, tp_rub
end

function pos_size(balance, sl_rub)
    --[[
        Function calculates position size based on ATR value.

        :param: balance - amount of money on account
        :param: sl_rub  - stop loss in rubles
    ]]

    pos = math.floor(balance * RISK_PERCENT / sl_rub)
    return pos
end

function price_step(futures_code, MOEX_CLASS)
    --[[
        Function return price step for certain instrument.
        :param: futures_code - futures_code from FUTURES_LIST[i][2].
        :param: MOEX_CLASS   - MOEX class constant.
    ]]

    local res = tonumber(getParamEx(MOEX_CLASS, futures_code, 'SEC_PRICE_STEP').param_value)
    return res
end

function price_step_cost(futures_code, MOEX_CLASS)
    --[[
        Function return price step cost for instrument.
        :param: futures_code - futures code from FUTURES_LIST[i][2].
        :param: MOEX_CLASS   - MOEX class constant.
    ]]

    local res = tonumber(string.format('%.4f', getParamEx(MOEX_CLASS, futures_code, 'STEPPRICE').param_value))
    return res
end

function lot_size(futures_code, MOEX_CLASS)
    --[[
        Function return lot size.
        :param: futures_code - futures code from FUTURES_LIST[i][2].
        :param: MOEX_CLASS   - MOEX class constant.
    ]]

    local res = tonumber(
        getParamEx(
        MOEX_CLASS,
        futures_code,
        'LOTSIZE'
        ).param_image
    )

    if res == nil then
        res = math.floor(
            tonumber(
                getParamEx(
                    MOEX_CLASS,
                    futures_code,
                    'LOTSIZE'
                ).param_value
            ) * 1000
        )
    end
    return res
end

function fill_params_table(table_id)
    --[[
        Function creates table with bot params.
        :param: table_id - ID of table.
    ]]

    AddColumn(table_id, 1, 'INSTRUMENT', true, QTABLE_STRING_TYPE, 20)
    AddColumn(table_id, 2, 'PREV_DAILY_CANDLE', true, QTABLE_STRING_TYPE, 20)
    AddColumn(table_id, 3, 'POS. SIZE', true, QTABLE_INT64_TYPE, 20)
    AddColumn(table_id, 4, 'PRICE STEP', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 5, 'PRICE STEP COST', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 6, 'LOT SIZE', true, QTABLE_INT64_TYPE, 20)
    AddColumn(table_id, 7, 'DAILY ATR', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 8, 'SL SIZE PIPS', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 9, 'TP_SIZE PIPS', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 10, 'SL SIZE RUB', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 11, 'TP SIZE RUB', true, QTABLE_DOUBLE_TYPE, 20)
    CreateWindow(table_id)

    Clear(table_id)
    SetWindowPos(table_id, PARAM_TABLE_X, PARAM_TABLE_Y, PARAM_TABLE_DX, PARAM_TABLE_DY)
    SetWindowCaption(table_id, 'Trading robot')

    n_instruments = #FUTURES_LIST

    if n_instruments == 0 then
        logging('ERROR', 'No instruments added to FUTURES_LIST, nothing to calculate.')
        return 0
    end

    for i = 1, n_instruments do
        local futures_code = tostring(FUTURES_LIST[i][2])
        local curr_pos = get_curr_position(ACCOUNT, futures_code)
        local atr_value = FUTURES_LIST[i][7]
        local pos_s = FUTURES_LIST[i][12] --position size
        local ps = FUTURES_LIST[i][14] --price step
        local psc = FUTURES_LIST[i][15] --price step cost
        local ls = lot_size(futures_code, MOEX_CLASS)

        InsertRow(table_id, -1)
        SetCell(table_id, i, 1, tostring(FUTURES_LIST[i].futures_name))
        SetCell(table_id, i, 2, tostring(FUTURES_LIST[i].prev_candle_color))
        SetCell(table_id, i, 3, tostring(FUTURES_LIST[i].pos_size))
        SetCell(table_id, i, 4, tostring(FUTURES_LIST[i].price_step))
        SetCell(table_id, i, 5, tostring(FUTURES_LIST[i].price_step_cost))
        SetCell(table_id, i, 6, tostring(ls))
        SetCell(table_id, i, 7, tostring(FUTURES_LIST[i].atr_value))
        SetCell(table_id, i, 8, tostring(FUTURES_LIST[i].sl_pips))
        SetCell(table_id, i, 9, tostring(FUTURES_LIST[i].tp_pips))
        SetCell(table_id, i, 10, tostring(FUTURES_LIST[i].sl_rub))
        SetCell(table_id, i, 11, tostring(FUTURES_LIST[i].tp_rub))
    end

    local n_rows, n_cols = GetTableSize(table_id)
    for i = 1, n_rows do
        if i % 2 == 0 then
            SetColor(
                table_id,
                i,
                QTABLE_NO_INDEX,
                COLOR_LIGHT_GREY,
                COLOR_BLACK,
                COLOR_BLUE,
                COLOR_BLACK
            )
        else
            SetColor(
                table_id,
                i,
                QTABLE_NO_INDEX,
                COLOR_WHITE,
                COLOR_BLACK,
                COLOR_BLUE,
                COLOR_BLACK
            )
        end
    end
    logging('INFO', 'Parameters table initialized')
end
