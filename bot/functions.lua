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
    [ ] Отрисовать таблицу настроек
    [ ] Заполнить таблицу настроек
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


function get_curr_balance()

    res = getItem('futures_client_limits', 0).cbplimit
    if res == nil then
         logging('ERROR', 'Could not find info about limits!')
        return 0
    end
    return res
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

    local futures_code = FUTURES_LIST[futures_index][2]
    local price_graph_name = FUTURES_LIST[futures_index][4]
    local atr_name = FUTURES_LIST[futures_index][5]
    local stock_open_hour = FUTURES_LIST[futures_index][10]
    local price_open = 0
    local price_close = 0

    local dt = {}
    dt.hour, dt.min, dt.sec = string.match(getInfoParam('SERVERTIME'), '(%d*):(%d*):(%d*)')
    local n_candles = getNumCandles(price_graph_name)
    local price_step = tonumber(getParamEx(MOEX_CLASS, futures_code, 'SEC_PRICE_STEP').param_value)
    local price_step_cost = tonumber(string.format('%.4f', getParamEx(MOEX_CLASS, futures_code, 'STEPPRICE').param_value))
    local money_on_acc = get_curr_balance()

    local candle_color = prev_candle_color(futures_index, price_graph_name, n_candles, dt.hour, stock_open_hour)
    local sl_pips, tp_pips, sl_rub, tp_rub = sl_tp_size(atr_name, dt.hour, stock_open_hour, price_step, price_step_cost)

    local pos = pos_size(money_on_acc, sl_rub)

    -- saving params to table
    FUTURES_LIST[futures_index][7] = sl_pips
    FUTURES_LIST[futures_index][8] = tp_pips
    FUTURES_LIST[futures_index][9] = sl_rub
    FUTURES_LIST[futures_index][10] = tp_rub
    FUTURES_LIST[futures_index][11] = pos
    FUTURES_LIST[futures_index][14] = price_step
    FUTURES_LIST[futures_index][15] = price_step_cost
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

    return sl_pips, tp_pips, sl_rub, tp_rub
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






function fill_params_table(table_id)
    --[[
        Function creates table with bot params.
    ]]

    AddColumn(table_id, 1, 'INSTRUMENT', true, QTABLE_STRING_TYPE, 20)
    AddColumn(table_id, 2, 'PREV_DAILY_CANDLE', true, QTABLE_STRING_TYPE, 20)
    AddColumn(table_id, 3, 'POS. SIZE', true, QTABLE_INT64_TYPE, 20)
    AddColumn(table_id, 4, 'PRICE STEP COST', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 5, 'PRICE STEP', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 6, 'LOT SIZE', true, QTABLE_INT64_TYPE, 20)
    AddColumn(table_id, 7, 'DAILY ATR', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 8, 'SL SIZE', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 9, 'TP_SIZE', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 10, 'STOP LOSS', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 11, 'TAKE PROFIT', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 12, 'MAX POS. SIZE', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 13, 'DAYS BEFORE EXP.', true, QTABLE_DOUBLE_TYPE, 20)
    AddColumn(table_id, 14, 'COMMENT', true, QTABLE_STRING_TYPE, 50)
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
        local futures_name = tostring(FUTURES_LIST[i][1])
        local futures_code = tostring(FUTURES_LIST[i][2])
        local atr_name = tostring(FUTURES_LIST[i][4])
        local curr_pos = get_curr_position(futures_code, ACCOUNT)
        local n_candles_atr = getNumCandles(atr_name)
        local atr_table, _, _ = getCandlesByIndex(atr_name, 0, n_candles_atr - 1, 1)
        local atr_value = tonumber(
            string.format(
                '%.4f',
                atr_table[0].close
            )
        )
        local max_by_n_candles = tostring(FUTURES_LIST[i][5])
        local min_by_n_candles = tostring(FUTURES_LIST[i][6])
        local days_before_exp = tonumber(days_before_expiration(i))
        local price_step_cost = tonumber(
            string.format(
                '%.4f',
                getParamEx(
                    MOEX_CLASS,
                    futures_code,
                    'STEPPRICE'
                ).param_value
            )
        )
        local price_step = tonumber(
            getParamEx(
                MOEX_CLASS,
                futures_code,
                'SEC_PRICE_STEP'
            ).param_value
        )
        local lot_size = tonumber(
            getParamEx(
                MOEX_CLASS,
                futures_code,
                'LOTSIZE'
            ).param_image
        )

        if n_candles_atr == nil then
            logging('ERROR', 'ATR unavailable. Please check graph ID and try again.')
            return 0
        end
        if lot_size == nil then
            lot_size = math.floor(
                tonumber(
                    getParamEx(
                        MOEX_CLASS,
                        futures_code,
                        'LOTSIZE'
                    ).param_value
                ) * 1000
            )
        end

        -- Max SL = daily ATR / 3
        local max_stop_loss = math.floor(atr_value / price_step * price_step_cost / 3)
        local deposit_amt = curr_balance()
        local max_risk = deposit_amt * RISK_PERCENT
        max_stop_loss = math.min(max_stop_loss, max_risk)
        FUTURES_LIST[i][7] = max_stop_loss
        
        local max_position_size = math.floor(max_risk / max_stop_loss)
        FUTURES_LIST[i][8] = max_position_size
        FUTURES_LIST[i][9] = days_before_exp
        local comment = NO_COMMENT
        if max_position_size > 5 then
            comment = 'Attention! Low SL, more then 5 lots available!'
            logging('WARNING', futures_name..': '..comment)
        elseif days_before_exp <= 1 then
            comment = 'Futures expires in '..days_before_exp..' days, please update!'
        end

        InsertRow(table_id, -1)
        SetCell(table_id, i, 1, futures_name)
        SetCell(table_id, i, 2, tostring(max_by_n_candles))
        SetCell(table_id, i, 3, tostring(min_by_n_candles))
        SetCell(table_id, i, 4, tostring(curr_pos))
        SetCell(table_id, i, 5, tostring(price_step_cost))
        SetCell(table_id, i, 6, tostring(price_step))
        SetCell(table_id, i, 7, tostring(lot_size))
        SetCell(table_id, i, 8, tostring(atr_value))
        SetCell(table_id, i, 9, tostring(max_stop_loss))
        SetCell(table_id, i, 10, tostring(max_position_size))
        SetCell(table_id, i, 11, tostring(days_before_exp))
        SetCell(table_id, i, 12, comment)

        calc_indicators(i)
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
