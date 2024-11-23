-- functions for bot

-- importing constants
dofile(getScriptPath()..'\\constants.lua')

function robot_body()
    -- main calculations

    local dt = {}
    dt.hour, dt.min, dt.sec = string.match(getInfoParam('SERVERTIME'), '(%d*):(%d*):(%d*)')

    local server_time = getInfoParam('SERVERTIME')
    local n_instruments = #FUTURES_LIST

    if server_time == nil or server_time == '' then
        logging('ERROR', 'Server time not received!')
    end

    if IsWindowClosed(table_id) then
        fill_params_table(table_id)
    end

    for i = 1, n_instruments do
        -- getting trade signal
        local ts = trading_signal(i, dt.hour, dt.min)
        if ts ~= NO_SIGNAL then
            if DEBUG_MODE == true then
                local dbg_msg = 'Got trading signal '..tostring(ts)..' | futures index: '..tostring(i)
                dbg_msg = dbg_msg..' | Time: '..tostring(dt.hour)..':'..tostring(dt.min)..':'..tostring(dt.sec)
                logging('DEBUG', dbg_msg)
            end
            send_transaction(i, ts)
        end
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
    local log_file_name = getScriptPath()..'\\logs\\log_'..tostring(server_date)..'.txt'
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
    local n_candles_atr = getNumCandles(FUTURES_LIST[futures_index].atr_d1_name)
    local ps = price_step(FUTURES_LIST[futures_index].futures_code, MOEX_CLASS)
    local psc = price_step_cost(FUTURES_LIST[futures_index].futures_code, MOEX_CLASS)
    local money_on_acc = curr_balance()

    local candle_color = prev_candle_color(
        futures_index,
        FUTURES_LIST[futures_index].d1_name,
        n_candles,
        dt.hour
    )
    local atr, sl_pips, tp_pips, sl_rub, tp_rub = sl_tp_size(
        FUTURES_LIST[futures_index].atr_d1_name,
        dt.hour,
        FUTURES_LIST[futures_index].open_hour,
        n_candles_atr,
        ps,
        psc
    )

    local max_pos = max_pos_size(money_on_acc, sl_rub)
    if DEBUG_MODE == true then
       logging('DEBUG', 'max position: '..tostring(max_pos)) 
    end

    -- saving params to table
    FUTURES_LIST[futures_index].atr_value = atr
    --FUTURES_LIST[futures_index].prev_candle_color = candle_color
    FUTURES_LIST[futures_index].sl_pips = sl_pips
    FUTURES_LIST[futures_index].tp_pips = tp_pips
    FUTURES_LIST[futures_index].sl_rub = sl_rub
    FUTURES_LIST[futures_index].tp_rub = tp_rub
    FUTURES_LIST[futures_index].max_pos_size = max_pos
    FUTURES_LIST[futures_index].price_step = ps
    FUTURES_LIST[futures_index].price_step_cost = psc
    logging('INFO', 'Instruments parameters saved.')
end

function prev_candle_color(futures_index, price_graph_name, n_candles, curr_hour)
    --[[
        Function returns color of previous daily candle.

        :param: futures_index    - futures number in FUTURES_LIST
        :param: price_graph_name - name of 1D graph
        :param: n_candles        - total number of candles on graph from the very beginning
        :param: curr_hour        - current hour from getInfoParam('SERVERTIME') table
    ]]
    
    local price_table, _, _ = getCandlesByIndex(
        price_graph_name,
        0,
        n_candles - 2,
        2
    )
    logging('DEBUG', tostring(price_table[0].close))

    local price_open = 0
    local price_close = 0
    local candle_color = 'no color'

    if tonumber(curr_hour) < FUTURES_LIST[futures_index].open_hour then
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

    if DEBUG_MODE == true then
        local msg = 'Function prev_candle_color. Open price: '..tostring(price_open)
        msg = msg..' | Close price: '..tostring(price_close)
        msg = msg..' | Candle color: '..tostring(candle_color)
        logging('DEBUG', msg)
    end

    return candle_color
end

function sl_tp_size(atr_name, curr_hour, stock_open_hour, n_candles_atr, price_step, price_step_cost)
    --[[
        Function calculates SL and TP size in rubles.

        :param: atr_name        - name of 1D ATR graph
        :param: curr_hour       - current hour from getInfoParam('SERVERTIME') table
        :param: stock_open_hour - hour when trades start for exact futures
        :param: n_candles - number of candles on graph
        :param: price_step      - price step for exact futures
        :param: price_step_cost - price step cost
    ]]

    local atr_value = 0
    local sl_pips = 0
    local tp_pips = 0
    local sl_rub = 0
    local tp_rub = 0

    local atr_table, _, _ = getCandlesByIndex(
        atr_name,
        0,
        n_candles_atr - 2,
        2
    )

    if tonumber(curr_hour) < stock_open_hour then
        atr_value = tonumber(string.format(
            '%.6f',    
            atr_table[1].close)
        )
    else
        atr_value = tonumber(string.format(
            '%.6f',    
            atr_table[0].close)
        )
    end

    sl_pips = tonumber(string.format('%6.f', atr_value / 3))
    tp_pips = tonumber(string.format('%6.f', atr_value * 2 / 3))

    sl_rub = sl_pips / price_step * price_step_cost
    tp_rub = tp_pips / price_step * price_step_cost

    if DEBUG_MODE == true then
        local msg = 'Function sl_tp_size. Atr value: '..tostring(atr_value)
        msg = msg..' | SL pips: '..tostring(sl_pips)
        msg = msg..' | TP pips: '..tostring(tp_pips)
        msg = msg..' | SL rub: '..tostring(sl_rub)
        msg = msg..' | TP rub: '..tostring(tp_rub)
        logging('DEBUG', msg)
    end

    return atr_value, sl_pips, tp_pips, sl_rub, tp_rub
end

function max_pos_size(balance, sl_rub)
    --[[
        Function calculates position size based on ATR value.

        :param: balance - amount of money on account
        :param: sl_rub  - stop loss in rubles
    ]]

    max_pos = math.floor(balance / #FUTURES_LIST * RISK_PERCENT / sl_rub)

    if DEBUG_MODE == true then
        logging('DEBUG', 'Function max_pos_size. Pos size: '..tostring(max_pos))
    end

    return max_pos
end

function price_step(futures_code, MOEX_CLASS)
    --[[
        Function return price step for certain instrument.
        :param: futures_code - futures_code from FUTURES_LIST[i][2].
        :param: MOEX_CLASS   - MOEX class constant.
    ]]

    local res = tonumber(getParamEx(MOEX_CLASS, futures_code, 'SEC_PRICE_STEP').param_value)

    if DEBUG_MODE == true then
        logging('DEBUG', 'Function price_step. Price step: '..tostring(res))
    end

    return res
end

function price_step_cost(futures_code, MOEX_CLASS)
    --[[
        Function return price step cost for instrument.
        :param: futures_code - futures code from FUTURES_LIST[i][2].
        :param: MOEX_CLASS   - MOEX class constant.
    ]]

    local res = tonumber(string.format('%.4f', getParamEx(
        MOEX_CLASS,
        futures_code,
        'STEPPRICE'
    ).param_value))

    if DEBUG_MODE == true then
        logging('DEBUG', 'Function price_step_cost. Price step cost: '..tostring(res))
    end

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
            )
        )
    end

    if DEBUG_MODE == true then
        logging('DEBUG', 'Function lot_size. Lot size: '..tostring(res))
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
    AddColumn(table_id, 9, 'TP SIZE PIPS', true, QTABLE_DOUBLE_TYPE, 20)
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
        local curr_pos = curr_position(ACCOUNT, FUTURES_LIST[i].futures_code)
        local ls = lot_size(FUTURES_LIST[i].futures_code, MOEX_CLASS)

        InsertRow(table_id, -1)
        SetCell(table_id, i, 1, tostring(FUTURES_LIST[i].futures_name))
        SetCell(table_id, i, 2, tostring(FUTURES_LIST[i].prev_candle_color))
        SetCell(table_id, i, 3, tostring(FUTURES_LIST[i].max_pos_size))
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
    logging('INFO', 'Parameters table initialized.')
end

function curr_price(futures_index)
    --[[
        Function returns last close price.
        :param: futures_index - instrument position in FUTURES_LIST table.
    ]]

    local n_candles = getNumCandles(FUTURES_LIST[futures_index].d1_name)
    local price_table, _, _ = getCandlesByIndex(
        FUTURES_LIST[futures_index].d1_name,
        0,
        n_candles - 1,
        n_candles
    )
    price_close = price_table[0].close
    return price_close
end

function trading_signal(futures_index, hour, minute)
    --[[
        Function returns trading signals.
        :param: futures_index - position of instrument in FUTURES_LIST table.concat
        :param: hour          - hour
        :param: minute        - minute
    ]]

    local trade_signal = NO_SIGNAL
    local c_price = curr_price(futures_index)
    local c_pos = curr_position(ACCOUNT, futures_index)

    if DEBUG_MODE == true then
        local dbg_msg = 'Function trading_signal | Current price: '..tostring(c_price)
        dbg_msg = dbg_msg..' | Current position: '..tostring(c_pos)
        dbg_msg = dbg_msg..' | Hour: '..tostring(hour)..' | Minute: '..tostring(minute)
        dbg_msg = dbg_msg..' | Open hour: '..tostring(FUTURES_LIST[futures_index].open_hour)
        dbg_msg = dbg_msg..' | Open minute: '..tostring(FUTURES_LIST[futures_index].open_minute)
        dbg_msg = dbg_msg..' | Was open: '..tostring(FUTURES_LIST[futures_index].was_open)
        logging('DEBUG', dbg_msg)
    end

    if (
        (tonumber(hour) == FUTURES_LIST[futures_index].open_hour) and
        (tonumber(minute) == FUTURES_LIST[futures_index].open_minute) and
        (c_pos == 0) and
        (FUTURES_LIST[futures_index].was_open == 0)
    ) then
        logging('DEBUG', 'Found condition for long or short')
        logging('DEBUG', 'candle color: '..FUTURES_LIST[futures_index].prev_candle_color)
        -- open long
        if FUTURES_LIST[futures_index].prev_candle_color == "green" then
            trade_signal = OPEN_LONG
        -- open short
        elseif FUTURES_LIST[futures_index].prev_candle_color == "red" then
            trade_signal = OPEN_SHORT
        end
    elseif (
        (tonumber(hour) >= FUTURES_LIST[futures_index].open_hour) and
        (tonumber(hour) < FUTURES_LIST[futures_index].close_hour)
    ) then
        -- close long
        if (
            (c_pos > 0) and
            (
                (c_price >= (FUTURES_LIST[futures_index].entry_price + FUTURES_LIST[futures_index].tp_pips)) or
                (c_price <= (FUTURES_LIST[futures_index].entry_price - FUTURES_LIST[futures_index].sl_pips))
            )
        ) then
            trade_signal = CLOSE_LONG
        -- close short
        elseif (
            (c_pos < 0) and
            (
                (c_price <= (FUTURES_LIST[futures_index].entry_price - FUTURES_LIST[futures_index].tp_pips)) or
                (c_price >= (FUTURES_LIST[futures_index].entry_price + FUTURES_LIST[futures_index].sl_pips))
            )
        ) then
            trade_signal = CLOSE_SHORT
        end
    -- close at close hour
    elseif (
        (tonumber(hour) == FUTURES_LIST[futures_index].close_hour) and
        (tonumber(minute) == 0) and
        (c_pos ~= 0)
    ) then
        if c_pos > 0 then
            trade_signal = CLOSE_LONG
        elseif c_pos < 0 then
            trade_signal = CLOSE_SHORT
        end
    end
    logging('DEBUG', 'Trading signal: '..trade_signal)
    return trade_signal
end

function send_transaction(futures_index, ts_type)
    --[[
        Function sends transaction to the server.
        :param: futures_index - futures position in FUTURES_LIST table
        :param: ts_type - trading signal
    ]]

    local actual_price = 0
    local c_pos = curr_position(ACCOUNT, futures_index)
    local operation_type = ''
    local volume = 0
    local msg = ''

    if ts_type == OPEN_LONG then
        actual_price = math.floor(curr_price(futures_index) + FUTURES_LIST[futures_index].spread_size)
        operation_type = 'B'
        volume = FUTURES_LIST[futures_index].max_pos_size + math.abs(c_pos)
    elseif ts_type == OPEN_SHORT then
        actual_price = math.floor(curr_price(futures_index) - FUTURES_LIST[futures_index].spread_size)
        operation_type = 'S'
        volume = FUTURES_LIST[futures_index].max_pos_size + math.abs(c_pos)
    elseif  ts_type == CLOSE_LONG then
        actual_price = math.floor(curr_price(futures_index) + FUTURES_LIST[futures_index].spread_size)
        operation_type = 'S'
        volume = math.abs(c_pos)
    elseif ts_type == CLOSE_SHORT then
        actual_price = math.floor(curr_price(futures_index) - FUTURES_LIST[futures_index].spread_size)
        operation_type = 'B'
        volume = math.abs(c_pos)
    end

    -- transaction params table
    transaction = {
        ['ACCOUNT'] = ACCOUNT,
        ['CLIENT_CODE'] = CLIENT_CODE,
        ['TYPE'] = 'L',
        ['TRANS_ID'] = '42',   -- no influence, just random number
        ['CLASSCODE'] = MOEX_CLASS,
        ['SECCODE'] = FUTURES_LIST[futures_index].futures_code,
        ['ACTION'] = 'NEW_ORDER',
        ['OPERATION'] = operation_type,   -- B - buy, S - sell
        ['PRICE'] = tostring(actual_price),   -- price
        ['QUANTITY'] = tostring(volume),   -- N lots to buy/sell
    }

    msg = 'Sending transaction. Account: '..tostring(ACCOUNT)
    msg = msg..' | CLIENT_CODE: '..tostring(CLIENT_CODE)
    msg = msg..' | TYPE: '..tostring('L')
    msg = msg..' | CLASSCODE: '..tostring(MOEX_CLASS)
    msg = msg..' | SECCODE: '..tostring(FUTURES_LIST[futures_index].futures_code)
    msg = msg..' | OPERATION: '..tostring(operation_type)
    msg = msg..' | PRICE: '..tostring(actual_price)
    msg = msg..' | QUANTITY: '..tostring(volume)
    logging('TRANSACTION', msg)

    if REAL_TRADING == true then
        local res = sendTransaction(transaction)
        FUTURES_LIST[futures_index].entry_price = actual_price
        if res ~= nil then
            msg = res
            logging('ERROR', msg)
        end
    else
        FUTURES_LIST[futures_index].entry_price = actual_price
    end
end
