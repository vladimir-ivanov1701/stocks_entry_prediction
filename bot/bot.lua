-- VERSION 0.1
-- Bot created

-- libraries
dofile(getScriptPath()..'\\functions.lua')

is_run = true
timer = 3

function OnInit()
    -- Actions done once before startup

    table_id = AllocTable()
    local n_instruments = #FUTURES_LIST

    for i = 1, n_instruments do
        calc_futures_data(i)
    end

    fill_params_table(table_id)
    logging('INFO', 'Robot launched!')
end

function OnTrade(TradeX)
    -- actions on new trade appearance    
end

function OnOrder(OrderX)
    -- actions on new order
end

function OnStopOrder()
    -- actions on stop-order
end

function OnStop()
    -- actions on robot turn-off
    is_run = false
    logging('INFO', 'Robot stopped!')
    return 0
end

function main()
    -- main actions
    while is_run == true do
        robot_body()
    end
end
