
local PARAM_TABLE_KEY = 7
local PARAM_TABLE_PREFIX = "SHIP_"

local takeoff_alti = 100
local land_alti = 30
local takeoff_initiated = false
local landing_initiated = false

local wp = mavlink_mission_item_int_t()
local to = mavlink_mission_item_int_t()
local land = mavlink_mission_item_int_t()
local wpland = mavlink_mission_item_int_t()
local home = Location()
wp:command(16)
to:command(22)
land:command(21)
wpland:command(16)
--wpglide:command(16)

--Atuomatic takeoff when pre-arm is good
function update_takeoff()

    if arming:pre_arm_checks() then
        arming:arm()
        home = ahrs:get_home()
        wp:x (home:lat() + 100)
        wp:y (home:lng() + 100)
        wp:z (takeoff_alti)
        to:z(takeoff_alti)
    end

    if arming:is_armed() and not takeoff_initiated then

        mission:set_item(mission:num_commands(), wp)
        mission:set_item(mission:num_commands(), to)
        gcs:send_text (0, "Mission added")

        vehicle:set_mode(10)
        vehicle:start_takeoff(takeoff_alti)
        gcs:send_text(0,string.format("Takeoff initiated"))
        takeoff_initiated = true
    end
end

function update_landing()

    wpland:x (home:lat() + 30000)
    wpland:y (home:lng() + 0)
    wpland:z (land_alti)
    land:x (home:lat() + 0)
    land:y (home:lng() + 0)
    land:z (0)

    if vehicle:get_mode() == 11 then
        vehicle:set_mode(10)
        mission:set_item(mission:num_commands(), wpland)
        --mission:set_item(mission:num_commands(), wpglide)
        mission:set_item(mission:num_commands(), land)

        landing_initiated = true
    end

end
--main update function
function update()
    update_takeoff()
    if not takeoff_initiated then
        return
    end
    update_landing()
    if not landing_initiated then
        return
    end

end

function loop()
    update()
    return loop, 50
end

return loop()