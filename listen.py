from pymavlink import mavutil
import math
import time
import matplotlib.pyplot as plt

# !!!!!!!!!!!!!!!!!!!!!!! denna funktionen kan introducera errors !!!!!!!!!!!!!!!!!!
def lat_long_distance(lat1, lon1, lat2, lon2): # taget härifrån https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    lat1 = lat1/1e7
    lon1 = lon1/1e7
    lat2 = lat2/1e7
    lon2 = lon2/1e7

    R = 6378.137; # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180)*math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000 # meters

update_freq = 1; # hz 
lat_incr = 500

# Start a connection listening on a UDP port
# the_connection = mavutil.mavlink_connection('udpin:localhost:14551')
the_connection = mavutil.mavlink_connection('udpin:localhost:14551')

# Wait for the first heartbeat 
#   This sets the system and component ID of remote system for the link
the_connection.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (the_connection.target_system, the_connection.target_component))

# while True:
    # msg = the_connection.recv_match(type='HOME_POSITION',blocking=True)   
    # # if not msg:
    # #     continue
    # if msg.get_type() == "BAD_DATA":
    #     if mavutil.all_printable(msg.data):
    #        print(msg.data)
    #         # sys.stdout.flush()
    # else:
    #     #Message is valid
    #     # Use the attribute
    # msg = the_connection.recv_match(blocking=True)
    # print(msg)


# the_connection.mav.command_int_send(the_connection.target_system, the_connection.target_component, 0, mavutil.mavlink.MAV_CMD_DO_SET_HOME, 0, 0, 0, 0, 0, 0, lat_long_alt[0]+lat_incr, lat_long_alt[1], 575) # flytta upp denna ovanför blocking för att alltid köra eller sät blockning = False

# the_connection.mav.command_int_send(the_connection.target_system, the_connection.target_component,0, mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,1 ,0 , 0, 0, 0, 0, 0, 0, 0) # borde ändra farten till 10m/s ground speed 

def arm(the_connection):
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,1,0,0,0,0,0,0)


# def mission_item_sender(the_connection, mission_item):
#     the_connection.mav.mission_item_int_send(mission_item)
# class mission_item:
    # self.target_system = 
def mission_item_waypoint():
    return the_connection.mav.mission_item_int_send(the_connection.target_system, the_connection.target_component,
                                                    0, # waypoint_id
                                                    0, # coordinate system
                                                    0, # command for waypoint MAV_CMD https://mavlink.io/en/messages/common.html#mav_commands
                                                    0, # current false:0, true:1
                                                    0, # autocontinue 
                                                    0, # PARAM 1
                                                    0, # PARAM 2
                                                    0, # PARAM 3
                                                    0, # PARAM 4
                                                    0, # PARAM 5
                                                    0, # PARAM 6
                                                    0, # PARAM 7
                                                    0, # Mission type 0:Mission, 1:FENCE, 2:RALLY                                                   
                                                    )
def mission_item_speedchange():
    return
def mission_item_landing():
    return

while True:
    
    
    the_connection.mav.command_int_send(the_connection.target_system, the_connection.target_component, 0, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 242, 0, 0, 0, 0, 0, 0, 0, 1)

    msg = the_connection.recv_match(type='HOME_POSITION',blocking=True)
    lat_long_alt = [msg.latitude, msg.longitude, msg.altitude]
    # print(lat_long_alt)
    # print(msg)


    ################# LÄNK https://ardupilot.org/dev/docs/mavlink-get-set-home-and-origin.html

    # the_connection.mav.send(mavutil.mavlink.MAVLink_command_long_message(the_connection.target_system, the_connection.target_component, mavutil.mavlink.MAV_CMD_DO_SET_HOME, 0, 0,0, 0, 0, lat_long_alt[0], lat_long_alt[1], lat_long_alt[2]))

    # the_connection.mav.command_int_send(the_connection.target_system, the_connection.target_component,0, mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 0, 1, 10, 0, 0, 0, 0, 0) # borde ändra farten till 10m/s ground speed 

    the_connection.mav.command_int_send(the_connection.target_system, the_connection.target_component,0, mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 0, 1, 10, 0, 0, 0, 0, 0) # borde ändra farten till 10m/s ground speed 

    the_connection.mav.command_int_send(the_connection.target_system, the_connection.target_component, 0, mavutil.mavlink.MAV_CMD_DO_SET_HOME, 0, 0, 0, 0, 0, 0, lat_long_alt[0]+lat_incr, lat_long_alt[1], 575) # flytta upp denna ovanför blocking för att alltid köra eller sät blockning = False
    # movement = distance(lat_long_alt[0]+lat_incr,lat_long_alt[1], lat_long_alt[0], lat_long_alt[1])
    # print(movement)
    time.sleep(1/update_freq)

    ## mavutil.mavlink.MAV_FRAME_GLOBAL
    # conf = the_connection.recv_match(type='COMMAND_ACK',blocking=True)
    # print(conf)
   # print(msg)

# Once connected, use 'the_connection' to get and send messagesS
    

