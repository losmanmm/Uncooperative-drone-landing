"""
Example of how to connect to the autopilot by using mavproxy's
--udpin:0.0.0.0:9000 endpoint from the companion computer itself
"""

# Disable "Bare exception" warning
# pylint: disable=W0702

import time
# Import mavutil
from pymavlink import mavutil


def wait_conn():
    """
    Sends a ping to stabilish the UDP communication and awaits for a response
    """
    msg = None
    while not msg:
        master.mav.ping_send(
            int(time.time() * 1e6), # Unix time in microseconds
            0, # Ping number
            0, # Request ping of all systems
            0 # Request ping of all components
        )
        msg = master.recv_match()
        time.sleep(0.5)

# Create the connection
#  Companion is already configured to allow script connections under the port 9000
# Note: The connection is done with 'udpout' and not 'udpin'.
#  You can check in http:192.168.1.2:2770/mavproxy that the communication made for 9000
#  uses a 'udp' (server) and not 'udpout' (client).
        
# master = mavutil.mavlink_connection('udpout:0.0.0.0:9000')
# master = mavutil.mavlink_connection('udpin:localhost:14540')
# master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200) # antagligen dumt sätt att göra det på

# master = mavutil.mavlink_connection('tcp:127.0.0.1:5760') # lägga ner krut på något sådant här 


# Send a ping to start connection and wait for any reply
#  This function is necessary when using 'udpout',
#  as described before, 'udpout' connects to 'udpin',
#  and needs to send something to allow 'udpin' to start
#  sending data.
wait_conn()

# Get some information !
while True:
    print('hallo')
    try:
        print(master.recv_match().to_dict())
    except:
        pass
    time.sleep(0.1)

# Output:
# {'mavpackettype': 'AHRS2', 'roll': -0.11364290863275528, 'pitch': -0.02841472253203392, 'yaw': 2.0993032455444336, 'altitude': 0.0, 'lat': 0, 'lng': 0}
# {'mavpackettype': 'AHRS3', 'roll': 0.025831475853919983, 'pitch': 0.006112074479460716, 'yaw': 2.1514968872070312, 'altitude': 0.0, 'lat': 0, 'lng': 0, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0, 'v4': 0.0}
# {'mavpackettype': 'VFR_HUD', 'airspeed': 0.0, 'groundspeed': 0.0, 'heading': 123, 'throttle': 0, 'alt': 3.129999876022339, 'climb': 3.2699999809265137}
# {'mavpackettype': 'AHRS', 'omegaIx': 0.0014122836291790009, 'omegaIy': -0.022567369043827057, 'omegaIz': 0.02394154854118824, 'accel_weight': 0.0, 'renorm_val': 0.0, 'error_rp': 0.08894175291061401, 'error_yaw': 0.0990816056728363}
        


# """
# Example of how to connect pymavlink to an autopilot via an UDP connection
# """

# # Disable "Bare exception" warning
# # pylint: disable=W0702

# import time
# # Import mavutil
# from pymavlink import mavutil

# # Create the connection
# #  If using a companion computer
# #  the default connection is available
# #  at ip 192.168.2.1 and the port 14550
# # Note: The connection is done with 'udpin' and not 'udpout'.
# #  You can check in http:192.168.2.2:2770/mavproxy that the communication made for 14550
# #  uses a 'udpbcast' (client) and not 'udpin' (server).
# #  If you want to use QGroundControl in parallel with your python script,
# #  it's possible to add a new output port in http:192.168.2.2:2770/mavproxy as a new line.
# #  E.g: --out udpbcast:192.168.2.255:yourport
# master = mavutil.mavlink_connection('udpin:127.0.0.1:9876')

# # Make sure the connection is valid
# master.wait_heartbeat()

# # Get some information !
# while True:
#     try:
#         print(master.recv_match().to_dict())
#     except:
#         pass
#     time.sleep(0.1)

# # Output:
# # {'mavpackettype': 'AHRS2', 'roll': -0.11364290863275528, 'pitch': -0.02841472253203392, 'yaw': 2.0993032455444336, 'altitude': 0.0, 'lat': 0, 'lng': 0}
# # {'mavpackettype': 'AHRS3', 'roll': 0.025831475853919983, 'pitch': 0.006112074479460716, 'yaw': 2.1514968872070312, 'altitude': 0.0, 'lat': 0, 'lng': 0, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0, 'v4': 0.0}
# # {'mavpackettype': 'VFR_HUD', 'airspeed': 0.0, 'groundspeed': 0.0, 'heading': 123, 'throttle': 0, 'alt': 3.129999876022339, 'climb': 3.2699999809265137}
# # {'mavpackettype': 'AHRS', 'omegaIx': 0.0014122836291790009, 'omegaIy': -0.022567369043827057, 'omegaIz': 0.02394154854118824, 'accel_weight': 0.0, 'renorm_val': 0.0, 'error_rp': 0.08894175291061401, 'error_yaw': 0.0990816056728363}




        