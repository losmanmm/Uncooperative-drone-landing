import cv2
from dt_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from geopy import distance
import pyproj           # ungefär lika dåliga resultat med
import pymap3d as pm    # dessa, pymap3d kanske har lite mer potential
from ultralytics import YOLO
# from PIL import Image




def lat_long_to_xy(lat1, lon1, lat2, lon2):
    
    # Define the UTM projection for zone 31 in France
    PROJ = '+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs'

    # Create a Proj object
    p1 = pyproj.Proj(PROJ, preserve_units=True)

    x1, y1 = p1(lon1, lat1) #- p1(lat2, lon2) # går inte att subtrahera tuplar
    x2, y2 = p1(lon2, lat2)
    x, y = x1-x2, y1-y2
    return x, y

def plot_point(point, angle, length):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.

     Will plot the line on a 10 x 10 plot.
     '''

     # unpack the first point
     x, y = point

     # find the end point
     endy = y + length * math.sin(angle)
     endx = x + length * math.cos(angle)

     # plot the points
    #  fig = plt.figure()
    #  ax = plt.subplot(111)
    #  plt.set_ylim([0, 10])   # set the bounds to be 10, 10
    #  plt.set_xlim([0, 10])
     plt.plot([x, endx], [y, endy])
     plt.draw()
    #  fig.show()

# def lat_long_distance(lat1,lon1,lat2,lon2):
#     lat1 = lat1/1e0
#     lon1 = lon1/1e0

#     lat2 = lat2/1e0
#     lon2 = lon2/1e0
#     return distance.distance((lat1,lon1), (lat2,lon2), ellipsoid='WGS-84').m



# denna kankse går att göra till bästa?????????????????????????????????????????????
# def lat_long_distance(lat1, lon1, lat2, lon2):

#     lat1 = lat1/1e0
#     lon1 = lon1/1e0

#     lat2 = lat2/1e0
#     lon2 = lon2/1e0
#     # print(lat2)

#     difflat = 111132.92-559.82*math.cos(2*(lat1))-0.0023*math.cos(6*(lat1))-111132.92-559.82*math.cos(2*(lat2))-0.0023*math.cos(6*(lat2))
#     difflong = 111412.84*math.cos(lon1)-93.5*math.cos(3*(lon1))+0.118*math.cos(5*(lon1))-111412.84*math.cos(lon2)-93.5*math.cos(3*(lon2))+0.118*math.cos(5*(lon2))
#     return   np.linalg.norm([difflat/lat1,difflong/lon1])

# from math import cos, asin, sqrt
# def lat_long_distance(lat1, lon1, lat2, lon2):
#     p = 0.017453292519943295
#     a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
#     return 12742 * asin(sqrt(a))*1000

# def lat_long_distance(lat1, lon1, lat2, lon2): # taget härifrån https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
#     lat1 = lat1/1e7
#     lon1 = lon1/1e7
#     lat2 = lat2/1e7
#     lon2 = lon2/1e7

#     R = 6378.137; # Radius of earth in KM
#     dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
#     dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
#     a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180)*math.sin(dLon/2) * math.sin(dLon/2)
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#     d = R * c
#     return d * 1000*1e6 # meters # la till 1e6 litar inte på denna 

y_model = YOLO('yolov8n.pt')
april_tags = False
visualization = True
r_earth = 6378137 # m

drone_times = []
boat_times = []

d_lat = []
d_long = []
rel_alt = [] # drone altitiude relative start?
b_lat = []
b_long = []
b_alt = []
b_bearing =[]

video_string = '0082' # 

capture = cv2.VideoCapture("/mnt/c/plugg/Examensarbete/Videor/Daaataflyg1/DJI_"+video_string+".MP4")
# capture = cv2.VideoCapture("/mnt/c/plugg/Examensarbete/Videor/Daaataflyg1/DJI_"+video_string+"LRF.MP4") # LRF
# capture = cv2.VideoCapture("/mnt/c/plugg/Examensarbete/Videor/Daaataflyg1/DJI_"+video_string+"1080p.MP4") # 1080p

# capture = cv2.VideoCapture("/mnt/c/plugg/Examensarbete/Videor/DJI_kalibrering/DJI_0089_3.087_utv_mtb_297.MP4") # utvärderings captures:  DJI_0087_11.102_utv, DJI_0088_7.390_utv, DJI_0089_3.087_utv_mtb_297
file_path_camera_srt = '/mnt/c/plugg/Examensarbete/Videor/Daaataflyg1/DJI_'+video_string+'.SRT'
file_path_phone =      '/mnt/c/plugg/Examensarbete/Videor/Daaataflyg1/gps_mob1.txt'


Camera_matrix = [[1.31754161e+03, 0.00000000e+00, 1.01639924e+03],[0.00000000e+00, 1.31547107e+03, 5.24436193e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
Camera_matrix = np.array(Camera_matrix)*2 # adjusting due to calibration being done in 1080p 1080p =*1, 4k = *2 ,720p = /1.5
# Camera_matrix = Camera_matrix/Camera_matrix[2,2] # hmmmmmmm
camera_params = (Camera_matrix[0,0], Camera_matrix[1,1], Camera_matrix[0,2], Camera_matrix[1,2])


with open(file_path_camera_srt) as file:
    format_string = '%Y-%m-%d %H:%M:%S.%f'
    for line in file:
        if line.startswith('2024-'):
            datetime_object = datetime.strptime(line.rstrip(), format_string) # rstrip to remove trailing newline
            drone_times.append(datetime_object) 

        if line.startswith('['):
            d_lat.append(float(line.split('latitude: ')[1][:8]))
            d_long.append(float(line.split('longitude: ')[1][:8]))
            rel_alt.append(float(line.split('rel_alt: ')[1][:4])) 
            

with open(file_path_phone) as file_p:
    format_string = '%Y-%m-%d %H:%M:%S'
    for line in file_p:
        splitted = line.split(',')
        if splitted[2] == 'latitude': # Skipping first entry in the file since it is a string of what it will contain 
            continue
        boat_times.append(datetime.strptime(splitted[1], format_string) + timedelta(hours=1)) 
        b_lat.append(float(splitted[2]))
        b_long.append(float(splitted[3]))
        b_alt.append(float(splitted[5]))
        b_bearing.append(float(splitted[8])) # added start value in file on second row was empty otherwise 

bools = np.logical_and(np.array(boat_times) >= np.array(drone_times[0]), np.array(boat_times) <= np.array(drone_times[-1])) # greater or equal then the first element and less or equal then last element

b_lat = np.array(b_lat)[bools]
b_long = np.array(b_long)[bools] 
boat_times = np.array(boat_times)[bools] 

# interpolate the mobile gps data so it gets as many points as there are frames(as the mavic drone has also)
# we will assume constand speed between the data points
b_interpolated_lat = []
b_interpolated_long = []
b_alt_interpolated = []
b_bearing_interpolated = []
# boat_times_interpolated =[]
num_interpolations = 25
for i in range(1, len(b_lat)):
    # interpolated_lat = np.linspace(b_lat[i-1],b_lat[i],num_interpolations)
    # interpolated_long = np.linspace(b_long[i-1],b_long[i],num_interpolations)
    b_interpolated_lat.append(np.linspace(b_lat[i-1],b_lat[i],num_interpolations))
    b_interpolated_long.append(np.linspace(b_long[i-1],b_long[i],num_interpolations))
    b_alt_interpolated.append(np.linspace(b_alt[i-1],b_alt[i],num_interpolations))
    b_bearing_interpolated.append(np.linspace(b_bearing[i-1],b_bearing[i],num_interpolations))
    # boat_times_interpolated.append()
    # interpolated_boat_lat = np.linspace(b_lat[i])

b_interpolated_lat = np.concatenate(b_interpolated_lat, axis=0)
b_interpolated_long = np.concatenate(b_interpolated_long, axis=0)
b_alt_interpolated = np.concatenate(b_alt_interpolated, axis=0)
b_bearing_interpolated = np.concatenate(b_bearing_interpolated, axis=0)
print(len(b_alt_interpolated))
#### TODO fixa interpolering av tid(kanske inte behövs men hade nog varit bra),
# sedan är målet att köra videon och flytta hemstationen enligt videons och gps datan samt printa den relativa positionen samtidigt. 




# plt.scatter(d_lat, d_long)
# plt.scatter(b_interpolated_lat, b_interpolated_long)

# plt.show()


# exit()




# distances = np.zeros(b_interpolated_lat.size)
# for i in range(len(b_interpolated_long)):
#     distances[i] = lat_long_distance(d_lat[i], d_long[i], b_interpolated_lat[i], b_interpolated_long[i])

x = np.zeros([b_interpolated_lat.size,1])
y = np.zeros([b_interpolated_lat.size,1])
for i in range(len(b_interpolated_long)):
    # print(i)

    x[i],y[i] = lat_long_to_xy(d_lat[i], d_long[i], b_interpolated_lat[i], b_interpolated_long[i])
    # print(x[0:400])



### Plotta i rätt riktning, vi måste veta i vilken frame vi adderar de locala xyz värderna, antagligen bör detta göras när den kollar rätt mot norr alternatict inågot gps koordinatformat
# Vi tar riktningen från gps punkterna så att den blir hyffsad finns det andra sätt??
# If tag så kan vi ta tagens position i skärmen och se hur långt ifrån det ät camera center och sedan plotta vinkeln från kamera center + båtens absoluta bearing
def boat_cam_angle(boat_bearing, tag, camera_matrix):
    x0, y0 = [camera_matrix[0,2],camera_matrix[1,2]] # should be [x0, y0] camera center
    # tag_pixel_offset = tag.center - [x0, y0]
    invcam = np.linalg.inv(camera_matrix)
    l_center = invcam.dot([x0, y0, 1.0])
    extended = np.append(tag.center, 1 )
    l_tag = invcam.dot(extended) # behöva concantenata denna??
    cos_angle = l_center.dot(l_tag) / (np.linalg.norm(l_center) * np.linalg.norm(l_tag))
    angle_radians = np.arccos(cos_angle)
    tag_deg_offset = np.rad2deg(angle_radians)
    # approx_drone_bearing = boat_bearing + tag_deg_offset
    return tag_deg_offset # rad2deg korrekt här?

def bbox_centerangle(bbox_center, camera_matrix):
    x0, y0 = [camera_matrix[0,2],camera_matrix[1,2]]
    invcam = np.linalg.inv(camera_matrix)
    l_center = invcam.dot([x0, y0, 1.0])
    xb, yb = bbox_center
    # print([yb, xb, 1])
    l_bbox = invcam.dot([yb, xb, 1])
    cos_angle = l_center.dot(l_bbox) / (np.linalg.norm(l_center) * np.linalg.norm(l_bbox))
    angle_radians = np.arccos(cos_angle)
    bbox_deg_offset = np.rad2deg(angle_radians)
    return bbox_deg_offset


def boat_angledim_compensator(boat_dimensions, boat_bearing, drone_bearing): #, boat_distance, bbox_centerangle):
    # TODO compensate for when the boat is turned in different ways towards the camera for example if its 90degrees bounding box should be length meters instead of width meters.
    drone_bearingdeg = np.rad2deg(drone_bearing)
    v = drone_bearingdeg%360 - boat_bearing # mod 360 fixes - negative degrees to postiive as we want ti
    print(v)
    b_length, b_width = boat_dimensions
    side_length = abs(b_width*math.sin(np.deg2rad(v)))+abs(b_length*math.cos(np.deg2rad(v)))# this length is not taking the perspective distorsions into account
    print(side_length)
    return side_length

def boat_distance(boat_dimensions, boat_bearing, drone_bearing, camera_matrix, bbox):
    # boat dimensions in [length,width]
    # boat bearing in degrees

    # compen_len = boat_angledim_compensator(boat_dimensions, boat_bearing, drone_bearing)  # this function is not working right now 
    compen_len = 4
    
    focal_l = camera_matrix[0][0] # behöver antagligen fixa så att kamera matrisen får formen: [f 0 0; 0 f 0; x x 1 ] tror inte det då pajjar man uplösnings kompensatinone bara
    dist = focal_l*compen_len/(bbox.xyxy[0][2]-bbox.xyxy[0][0])  # blir en tensor här pga bbox maybe bad?
    return dist



def tag_center_angle(tag):
    x, y, z = tag.pose_t
    return np.rad2deg(math.acos(z/np.linalg.norm([y,z])))


if (capture.isOpened()== False):  
    print("could not open video file")

at_detector = Detector(families='tag36h11',
                       nthreads=12,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


# plt.scatter(d_long,d_lat,color='red')# , label='drone gps') # 
# plt.scatter(b_interpolated_long,b_interpolated_lat,color='blue')#,label='boat gps') # boats coordinates
# plt.legend(['drone gps', 'boat gps', 'April tags + drone gps'],loc='upper right')
# plt.show()
# exit()


start_frame_number = 1700 # 2000 är bra för 0085 # 1700 är bra för 0082
frame_number = start_frame_number
bearing_window = 25
drone_bearing = 0 # initial bearing before getting any values
capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
while capture.isOpened():
    # print(distances[frame_number])
    # print(x[frame_number],y[frame_number])

    gpspos = pm.geodetic2enu( d_lat[frame_number], d_long[frame_number], rel_alt[frame_number],b_interpolated_lat[frame_number], b_interpolated_long[frame_number], b_alt_interpolated[frame_number]) # båda höjderna till 0 pga dålig data
    # print(gpspos)



    plt.draw()
    plt.pause(0.00001)
    frame_number += 1
    if frame_number % 50 == 0: # clear the figure every 50 frames due to performence loss
        plt.clf()
        

        print(drone_times[frame_number], boat_times[int(frame_number/25)]) # printar ut för att se att tiderna är hyffsat synkade


    # if frame_number > bearing_window:
    if frame_number % bearing_window == 0:
        drone_bearing = math.atan2(d_long[frame_number]-d_long[frame_number-bearing_window], d_lat[frame_number]-d_lat[frame_number-bearing_window])

        # drone_bearing = math.atan2(math.sin(d_long[frame_number]-d_long[frame_number-bearing_window])*math.cos(d_lat[frame_number]),
        #                            math.cos(d_lat[frame_number-bearing_window])*math.sin(d_lat[frame_number]) - math.sin(frame_number-bearing_window)*math.cos(frame_number)*math.cos(d_long[frame_number]-d_long[frame_number- bearing_window]))
        plot_point((d_long[frame_number],d_lat[frame_number]),drone_bearing,0.0001)


    # new_latitude  = d_lat[frame_number]  + (dy / r_earth) * (180 / math.pi)
    # new_longitude = d_long[frame_number] + (dx / r_earth) * (180 / math.pi) / math.cos(d_lat[frame_number] * math.pi/180)    

    
    # drone_bearing = math.atan2(math.sin(d_long[frame_number]-d_long[frame_number-1])*math.cos(d_lat[frame_number]),math.cos(d_lat[frame_number-1])*math.sin(d_lat[frame_number]) - math.sin(frame_number-1)*math.cos(frame_number)*math.cos(d_long[frame_number]-d_long[frame_number-1]))
    
    # using machine learning yolov8n model
    ret, frame = capture.read()
    if ret == True:
        results = y_model(frame)
        # plot(conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=True, probs=True, show=False, save=False, filename=None)
        for result in results:
            # print(result.boxes.xyxy[0][0])
            im_array = result.plot()
            # print(bool(result.boxes))
        if result.boxes:
            frame = im_array
            
            bbox_center = int((result.boxes.xyxy[0][1]+result.boxes.xyxy[0][3])/2), int((result.boxes.xyxy[0][0]+result.boxes.xyxy[0][2])/2) # egentligen kan jag byta ordning på dessa så att det blir [y,x] men gör det i funktionen bbox_centerangle nu
            bbox_cent_offset = bbox_centerangle(bbox_center, Camera_matrix)
            b_dist = boat_distance([12,4], b_bearing_interpolated[frame_number], drone_bearing, Camera_matrix, result.boxes) 

            yolo_dist = distance.distance(meters=b_dist).destination((d_lat[frame_number],d_long[frame_number]),np.rad2deg(drone_bearing)+bbox_cent_offset)
            yolo_lat, yolo_long = yolo_dist.latitude, yolo_dist.longitude
            plt.scatter(yolo_long, yolo_lat ,color='purple')#, label='April tag + drone gps') # och denna 

        # frame = im_array[int(result.boxes.xyxy[0][1]):int(result.boxes.xyxy[0][3]),int(result.boxes.xyxy[0][0]):int(result.boxes.xyxy[0][2])] # croppar framen utifrån object detection rutan TODO FIXA SÅ ATT DEN BARA CROPPAR RUNT BÅT detections
        
        
        # cv2.SOLVEPNP_EPNP
        if april_tags:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(frame, True, camera_params, 0.1735) #  130 mm är våran man mäter insidan och den anges i meter https://github.com/AprilRobotics/apriltag?tab=readme-ov-file#pose-estimation (blir 3 pixlar man mäter)
            # print(tags) # https://github.com/duckietown/lib-dt-apriltags/tree/daffy
                        # pose_t är på formen X,Y,Z och med högerhandkoordinatsystem blir det: [högerled_från kameran, höjdled, avstånd pekar ut ur kameran]
            for tag in tags:
                # print(tag.pose_t) # prints the pose_t probably on the form X,Y,Z which translates to [right, down, distance(depth)]
                # print(tag)
                dx = tag.pose_t[0] # X coordinate 
                dy = tag.pose_t[2] # Z coordinate boats relative position to the camera (no altitude yet)
                bca = boat_cam_angle(drone_bearing, tag, Camera_matrix)
                approx_drone_bearing = np.rad2deg(drone_bearing) + bca#tag_center_angle(tags[0])

                m = (1 / ((2 * math.pi / 360) * r_earth/1000)) / 1000
                
                newlatlong = distance.distance(meters=tag.pose_t[2][0]).destination((d_lat[frame_number],d_long[frame_number]),approx_drone_bearing)
                new_latitude, new_longitude = newlatlong.latitude, newlatlong.longitude
                plt.scatter(new_longitude,new_latitude,color='green')#, label='April tag + drone gps') # och denna 

                for idx in range(len(tag.corners)):
                    cv2.line(frame, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

                cv2.putText(frame, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))

        if visualization:
            cv2.imshow('Detected tags', frame)
            # k = cv2.waitKey(0)
            # if tags:
            plt.scatter(d_long[frame_number],d_lat[frame_number],color='red' , label='drone gps') # 
            plt.scatter(b_interpolated_long[frame_number],b_interpolated_lat[frame_number],color='blue',label='boat gps') # boats coordinates
            
            # if tags: 
                # plt.scatter(new_longitude,new_latitude,color='green')#, label='April tag + drone gps') # och denna 

            plt.legend(['drone gps', 'boat gps', 'April tags + drone gps'],loc='upper right')
            

        if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    # else: 
        #  break 


