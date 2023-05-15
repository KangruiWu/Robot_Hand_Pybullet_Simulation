import pybullet as p
import cv2
import mediapipe as mp
import time
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import math
import csv
p.connect(p.GUI)
    
move = 0.01

p.setGravity(0,0,0)

objects = p.loadMJCF("MPL.xml",flags=0)
hand=objects[0]  #1 total
#obj_po = p.getBasePositionAndOrientation(obj_to_classify)
hand_cid = p.createConstraint(hand,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])

pi = 3.14159
hand_po = p.getBasePositionAndOrientation(hand)
ho = p.getQuaternionFromEuler([-pi/2, 0, 0.0])

# keeps it from moving back go origin
p.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]),ho, maxForce=200)

p.setRealTimeSimulation(0)
testAngleTargetId=[[] for i in range(46)]
for i in range(46):
    name="joint"+str(i)
    testAngleTargetId[i]=p.addUserDebugParameter(paramName=name, rangeMin=-pi/2, rangeMax=pi/2, startValue=0)
num_joints = p.getNumJoints(hand)
    
offset = 0 #0.02
indexEndID = 21 # Need get position and orientation from index finger parts
fig = plt.figure()
ax = plt.axes(projection='3d')

plt.xlim(0,640)
plt.ylim(-320,320)
plt.ion()

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
pTime = 0
cTime = 0
count=0
coordinates=np.zeros([21,3])
coordinates_plt=np.zeros([21,3])
coordinates_raw=np.zeros([21,3])

def ahead_view():
    link_state = p.getLinkState(hand,indexEndID)
    link_p = link_state[0]
    link_o = link_state[1]
    handmat = p.getMatrixFromQuaternion(link_o)

    axisX = [handmat[0],handmat[3],handmat[6]]
    axisY = [-handmat[1],-handmat[4],-handmat[7]] # Negative Y axis
    axisZ = [handmat[2],handmat[5],handmat[8]]

    eye_pos    = [link_p[0]+offset*axisY[0],link_p[1]+offset*axisY[1],link_p[2]+offset*axisY[2]]
    target_pos = [link_p[0]+axisY[0],link_p[1]+axisY[1],link_p[2]+axisY[2]] # target position based by axisY, not X
    up = axisZ # Up is Z axis
    viewMatrix = p.computeViewMatrix(eye_pos,target_pos,up)

    p.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.05) # Debug line in camera direction    

    return viewMatrix

kf_dict = {}
filtered_state_mean_dict = {}
filtered_state_covariance_dict = {}

mse_values = []  # List to store MSE values

# Set the start time for the timer
start_time = time.time()

while count<=1500:
    count=count+1
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark):

                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    zPos = int(lm.z * imgWidth)
                    #print(lm.z)

                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    #coordinates[i]=[lm.x,lm.y,lm.z]
                    coordinates[i]=[xPos,yPos,zPos]
                    coordinates_plt[i]=[xPos,imgHeight-yPos,zPos] 
                    coordinates_raw[i]=[xPos,yPos,zPos]

                    
        # if result.multi_hand_world_landmarks:
        #     for handLms in result.multi_hand_world_landmarks:
        #         for i, lm in enumerate(handLms.landmark):

        #             xPos = int(lm.x * imgWidth)
        #             yPos = int(lm.y * imgHeight)
        #             zPos = int(lm.z * imgWidth)
        #             coordinates[i]=[lm.x,lm.y,lm.z]
        #             #coordinates_plt[i]=[xPos,imgHeight-yPos,zPos]
        
        
        
        
        def find_projection (ref_v,tgt_v):
            ref_v_array=np.array(ref_v)
            tgt_v_array=np.array(tgt_v)
            if np.dot(ref_v_array,tgt_v_array)!=0:
                prj_v=tgt_v_array-(np.dot(ref_v_array,tgt_v_array)/np.dot(ref_v_array,ref_v_array))*ref_v_array
            else:
                prj_v=tgt_v
            return prj_v 
        
        finger_length_real=[0,70.34912934784623,67.36467917239716,52.66877632905477,43.749285708454714,123.97983707038819,65.21502894272147,43.520110293977886,37.12142238654117,156.16017418023074,72.91090453423274,48.19751030914356,41.67733196834941,177.2343081911625,66.87301398920195,45.48626166217664,41.66533331199932,177.86230629337965,52.820450584977024,34.899856733230294,33.24154027718932]

        finger_length_world=[0,0.0472064,0.031173,0.0345788,0.0258304,0.0606378,0.031548,0.0209093,0.0341975,0.0913625,0.0409461,0.0285443,0.0291958,0.101185,0.0337896,0.0276152,0.0294784,0.102198,0.0229053,0.0219749,0.0242347]
        
        zz=np.array([0,0,-1])
        exclude=[0,5,9,13,17,21]
        ref_length_now=np.linalg.norm(coordinates[5]-coordinates[0])
        ref_length_real=160.3558542741736

        for i in range(21):
            if not(i in exclude):
                tgt_v=coordinates[i]-coordinates[i-1]
                length_now=np.linalg.norm(tgt_v)
                length_ideal=finger_length_real[i]*ref_length_now/ref_length_real
                length_projection=np.linalg.norm(find_projection(zz, tgt_v))
                if length_ideal>=length_projection:
                    real_z_offset=math.sqrt(length_ideal*length_ideal-length_projection*length_projection)
                    now_z_offset=coordinates[i,2]-coordinates[i-1,2]+0.00001
                    coordinates[i,2]=coordinates[i-1,2]+now_z_offset/abs(now_z_offset)*real_z_offset
                else:
                    coordinates[i]=coordinates[i-1]+tgt_v*length_ideal/length_now
                    
                if not(i+1 in exclude):
                    coordinates[i+1]=coordinates[i+1]+coordinates[i]-coordinates_raw[i]
                        
        coordinates_plt[:,2]=coordinates[:,2]     
                

        # for i in range(1,21):
        #     finger_length_real[i]=np.linalg.norm(coordinates[i]-coordinates[i-1])
                   
        #MCP angle
        index_and_vv = [(7, coordinates[3] - coordinates[0]),
                        (17, coordinates[6] - coordinates[5]),
                        (24, coordinates[10] - coordinates[9]),
                        (32, coordinates[14] - coordinates[13]),
                        (40, coordinates[18] - coordinates[17])]
        #print(np.linalg.norm(coordinates[5] - coordinates[0])/np.linalg.norm(coordinates[19] - coordinates[18]+[0,0,0.001]))
        
        #ABD angle
        def normalize(v):
            if np.linalg.norm(v)!=0:
                return v/np.linalg.norm(v)
            else:
                return v
        def find_projection (ref_v,tgt_v):
            ref_v_array=np.array(ref_v)
            tgt_v_array=np.array(tgt_v)
            if np.dot(ref_v_array,tgt_v_array)!=0:
                prj_v=tgt_v_array-(np.dot(ref_v_array,tgt_v_array)/np.dot(ref_v_array,ref_v_array))*ref_v_array
            else:
                prj_v=tgt_v
            return prj_v
        
        
        palm_ref=normalize(np.cross(np.array(coordinates[5] - coordinates[0]), np.array(coordinates[17] - coordinates[0])))
        #abd_ref=normalize(find_projection(palm_ref,coordinates[9] - coordinates[0]))
        abd_ref=normalize(coordinates[9] - coordinates[0])
        index_and_abd_ref = [(9, find_projection(palm_ref,normalize(coordinates[3] - coordinates[2])),normalize(coordinates[2] - coordinates[1])),
                            (15, find_projection(palm_ref,normalize(coordinates[6] - coordinates[5])),abd_ref),
                            (30, find_projection(palm_ref,normalize(coordinates[14] - coordinates[13])),abd_ref),
                            (38, find_projection(palm_ref,normalize(coordinates[18] - coordinates[17])),abd_ref)]
        
        #Rest angle
        index_and_dip_next = [(11, coordinates[3] - coordinates[2],coordinates[2] - coordinates[1]),
                              (13, coordinates[4] - coordinates[3],coordinates[3] - coordinates[2]),
                            
                              (19, coordinates[7] - coordinates[6],coordinates[6] - coordinates[5]),
                              (21, coordinates[8] - coordinates[7],coordinates[7] - coordinates[6]),
                            
                              (26, coordinates[11] - coordinates[10],coordinates[10] - coordinates[9]),
                              (28, coordinates[12] - coordinates[11],coordinates[11] - coordinates[10]),
                            
                              (34, coordinates[15] - coordinates[14],coordinates[14] - coordinates[13]),
                              (37, coordinates[16] - coordinates[15],coordinates[15] - coordinates[14]),
                            
                              (42, coordinates[19] - coordinates[18],coordinates[18] - coordinates[17]),
                              (44, coordinates[20] - coordinates[19],coordinates[19] - coordinates[18])]
                
        # Initialize Kalman filters for each index
        for index in set([x[0] for x in index_and_vv] + [x[0] for x in index_and_abd_ref] + [x[0] for x in index_and_dip_next]):
            kf_dict[index] = KalmanFilter(initial_state_mean=[0], initial_state_covariance=[1], 
                                        transition_matrices=[1], observation_matrices=[1], 
                                        transition_covariance=[0.01], observation_covariance=[0.1])
            filtered_state_mean_dict[index] = kf_dict[index].initial_state_mean
            filtered_state_covariance_dict[index] = kf_dict[index].initial_state_covariance
        
        def mapping(index, angle):
            # if index == 7:
            #     mapped_angle = (angle - 0) * (np.pi/2 - 0) / (np.pi/4 - 0) + 0
            # elif index == 9:
            #     mapped_angle = (angle - 0) * (np.pi/2 - 0) / (np.pi/4 - 0) + 0
            # elif index in [11,13,17,24,32,40]:
            #     mapped_angle = (angle - 0) * (np.pi  - 0) / (np.pi/2 - 0) + 0
            # elif index in [15,30,38]:
            #     mapped_angle = (angle - 0) * (np.pi/2  - 0) / (np.pi - 0) + 0
            # else:
            #     mapped_angle = angle
            mapped_angle = angle
            return mapped_angle


        def set_joint_motor_control(hand, index_and_vv, index_and_abd_ref, index_and_dip_next):
            global filtered_state_mean, filtered_state_covariance
            
            mse_sum = 0.0  # Initialize sum of squared errors
            counts = 0     # Initialize count of data points

            # MCP angle
            v1 = np.array(coordinates[5] - coordinates[0])
            v2 = np.array(coordinates[9] - coordinates[0])

            for index, vv in index_and_vv:
                cross_product = np.cross(v1, v2)
                if np.linalg.norm(cross_product) * np.linalg.norm(vv) != 0:
                    dot_product = np.dot(cross_product, vv)
                    arccos = np.arccos(dot_product / (np.linalg.norm(cross_product) * np.linalg.norm(vv)))
                    angle = np.degrees(arccos)
                    #print(f"Angle for index {index}: {angle}")
                    
                    # Store unfiltered angle for the index
                    unfiltered_angle = np.pi / 2 - arccos

                    # Update the Kalman filter and get the filtered angle
                    filtered_state_mean_dict[index], filtered_state_covariance_dict[index] = kf_dict[index].filter_update(
                        filtered_state_mean_dict[index], filtered_state_covariance_dict[index], observation=[np.pi / 2 - arccos]
                    )
                    filtered_angle_rad = filtered_state_mean_dict[index][0]

                    # Add angle to the specified index
                    mapped_filtered_angle_rad = mapping(index, filtered_angle_rad)
                    p.setJointMotorControlArray(hand, [index], p.POSITION_CONTROL, [filtered_angle_rad], [0])
                    # print("=====================")
                    # print(f"Angle for index {index}: {arccos}")
                    # print("vvvvvvvvvvvvvvvvvvvvvvvvv")
                    # print(f"Angle for index {index}: {mapped_filtered_angle_rad}")
                    # print("=====================")
                    
                    # Calculate squared error and add to the sum
                    mse_sum += (filtered_angle_rad - unfiltered_angle) ** 2
                    counts += 1

            # ABD angle
            for index, abd, ref in index_and_abd_ref:
                if np.linalg.norm(ref) * np.linalg.norm(abd) != 0:
                    dot_product = np.dot(ref, abd)
                    norm_ref = np.linalg.norm(ref)
                    norm_abd = np.linalg.norm(abd)
                    cosine_angle = dot_product / (norm_ref * norm_abd)
                    if norm_ref>=0.5:
                        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle potential floating point errors
                    else: angle_rad=0
                    if angle_rad>pi/2:
                        angle_rad=angle_rad-pi/2
                    angle_deg = np.degrees(angle_rad)
                    #print(f"Angle for index {index}: {angle_deg}")
                    # Store unfiltered angle for the index
                    unfiltered_angle = angle_rad
                    # Update the Kalman filter and get the filtered angle
            
                    filtered_state_mean_dict[index], filtered_state_covariance_dict[index] = kf_dict[index].filter_update(
                        filtered_state_mean_dict[index], filtered_state_covariance_dict[index], observation=[angle_rad]
                    )
                    filtered_angle_rad1 = filtered_state_mean_dict[index][0]

                    # Add angle to the specified index
                    mapped_filtered_angle_rad1 = mapping(index, filtered_angle_rad1)
                    p.setJointMotorControlArray(hand, [index], p.POSITION_CONTROL, [filtered_angle_rad1], [0])
                    # print("=====================")
                    # print(f"Angle for index {index}: {angle_rad}")
                    # print("vvvvvvvvvvvvvvvvvvvvvvvvv")
                    # print(f"Angle for index {index}: {mapped_filtered_angle_rad1}")
                    # Calculate squared error and add to the sum
                    mse_sum += (filtered_angle_rad - unfiltered_angle) ** 2
                    counts += 1
                    
            # Rest angle
            for index, dip, next in index_and_dip_next:
                if np.linalg.norm(next) * np.linalg.norm(dip) != 0:
                    dot_product = np.dot(next, dip)
                    norm_next = np.linalg.norm(next)
                    norm_dip = np.linalg.norm(dip)
                    cosine_angle2 = dot_product / (norm_next * norm_dip)
                    angle_rad2 = np.arccos(np.clip(cosine_angle2, -1.0, 1.0))  # Clip to handle potential floating point errors
                    angle_deg2 = np.degrees(angle_rad2)
                    #print(f"Angle for index {index}: {angle_deg2}")
                    # Store unfiltered angle for the index
                    unfiltered_angle = angle_rad2

                    # Update the Kalman filter and get the filtered angle
                    
                    filtered_state_mean_dict[index], filtered_state_covariance_dict[index] = kf_dict[index].filter_update(
                        filtered_state_mean_dict[index], filtered_state_covariance_dict[index], observation=[angle_rad2]
                    )
                    filtered_angle_rad2 = filtered_state_mean_dict[index][0]

                    # Add angle to the specified index
                    mapped_filtered_angle_rad2 = mapping(index, filtered_angle_rad2)
                    p.setJointMotorControlArray(hand, [index], p.POSITION_CONTROL, [filtered_angle_rad2], [0])
                    
                    # Calculate squared error and add to the sum
                    mse_sum += (filtered_angle_rad2 - unfiltered_angle) ** 2
                    counts += 1

            # Calculate mean squared error if there are valid data points
            if counts > 0:
                mse = mse_sum / counts
                mse_values.append(mse)
                print("Mean Squared Error (MSE):", mse)
            else:
                print("No valid data points for MSE calculation.")
        # Call the set_joint_motor_control function with the appropriate data
        set_joint_motor_control(hand, index_and_vv, index_and_abd_ref, index_and_dip_next)
        
        key = p.getKeyboardEvents()
        viewMatrix = ahead_view()
        p.stepSimulation()
        
        #####
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)


    
    
    #plt.clf()
    # ax.plot3D(coordinates_plt[0:5,0], coordinates_plt[0:5,2], coordinates_plt[0:5,1], 'gray',marker='o')
    # ax.plot3D(coordinates_plt[5:9,0], coordinates_plt[5:9,2], coordinates_plt[5:9,1], 'gray',marker='o')
    # ax.plot3D(coordinates_plt[9:13,0], coordinates_plt[9:13,2], coordinates_plt[9:13,1], 'gray',marker='o')
    # ax.plot3D(coordinates_plt[13:17,0], coordinates_plt[13:17,2], coordinates_plt[13:17,1], 'gray',marker='o')
    # ax.plot3D(coordinates_plt[17:21,0], coordinates_plt[17:21,2], coordinates_plt[17:21,1], 'gray',marker='o')
    # ax.set_title('3D plot')
    # plt.show()
    # plt.pause(0.01)
    
    
    
    if cv2.waitKey(1) == ord('q'):
        break

    # Check if the timer has reached 20 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time >= 30:
        break

#cv2.destroyALLWindows()

# Save MSE values to a CSV file
csv_file = r'C:\Users\foser\Downloads\robot_hand\data\kf_mse_data.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'MSE'])
    for i, mse in enumerate(mse_values):
        writer.writerow([i, mse])

print("MSE data saved to:", csv_file)

p.disconnect()