import numpy

def calculate_angle(u, v, w):

    vec1 = v - u
    vec2 = v - w

    numerator = numpy.linalg.norm(numpy.cross(vec1, vec2))
    denominator = numpy.vdot(vec1, vec2)

    return numpy.arctan2(numerator, denominator)



def get_half_skel_joint_angles(data):

    joints = ['chestbottom', 'thigh', 'upperarm', 'lowerleg', 'forearm', 'hip']
    angles = numpy.zeros((data.shape[0], data.shape[1], 4, 3))

    for i in numpy.arange(0, data.shape[0]):

        for j in numpy.arange(0, data.shape[1]):

            #hip-thigh-lowerleg
            angles[i, j, 0, 0] = calculate_angle(data[i, j, 3, :2], data[i, j, 1, :2], data[i, j, 5, :2])
            angles[i, j, 0, 1] = calculate_angle(data[i, j, 3, 0:3:2], data[i, j, 1, 0:3:2], data[i, j, 5, 0:3:2])
            angles[i, j, 0, 2] = calculate_angle(data[i, j, 3, 1:], data[i, j, 1, 1:], data[i, j, 5, 1:])


            #chestbottom-hip-thigh
            angles[i, j, 1, 0] = calculate_angle(data[i, j, 1, :2], data[i, j, 5, :2], data[i, j, 0, :2])
            angles[i, j, 1, 1] = calculate_angle(data[i, j, 1, 0:3:2], data[i, j, 5, 0:3:2], data[i, j, 0, 0:3:2])
            angles[i, j, 1, 2] = calculate_angle(data[i, j, 1, 1:], data[i, j, 5, 1:], data[i, j, 0, 1:])

            

            #forearm-chestbottom-hip
            angles[i, j, 2, 0] = calculate_angle(data[i, j, 5, :2], data[i, j, 0, :2], data[i, j, 4, :2])
            angles[i, j, 2, 1] = calculate_angle(data[i, j, 5, 0:3:2], data[i, j, 0, 0:3:2], data[i, j, 4, 0:3:2])
            angles[i, j, 2, 2] = calculate_angle(data[i, j, 5, 1:], data[i, j, 0, 1:], data[i, j, 4, 1:])


            #upperarm-chestbottom-hip
            angles[i, j, 3, 0] = calculate_angle(data[i, j, 5, :2], data[i, j, 0, :2], data[i, j, 2, :2])
            angles[i, j, 3, 1] = calculate_angle(data[i, j, 5, 0:3:2], data[i, j, 0, 0:3:2], data[i, j, 2, 0:3:2])
            angles[i, j, 3, 2] = calculate_angle(data[i, j, 5, 1:], data[i, j, 0, 1:], data[i, j, 2, 1:])


    angles = numpy.degrees(angles)

    #print(angles)


    return angles    

    
