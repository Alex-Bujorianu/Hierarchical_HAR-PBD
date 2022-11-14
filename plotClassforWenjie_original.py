



import sys
sys.path.append(r"C:\Users\uclicadmin\Desktop\Temi EnTimeMent\coding\Python\Classes\homestudy")
sys.path.append(r"C:\Users\uclicadmin\Desktop\Temi Code\data")
from emopainathomeDataClass import EmoPainAtHomedata
from dataClass import *


import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\uclicadmin\AppData\Local\Programs\Python\Python38\Scripts\ffmpeg.exe'
savefilepathgif = r"C:\Users\uclicadmin\Desktop\testplot.gif"
#savefilepathmp4 = r"C:\Users\uclicadmin\Desktop\testplot2.mp4"

class Plot():

    
    def draw_halfskel(halfaskel, ax, plotvars):


        ax.clear()
        ax.set(xlim3d=(-1, 1), xlabel='X')
        ax.set(ylim3d=(-1, 1), ylabel='Y')
        ax.set(zlim3d=(-1, 1), zlabel='Z')

        halfaskel = numpy.reshape(halfaskel, (-1, 6, 3))
        t=0

        chestbottom=0
        thigh=1
        upperarm=2
        lowerleg=3
        forearm=4
        hip=5

        x = 0
        y = 2
        z = 1
        
        plotvarschest = 'ko-'
        #plotvarschest = plotvars

        skel = []


        chest_hip = ax.plot([halfaskel[t, chestbottom, x], halfaskel[t, hip, x]],
                [halfaskel[t, chestbottom, y], halfaskel[t, hip, y]],
                [halfaskel[t, chestbottom, z], halfaskel[t, hip, z]], plotvarschest)
        
        hip_thigh = ax.plot([halfaskel[t, hip, x], halfaskel[t, thigh, x]],
                [halfaskel[t, hip, y], halfaskel[t, thigh, y]],
                [halfaskel[t, hip, z], halfaskel[t, thigh, z]], plotvars)

        thigh_leg = ax.plot([halfaskel[t, thigh, x], halfaskel[t, lowerleg, x]],
                [halfaskel[t, thigh, y], halfaskel[t, lowerleg, y]],
                [halfaskel[t, thigh, z], halfaskel[t, lowerleg, z]], plotvars)

        arm_arm = ax.plot([halfaskel[t, upperarm, x], halfaskel[t, forearm, x]],
                [halfaskel[t, upperarm, y], halfaskel[t, forearm, y]],
                [halfaskel[t, upperarm, z], halfaskel[t, forearm, z]], plotvars)

        #print(chest_hip)
           
        skel.extend(chest_hip)
            
        skel.extend(hip_thigh)

        skel.extend(thigh_leg)

        skel.extend(arm_arm)


            


        return skel


        
    def update_skel(tau, single_instance, skel, ax, plotvars):
    
        newskel = Plot.draw_halfskel(single_instance[tau, :, :], ax, plotvars)
    
        for oldp, newp in zip(skel, newskel):

            oldp.set_data_3d(newp.get_data_3d())
  

        return skel
    







###preparing my plot input data
#remove this lines and replace 'single_instance' variable
#with your own input data
datafolderpath = r"C:\Users\uclicadmin\Desktop\Temi EnTimeMent\coding\Python\Data\EmoPain\home study"
labelsfile = r"C:\Users\uclicadmin\Desktop\Temi EnTimeMent\coding\Python\Data\EmoPain\home study\labels.csv"
dataobj = EmoPainAtHomedata(datafolderpath, labelsfile)
datatable = dataobj.createdatatable()

noofattrs = 2400*6*3
single_instance = datatable[:, :noofattrs]
single_instance = numpy.reshape(single_instance, (-1, 2400, 6, 3))
single_instance = single_instance[0, :, :, :]


###plotting
#setting up my plot and animation parameters
plotvars = 'b.-'
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_title('Test')
metadata = dict(title='Notch Half Skeleton Animation', artist='Matplotlib')

#initializing the animation
skel = Plot.draw_halfskel(single_instance[0, :, :], ax, plotvars)
        
anime = animation.FuncAnimation(fig, Plot.update_skel, frames=single_instance.shape[0], 
                                fargs=(single_instance, skel, ax, plotvars), 
                                interval=25, repeat=False)
#saving the animation as a gif
#you can choose to save as mp4, but you would need to use the FFMpegWriter for that
#instead of the PillowWriter, but you need to install ffmpeg for this
#https://www.gyan.dev/ffmpeg/builds/#release-builds
#and put in the python folder specified in plt.rcParams['animation.ffmpeg_path']
#above (near the import statements at the start of script
anime.save(savefilepathgif, writer=animation.PillowWriter(fps=15, metadata=metadata))
#anime.save(savefilepathmp4, writer=animation.FFMpegFileWriter(fps=15, metadata=metadata))
plt.show()

        

    

