'''
================
三维用户运动轨迹
================
'''

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.dates as mdate
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
lon1=[]
lat1=[]
time1=[]
with open("data.txt", 'r') as fp:
    for line in fp.readlines():
        lon,lat,time = line.split(',')
        lon1.append(float(lon.strip()))
        lat1.append(float(lat.strip()))
        time1.append(mpl.dates.date2num(datetime.strptime(time.strip(),'%Y-%m-%d %H:%M:%S')))
# print('lon1=%s, lat1=%s, time1=%s' %(lon1, lat1, time1))
# print('time1=%s' %(time1))
ax.zaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
# ax.zaxis.set_major_locator(mdate.DateLocator())
ax.scatter(lon1, lat1,time1, label='user')
ax.set_xlabel("lon")
ax.set_ylabel("lat")
ax.legend()
plt.show()