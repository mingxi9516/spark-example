'''
================
三维用户运动轨迹
================

'''
from pyspark.sql import SparkSession
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.dates as mdate
if __name__ == "__main__":
    mpl.rcParams['legend.fontsize'] = 10
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .enableHiveSupport() \
        .getOrCreate()


    # Displays the content of the DataFrame to stdout
    pandasDf = spark.sql("""SELECT longitude,
                   latitude,
                   currentTime
                   FROM dc.dwd_user_gps_tmp
                   WHERE phoneModel="860191034777629" AND cityCode="610303" """).toPandas()
    # sparkDf = spark.read.format("csv") \
    #     .option("header", "true") \
    #     .load("data.txt")
    # pandasDf = sparkDf.toPandas()
    lon=np.array(pandasDf['longitude']).tolist()
    lat=np.array(pandasDf['latitude']).tolist()
    timestamp=np.array(pandasDf['currentTime']).tolist()
    lon1=[]
    for la in lon:
        lon1.append(float(la.strip()))
    lat1=[]
    for la in lat:
        lat1.append(float(la.strip()))
    time1=[]
    for time in timestamp:
        time1.append(mpl.dates.date2num(datetime.strptime(time.strip(),'%Y-%m-%d %H:%M:%S')))
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.zaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax.scatter(lon1, lat1,time1, label='user')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend()
    plt.show()
    spark.stop()
