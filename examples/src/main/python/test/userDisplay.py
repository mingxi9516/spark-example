'''
================
三维用户运动轨迹
================

'''
from pyspark.sql import SparkSession
import matplotlib as mpl
import matplotlib.pyplot as plt
if __name__ == "__main__":
    mpl.rcParams['legend.fontsize'] = 10
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .getOrCreate()
    df = spark.read.json("people.json").toPandas()
    # Displays the content of the DataFrame to stdout
    df.plot()
    # df = spark.sql("""SELECT trim(phoneModel) phoneModel,
    #                cast(trim(longitude) as double) longitude,
    #                cast(trim(latitude) as double) latitude,
    #                timestamp(trim(currentTime)) currentTime
    #                FROM dc.dwd_user_gps_tmp
    #                WHERE phoneModel=865091024998281 AND cityCode=0755""").toPandas
    plt.show()
    spark.stop()
