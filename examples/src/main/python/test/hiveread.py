#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

# $example on:spark_hive$
from os.path import expanduser, join, abspath

from pyspark.sql import SparkSession
from pyspark.sql import Row

# $example off:spark_hive$

"""
A simple example demonstrating Spark SQL Hive integration.
Run with:
  ./bin/spark-submit examples/src/main/python/sql/hive.py
"""

if __name__ == "__main__":
    # $example on:spark_hive$
    # warehouse_location points to the default location for managed databases and tables
    warehouse_location = abspath('spark-warehouse')

    spark = SparkSession \
        .builder \
        .config("spark.sql.warehouse.dir", warehouse_location) \
        .enableHiveSupport() \
        .getOrCreate()

    # spark is an existing SparkSession
    # spark.sql("SELECT trim(userId) userId,"
    #           "trim(phoneModel) phoneModel,"
    #           "trim(cityCode) cityCode,"
    #           "cast(trim(longitude) as double) longitude,"
    #           "cast(trim(latitude) as double) latitude,"
    #           "timestamp(trim(currentTime)) currentTime,"
    #           "trim(geocode6) region"
    #           "FROM dc.dwd_user_gps_tmp").show
    spark.sql("select * from  dc.dwd_user_gps_tmp").show
    spark.stop()
