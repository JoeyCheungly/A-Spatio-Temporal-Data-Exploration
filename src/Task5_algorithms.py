"""
INFS7205 Individual Project
Semester 1, 2023
"""

__author__ = "Lok Yee Joey Cheung"
__email__ = "s4763354@student.uq.edu.au"
__date__ = "06/05/2023"

import pandas as pd
import csv
import datetime
import psutil
from geopy.distance import great_circle
from rtree import index
from scipy.spatial import KDTree
from pyqtree import Index
from sqlalchemy import create_engine
import numpy as np




#--------------------------------------TASK5--------------------------------------#
# Find top k popular venue catgories having the most data-points within a time window in a rectangular area.

def task5_qtree(filename, point, k, time_month, time_day):
    """Find top k popular venue catgories having the most data-points within a time window in a rectangular area. (Quad tree)

    Parameters:
        filename (str): input csv file as dataset
        point (tuple): bounding box of the given rectangular area in the form of min longitude, min latitude, max longitude, max langitude 
        k(int): number of popular venues
        time_month (str): month of the given time
        time_day(str): day of the given time
    """

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    #Build the quad tree index
    index = Index(bbox=[-74.27476645, 40.55085247, -73.6838252, 40.98833172])
    with open(filename, 'r') as file:
        checkin = csv.DictReader(file)
        for row in checkin:
            index.insert(item = (row['venueCategory'], float(row['longitude']), float(row['latitude']), row['utcTimestamp']),bbox = (float(row['longitude']), float(row['latitude']), float(row['longitude']), float(row['latitude'])))
    
    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()
    
    #Query the quad tree to find all data points whose bounding boxes intersect the given bounding box of 'points'
    results = index.intersect(point)

    venue = row['venueCategory']
    longitude =  float(row['longitude'])
    latitude = float(row['latitude'])
    timestamp = row['utcTimestamp']
    filtered_dp = []
    venue_count = {}

    #Filter the data on the specified date
    for i in results:
        venue, longitude, latitude, timestamp  = i
        month = timestamp.split(" ")[1]
        day = timestamp.split(" ")[2]
        if month == time_month and day == time_day:
            filtered_dp.append(i)
    
    for dp in filtered_dp:
        venueCategory = dp[0]
        if venueCategory not in venue_count:
            venue_count[venueCategory] = 1
        else:
            venue_count[venueCategory] += 1
    
    #Sort the result in terms of degree of popularity
    result = sorted(venue_count.items(), key=lambda x:x[1], reverse=True)

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    #Display result
    result = pd.DataFrame(result, columns=['venue','num_points'])
    result = result.reset_index(drop=True)
    result.index += 1  
    print(result.head(k))

    return result.head(k)









def task5_rtree(filename, point, k, time_month, time_day):
    """Find top k popular venue catgories having the most data-points within a time window in a rectangular area. (R-tree)

    Parameters:
        filename (str): input csv file as dataset
        point (tuple): bounding box of the given rectangular area in the form of min longitude, min latitude, max longitude, max langitude 
        k(int): number of popular venues
        time_month (str): month of the given time
        time_day(str): day of the given time
    """
    df = pd.read_csv(filename)

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    rtree_index = index.Index()
    for i, row in df.iterrows():
        rtree_index.insert(i, (row['latitude'], row['longitude'], row['latitude'], row['longitude']))

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    #Query the r tree to find all data points whose bounding boxes intersect the given bounding box of 'points'
    results = list(rtree_index.intersection(point))

    #Get data points within the specified tiime
    result_dp = []
    for i in results:
        timestamp = df.loc[i, 'utcTimestamp']
        month = timestamp.split(" ")[1]
        day = timestamp.split(" ")[2]
        if month == time_month and day == time_day:
            result_dp.append(i)

    venue_count = {}
    for dp in result_dp:
        venue = df.loc[dp, 'venueCategory']  
        if venue not in venue_count:
            venue_count[venue] = 1
        else:
            venue_count[venue] += 1

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    result = sorted(venue_count.items(), key=lambda x:x[1], reverse=True)
    result = pd.DataFrame(result, columns=['venue','num_points'])
    result = result.reset_index(drop=True)
    result.index += 1  
    print(result.head(k))

    return result.head(k)




def task5_linearscan(filename, point, k, time_month, time_day):
    """Find top k popular venue catgories having the most data-points within a time window in a rectangular area. (Grid-based indexing)

    Parameters:
        filename (str): input csv file as dataset
        point (tuple): bounding box of the given rectangular area in the form of min longitude, min latitude, max longitude, max langitude 
        k: number of popular venues
    """
    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    process = psutil.Process()
    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    checkin = []
    #Query for data points 
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            timestamp = row['utcTimestamp']
            
            # Check if the check-in point is within the bounding box
            if (point[0] <= longitude <= point[2]) and (point[1] <= latitude <= point[3]):                
                # Filter data based on date
                month = timestamp.split(" ")[1]
                day = timestamp.split(" ")[2]
                if month == time_month and day == time_day:
                    checkin.append(row)
    # Count data based on venues
    venue_count = {}
    for dp in checkin:
        venue = dp['venueCategory']
        if venue not in venue_count:
            venue_count[venue] = 1
        else:
            venue_count[venue] += 1  

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    result = sorted(venue_count.items(), key=lambda x:x[1], reverse=True)
    result = pd.DataFrame(result, columns=['venue','num_points'])
    result = result.reset_index(drop=True)
    result.index += 1  
    print(result.head(k))
    
    return result.head(k)








#--------------------------------------VALIDATION--------------------------------------#

def validation(ground_truth_file, prediction, task):
    # Obtain ground truth for the query
    gt = pd.read_csv(ground_truth_file)
    ground_truth = gt['num_points'].tolist()

    # Compare the results
    tp1 = prediction[prediction['num_points'].isin(ground_truth)].shape[0]
    fp1 = prediction[~prediction['num_points'].isin(ground_truth)].shape[0]
    fn1 = len(ground_truth) - tp1
    precision = tp1 / (tp1 + fp1)
    recall = tp1 / (tp1 + fn1)
    f1_score = (2 * precision * recall) / (precision + recall)
    
    print(f"Correctness of {task}: Precision={precision}, Recall={recall}, F-measure={f1_score}")




def main():
    """ Main function: call other functions to perform desired actions.
    """
   #----------------Task 5----------------------#
    # Test Case 1: Find 5 most popular venues that people love to go to during Thanksgiving in Manhattan.
    task5_qtree_result = task5_qtree('dataset_TSMC2014_NYC.csv',(-74.02, 40.68, -73.93, 40.88), 5, 'Nov', '22')

    task5_rtree_result = task5_rtree('dataset_TSMC2014_NYC.csv',(40.68, -74.02, 40.88, -73.93), 5 ,'Nov', '22')

    task5_linearscan_result = task5_linearscan('dataset_TSMC2014_NYC.csv',(-74.02, 40.68, -73.93, 40.88), 5, 'Nov', '22')

    #Test Case 2: Find 3 most popular venues that people love to go to during Mother's Day in Brooklyn.
    task5_qtree_result2 = task5_qtree('dataset_TSMC2014_NYC.csv',(-73.9764, 40.5707, -73.8331, 40.7395), 3, 'May', '13')
    
    task5_rtree_result2 = task5_rtree('dataset_TSMC2014_NYC.csv',(40.5707, -73.9764, 40.7395, -73.8331), 3 ,'May', '13')

    task5_linearscan_result2 = task5_linearscan('dataset_TSMC2014_NYC.csv',(-73.9764, 40.5707, -73.8331, 40.7395), 3, 'May', '13')

    #Validation
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task5_groundtruth.csv',task5_qtree_result ,'Task5 Test Case 1 using qtree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task5_groundtruth.csv',task5_rtree_result,'Task5 Test Case 1 using r tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task5_groundtruth.csv',task5_linearscan_result,'Task5 Test Case 1 using linear scan')

    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task5_groundtruth2.csv',task5_qtree_result2 ,'Task5 Test Case 2 using qtree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task5_groundtruth2.csv',task5_rtree_result2,'Task5 Test Case 2 using r tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task5_groundtruth2.csv',task5_linearscan_result2,'Task5 Test Case 2 using linear scan')

if __name__ == "__main__":
    main()


        