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
from rtree import index
from pyqtree import Index
from sqlalchemy import create_engine
from collections import defaultdict



#--------------------------------------TASK4--------------------------------------#
def task4_qtree(filename,point):
    """Find the number of data-points within a rectangular area in certain period of times. (Quad tree)

    Parameters:
        filename (str): input csv file as dataset
        point (tuple): latitude and longitude of the given rectangle
    """
    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    #Build the quad tree index
    index = Index(bbox=[-74.27476645, 40.55085247, -73.6838252, 40.98833172]) #New york
    with open(filename, 'r') as file:
        checkin = csv.DictReader(file)
        for row in checkin:
            index.insert(item = (float(row['longitude']), float(row['latitude']), row['utcTimestamp']),bbox = (float(row['longitude']), float(row['latitude']), float(row['longitude']), float(row['latitude'])))
    
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

    longitude =  float(row['longitude'])
    latitude = float(row['latitude'])
    timestamp = row['utcTimestamp']
    seasons_count = {}
    seasons = {'Spring': ['Mar','Apr','May'], 'Summer': ['Jun','Jul','Aug'], 'Fall': ['Sep','Oct','Nov'], 'Winter': ['Dec','Jan','Feb']}
    
    #Filter the data base on seasons
    for i in results:
        longitude, latitude, timestamp  = i
        month = timestamp.split(' ')[1]
        for key, items in seasons.items():
            if month in items:
                if key not in seasons_count:
                    seasons_count[key] = 1
                else:
                    seasons_count[key] += 1
    
    # for season, count in seasons_count.items():
    #     print(f"Number of check-ins in {season}: {count}")

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    result = pd.DataFrame(seasons_count.items(), columns=['season','num_points'])
    result = result.sort_values("season",ascending=True)
    result = result.reset_index(drop=True)
    result.index += 1
    print(result)   

    return result




def task4_rtree(filename,point):
    """Find the number of data-points within a rectangular area in certain period of times. (R-tree)

    Parameters:
        filename (str): input csv file as dataset
        point (tuple): latitude and longitude of the given rectangle
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

    #Query the r tree to find data points intersecting the bounding box of rectangular area
    results = list(rtree_index.intersection(point))

    seasons_count = {}
    seasons = {'Spring': ['Mar','Apr','May'], 'Summer': ['Jun','Jul','Aug'], 'Fall': ['Sep','Oct','Nov'], 'Winter': ['Dec','Jan','Feb']}
        
    for i in results:
        timestamp = df.loc[i, 'utcTimestamp']
        month = timestamp.split(' ')[1]
        for key, items in seasons.items():
            if month in items:
                if key not in seasons_count:
                    seasons_count[key] = 1
                else:
                    seasons_count[key] += 1

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    result = pd.DataFrame(seasons_count.items(), columns=['season','num_points'])
    result = result.sort_values("season",ascending=True)
    result = result.reset_index(drop=True)
    result.index += 1
    print(result)  

    return result





def task4_linearscan(filename,point):
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

    # Define seasons and set a dictionary to store the counts for each cell and season
    seasons = {'Spring': ['Mar','Apr','May'], 'Summer': ['Jun','Jul','Aug'], 'Fall': ['Sep','Oct','Nov'], 'Winter': ['Dec','Jan','Feb']}
    checkin = defaultdict(int) 
    #Query for data points 
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            timestamp = row['utcTimestamp']
            
            # Check if the check-in point is within the bounding box
            if (point[0] <= longitude <= point[2]) and (point[1] <= latitude <= point[3]):                
                # Increment the count for the corresponding season
                month = timestamp.split(' ')[1]
                for s, months in seasons.items():
                    if month in months:
                        season = s
                        break
                if season not in checkin:
                    checkin[season] = 0
                checkin[season] += 1

        # Display computational complexity of executing query
        print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
        print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
        end_disk_io = psutil.disk_io_counters()
        disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
        print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

        result = pd.DataFrame(checkin.items(), columns=['season','num_points'])
        result = result.sort_values("season",ascending=True)
        result = result.reset_index(drop=True)
        result.index += 1
        print(result) 

        return result 



#--------------------------------------VALIDATION--------------------------------------#
def validation_t4(ground_truth_file, prediction, task):
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

    #----------------Task 4----------------------#
    #Test Case 1: Find the number of people travelling from John F. Kennedy International Airport in each season. 
    task4_qtree_result = task4_qtree('dataset_TSMC2014_NYC.csv',(-73.810, 40.660, -73.390, 40.670))

    task4_rtree_result = task4_rtree('dataset_TSMC2014_NYC.csv',(40.660, -73.810, 40.670, -73.390))

    task4_linearscan_result = task4_linearscan('dataset_TSMC2014_NYC.csv',(-73.810, 40.660, -73.390, 40.670))

    #Test Case 2: Find the number of people travelling from LaGuardia Airport in each season. 
    task4_qtree_result2 = task4_qtree('dataset_TSMC2014_NYC.csv',(-73.890441, 40.773110, -73.849975, 40.785975))

    task4_rtree_result2 = task4_rtree('dataset_TSMC2014_NYC.csv',(40.773110, -73.890441, 40.785975, -73.849975))

    task4_linearscan_result2 = task4_linearscan('dataset_TSMC2014_NYC.csv',(-73.890441, 40.773110, -73.849975, 40.785975))


    #Validation
    validation_t4('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task4_groundtruth.csv',task4_qtree_result,'Task4 Test Case 1 using Quad tree')
    validation_t4('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task4_groundtruth.csv',task4_rtree_result,'Task4 Test Case 1 using R tree')
    validation_t4('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task4_groundtruth.csv',task4_linearscan_result,'Task4 Test Case 1 using linear scan')

    validation_t4('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task4_groundtruth2.csv',task4_qtree_result2,'Task4 Test Case 2 using Quad tree')
    validation_t4('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task4_groundtruth2.csv',task4_rtree_result2,'Task4 Test Case 2 using R tree')
    validation_t4('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task4_groundtruth2.csv',task4_linearscan_result2,'Task4 Test Case 2 using linear scan')

if __name__ == "__main__":
    main()
