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



#--------------------------------------TASK2--------------------------------------#

def task2_rtree(filename, max_distance, start_time, end_time, lat, lon):
    """Within a time window, find all check-ins within a maximum distance of d kilometers to a given location. (R tree)

    Parameters:
        filename (str): input csv file as dataset
        max_distance (int): maximum distance in kilometers
        start_time (str): start time of the specified time window
        end_time (str): end time of the specified time window
        lat (float): latitude of given location
        lon (float): longitude of given location
    """
    # Define the time window
    start_time = pd.Timestamp(start_time) 
    end_time = pd.Timestamp(end_time)

    # Read file and convert the timestamp column to datetime format
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    #Build the R-tree index
    rtree_index = index.Index()
    for i, checkin in df.iterrows():
        rtree_index.insert(i, (checkin['latitude'], checkin['longitude'], checkin['latitude'], checkin['longitude']))

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    #Query the index to find required data points
    checkins = []
    for r_index in rtree_index.intersection((lat-0.05, lon-0.05,lat+0.05, lon+0.05)):
        checkin = df.loc[r_index]
        distance = great_circle((lat,lon), (checkin['latitude'], checkin['longitude'])).kilometers
        if distance <= max_distance and start_time <= checkin['utcTimestamp'] <= end_time:
            checkins.append(checkin)
    
    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    result = pd.DataFrame(checkins)
    print(result)

    return result









def task2_qtree(filename, max_distance, start_time, end_time, lat, lon):
    """Within a time window, find all check-ins within a maximum distance of d kilometers to a given location. (Quad tree)

    Parameters:
        filename (str): input csv file as dataset
        max_distance (int): maximum distance in kilometers
        start_time (str): start time of the specified time window
        end_time (str): end time of the specified time window
        lat (float): latitude of given location
        lon (float): longitude of given location
    """
    # Define the time window
    start_time = pd.Timestamp(start_time) 
    end_time = pd.Timestamp(end_time)

    # Read file and convert the timestamp column to datetime format
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')

    #Put coordinates into a list to prepare for insertion to index
    checkins = []
    for index, row in df.iterrows():
        checkins.append({
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
        })

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Create the Quad tree index
    qtree = Index(bbox=[-74.27476645, 40.55085247, -73.6838252, 40.98833172]) #New york

    nearby_checkins = []
    for i, checkin in enumerate(checkins):
        qtree.insert(i, (checkin['latitude'], checkin['longitude'], checkin['latitude'], checkin['longitude']))

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    
    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    #Query the index to find required data points
    for item in qtree.intersect((lat-0.05, lon-0.05,lat+0.05, lon+0.05)):
        checkin = df.loc[item]
        distance = great_circle((lat,lon), (checkin['latitude'], checkin['longitude'])).kilometers
        if distance <= max_distance and start_time <= checkin['utcTimestamp'] <= end_time:
            nearby_checkins.append(checkin)

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    # Create DataFrame from nearby_checkins
    result = pd.DataFrame(nearby_checkins)
    print(result)

    return result





def task2_grid(filename, max_distance, start_time, end_time, lat,lon):
    """Within a time window, find all check-ins within a maximum distance of d kilometers to a given location. (Grid-based indexing)

    Parameters:
        filename (str): input csv file as dataset
        max_distance (int): maximum distance in kilometers
        start_time (str): start time of the specified time window
        end_time (str): end time of the specified time window
        lat (float): latitude of given location
        lon (float): longitude of given location
    """
    # Define the time window
    start_time = pd.Timestamp(start_time) 
    end_time = pd.Timestamp(end_time)

    # Load the check-in data from CSV
    checkin_csv = []
    checkin_indices = {}

    with open(filename,'r') as checkin_file:
        checkin_reader = csv.DictReader(checkin_file)
        for num, row in enumerate(checkin_reader):
            checkin_csv.append({'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'timestamp': datetime.datetime.strptime(row['utcTimestamp'], '%a %b %d %H:%M:%S %z %Y'), #convert timestamp to datetime format
            })
            checkin_indices[num] = len(checkin_csv ) - 1 #record the indices of data

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Define size of grid cell in degress and create the grid index
    grid_size = 0.1  
    #Dictionary storing cell coordinates and associated check-in data
    grid_index = {}

    # Insert check-in points into associated grid cells
    for item in checkin_csv:
        point_lat = int(item['latitude']/grid_size)
        point_lon = int(item['longitude']/grid_size)
        if (point_lat,point_lon) in grid_index:
            grid_index[(point_lat,point_lon)].append(item)
        else:
            grid_index[(point_lat,point_lon)] = [item]

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Perform the query by iterating over each neighboring cell to examine each check-in data point
    neighbour_point = []
    neighbour_point_index = []
    for x in range(point_lat -1, point_lat+2):
        for y in range(point_lon-1, point_lon+2):
            if (x,y) in grid_index:
                checkin_data = grid_index[(x,y)]
                # Check the distance and timestamp of each data point and filter
                for i in checkin_data:
                    distance = great_circle((lat,lon), (i['latitude'], i['longitude'])).kilometers
                    if distance <= max_distance and start_time <= i['timestamp'] <= end_time:
                        neighbour_point.append(i)
                        neighbour_point_index.append(checkin_indices[checkin_csv.index(i)])


    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    indices = []
    for i, j in zip(neighbour_point,neighbour_point_index):
        indices.append(j)

    df = pd.read_csv('dataset_TSMC2014_NYC.csv')
    result_df = df.iloc[indices, :]
    print(result_df)

    return result_df






#--------------------------------------VALIDATION--------------------------------------#
def validation(ground_truth_file, prediction, task):
    # Obtain ground truth for the query
    gt = pd.read_csv(ground_truth_file)
    ground_truth = gt['index'].tolist()

    # Compare the results
    tp1 = prediction[prediction.index.isin(ground_truth)].shape[0]
    fp1 = prediction[~prediction.index.isin(ground_truth)].shape[0]
    fn1 = len(ground_truth) - tp1
    precision = tp1 / (tp1 + fp1)
    recall = tp1 / (tp1 + fn1)
    f1_score = (2 * precision * recall) / (precision + recall)
    
    print(f"Correctness of {task}: Precision={precision}, Recall={recall}, F-measure={f1_score}")



def main():
    """ Main function: call other functions to perform desired actions.
    """

    #----------------Task 2----------------------#
    #Test Case 1: Find all check-ins within a maximum distance of 5 km to Museum of the City of New York in the afternoon of Independence Day (4th July). 
    task2_rtree_result = task2_rtree('dataset_TSMC2014_NYC.csv', 5.0, 'Wed Jul 04 12:00:00 +0000 2012','Wed Jul 04 17:00:00 +0000 2012',40.792406, -73.952038)

    task2_qtree_result = task2_qtree('dataset_TSMC2014_NYC.csv', 5.0, 'Wed Jul 04 12:00:00 +0000 2012','Wed Jul 04 17:00:00 +0000 2012',40.792406, -73.952038)

    task2_grid_result = task2_grid('dataset_TSMC2014_NYC.csv', 5.0, 'Wed Jul 04 12:00:00 +0000 2012','Wed Jul 04 17:00:00 +0000 2012',40.792406, -73.952038)

    #Test Case 2: Find all check-ins within a maximum distance of 10 km to Central Park in Valentines' Day.
    task2_rtree_result2 = task2_rtree('dataset_TSMC2014_NYC.csv', 3.0, 'Thu Feb 14 00:00:00 +0000 2013','Thu Feb 14 23:59:59 +0000 2013',40.781512, -73.966515)

    task2_qtree_result2 = task2_qtree('dataset_TSMC2014_NYC.csv', 3.0, 'Thu Feb 14 00:00:00 +0000 2013','Thu Feb 14 23:59:59 +0000 2013',40.781512, -73.966515)

    task2_grid_result2 = task2_grid('dataset_TSMC2014_NYC.csv', 3.0, 'Thu Feb 14 00:00:00 +0000 2013','Thu Feb 14 23:59:59 +0000 2013',40.781512, -73.966515)


    #Validation
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task2_groundtruth.csv',task2_rtree_result,'Task2 Test case 1 using R tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task2_groundtruth.csv',task2_qtree_result,'Task2 Test case 1 using Quad tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task2_groundtruth.csv',task2_grid_result, 'Task2 Test case 1 using Grid-based indexing')

    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task2_groundtruth2.csv',task2_rtree_result2,'Task2 Test case 2 using R tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task2_groundtruth2.csv',task2_qtree_result2,'Task2 Test case 2 using Quad tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task2_groundtruth2.csv',task2_grid_result2, 'Task2 Test case 2 using Grid-based indexing')
if __name__ == "__main__":
    main()

