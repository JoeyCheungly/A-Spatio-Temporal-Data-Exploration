__author__ = "Lok Yee Joey Cheung"
__email__ = "cheunglokyeejoey@gmail.com"
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

#-------------------------Import to PostgreSQL--------------------------#
# df_nyc = pd.read_csv('dataset_TSMC2014_NYC.csv')
# df.columns = [c.lower() for c in df_nyc.columns] # PostgreSQL doesn't like capitals or spaces

# engine = create_engine('postgresql://joeycly:123456@localhost:5432/7205project')

# df_nyc.to_sql("db_checkin", engine)



#--------------------------------------TASK1--------------------------------------#
# Find all check-ins of a particular venue in a rectangular area within a certain time window. 

def task1_rtree(filename, min_lat, min_lon, max_lat, max_lon, start_time, end_time, venueCategory):
    """Perfrom query task 1 by R-tree.

    Parameters:
        filename (str): input csv file as dataset
        min_lat (float): minimum latitude of query area
        min_lon (float): minimum longitude of query area
        max_lat (float): maximum latitude of query area
        max_lon (float): maximum longitude of query area
        start_time (str): start time of query window
        end_time (str): example end time of query window
        venueCategory (str): particular venue category
    """
    # Read file and convert the timestamp column to datetime format
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # Define the time window
    start_time = pd.Timestamp(start_time) 
    end_time = pd.Timestamp(end_time)  

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Build the R-tree index with restaurant check-in locations
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
    
    # Perform the spatial and temporal query
    results = list(rtree_index.intersection((min_lat, min_lon, max_lat, max_lon)))
    result_dp = []
    for i in results:
        timestamp = df.loc[i, 'utcTimestamp']
        if start_time <= timestamp <= end_time:
            result_dp.append(i)
     
    # Filter the dataset to include the specific venue category
    result_df = df.iloc[result_dp, :]
    result_df = result_df[result_df['venueCategory']== venueCategory]

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    result = result_df.copy()
    result['utcTimestamp'] = result_df['utcTimestamp'].dt.strftime('%a %b %d %H:%M:%S %z %Y')
    print(result)
    
    return result







def task1_qtree(filename, min_lat, min_lon, max_lat, max_lon, start_time, end_time, venueCategory):
    """ Perfrom query task 1 by Quad tree

    Parameters:
        filename (str): input csv file as dataset
        min_lat (float): minimum latitude of query area
        min_lon (float): minimum longitude of query area
        max_lat (float): maximum latitude of query area
        max_lon (float): maximum longitude of query area
        start_time (str): start time of query window
        end_time (str): example end time of query window
        venueCategory (str): particular venue category
    """
    # Read file and convert the timestamp column to datetime format
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # Define the time window
    start_time = pd.Timestamp(start_time) 
    end_time = pd.Timestamp(end_time)  

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
    
    # Perform the spatial and temporal query
    results = list(qtree.intersect((min_lat, min_lon, max_lat, max_lon)))
    result_dp = []
    for i in results:
        timestamp = df.loc[i, 'utcTimestamp']
        if start_time <= timestamp <= end_time:
            result_dp.append(i)
     
    # Filter the dataset to include the specific venue category
    result_df = df.iloc[result_dp, :]
    result_df = result_df[result_df['venueCategory']== venueCategory]

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    result = result_df.copy()
    result['utcTimestamp'] = result_df['utcTimestamp'].dt.strftime('%a %b %d %H:%M:%S %z %Y')
    print(result)
    
    return result

   

def task1_kdtree(filename, min_lat, min_lon, max_lat, max_lon, start_time, end_time, venueCategory):
    """Find top k popular venue catgories having the most data-points within a time window in a rectangular area. (Kd-tree)

    Parameters:
        filename (str): input csv file as dataset
        point (tuple): bounding box of the given rectangular area in the form of min longitude, min latitude, max longitude, max langitude 
        k(int): number of popular venues
        time_month (str): month of the given time
        time_day(str): day of the given time
    """
    # Load data from database
    data = pd.read_csv(filename)
    # Read file and convert the timestamp column to datetime format
    data['utcTimestamp'] = pd.to_datetime(data['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # Define the time window
    start_time = pd.Timestamp(start_time) 
    end_time = pd.Timestamp(end_time) 

    # Convert data to numpy array
    coords = data[['latitude', 'longitude']].values

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost   
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Build the kd-Tree
    tree = KDTree(coords)

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()
    
    # Define query point as the center of Brooklyn
    center_lat = (min_lat+max_lat)/2
    center_lon = (min_lon+max_lon)/2
    query_point = np.array([center_lat, center_lon, pd.Timestamp.now().timestamp()])

    # Query the KD-Tree for all points within Brooklyn
    brooklyn_indices = tree.query_ball_point(query_point[:2], r=0.1)

    #Get data points within the specified tiime
    result_dp = []
    for i in brooklyn_indices:
        timestamp = data.loc[i, 'utcTimestamp']
        venue = data.loc[i, 'venueCategory']  
        if start_time <= timestamp <= end_time and venue == venueCategory:
            result_dp.append(i)
    
    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")
    
    result_df = data.iloc[result_dp, :]
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
    #----------------Task 1----------------------#
    # Test Case 1: Find all check-ins in Bars within Brooklyn during New Year’s Eve and New Year’s Day. 
    task1_result = task1_rtree('dataset_TSMC2014_NYC.csv',  40.5707, - 73.9764, 40.7395, - 73.8331, 'Mon Dec 31 00:00:00 +0000 2012', 'Tue Jan 01 23:59:59 +0000 2013', 'Bar')

    task1_qtree_result = task1_qtree('dataset_TSMC2014_NYC.csv',  40.5707, - 73.9764, 40.7395, - 73.8331, 'Mon Dec 31 00:00:00 +0000 2012', 'Tue Jan 01 23:59:59 +0000 2013', 'Bar')

    task1_kdtree_result = task1_kdtree('dataset_TSMC2014_NYC.csv',  40.5707, - 73.9764, 40.7395, - 73.8331, 'Mon Dec 31 00:00:00 +0000 2012', 'Tue Jan 01 23:59:59 +0000 2013', 'Bar')
    

    # Test Case 2: Find all check-ins in 'Park' in Manhattan during the first two days of 2013
    task1_result2 = task1_rtree('dataset_TSMC2014_NYC.csv',  40.68, -74.02, 40.88, -73.93, 'Tue Jan 01 00:00:00 +0000 2013', 'Wed Jan 02 23:59:59 +0000 2013', 'Park')

    task1_qtree_result2 = task1_qtree('dataset_TSMC2014_NYC.csv',  40.68, -74.02, 40.88, -73.93, 'Tue Jan 01 00:00:00 +0000 2013', 'Wed Jan 02 23:59:59 +0000 2013', 'Park')

    task1_kdtree_result2 = task1_kdtree('dataset_TSMC2014_NYC.csv',  40.68, -74.02, 40.88, -73.93, 'Tue Jan 01 00:00:00 +0000 2013', 'Wed Jan 02 23:59:59 +0000 2013', 'Park')
    

    #Validation
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task1_groundtruth.csv',task1_result, 'Task1 Test case 1 using R tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task1_groundtruth.csv',task1_qtree_result, 'Task1 Test case 1 using Quad tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task1_groundtruth.csv',task1_kdtree_result, 'Task1 Test case 1 using  Kd tree')

    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task1_groundtruth2.csv',task1_result2, 'Task1 Test case 2 using R tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task1_groundtruth2.csv',task1_qtree_result2, 'Task1 Test case 2 using Quad tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task1_groundtruth2.csv',task1_kdtree_result2, 'Task1 Test case 2 using  Kd tree')
if __name__ == "__main__":
    main()


    

