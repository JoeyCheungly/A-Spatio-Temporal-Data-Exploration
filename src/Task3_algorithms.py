"""
INFS7205 Individual Project
Semester 1, 2023
"""

__author__ = "Lok Yee Joey Cheung"
__email__ = "s4763354@student.uq.edu.au"
__date__ = "06/05/2023"

import pandas as pd
import datetime
import psutil
from rtree import index
from scipy.spatial import KDTree
from sqlalchemy import create_engine
#from sklearn.neighbors import BallTree
import geohash
from geopy.distance import geodesic

#--------------------------------------TASK3--------------------------------------#

def task3(filename, location, date, k):
    """ Find k nearest neighbour of the given check-in within the same day. (KD tree)

    Parameters:
        filename (str): input csv file as dataset
        location (list): latitude and longitude of the given check-in location
        date (str): date of the given check-in
        k (int): number of nearest neighbour to be found
    """
    # Read file and convert the timestamp column to datetime format
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # Filter the dataset based on the given date
    new_df = df[df['utcTimestamp'].dt.date == pd.to_datetime(date).date()]

    # Filter out the given location from the dataset
    new_df = new_df[(new_df['latitude'] != location[0]) | (new_df['longitude'] != location[1])]

    # Create a numpy array of the latitude and longitude 
    point = new_df[['latitude', 'longitude']].to_numpy()

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost    
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Create a KDTree object using the coordinates
    tree = KDTree(point)

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Query the KDTree object to find the k-nearest neighbors
    distance, index = tree.query([location], k)

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    # Retrieve and display the rows corresponding to the nearest neighbors
    nearest_neighbors = new_df.iloc[index[0]]
    result_knn = nearest_neighbors.copy()
    result_knn['utcTimestamp'] = nearest_neighbors['utcTimestamp'].dt.strftime('%a %b %d %H:%M:%S %z %Y')
    
    #Convert distance from coordinates to meters 
    distance_m = distance[0]*11139
    result_knn['distance'] = distance_m.tolist()
    print(result_knn)

    return result_knn








def task3_rtree(filename, location, date, k):
    """ Find k nearest neighbour of the given check-in within the same day. (R tree)

    Parameters:
        filename (str): input csv file as dataset
        location (list): latitude and longitude of the given check-in location
        date (str): date of the given check-in
        k (int): number of nearest neighbour to be found
    """
    # Read file and convert the timestamp column to datetime format
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')    

    # Filter the dataset based on the given date
    new_df = df[df['utcTimestamp'].dt.date == pd.to_datetime(date).date()]

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost    
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Build the R-tree index with restaurant check-in locations
    rtree_index = index.Index()
    for i, row in new_df.iterrows():
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

    # Perform the nearest neighbor search
    nearest_neighbors = list(rtree_index.nearest((location[0],location[1]), k))

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    # Filter the dataset based on the given date
    result_df = df.iloc[nearest_neighbors, :]
    print(result_df)

    return result_df






def task3_geohash(filename, location, date, k):
    """ Find k nearest neighbour of the given check-in within the same day. (Geohash)

    Parameters:
        filename (str): input csv file as dataset
        location (list): latitude and longitude of the given check-in location
        date (str): date of the given check-in
        k (int): number of nearest neighbour to be found
    """

    # Load data from CSV file
    df = pd.read_csv(filename)
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')   
    
    # Filter the dataset based on the given date
    new_df = df[df['utcTimestamp'].dt.date == pd.to_datetime(date).date()]

    #Calculate computational complexity i.e. time cost, memory cost, I/O cost    
    start = datetime.datetime.now()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Encode latitude and longitude to Geohash values
    new_df['geohash'] = new_df.apply(lambda row: geohash.encode(row['latitude'], row['longitude']), axis=1)

    # Encode query location to Geohash
    query_geohash = geohash.encode(location[0], location[1])

    # Display computational complexity of building index
    print('Time cost of building index:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of building index:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of building index: {disk_io_cost / 1024 / 1024:.2f} MB")

    start = datetime.datetime.now()
    mem_before = process.memory_info().rss
    start_disk_io = psutil.disk_io_counters()

    # Filter data within a certain radius from the query location using Geohash prefix
    prefix_length =6 
    prefix = query_geohash[:prefix_length]
    filtered_data = new_df[new_df['geohash'].str.startswith(prefix)]

    # Calculate distances between the query location and filtered data points
    filtered_data['distance'] = filtered_data.apply(lambda row: geodesic(location, [row['latitude'], row['longitude']]).meters, axis=1)

    # Sort the filtered data by distance and retrieve the nearest neighbors
    nearest_neighbors = filtered_data.nsmallest(k, 'distance')

    # Display computational complexity of executing query
    print('Time cost of executing query:',(datetime.datetime.now() - start).total_seconds())
    print('Memory cost of executing query:',(process.memory_info().rss - mem_before)) 
    end_disk_io = psutil.disk_io_counters()
    disk_io_cost = end_disk_io.read_bytes - start_disk_io.read_bytes + end_disk_io.write_bytes - start_disk_io.write_bytes
    print(f"Disk I/O cost of executing query: {disk_io_cost / 1024 / 1024:.2f} MB")

    # Print the nearest neighbors
    print(nearest_neighbors)

    return nearest_neighbors






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
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    
    print(f"Correctness of {task}: Precision={precision}, Recall={recall}, F-measure={f1_score}")




def main():
    """ Main function: call other functions to perform desired actions.
    """

    #----------------Task 3----------------------#
    #Test Case 1: Find 5 nearest neighbour of the the given check-in at location [40.78430372 , -73.97968281] within the same day which is 4th of April 2012. 
    task3_result = task3('dataset_TSMC2014_NYC.csv', [40.78430372 , -73.97968281], '2012-04-04', 5)

    task3_rtree_result = task3_rtree('dataset_TSMC2014_NYC.csv', [40.78430372 , -73.97968281], '2012-04-04', 5)

    task3_geohash_result = task3_geohash('dataset_TSMC2014_NYC.csv', [40.78430372 , -73.97968281], '2012-04-04', 5)
    

    #Test Case 2: Find 3 nearest neighbour of the the given check-in at location [40.96537083,-74.06281492] within the same day which is 14th of May 2012. 
    task3_result2 = task3('dataset_TSMC2014_NYC.csv', [40.96537083,-74.06281492], '2012-05-14', 3)

    task3_rtree_result2 = task3_rtree('dataset_TSMC2014_NYC.csv', [40.96537083,-74.06281492], '2012-05-14', 3)

    task3_geohash_result2 = task3_geohash('dataset_TSMC2014_NYC.csv', [40.96537083,-74.06281492], '2012-05-14', 3)

    # #Validation
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task3_groundtruth.csv',task3_result, 'Task3 Test Case 1 using kd tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task3_groundtruth.csv',task3_rtree_result, 'Task3 Test Case 1 using r tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task3_groundtruth.csv',task3_geohash_result, 'Task3 Test Case 1 using geohash')

    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task3_groundtruth2.csv',task3_result2, 'Task3 Test Case 2 using kd tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task3_groundtruth2.csv',task3_rtree_result2, 'Task3 Test Case 2 using r tree')
    validation('/Users/joeycly/Desktop/Sem1/INFS7205/report/ground_truth/task3_groundtruth2.csv',task3_geohash_result2, 'Task3 Test Case 2 using geohash')


if __name__ == "__main__":
    main()


    