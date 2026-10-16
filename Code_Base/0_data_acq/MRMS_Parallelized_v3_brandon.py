####################################################################################################################################
#
# Author: Marina Vicens-Miquel
# Date: August 2024
# 
# Downloading MRMS data for a single variable from the NOAA AWS S3 Bucket (https://noaa-mrms-pds.s3.amazonaws.com/index.html)
#
####################################################################################################################################

# Importing the necessary libraries
import os
import sys
import glob
import boto3
import botocore
import multiprocess as mp
import time
from datetime import datetime, timedelta
import threading
import gzip
import shutil

# Creating a lock object to safely write the names of missing files into a txt file
lock = threading.Lock()

# This function downloads the MRMS data and writes the names of missing files into a txt file
def downloadingData(variable, date, geographicLocation):

    # Defining the AWS S3 bucket and the file to download
    bucket_name = 'noaa-mrms-pds'
    s3_folder = geographicLocation + '/' + variable + '/' + date + '/'
    savePath = '/ourdisk/hpc/ai2es/datasets/MRMS/' + date[:4] + '/' + date[:8] + '/' + variable + '/compressed'
    print(savePath) 

    # Checking if the directory where we want to store the data exists. If not, create it
    os.makedirs(savePath, exist_ok=True)
    
    # Initialize the S3 client with anonymous access
    s3 = boto3.client('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=s3_folder)
    
    # Full AWS S3 bucket path
    
    for item in response.get('Contents', []):
        try:
            filename = item['Key'].removeprefix(s3_folder)
            # Downloading the file from AWS S3 bucket
            s3_key = s3_folder + filename
            local_file_path = os.path.join(savePath, filename)
            if os.path.isfile(local_file_path)==False:
                print(f'Currently downloading {filename}')
                s3.download_file(bucket_name, s3_key, local_file_path)
                print("File downloaded successfully.")
    
        except Exception as e:
            print("Error occurred while downloading the file:", e)

        # In case the file we want does not exist, we will write the file name into a text file to keep track of it. 
        # Because we are using multithreading, we need to ensure that only one thread at a time accesses the text file. 
        # Therefore, we use a lock that only allows one thread at a time. If the lock is acquired, other threads wait
        
        # I don't want the locking routine for reflectivity data, because I don't care about missing files as much...just need all data at whatever available interval

        #if lock.acquire(blocking=False):
        #    print('Moving to missing file routine...')
        #    try:
        #        with open('/ourdisk/hpc/ai2es/datasets/MRMS/' + variable + '/missingFiles.txt', 'a') as file:
        #            file.write(filename + '\n')
        #    finally:
                # Releasing the lock after writing
        #        lock.release()
        #else:  
            # Blocking until the lock is available
        #    with lock:  
        #        with open('../../ourdisk/hpc/ai2es/datasets/MRMS/' + variable + '/missingFiles.txt', 'a') as file:
        #            file.write(filename + '\n')

# Extract .gz data from downloaded files
def extractingData(variable, date):

    # Set the file path to read from for each date
    inPath = '/ourdisk/hpc/ai2es/datasets/MRMS/' + date[:4] + '/' + date[:8] + '/' + variable + '/compressed'
    os.chdir(inPath)
    #print(inPath) 

    # Extract .gz files in directory
    files = glob.glob(os.path.join(inPath+'/*.gz'))
    #print(files)
    for file in files:
        try:
            file_out = file.rstrip('.gz') 
            with gzip.open(file, 'rb') as f_in:
                with open(file_out, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    os.remove(file)
        except Exception as e:
            print("Error occurred while extracting the file:", e)

# Executing the task that we want to parallelize
def execute_task(task, variable, geographicLocation):
    
    date = task 
    # downloadingData(variable, date, geographicLocation)
    extractingData(variable,date)

# This function is responsible for retrieving all the dates for which we want to download the data
def gettingAllDates(startDate, endDate):

    allDates = []
    
    # Converting the dates from strings to datetime objects
    date = datetime.strptime(startDate, '%Y%m%d')
    endDate = datetime.strptime(endDate, '%Y%m%d')

    while date <= endDate:
        date = date + timedelta(days=1)

        # Converting all the dates back to their original string format and appending them to a list
        stringDate = date.strftime('%Y%m%d')
        allDates.append(stringDate)

    return allDates
    
def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Download', fromfile_prefix_chars='@')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    return args.idx
# Driver function
def main():

    idx = create_parser()

    # Variables definition
    variables = [
                'MergedZdr_00.50',#0
                'MergedZdr_00.75',#1
                'MergedZdr_01.00',#2
                'MergedZdr_01.25',#3
                'MergedZdr_01.50',#4
                'MergedZdr_01.75',#5
                'MergedZdr_02.00',#6
                'MergedZdr_02.25',#7
                'MergedZdr_02.50',#8
                'MergedZdr_02.75',#9
                'MergedZdr_03.00',#10
                'MergedZdr_03.50',#11
                'MergedZdr_04.00',#12
                'MergedZdr_04.50',#13
                'MergedZdr_05.00',#14
                'MergedZdr_05.50',#15
                'MergedZdr_06.00',#16
                'MergedZdr_06.50',#17
                'MergedZdr_07.00',#18
                'MergedZdr_07.50',#19
                'MergedZdr_08.00',#20
                'MergedZdr_08.50',#21
                'MergedZdr_09.00',#22
                'MergedZdr_10.00',#23
                'MergedZdr_11.00',#24
                'MergedZdr_12.00',#25
                'MergedZdr_13.00',#26
                'MergedZdr_14.00',#27
                'MergedZdr_15.00',#28
                'MergedZdr_16.00',#29
                'MergedZdr_17.00',#30
                'MergedZdr_18.00'#31
                ]
    variable = variables[idx]
    print('extracting for variable:',variable)
    geographicLocation = 'CONUS'

    startDate = '20201014' #yyyymmdd for Reflectivity, which has uneven intervals
    endDate =  '20241231'

    dates = gettingAllDates(startDate, endDate)
    print(dates)
    # Creating a queue with all the files that we want to download
    tasks_queue = [(date) for date in dates]

    # Maximum number of parallel processes (number of threads)
    num_processes = 100

    # Create a list to hold the process objects
    processes = []

    # Starting the parallelization processes
    while tasks_queue or processes:

        # Start new processes up to the maximum number of threads
        while len(processes) < num_processes and tasks_queue:
            task = tasks_queue.pop(0)
            process = mp.Process(target=execute_task, args=(task, variable, geographicLocation))
            processes.append(process)
            process.start()

        # Removing the completed processes from the list, keeping only the active ones
        processes = [process for process in processes if process.is_alive()]

        # Wait for a short time before checking again
        time.sleep(1)

    # Wait for the remaining training processes to complete
    for process in processes:
        process.join()

    print("All threads have completed their tasks.")


if __name__ == "__main__":
    main()
