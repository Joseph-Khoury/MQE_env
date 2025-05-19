import os
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

default_url = "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/2025/hmiigr/"
default_save_dir = "sdo_hmi_jpgs"

#Function to download images from the SOHO archive
def fetch_images(data_bank_url: str = default_url, save_dir: str = default_save_dir, start_date: datetime = None, end_date: datetime = None, cadence: timedelta = timedelta(hours=1.5)):

    os.makedirs(save_dir, exist_ok=True)

    #Generate all of the timestamps for each of the desired files
    timestamps = []
    _timestamp = start_date
    while _timestamp < end_date:
        timestamps.append(_timestamp)
        _timestamp += cadence

    #Begin download
    with tqdm(total=len(timestamps), desc="Downloading HMI images...", unit = "img") as pbar: 
        for current in timestamps:
            try:
                #Construct image URL
                date_str = current.strftime(r"%Y%m%d")
                timestamp = current.strftime(r"%Y%m%d_%H%M")
                url = f"{data_bank_url}{date_str}/{timestamp}_hmiigr_512.jpg"
                
                #Make the day directory
                day_dir = os.path.join(save_dir,date_str)
                os.makedirs(day_dir, exist_ok=True)
                
                file_name = os.path.join(day_dir, f"{timestamp}.jpg")
                
                #Check if the file exists already
                if os.path.exists(file_name):
                    pbar.update(1)
                    continue
                
                #Download the image
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(file_name, 'wb') as f:
                        f.write(response.content)
                    print(f"successfully downloaded {file_name}")
                else:
                    tqdm.write(f"Achtung! Image not available: {url}")
                    
            except Exception as e:
                print(f"Error at {current}: {e}")
            pbar.update(1)
    print("All requested files successfully downloaded!")
    

# Function to extract the image paths and their timestamps
def get_files_with_times(root_dir: str = "sdo_hmi_jpgs"):
    file_paths = []
    times = []
    for day_dir in sorted(os.listdir(root_dir)):
        day_path = os.path.join(root_dir,day_dir)
        if os.path.isdir(day_path):
            for file in sorted(os.listdir(day_path)):
                if file.endswith(".jpg"):
                    time_str = file.split('_')[1] # Extract time (hhmm)
                    time_str = time_str.removesuffix('.jpg')
                    time = datetime.strptime(f"{day_dir}_{time_str}", r"%Y%m%d_%H%M")
                    file_paths.append(os.path.join(day_path,file))
                    times.append(time)
    return file_paths, times