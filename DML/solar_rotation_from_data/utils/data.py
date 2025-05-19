import os
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
import json
from astropy.coordinates import SkyCoord

from utils.solar_geometry import calculate_angular_velocity

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

#Function to merge tracks based on proximity to others
def track_association(data: list[dict] = None, max_gap_hours = 3, max_distance_deg = 5):

# Create a table to stitch short tracks into longer ones
    merged_tracks = []
    used = set()

    for i, t1 in enumerate(data):
        if i in used:
            continue
        merged = t1.copy()
        used.add(i)
        for j, t2 in enumerate(data):
            if j in used or i == j:
                continue
            # Compare end of t1 to start of t2
            time_gap = (t2["times"][0] - t1["times"][-1]).total_seconds() / 3600.0
            # print(i,j,t2["times"][0], t1["times"][-1],time_gap)
            if 0 < time_gap <= max_gap_hours:
                dist = t1["positions_helio"][-1].separation(t2["positions_helio"][0]).deg
                if dist <= max_distance_deg:
                    # Merge t2 into merged
                    
                    '''
                    calculate velocity between them,
                    add velocity,
                    then merge.
                    '''
                    #calculate the velocity between the merging points
                    new_velocity = calculate_angular_velocity(
                        t1['positions_helio'][-1],
                        t1['times'][-1],
                        t2['positions_helio'][0], 
                        t2['times'][0]
                        )
                    merged['velocities'].append(new_velocity)
                    
                    #then merge the rest
                    for key in t1.keys():
                        merged[key].extend(t2[key])
                    used.add(j)
        merged_tracks.append(merged)
    return merged_tracks

#function to convert to JSON
def toJSON(data: list[dict] = None, file_name: str = None):
    #First convert necessary datatypes to str
    for entry in data:
        if 'times' in entry:
            entry['times'] = [t.isoformat() if isinstance(t, datetime) else t for t in entry['times']]
        if 'positions_helio' in entry:
            entry['positions_helio'] = [skycoord_to_dict(coord) if isinstance(coord, SkyCoord) else coord for coord in entry['positions_helio']]

    with open(file=file_name,mode='w') as f:
        json.dump(data, f, indent=4)

#Convert JSON data back into useable format
def fromJSON(data_path: str = "sunspot_data.json"):
    with open(data_path, 'r') as f:
        data = json.load(f)

    #Convert strings back into necessary datatypes
    for entry in data:
        if 'times' in entry:
            entry['times'] = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in entry['times']]
        if 'positions_helio' in entry:
            entry['positions_helio'] = [dict_to_skycoord(coord) if isinstance(coord, dict) else coord for coord in entry['positions_helio']]
    return data


#This is for converting to JSON
def skycoord_to_dict(coord):
    #This function is for saving to JSON
    return {
        'lon': coord.lon.deg,
        'lat': coord.lat.deg,
        'frame': coord.frame.name,
        'unit': 'deg'
    }
    
#This is for converting from JSON
def dict_to_skycoord(d):
    return SkyCoord(lon=d['lon'], lat=d['lat'],
                    frame=d['frame'], unit=d['unit'])