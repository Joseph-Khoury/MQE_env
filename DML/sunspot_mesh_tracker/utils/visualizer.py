import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def plot_track_longitudes(tracks):
    plt.figure(figsize=(12, 6))
    for track in tracks:
        times = [t[0] for t in track]
        lons = [t[3] for t in track]
        plt.plot(times, lons, 'o-', markersize=2, alpha=0.7)
    
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xlabel('Time')
    plt.ylabel('Carrington Longitude (°)')
    plt.title('Sunspot Longitudinal Motion')
    plt.grid(True)
    plt.show()

def plot_rotation_profile(latitudes, periods):
    plt.figure(figsize=(10, 6))
    plt.scatter(latitudes, periods, alpha=0.5, edgecolor='k')
    plt.xlabel('Latitude (°)')
    plt.ylabel('Rotation Period (days)')
    plt.title('Solar Differential Rotation')
    plt.ylim(20, 35)
    plt.grid(True)
    plt.show()