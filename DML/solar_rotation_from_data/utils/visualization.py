import os
from ipywidgets import interact, Dropdown
import matplotlib.pyplot as plt
import cv2

from utils.image_processing import detect_sunspots

def show_sunspot_images(file_paths: list[str] = None, data: list[dict] = None):
    
    #Get the days 
    day_dirs = []
    for path in file_paths:
        day = path.split('/')[1]
        if day not in day_dirs:
            day_dirs.append(day)
            
    @interact(day=Dropdown(options=day_dirs, description="Select Day:"))
    def show_day_images(day):
        files = sorted([f for f in file_paths if day in f])[:16] #only the first 16 images
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        for ax, file in zip(axes.flat, files):
            img, centroids, _, _ = detect_sunspots(file)
            print(centroids)
            
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.scatter([c[0] for c in centroids], [c[1] for c in centroids], s=5, c='blue')
            
            #If processed data, match the sunspots to tracked features by coordinate matching (odds of mismatch are very small)
            if data:
                matches = []
                for i, coords in enumerate(centroids):
                    for track_idx, track in enumerate(data):
                        for _, pos_px in enumerate(track['positions_px']):
                            if list(coords) == pos_px:
                                matches.append([i,track_idx])
                # print(matches)
                
                for centroid_idx, track_id in matches:
                    x,y = centroids[centroid_idx]
                    ax.text(x + 3, y + 3, str(track_id), color='blue', fontsize=8, weight='bold')
            
            ax.set_title(file.split('_')[-1].removesuffix('.jpg'))  # Show time (hhmm)
            ax.axis('off')
        plt.tight_layout()
        
if __name__ == "__main__()":
    pass