import numpy as np
from scipy.spatial import KDTree
from utils.solar_geometry import pixel_to_heliographic, calculate_angular_velocity

class SunspotTracker:
    def __init__(self, solar_center_px, solar_radius_px, max_angular_speed=2):  # deg/hr
        """
        solar_radius_px: Radius of Sun in pixels
        max_angular_speed: Maximum expected angular speed (deg/hr)
        """
        self.solar_center = solar_center_px
        self.solar_radius = solar_radius_px
        self.max_speed = max_angular_speed
        self.tracks = []  # List of dicts: {'positions': [], 'times': [], 'velocities': []}
        
    def process_frame(self, frame_time, centroids):
        """Process a new frame of sunspot positions"""
        centroids = self.filter_limb_features(centroids) #This is to try and eliminate negative velocities
        
        if not self.tracks:  # First frame initialization
            for pos in centroids:
                pos_helio = self._pixel_to_angular(pos,frame_time)
                self.tracks.append({
                    'positions_px': [pos],
                    'positions_helio': [pos_helio],
                    'times': [frame_time],
                    'velocities': []
                })
            return
        
        #Convert new centroids into heliographic coords
        current_angular = [self._pixel_to_angular(c, frame_time) for c in centroids]
        current_angular_tuple = [(p.lon.deg % 360, p.lat.deg) for p in current_angular]
        
        
        # Convert existing tracks to helio coords
        prev_angular = [t['positions_helio'][-1] for t in self.tracks]
        prev_angular_tuple = [(p.lon.deg % 360, p.lat.deg) for p in prev_angular]
        
        
        # Find nearest neighbors
        tree = KDTree(current_angular_tuple) 
        distances, indices = tree.query(prev_angular_tuple, distance_upper_bound=self.max_speed)
        
        
        # Update tracks
        updated = set()
        for track_idx, (dist, match_idx) in enumerate(zip(distances, indices)):
            '''Perform validation checks'''
            #Check if there are more matches than centroids
            if match_idx >= len(centroids):
                continue
            
            new_pos = centroids[match_idx]
            assert isinstance(self.tracks[track_idx]['times'], list) #Some type protection because of a bug where a list was just datetime
            prev_time = self.tracks[track_idx]['times'][-1]
            
            
            #Time gap validation (max 3 hours)
            if (frame_time - prev_time).total_seconds() > 3*3600:
                continue
            
            #Velocity validation
            # print(f"prev_angular = {prev_angular}\ncurrent_angular = {current_angular}\ntrack_idx = {track_idx}")
            velocity = calculate_angular_velocity(prev_angular[track_idx], prev_time, current_angular[match_idx], frame_time)
            if abs(velocity) > 15: #deg/day
                continue
            
            '''Update tracks'''
            self.tracks[track_idx]['positions_px'].append(new_pos)
            self.tracks[track_idx]['positions_helio'].append(current_angular[match_idx])
            self.tracks[track_idx]['times'].append(frame_time)
            self.tracks[track_idx]['velocities'].append(velocity)
            updated.add(match_idx)
            
        #Start new tracks for unmatched centroids
        for i, pos in enumerate(centroids):
            if i not in updated:
                self.tracks.append({
                    'positions_px': [pos],
                    'positions_helio': [current_angular[i]],
                    'times': [frame_time],
                    'velocities': []
                })
        

    def _pixel_to_angular(self, position, time):
        """Convert pixel position to angular coordinates (degrees from center)"""
        x, y = position
        coords = pixel_to_heliographic(x, y, time, self.solar_center, self.solar_radius)
        if coords and not (np.isnan(coords.lon.deg) or np.isnan(coords.lat.deg)):
            return coords
        return None

    # def _update_track(self, track_idx, new_position, new_time):
    #     try:
    #         prev_time = self.tracks[track_idx]['times'][-1]
    #         prev_angular = self.tracks[track_idx]['positions_helio'][-1]
    #         new_angular = self._pixel_to_angular(new_position, new_time)
            
    #         if None in (prev_angular, new_angular):
    #             return
            
    #         # Physical velocity constraints
    #         velocity = calculate_angular_velocity(prev_angular, 
    #                                             prev_time,
    #                                             new_angular, new_time)
    #         # print("here 3")
    #         if abs(velocity) > 15:  # Max solar surface speed ~2 km/s ≈ 20 deg/day
    #             return
            
    #         #Check for large time jump to avoid rematching incorrectly
    #         max_gap = 3*3600 #3 hour max?
    #         if (new_time - prev_time).total_seconds() > max_gap:
    #             return
            
    #         # Update track
    #         self.tracks[track_idx]['positions_px'].append(new_position)
    #         self.tracks[track_idx]['positions_helio'].append(new_angular)
    #         self.tracks[track_idx]['times'].append(new_time)
    #         self.tracks[track_idx]['velocities'].append(velocity)
    #         # print("here 4")
    #     except Exception as e:
    #         print(f"Error updating track {track_idx}: {str(e)}")
        
    def get_all_velocities(self):
        """Return all velocity measurements in deg/hr"""
        return [track['velocities'] for track in self.tracks]
    
    def filter_limb_features(self, centroids):
        center = self.solar_center
        filtered = []
        for x, y in centroids:
            r = np.hypot(x - center[0], y - center[1]) / self.solar_radius
            if r <= 0.85:
                filtered.append((x, y))
        return filtered
