import numpy as np
from scipy.spatial import KDTree
from utils.solar_geometry import pixel_to_heliographic, calculate_angular_velocity

class SunspotTracker:
    def __init__(self, solar_center_px, solar_radius_px, max_angular_speed=1):  # deg/hr
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
                self.tracks.append({
                    'positions': [pos],
                    'times': [frame_time],
                    'velocities': []
                })
            return
        
        # Convert existing tracks to angular coordinates
        prev_angular = [self._pixel_to_angular(t['positions'][-1], t['times'][-1]) for t in self.tracks]
        current_angular = [self._pixel_to_angular(c, frame_time) for c in centroids]
        
        prev_angular_tuple = [(p.lon.deg % 360, p.lat.deg % 360) for p in prev_angular]
        current_angular_tuple = [(p.lon.deg % 360, p.lat.deg % 360) for p in current_angular]
        
        # Find nearest neighbors
        tree = KDTree(current_angular_tuple) 
        distances, indices = tree.query(prev_angular_tuple, distance_upper_bound=self.max_speed)
        
        # Update tracks
        updated = set()
        for track_idx, (dist, current_idx) in enumerate(zip(distances, indices)):
            if current_idx < len(centroids):
                self._update_track(track_idx, centroids[current_idx], frame_time)
                updated.add(current_idx)
        
        # Start new tracks for unmatched detections (including splits)
        for i, pos in enumerate(centroids):
            if i not in updated:
                self.tracks.append({
                    'positions': [pos],
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

    def _update_track(self, track_idx, new_position, new_time):
        try:
            prev_time = self.tracks[track_idx]['times'][-1]
            prev_position = self.tracks[track_idx]['positions'][-1]
            prev_angular = self._pixel_to_angular(prev_position, prev_time)
            new_angular = self._pixel_to_angular(new_position, new_time)
            if None in (prev_angular, new_angular):
                return
            # Physical velocity constraints
            # print(f"prev_angular = {prev_angular}\n\
            #       prev_time = {prev_time}\n\
            #           new_angular = {new_angular}\n\
            #               new_time = {new_time}")
            
            velocity = calculate_angular_velocity(prev_angular, 
                                                prev_time,
                                                new_angular, new_time)
            # print("here 3")
            if abs(velocity) > 15:  # Max solar surface speed ~2 km/s ≈ 20 deg/day
                return
            
            # Update track
            self.tracks[track_idx]['positions'].append(new_position)
            self.tracks[track_idx]['times'].append(new_time)
            self.tracks[track_idx]['velocities'].append(velocity)
            # print("here 4")
        except Exception as e:
            print(f"Error updating track {track_idx}: {str(e)}")
        
        
        # track = self.tracks[track_idx]
        # prev_time = track['times'][-1]
        
        # # Calculate angular velocity
        # prev_angular = self._pixel_to_angular(track['positions'][-1], track['times'][-1])
        # new_angular = self._pixel_to_angular(new_position, new_time)
        
        # angular_velocity = calculate_angular_velocity(prev_angular, prev_time, new_angular, new_time)
        
        # # Update track
        # track['positions'].append(new_position)
        # track['times'].append(new_time)
        # track['velocities'].append(angular_velocity)

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
