# utils/solar_geometry.py
from sunpy.coordinates import sun
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from datetime import datetime

def pixel_to_heliographic(x, y, time, image_center, solar_radius_px):
    """
    Convert pixel coordinates to Carrington heliographic coordinates.
    
    Args:
        x, y: Pixel coordinates (float)
        time: Observation time (datetime)
        image_center: Tuple (x_center, y_center) in pixels
        solar_radius_px: Solar radius in pixels (float)
    
    Returns:
        SkyCoord: Heliographic coordinates (lon, lat in degrees)
        or None if invalid
    """
    # Calculate offset from disk center
    dx = x - image_center[0]
    dy = image_center[1] - y  # Flip y-axis
    r = np.hypot(dx, dy)
    rho = r / solar_radius_px

    if rho >= 1.0:  # Point is outside the Sun
        return None

    # Solar orientation parameters (radians)
    B0 = sun.B0(time).to(u.rad).value
    L0 = sun.L0(time).to(u.rad).value
    P = sun.P(time).to(u.rad).value

    # Position angle from solar axis
    theta = np.arctan2(dy, dx) - P

    # Calculate latitude (ψ)
    sin_psi = np.sin(B0) * np.sqrt(1 - rho**2) + np.cos(B0) * rho * np.sin(theta)
    psi = np.arcsin(sin_psi)

    # Calculate longitude (φ)
    numerator = rho * np.cos(theta)
    denominator = np.cos(B0)*np.sqrt(1 - rho**2) - np.sin(B0)*rho*np.sin(theta)
    delta_phi = np.arctan2(numerator, denominator)
    phi = (L0 + delta_phi) % (2 * np.pi)  # [0, 2π]

    return SkyCoord(phi*u.rad, psi*u.rad, frame="heliographic_carrington")


def calculate_angular_velocity(coord1: SkyCoord, time1: datetime,
                              coord2: SkyCoord, time2: datetime) -> float:
    """
    Calculate angular velocity in degrees/day between two observations.
    
    Args:
        coord1: First coordinate (SkyCoord)
        time1: Time of first observation
        coord2: Second coordinate (SkyCoord)
        time2: Time of second observation
    
    Returns:
        Angular velocity (degrees/day)
    """
    # Validate inputs
    if None in (coord1, coord2) or time2 <= time1:
        return np.nan

    # Time difference in days
    delta_days = (time2 - time1).total_seconds() / 86400
    
    # Longitude difference (handle 360° wrap)
    lon1 = coord1.lon.deg % 360
    lon2 = coord2.lon.deg % 360
    delta_lon = ((lon2 - lon1 + 180) % 360) - 180  # [-180, 180]
    
    return - delta_lon / delta_days