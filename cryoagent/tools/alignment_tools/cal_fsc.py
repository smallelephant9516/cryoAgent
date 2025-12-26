import numpy as np
from numpy.fft import fftn, fftshift



def calculate_fsc(map1, map2, pixel_size, shells=50):
    """
    Calculate FSC curve between two 3D density maps.
    
    Parameters:
    -----------
    map1, map2 : numpy.ndarray
        3D density maps with shape (N,N,N)
    pixel_size : float
        Pixel size in Angstroms
    shells : int
        Number of shells for FSC calculation
        
    Returns:
    --------
    spatial_freq : numpy.ndarray
        Spatial frequencies in 1/Angstrom
    fsc : numpy.ndarray
        FSC correlation values
    """
    # Check input dimensions
    assert map1.shape == map2.shape, "Maps must have the same dimensions"
    assert len(map1.shape) == 3, "Maps must be 3D"
    N = map1.shape[0]
    
    # Calculate Fourier transforms
    ft1 = fftshift(fftn(map1))
    ft2 = fftshift(fftn(map2))
    
    # Create distance matrix from center
    center = N // 2
    x, y, z = np.ogrid[-center:center, -center:center, -center:center]
    r = np.sqrt(x*x + y*y + z*z)
    
    # Maximum radius and shell thickness
    max_r = center
    shell_thickness = max_r / shells
    
    # Initialize arrays for results
    fsc = np.zeros(shells)
    shell_volumes = np.zeros(shells)
    
    # Calculate FSC for each shell
    for i in range(shells):
        r_min = i * shell_thickness
        r_max = (i + 1) * shell_thickness
        shell_mask = (r >= r_min) & (r < r_max)
        
        # Extract complex values in the shell
        f1_shell = ft1[shell_mask]
        f2_shell = ft2[shell_mask]
        
        # Calculate correlation
        numerator = np.abs(np.sum(f1_shell * f2_shell.conj()))
        denominator = np.sqrt(np.sum(np.abs(f1_shell)**2) * np.sum(np.abs(f2_shell)**2))
        
        if denominator != 0:
            fsc[i] = numerator / denominator
        shell_volumes[i] = np.sum(shell_mask)
    
    # Calculate spatial frequencies (in 1/Angstrom)
    spatial_freq = np.arange(shells) * shell_thickness / (N * pixel_size)
    
    return spatial_freq, fsc

# Example usage:
if __name__ == '__main__':
    # For demonstration, create two synthetic 3D density maps.
    # In practice, you would load your density maps (e.g., from .mrc files or similar).
    import mrcfile
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("map1", help="Input MRC map 1")
    p.add_argument("map2", help="Input MRC map 2")
    args = p.parse_args()

    map1_meta = mrcfile.open(args.map1)
    map2_meta = mrcfile.open(args.map2)
    map1 = map1_meta.data
    map2 = map2_meta.data
    pixel_size = map1_meta.voxel_size.x
    print(f"Pixel size: {pixel_size} Å")

    spatial_freq, fsc = calculate_fsc(map1, map2, pixel_size)

    gold_standard = np.argwhere(fsc < 0.5).min()
    

    resolution_20A = np.argwhere(spatial_freq < 0.05).max()
    fsc_20A = fsc[:resolution_20A]
    if len(fsc_20A[fsc_20A<0.85]) > 0:
        gold_standard = np.argwhere(fsc_20A < 0.85).min()
    
    try:
        resolution = (spatial_freq[gold_standard] + spatial_freq[gold_standard+1]) / 2
    except:
        resolution = (spatial_freq[gold_standard] + spatial_freq[gold_standard-1]) / 2
    #resolution = spatial_freq[gold_standard]
    resolution = round(1/resolution, 2)

    print(f"Gold standard: {gold_standard}")
    print(f"FSC: {fsc}")
    print(f"Resolution: {resolution} Å")