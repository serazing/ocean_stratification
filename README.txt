# ocean_stratification

List of library files:
---------------------
- isas_io.py: list of functions to open the ISAS dataset and write outputs
- profile_tools.py: list of functions to analyse the vertical profiles
- plot.py: list of plotting functions

List of batch files:
-------------------
1-make_stratification_profiles.py: Construct the stratification profiles on the profile dataset
2-compute_mlds.py: Compute the mixed layer depth estimates on the profile dataset
3-compute_n2_peaks.py: Capture all the different stratification peaks with their properties (intensity, depth, thickness)
4-compute_uop_boundary_variables.py: Compute the variables at the top and the bottom of stratification peaks
5-compute_mld_variables.py: Estimate variables at 10 m and at the mixed layer base
6-grid_dat.py: Binned the data onto a grid for producing a climatology


