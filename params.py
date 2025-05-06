import numpy as np
import os

### UNIT SYSTEM #########
# Length - mm
# Mass - gm
# Time - sec
# Pressure - Pa or mm of H20 1Pa ~ 1 mm of H20

#### PHYSIOLOGICAL PARAMETERS (WORKING) #######
# LUNG_MASS = 1200. #gm
# PRESSURE_AMP = 1.5 * 10 # Pressure variation in cm H20 to mm H20  (or equivalent Pa)
# BREATHING_TIME = 5.0 #sec

# ######## MODEL PARAMETERS #################
# # === VORONOI PARAMETERS ===
# NUM_POINTS = 1400.
# NUM_NEIGHBORS = 6

# ### SIMULATION PARAMETERS 
# dt = 0.01
# noofCycles = 1
# omega = (2.0 * np.pi) / BREATHING_TIME

# steps = int(noofCycles*BREATHING_TIME/dt)
# mass = LUNG_MASS / NUM_POINTS

# k_spring = 0.014 #gm/sec2
# k_spring_interlobar = 0.1 * 0.1 #gm/sec2
# damping = 0.75 #g/s

# writeInterval = 10
####################################


############# EXPERIMENT ################
LUNG_MASS = 1200. #gm
PRESSURE_AMP = 1.5 * 10 # Pressure variation in cm H20 to mm H20  (or equivalent Pa)
BREATHING_TIME = 5.0 #sec

######## MODEL PARAMETERS #################
# === VORONOI PARAMETERS ===
NUM_POINTS = 1400.
NUM_NEIGHBORS = 6

### SIMULATION PARAMETERS 
dt = 0.01
noofCycles = 10
omega = (2.0 * np.pi) / BREATHING_TIME

steps = int(noofCycles*BREATHING_TIME/dt)
mass = LUNG_MASS / NUM_POINTS

k_spring = 0.008 #gm/sec2
k_spring_interlobar = 0.1 * 0.1 #gm/sec2
damping = 0.75 #g/s

writeInterval = 25
####################################


#outputDataFolder = f'/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/RESULT/OUTPUT2'
outputDataFolder = f'/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/RESULT/PARAMETRIC/OUTPUT_spring/{k_spring}'

### LOBES ACTIVE ###
LL = True
LU = True
RL = True
RM = True
RU = True