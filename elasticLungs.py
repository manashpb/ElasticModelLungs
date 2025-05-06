import numpy as np
import matplotlib.pyplot as plt
np.bool = bool  # Temporary fix for VTK compatibility with NumPy >= 1.24

import pyvista as pv
import pyvoro
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import os
import shutil
import pathlib
import trimesh
import params



############################################################
def manageDataFolder():
    folder = pathlib.Path(params.outputDataFolder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created new folder: {folder}")
    else:
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)  
            else:
                item.unlink()  
        print(f"Deleted previous folder contents")


###########################################################
def save_to_vtk(positions, velocities, step,outputFile):
    # Create a PolyData object with point coordinates
    point_cloud = pv.PolyData(positions)
    
    # Add velocity as a vector field
    point_cloud['velocity'] = velocities

    # Save to VTU format
    filename = f"{outputFile}_{step:04d}.vtk"
    point_cloud.save(filename)

def save_enclosing_surface(positions,step, outputFile, alpha = 5.0):
    point_cloud = pv.PolyData(positions)
    surface = point_cloud.delaunay_3d(alpha=0)  # Tune alpha as needed
    shell = surface.extract_geometry()

    # Save to VTU format
    filename = f"{outputFile}-EN_{step:04d}.vtp"
    shell.save(filename)



def saveInterLobar_to_vtk(G, A_points, B_points, step,outputFile):

    # === First collect unique indices ===
    unique_A_indices = set()
    unique_B_indices = set()

    for u, v in G.edges():
        unique_A_indices.add(u)  # u from A_points
        unique_B_indices.add(v)  # v from B_points

    # === Extract coordinates ===
    A_coords = np.array([A_points[i] for i in unique_A_indices])
    B_coords = np.array([B_points[j] for j in unique_B_indices])

    # Combine all points
    all_coords = np.vstack((A_coords, B_coords))

    # === Create PyVista point cloud ===
    point_cloud = pv.PolyData(all_coords)

    # Save to VTU format
    filename = f"{outputFile}_{step:04d}.vtk"
    point_cloud.save(filename)



def saveInterLobar_to_vtk2(G, A_points, B_points, indA, indB, step,outputFile):


    # === Extract coordinates ===
    A_coords = np.array([A_points[i] for i in indA])
    B_coords = np.array([B_points[j] for j in indB])

    # Combine all points
    all_coords = np.vstack((A_coords, B_coords))

    # === Create PyVista point cloud ===
    point_cloud = pv.PolyData(all_coords)

    # Save to VTU format
    filename = f"{outputFile}_{step:04d}.vtk"
    point_cloud.save(filename)



################# PREPARATION ##################################

############# NEW POINT DETECTION METHOD #############

def fast_evenly_distributed_points_in_stl(file_path, num_points, pitch=None):
    random_seed=42  ### DEFAULT SEED

    mesh = trimesh.load(file_path)
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight for volume sampling.")

    # Estimate pitch if not given
    if pitch is None:
        volume = mesh.volume
        pitch = (volume / num_points) ** (1/8)  # cubic root of volume per point

    # Create a voxelized version of the mesh
    voxels = mesh.voxelized(pitch)
    filled = voxels.points  # center of filled voxels

    if len(filled) < num_points:
        raise RuntimeError("Voxel resolution too coarse. Try reducing `pitch` or check mesh quality.")

    # Shuffle and select N points
    np.random.seed(random_seed)
    np.random.shuffle(filled)
    selected_points = filled[:num_points]
    return selected_points


############################################################

def constructPoints(MESH_FILE, N_POINTS=100, K_NEIGHBORS = 6):
    # === 1. LOAD AND PREPARE MESH ===
    mesh = pv.read(MESH_FILE)
    print(f"Mesh bounds: {mesh.bounds}")
    print(f"Mesh is{'' if mesh.is_manifold else ' not'} manifold (watertight)")
    print(f"Mesh center: {mesh.center}")

    # === 2. BULLETPROOF POINT GENERATION ===
    def generate_points_in_mesh(mesh, n_points):
        # Get mesh information
        bounds = mesh.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        center = mesh.center
        
        # Method 1: Use mesh vertices as a starting point
        print("Generating points from mesh vertices...")
        vertices = mesh.points
        inside_points = []
        
        # Ray casting method to check if point is inside
        def is_inside(mesh, point):
            # Cast rays in 6 directions to be extra sure
            directions = [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ]
            
            intersections = []
            for direction in directions:
                try:
                    points, indices = mesh.ray_trace(point, direction)
                    intersections.append(len(indices) % 2 == 1)
                except:
                    # If ray tracing fails, consider the point outside
                    intersections.append(False)
                    
            # Point is inside if majority of rays indicate it's inside
            return sum(intersections) > len(directions) // 2
        
        # Try to generate points from random positions within the mesh bounding box
        print("Generating random points within bounding box...")
        attempts = 0
        max_attempts = 10000
        
        while len(inside_points) < n_points and attempts < max_attempts:
            # Generate random point in bounding box
            point = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
                np.random.uniform(zmin, zmax)
            ])
            
            # Check if point is inside mesh
            try:
                if is_inside(mesh, point):
                    inside_points.append(point)
                    if len(inside_points) % 10 == 0:
                        print(f"Found {len(inside_points)} points...")
            except Exception as e:
                print(f"Error checking point: {str(e)}")
                
            attempts += 1
        
        if len(inside_points) >= n_points:
            print(f"Successfully generated {n_points} points inside the mesh")
            return np.array(inside_points[:n_points])
        
        # Fallback: Generate points from mesh vertices with displacement
        print(f"Only found {len(inside_points)} points. Using vertex-based approach...")
        
        # Use existing inside points
        points = inside_points.copy()
        
        # Generate remaining points by interpolating between random pairs of vertices
        while len(points) < n_points:
            if len(vertices) >= 2:
                # Pick two random vertices
                idx1, idx2 = np.random.choice(len(vertices), 2, replace=False)
                # Interpolate between them with random factor
                t = np.random.uniform(0.3, 0.7)
                point = vertices[idx1] * t + vertices[idx2] * (1-t)
                
                # Move slightly toward center
                point = point + 0.1 * (center - point)
                
                # Add the point without checking (fallback)
                points.append(point)
            else:
                # If not enough vertices, use center with random displacement
                displacement = np.random.uniform(-0.2, 0.2, 3)
                point = center + displacement
                points.append(point)
        
        return np.array(points)

    print('Generating points...')
    #points = generate_points_in_mesh(mesh, N_POINTS)
    points = fast_evenly_distributed_points_in_stl(MESH_FILE, N_POINTS)
    print(f'Successfully generated {len(points)} points')

    # === 3. BUILD GRAPH ===
    print('Building graph...')
    if len(points) < 2:
        raise ValueError("Not enough points generated to build graph")

    neighbors = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(points)-1))
    neighbors.fit(points)
    distances, indices = neighbors.kneighbors(points)

    G = nx.Graph()
    for i, neighbors_i in enumerate(indices):
        for j in neighbors_i:
            if i != j:
                G.add_edge(i, j, weight=np.linalg.norm(points[i] - points[j]))

    # === 4. VORONOI CALCULATION ===
    print('Computing Voronoi cells...')
    padding = 5.0
    domain = [
        [mesh.bounds[0]-padding, mesh.bounds[1]+padding],
        [mesh.bounds[2]-padding, mesh.bounds[3]+padding],
        [mesh.bounds[4]-padding, mesh.bounds[5]+padding]
    ]

    cells = pyvoro.compute_voronoi(points.tolist(), domain, 1.0)

    # === 5. VOLUME CALCULATION ===
    print('Calculating volumes...')
    volumes = []
    for cell in cells:
        volumes.append(cell['volume'])

    return mesh, G, points, indices, volumes




def export_graph_to_vtk(G, filename="graph.vtp"):
    """
    Exports a NetworkX graph with 3D positions to a VTK PolyData file.
    Nodes must have a 'pos' attribute.
    """

    # Extract node positions
    nodes = list(G.nodes)
    if len(nodes) == 0:
        print('No points within cutoff!')
        return None
    points = np.array([G.nodes[n]['pos'] for n in nodes])
    
    # Create lines from edges (as VTK expects cell format: [N, pt0, pt1])
    edges = []
    for u, v in G.edges():
        idx_u = nodes.index(u)
        idx_v = nodes.index(v)
        edges.append([2, idx_u, idx_v])  # 2 = number of points in the line

    # Create PolyData object
    poly = pv.PolyData()
    poly.points = points
    poly.lines = np.array(edges)

    # Optional: add point labels as scalar data
    poly['labels'] = np.arange(len(points))

    # Save to file
    poly.save(filename)
    print(f"Graph saved to {filename}")


def export_graph_nodes_to_xyz(G, filename):
    with open(filename, 'w') as f:
        total = len(G.nodes)
        f.write(str(total)+'\n')
        for node in G.nodes():
            pos = G.nodes[node]['pos']
            f.write(f"1 {pos[0]} {pos[1]} {pos[2]}\n")
    print(f"Graph nodes exported to: {filename}")



def stlVolume(stl_path):
    mesh = pv.read(stl_path)
    if mesh.is_manifold:
        volume = mesh.volume
    else:
        volume = 0.0
    return volume

############### TIME SIMULATION ################################

def calcPg(volumes):
    #Pg = (n * R * T) / np.mean(np.array(volumes))
    Pg = params.PRESSURE_AMP
    print (f'Pg : {Pg}')
    return Pg

def calcSpringForce(positions, G, alpha, P_gas_curr, forces):
    for i, j in G.edges:
        xi, xj = positions[i], positions[j]
        delta = xj - xi
        dist = np.linalg.norm(delta)
        if dist == 0:
            continue
        direction = delta / dist
        r_eq = alpha[(i, j)]  * P_gas_curr
        F = params.k_spring * (dist - r_eq) * direction
        forces[i] += F
        forces[j] -= F
    return forces

def calcDamperForces(velocities, forces):
    forces -= params.damping * velocities
    return forces

def calcPressureWaveForm(Pg, time):
    #Pg_curr = ((Pg * (1. + np.sin(omega*time)) + Pp)) - (Pg + Pp)   
    Pg_curr = Pg * (0.0 + np.sin(params.omega*time - (np.pi/2.0)) )
    return Pg_curr

def calcAlpha(G, positions, P = 1.0):
    alpha = {}
    for i, j in G.edges:
        alpha[(i, j)] = (((np.linalg.norm(positions[i] - positions[j]))) / P)* 1.0
    return alpha


def anchorPoints(forces, velocities, index_point):
    forces[index_point] = 0
    velocities[index_point] = 0
    return forces, velocities


def updateVelHalf(velocities, forces, dt):
    # Integration: Verlet half step
    velocities += 0.5 * (forces / params.mass) * dt

    return velocities

def updateVelPos(positions, velocities, forces, dt):
    # Integration
    velocities += (forces / params.mass) * dt
    positions += velocities * dt

    return positions, velocities

def updateVelPosHalf(positions, velocities, forces, dt):
    # Integration: Verlet half step for position and velocity
    positions += velocities * dt
    velocities += 0.5 * (forces / params.mass) * dt

    return positions, velocities


################ INTERLOBAR FUNCTIONS ###############
def create_filtered_graph(A_points, B_points, cutoff):
    # Build KDTree for B to efficiently find neighbors of A
    tree_B = KDTree(B_points)

    # Query all neighbors within cutoff
    pairs = []
    for i, point_A in enumerate(A_points):
        indices = tree_B.query_radius([point_A], r=cutoff)[0]
        for j in indices:
            pairs.append((i,j))

    # Create graph with only those nodes involved in the pairs
    G = nx.Graph()

    indexA = []
    indexB = []
    for i, j in pairs:
        indexA.append(i)
        indexB.append(j)
        A_index = i
        B_index = j
        G.add_node(A_index, pos=A_points[i])
        G.add_node(B_index, pos=B_points[j])
        dist = np.linalg.norm(A_points[i] - B_points[j])
        G.add_edge(A_index, B_index, weight=dist)

    indexA = list(set(indexA))
    indexB = list(set(indexB))

    return G, indexA, indexB, pairs


def calcInterLobarAlpha(G,A_points, B_points, pairs, P = 1.0):
    alpha = {}
    for i, j in pairs:
        pos_u = A_points[i]
        pos_v = B_points[j]

        vec = pos_v - pos_u
        current_length = np.linalg.norm(vec)
        alpha[(i, j)] = current_length / P
    return alpha

def calcInterLobarForce(G, alpha_interlobar, A_points, B_points, pairs, P_gas_curr, forcesA, forcesB):
    for i, j in pairs:
        xi, xj = A_points[i], B_points[j]
        delta = xj - xi
        dist = np.linalg.norm(delta)
        if dist == 0:
            continue
        direction = delta / dist
        #r_eq = alpha_interlobar[(i, j)]  * P_gas_curr
        r_eq = alpha_interlobar[(i, j)] * 1.0
        F = params.k_spring_interlobar * (dist - r_eq) * direction
        forcesA[i] += F
        forcesB[j] -= F
    return forcesA, forcesB

##### METRIC CALCULATION ######
def compute_metrics(points):
    # Convex Hull Volume
    if len(points) >= 4:  # ConvexHull needs at least 4 non-coplanar points
        hull = ConvexHull(points)
        volume = hull.volume
    else:
        volume = 0.0

    # RMS distance from centroid
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    rms = np.sqrt(np.mean(distances ** 2))

    return volume, rms

def write_metric(volArr, rmsArr, output):
    timeArr = np.arange(0,params.steps, params.dt)
    f = open(output,'w')
    for i in range(len(volArr)):
        f.write(f'{timeArr[i]},{volArr[i]},{rmsArr[i]}'+'\n')
    f.close()


################### START MAIN PART ################################################

def simulate():
    manageDataFolder()

    llVol = 0.0
    luVol = 0.0
    rlVol = 0.0
    rmVol = 0.0
    ruVol = 0.0

    if params.LL:
        INPUT_MESH_FILE_LL = '/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/STLs/LL.stl'  
        llVol = stlVolume(INPUT_MESH_FILE_LL)
    if params.LU:
        INPUT_MESH_FILE_LU = '/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/STLs/LU.stl'
        luVol = stlVolume(INPUT_MESH_FILE_LU)
    if params.RL:
        INPUT_MESH_FILE_RL = '/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/STLs/RL.stl' 
        rlVol = stlVolume(INPUT_MESH_FILE_RL)
    if params.RM:
        INPUT_MESH_FILE_RM = '/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/STLs/RM.stl'  
        rmVol = stlVolume(INPUT_MESH_FILE_RM)
    if params.RU:
        INPUT_MESH_FILE_RU = '/home/manash/DATA/SIMONE/COLLAB/2024/LUNG_NEW_MODEL_FULL/ELASTIC MODEL/STLs/RU.stl'  
        ruVol = stlVolume(INPUT_MESH_FILE_RU)

    totalVolume = llVol + luVol + rlVol + rmVol + ruVol
    percentll = llVol / totalVolume
    percentlu = luVol / totalVolume
    percentrl = rlVol / totalVolume
    percentrm = rmVol / totalVolume
    percentru = ruVol / totalVolume

    print (f'STL volumes : {llVol}, {luVol}, {rlVol}, {rmVol}, {ruVol}')


    ## LL ##
    if params.LL:
        print ('Computing for Lobe LL')
        meshLL, GLL, pointsLL, indicesLL, volumesLL = constructPoints(INPUT_MESH_FILE_LL, int(params.NUM_POINTS*percentll), params.NUM_NEIGHBORS)
        outputFileNameLL =  os.path.join(params.outputDataFolder, 'LL')

    ## LU ##
    if params.LU:
        print ('Computing for Lobe LU')
        meshLU, GLU, pointsLU, indicesLU, volumesLU = constructPoints(INPUT_MESH_FILE_LU, int(params.NUM_POINTS*percentlu), params.NUM_NEIGHBORS)
        outputFileNameLU = os.path.join(params.outputDataFolder, 'LU')

    ## RL ##
    if params.RL:  
        print ('Computing for Lobe RL')
        meshRL, GRL, pointsRL, indicesRL, volumesRL = constructPoints(INPUT_MESH_FILE_RL, int(params.NUM_POINTS*percentrl), params.NUM_NEIGHBORS)
        outputFileNameRL = os.path.join(params.outputDataFolder, 'RL')

    ## RM ##
    if params.RM: 
        print ('Computing for Lobe RM')
        meshRM, GRM, pointsRM, indicesRM, volumesRM = constructPoints(INPUT_MESH_FILE_RM, int(params.NUM_POINTS*percentrm), params.NUM_NEIGHBORS)
        outputFileNameRM = os.path.join(params.outputDataFolder, 'RM')

    ## RU ##
    if params.RU: 
        print ('Computing for Lobe RU')
        meshRU, GRU, pointsRU, indicesRU, volumesRU = constructPoints(INPUT_MESH_FILE_RU, int(params.NUM_POINTS*percentru), params.NUM_NEIGHBORS)
        outputFileNameRU = os.path.join(params.outputDataFolder, 'RU')




    ######### INTERLOBAR GRAPH INITIALIZE ############
    ## Left Upper-Lower pair
    if params.LU and params.LL:
        G_interLobar_LeftLU, indLL_LeftLU, indLU_LeftLU, pairs_LeftLU = create_filtered_graph(pointsLL, pointsLU, cutoff=15)
        alpha_interlobar_LeftLU = calcInterLobarAlpha(G_interLobar_LeftLU,pointsLL, pointsLU, pairs_LeftLU)
        outputFileNameInterLobar_LeftLU = os.path.join(params.outputDataFolder, 'Inter_LeftLU')

    ## Right Upper-Middle pair
    if params.RU and params.RM:
        G_interLobar_RightUM, indRU_RightUM, indRM_RightUM, pairs_RightUM = create_filtered_graph(pointsRU, pointsRM, cutoff=15)
        alpha_interlobar_RightUM = calcInterLobarAlpha(G_interLobar_RightUM,pointsRU, pointsRM, pairs_RightUM)
        outputFileNameInterLobar_RightUM = os.path.join(params.outputDataFolder, 'Inter_RightUM') 

    ## Right Middle-Lower pair
    if params.RM and params.RL:
        G_interLobar_RightML, indRM_RightML, indRL_RightML, pairs_RightML = create_filtered_graph(pointsRM, pointsRL, cutoff=15)
        alpha_interlobar_RightML = calcInterLobarAlpha(G_interLobar_RightML,pointsRM, pointsRL, pairs_RightML)
        outputFileNameInterLobar_RightML = os.path.join(params.outputDataFolder, 'Inter_RightML')


    ########  TIME LOOP ###########

    ######## Lobe LL ########
    if params.LL:
        PgLL = calcPg(volumesLL)
        positionsLL = pointsLL.copy()
        velocitiesLL = np.zeros_like(positionsLL)
        inlet_indexLL = np.argmax(positionsLL[:, 2])  ### Highest x point
        alphaLL = calcAlpha(GLL, positionsLL,P=1.0)

        volumesLL = []
        rms_valuesLL = []


    ######## Lobe LU ########
    if params.LU:
        PgLU = calcPg(volumesLU)
        positionsLU = pointsLU.copy()
        velocitiesLU = np.zeros_like(positionsLU)
        inlet_indexLU = np.argmax(positionsLU[:, 2])  ### Highest x point
        alphaLU = calcAlpha(GLU, positionsLU,P=1.0)

        volumesLU = []
        rms_valuesLU = []


    ######## Lobe RL ########
    if params.RL:
        PgRL = calcPg(volumesRL)
        positionsRL = pointsRL.copy()
        velocitiesRL = np.zeros_like(positionsRL)
        inlet_indexRL = np.argmax(positionsRL[:, 2])  ### Highest x point
        alphaRL = calcAlpha(GRL, positionsRL)

        volumesRL = []
        rms_valuesRL = []

    ######## Lobe RM ########
    if params.RM:
        PgRM = calcPg(volumesRM)
        positionsRM = pointsRM.copy()
        velocitiesRM = np.zeros_like(positionsRM)
        inlet_indexRM = np.argmax(positionsRM[:, 2])  ### Highest x point
        alphaRM = calcAlpha(GRM, positionsRM)

        volumesRM = []
        rms_valuesRM = []


    ######## Lobe RU ########
    if params.RU:
        PgRU = calcPg(volumesRU)
        positionsRU = pointsRU.copy()
        velocitiesRU = np.zeros_like(positionsRU)
        inlet_indexRU = np.argmax(positionsRU[:, 2])  ### Highest x point
        alphaRU = calcAlpha(GRU, positionsRU)

        volumesRU = []
        rms_valuesRU = []



    time = 0.0
    for step in range(params.steps):
        print (f'Step : {step} , Time : {time}')

        ####### SAVE OUTPUT ############
        if params.LL and step%params.writeInterval == 0: save_enclosing_surface(positionsLL,step, outputFileNameLL)
        if params.LU and step%params.writeInterval == 0: save_enclosing_surface(positionsLU,step, outputFileNameLU)
        if params.RL and step%params.writeInterval == 0: save_enclosing_surface(positionsRL,step, outputFileNameRL)
        if params.RM and step%params.writeInterval == 0: save_enclosing_surface(positionsRM,step, outputFileNameRM)
        if params.RU and step%params.writeInterval == 0: save_enclosing_surface(positionsRU,step, outputFileNameRU)


        if params.LL and step%params.writeInterval == 0: save_to_vtk(positionsLL,velocitiesLL,step, outputFileNameLL)
        if params.LU and step%params.writeInterval == 0: save_to_vtk(positionsLU,velocitiesLU,step, outputFileNameLU)
        if params.RL and step%params.writeInterval == 0: save_to_vtk(positionsRL,velocitiesRL,step, outputFileNameRL)
        if params.RM and step%params.writeInterval == 0: save_to_vtk(positionsRM,velocitiesRM,step, outputFileNameRM)
        if params.RU and step%params.writeInterval == 0: save_to_vtk(positionsRU,velocitiesRU,step, outputFileNameRU)


        #if LL and LU: saveInterLobar_to_vtk2(G_interLobar_LeftLU, positionsLL, positionsLU, indLL_LeftLU, indLU_LeftLU, step,outputFileNameInterLobar_LeftLU)
        #if RU and RM: saveInterLobar_to_vtk2(G_interLobar_RightUM, positionsRU, positionsRM, indRU_RightUM, indRM_RightUM, step,outputFileNameInterLobar_RightUM)
        #if RM and RL: saveInterLobar_to_vtk2(G_interLobar_RightML, positionsRM, positionsRL, indRM_RightML, indRL_RightML, step,outputFileNameInterLobar_RightML)

        ######## COMPUTE METRICS #########
        if params.LL and step%params.writeInterval == 0:
            volume, rms = compute_metrics(positionsLL)
            volumesLL.append(volume)
            rms_valuesLL.append(rms)

        if params.LU and step%params.writeInterval == 0:
            volume, rms = compute_metrics(positionsLU)
            volumesLU.append(volume)
            rms_valuesLU.append(rms)

        if params.RL and step%params.writeInterval == 0:
            volume, rms = compute_metrics(positionsRL)
            volumesRL.append(volume)
            rms_valuesRL.append(rms)

        if params.RM and step%params.writeInterval == 0:
            volume, rms = compute_metrics(positionsRM)
            volumesRM.append(volume)
            rms_valuesRM.append(rms)
        
        if params.RU and step%params.writeInterval == 0:
            volume, rms = compute_metrics(positionsRU)
            volumesRU.append(volume)
            rms_valuesRU.append(rms)


        ### INITIALIZE FORCES ###
        if params.LL: forcesLL = np.zeros_like(positionsLL)
        if params.LU: forcesLU = np.zeros_like(positionsLU)
        if params.RL: forcesRL = np.zeros_like(positionsRL)
        if params.RM: forcesRM = np.zeros_like(positionsRM)
        if params.RL: forcesRU = np.zeros_like(positionsRU)


        time = time + params.dt

        if params.LL: P_gas_currLL = calcPressureWaveForm(PgLL, time)  
        if params.LU: P_gas_currLU = calcPressureWaveForm(PgLU, time)  
        if params.RL: P_gas_currRL = calcPressureWaveForm(PgRL, time)  
        if params.RM: P_gas_currRM = calcPressureWaveForm(PgRM, time) 
        if params.RU: P_gas_currRU = calcPressureWaveForm(PgRU, time)  

        ### Half step Update #######
        if params.LL: velocitiesLL = updateVelHalf(velocitiesLL, forcesLL, params.dt)
        if params.LU: velocitiesLU = updateVelHalf(velocitiesLU, forcesLU, params.dt)
        if params.RL: velocitiesRL = updateVelHalf(velocitiesRL, forcesRL, params.dt)
        if params.RM: velocitiesRM = updateVelHalf(velocitiesRM, forcesRM, params.dt)
        if params.RU: velocitiesRU = updateVelHalf(velocitiesRU, forcesRU, params.dt)

        ########## SPRING FORCE #########
        if params.LL: forcesLL = calcSpringForce(positionsLL, GLL, alphaLL, P_gas_currLL, forcesLL)
        if params.LU: forcesLU = calcSpringForce(positionsLU, GLU, alphaLU, P_gas_currLU, forcesLU)
        if params.RL: forcesRL = calcSpringForce(positionsRL, GRL, alphaRL, P_gas_currRL, forcesRL)
        if params.RM: forcesRM = calcSpringForce(positionsRM, GRM, alphaRM, P_gas_currRM, forcesRM)
        if params.RU: forcesRU = calcSpringForce(positionsRU, GRU, alphaRU, P_gas_currRU, forcesRU)

        # Damping
        if params.LL: forcesLL = calcDamperForces(velocitiesLL, forcesLL)
        if params.LU: forcesLU = calcDamperForces(velocitiesLU, forcesLU)
        if params.RL: forcesRL = calcDamperForces(velocitiesRL, forcesRL)
        if params.RM: forcesRM = calcDamperForces(velocitiesRM, forcesRM)
        if params.RU: forcesRU = calcDamperForces(velocitiesRU, forcesRU)

        ## INTERLOBAR FORCE ###
        if params.LL and params.LU: forcesLL, forcesLU = calcInterLobarForce(G_interLobar_LeftLU, alpha_interlobar_LeftLU, positionsLL, positionsLU, pairs_LeftLU, PgLU, forcesLL, forcesLU)  ## P gas taken for the LU lobe  ##Left UL
        if params.RM and params.RU: forcesRU, forcesRM = calcInterLobarForce(G_interLobar_RightUM, alpha_interlobar_RightUM, positionsRU, positionsRM, pairs_RightUM, PgRU, forcesRU, forcesRM)  ## P gas taken for the RU lobe  ##Right UM
        if params.RM and params.RL: forcesRM, forcesRL = calcInterLobarForce(G_interLobar_RightML, alpha_interlobar_RightML, positionsRM, positionsRL, pairs_RightML, PgRL, forcesRM, forcesRL)  ## P gas taken for the RL lobe  ##Right ML

        # Anchor inlet point
        if params.LL: forcesLL, velocitiesLL = anchorPoints(forcesLL, velocitiesLL, inlet_indexLL)
        if params.LU: forcesLU, velocitiesLU = anchorPoints(forcesLU, velocitiesLU, inlet_indexLU)
        if params.RL: forcesRL, velocitiesRL = anchorPoints(forcesRL, velocitiesRL, inlet_indexRL)
        if params.RM: forcesRM, velocitiesRM = anchorPoints(forcesRM, velocitiesRM, inlet_indexRM)
        if params.RU: forcesRU, velocitiesRU = anchorPoints(forcesRU, velocitiesRU, inlet_indexRU)

        ### Update #######
        # if LL: positionsLL, velocitiesLL = updateVelPos(positionsLL, velocitiesLL, forcesLL, dt)
        # if LU: positionsLU, velocitiesLU = updateVelPos(positionsLU, velocitiesLU, forcesLU, dt)
        # if RL: positionsRL, velocitiesRL = updateVelPos(positionsRL, velocitiesRL, forcesRL, dt)
        # if RM: positionsRM, velocitiesRM = updateVelPos(positionsRM, velocitiesRM, forcesRM, dt)
        # if RU: positionsRU, velocitiesRU = updateVelPos(positionsRU, velocitiesRU, forcesRU, dt)

        if params.LL: positionsLL, velocitiesLL = updateVelPosHalf(positionsLL, velocitiesLL, forcesLL, params.dt)
        if params.LU: positionsLU, velocitiesLU = updateVelPosHalf(positionsLU, velocitiesLU, forcesLU, params.dt)
        if params.RL: positionsRL, velocitiesRL = updateVelPosHalf(positionsRL, velocitiesRL, forcesRL, params.dt)
        if params.RM: positionsRM, velocitiesRM = updateVelPosHalf(positionsRM, velocitiesRM, forcesRM, params.dt)
        if params.RU: positionsRU, velocitiesRU = updateVelPosHalf(positionsRU, velocitiesRU, forcesRU, params.dt)

    print (f'Time loop complete - steps : {step}')


    #### WRITING THE METRICS ###
    if params.LL: write_metric(volumesLL, rms_valuesLL, os.path.join(params.outputDataFolder,'metricLL.csv'))
    if params.LU: write_metric(volumesLU, rms_valuesLU, os.path.join(params.outputDataFolder,'metricLU.csv'))
    if params.RL: write_metric(volumesRL, rms_valuesRL, os.path.join(params.outputDataFolder,'metricRL.csv'))
    if params.RM: write_metric(volumesRM, rms_valuesRM, os.path.join(params.outputDataFolder,'metricRM.csv'))
    if params.RU: write_metric(volumesRU, rms_valuesRU, os.path.join(params.outputDataFolder,'metricRU.csv'))



if __name__=="__main__":
    simulate()