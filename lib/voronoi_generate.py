import math
import numpy as np
import scipy
from scipy.spatial import Voronoi, ConvexHull, distance
from scipy.interpolate import griddata as gd
import cv2
import matplotlib.pyplot as plt


def cw_rotate(pos, ang):

    ang = math.radians(ang) 
    new_pos = []
    for i in range(len(pos)):
        new_x = round(pos[i][0] * math.cos(ang) + pos[i][1] * math.sin(ang), 5)
        new_y = round(-pos[i][0] * math.sin(ang) + pos[i][1] * math.cos(ang), 5)
        new_pos.append([new_x, new_y])
    
    return new_pos


class TransformVoronoi_331:

    def __init__(self, borderScale=1.1):
        super(TransformVoronoi_331, self).__init__()
        self.borderScale = borderScale
        self.medianX = []

    def transform(self, X):
        # Voronoi tesselation function
        # Input: X = (x,y) pin positions (centroids)
        # Returns: Y = (x,y,A) pin positions and cell areas
        #          C cells are lists of vertices

        X = np.squeeze(X)
        _, unindx = np.unique(X, return_index=True, axis=0)
        unindx = np.sort(unindx)
        X = X[unindx]
        X = (X - np.median(X))

        # apply voronoi to data + boundary: V vertices, C cells
        B = ConvexHull(X)  ## calculate the convexhull to extract the outter circle nodes
        BX = np.transpose([X[B.vertices, 0] * self.borderScale, X[B.vertices, 1] * self.borderScale])  ## new round border nodes
        X_BX = np.vstack((X, BX))  ## combine the boundary nodes with the origin nodes

        vor = Voronoi(X_BX, qhull_options='Qbb')  ## Voronoi vertices and Voronoi cell
        BV = vor.vertices  ## Coordinates of the Voronoi vertices
        BC = vor.regions  ## Indices of the Voronoi vertices forming each Voronoi region
        BXY = vor.points  ## Input points position, equal to X_BX
        BXY_index = vor.point_region  ## Index of the Voronoi region for each input point

        # prune edges outside boundary
        Cx = np.asarray([BV[BC[BXY_index[i]], 0] for i in range(len(X))], dtype=object)
        Cy = np.asarray([BV[BC[BXY_index[i]], 1] for i in range(len(X))], dtype=object)

        # calculate area of each Voronoi region based on Voronoi vertices coordinates
        A = [TransformVoronoi_331.polyarea(Cx[indx], Cy[indx]) for indx, val in enumerate(Cx)]

        return A, Cx, Cy, X

    @staticmethod
    def polyarea(x, y):
        # computes area of polygons
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



class TransformVoronoi_127:

    def __init__(self, borderScale=1.1):
        super(TransformVoronoi_127, self).__init__()
        self.borderScale = borderScale
        self.medianX = []

    def transform(self, X):
        # Voronoi tesselation function
        # Input: X = (x,y) pin positions (centroids)
        # Returns: Y = (x,y, A) pin positions and cell areas
        #          C cells are lists of vertices

        X = np.squeeze(X)
        _, unindx = np.unique(X, return_index=True, axis=0)
        unindx = np.sort(unindx)
        X = X[unindx]
        X = (X - np.median(X))

        # Apply voronoi to data + boundary: V vertices, C cells
        B = ConvexHull(X)  ## Calculate the convexhull to extract the outter circle nodes
        BX = np.transpose([X[B.vertices, 0] * self.borderScale, X[B.vertices, 1] * self.borderScale])  ## New border circle nodes
        # Build virtual hexagonal boundary
        BX_5 = np.array(cw_rotate(BX, ang=7))*0.95
        BX_10 = np.array(cw_rotate(BX, ang=12))*0.9
        BX_20 = np.array(cw_rotate(BX, ang=22))*0.85
        BX_30 = np.array(cw_rotate(BX, ang=30))*0.85
        BX_40 = np.array(cw_rotate(BX, ang=38))*0.85
        BX_50 = np.array(cw_rotate(BX, ang=48))*0.9
        BX_55 = np.array(cw_rotate(BX, ang=53))*0.95
        BX = np.vstack((BX, BX_5, BX_10, BX_20, BX_30, BX_40, BX_50, BX_55))  ## New hexagonal border nodes
        X_BX = np.vstack((X, BX))  ## Combine the boundary nodes with the origin nodes

        vor = Voronoi(X_BX, qhull_options='Qbb')  ## Voronoi vertices and Voronoi cell
        BV = vor.vertices  ## Coordinates of the Voronoi vertices
        BC = vor.regions  ## Indices of the Voronoi vertices forming each Voronoi region
        BXY = vor.points  ## input points position, equal to X_BX
        BXY_index = vor.point_region  ## Index of the Voronoi region for each input point

        # Prune edges outside boundary
        Cx = np.asarray([BV[BC[BXY_index[i]], 0] for i in range(len(X))], dtype=object)
        Cy = np.asarray([BV[BC[BXY_index[i]], 1] for i in range(len(X))], dtype=object)

        # Calculate area of each Voronoi region based on Voronoi vertices coordinates
        A = [TransformVoronoi_127.polyarea(Cx[indx], Cy[indx]) for indx, val in enumerate(Cx)]

        return A, Cx, Cy, X

    @staticmethod
    def polyarea(x, y):
        # Computes area of polygons
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    


def cal_3d_Voronoi(Axx_canon, Cxx_canon, Cyy_canon, pool_neighbours, num_interp_points):
    cen_coord_canon = np.zeros((len(Axx_canon), 2))

    for ii in range(len(Cxx_canon)):
        cen_coord_canon[ii, 0] = np.mean(Cxx_canon[ii])
        cen_coord_canon[ii, 1] = np.mean(Cyy_canon[ii])

    vorarea_canon = np.asarray(Axx_canon)

    # Interpolate voronoi cell areas over neighbours
    dist_neighbourM_canon = scipy.spatial.distance.cdist(cen_coord_canon, cen_coord_canon, metric='euclidean')

    vorarea_canon_interp = np.zeros(np.shape(vorarea_canon))  ## Interlolate

    for ii in range(len(dist_neighbourM_canon)):
        indx_neighbour_canon = np.argsort(dist_neighbourM_canon[ii, :])[1:pool_neighbours+1]  ## Index from close neighbor
        vorarea_canon_interp[ii] = (1*vorarea_canon[ii] + np.sum(vorarea_canon[indx_neighbour_canon]))*np.divide(1, pool_neighbours+1)

    # Create uniform grid to interpolate on
    Xgrid, Ygrid = np.meshgrid(
        np.linspace(np.min(cen_coord_canon[:, 0]), np.max(cen_coord_canon[:, 0]), num_interp_points),
        np.linspace(np.min(cen_coord_canon[:, 1]), np.max(cen_coord_canon[:, 1]), num_interp_points))

    # Fit surface
    Z_canon = gd(cen_coord_canon, vorarea_canon, (Xgrid, Ygrid), 'cubic', fill_value=np.min(vorarea_canon))
    Z_canon = cv2.GaussianBlur(Z_canon, (5, 5), 0)
    Z_canon = Z_canon - np.min(Z_canon)
    Z_canon = Z_canon / np.max(Z_canon)

    return Xgrid, Ygrid, Z_canon



def plot_3d_Voronoi(Xgrid, Ygrid, Z_canon, vmin=0.1, vmax=0.6):

    figg = plt.figure(1, figsize=(8, 8))
    figg.subplots_adjust(wspace=0, hspace=0, top=0.99, bottom=0.01)

    # Select one of the colormap types below
    # cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", ['g', 'b', 'r'])
    cmap = plt.cm.get_cmap('jet')
    # cmap = plt.cm.get_cmap('coolwarm')
    # cmap = plt.cm.get_cmap('seismic')
    # cmap = plt.cm.get_cmap('bwr')
    # cmap = plt.cm.get_cmap('hot')
    # cmap = plt.cm.get_cmap('Reds')
    # cmap = plt.cm.get_cmap('inferno')
    # cmap = plt.cm.get_cmap('magma')
    # cmap = plt.cm.get_cmap('rainbow')

    ax = figg.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(Xgrid, Ygrid, Z_canon, cmap=cmap,
                    linewidth=1, antialiased=True, shade=False, vmin=vmin, vmax=vmax)

    plt.axis('on')
    plt.tight_layout()
    # ax.view_init(elev=71, azim=-110)
    ax.view_init(elev=90, azim=-90)
    plt.show()


def set_zero_VoronoiRecons_outside_ROI(Xgrid, Ygrid, Z, num_interp_pts):

    Z_flatten = Z.flatten()
    grid_cent = int(np.round(num_interp_pts/2) + 1)
    dist_points = np.sqrt(np.sum(((np.squeeze(np.array([[Xgrid.flatten()], [Ygrid.flatten()]])).T - \
                                   np.squeeze(np.array([[Xgrid[grid_cent, grid_cent]], [Ygrid[grid_cent, grid_cent]]])).T) ** 2),
                                 axis=1))
    for jj in range(len(dist_points)):
        if abs(dist_points[jj]) > 400:
            Z_flatten[jj] = 0
    Z = np.reshape(Z_flatten, (num_interp_pts, num_interp_pts))
    
    return Z