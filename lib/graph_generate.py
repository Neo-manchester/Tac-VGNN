import numpy as np
import math
import scipy
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, ConvexHull, distance
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcol
import matplotlib.cm as cm



def fast_knn(nod_pos, k=6):

        node_edges = []

        dist_neighbourM_canon = scipy.spatial.distance.cdist(nod_pos, nod_pos, metric='euclidean')

        for ii in range(len(dist_neighbourM_canon)):

            indx_neighbour_canon = np.argsort(dist_neighbourM_canon[ii, :])[1:k+1]  ## index from close neighbor, except itself

            for e in range(k):
            
                node_edges.append([ii, indx_neighbour_canon[e]])

        return node_edges



def cw_rotate(pos, ang):

    ang = math.radians(ang) 
    new_pos = []
    for i in range(len(pos)):
        new_x = round(pos[i][0] * math.cos(ang) + pos[i][1] * math.sin(ang), 5)
        new_y = round(-pos[i][0] * math.sin(ang) + pos[i][1] * math.cos(ang), 5)
        new_pos.append([new_x, new_y])
    
    return new_pos



def remove_repeat_edge(raw_nodes_edges):

    c = np.array(raw_nodes_edges)
    x = c[:,0] + c[:,1]*1j
    idx=np.unique(x,return_index=True)[1]
    clean_nodes_edges = c[idx]

    return clean_nodes_edges



def Delaunay_graph_generate(nodes_pos):

    tri = Delaunay(nodes_pos)
    node_edges = []
    for i in range(len(tri.simplices)):
        for triangle in itertools.product(tri.simplices[i], tri.simplices[i]):
            if triangle[0] != triangle[1]:
                node_edges.append(triangle)
    node_edges = remove_repeat_edge(node_edges)

    return node_edges



def Convex_delaunay_graph_generate(nodes_pos):

    tri = Delaunay(nodes_pos)  ## Delaunay or kNN
    node_edges = []
    for i in range(len(tri.simplices)):
        for triangle in itertools.product(tri.simplices[i], tri.simplices[i]):
            if triangle[0] != triangle[1] and triangle[0]-127 < 0 and triangle[1]-127 < 0:
                node_edges.append(triangle)

    return node_edges


def voronoi_graph_built(nodes_pos):

    nodes_pos_norm = np.squeeze(nodes_pos)
    _, unindx = np.unique(nodes_pos_norm, return_index=True, axis=0)
    unindx = np.sort(unindx)
    nodes_pos_norm = nodes_pos_norm[unindx]
    nodes_pos_norm = (nodes_pos_norm - np.median(nodes_pos_norm))

    clean_nodes_edges = Delaunay_graph_generate(nodes_pos_norm)

    return clean_nodes_edges, nodes_pos_norm



def hexagon_voronoi_graph_built(nodes_pos):

    nodes_pos_norm = np.squeeze(nodes_pos)
    _, unindx = np.unique(nodes_pos_norm, return_index=True, axis=0)
    unindx = np.sort(unindx)
    nodes_pos_norm = nodes_pos_norm[unindx]
    nodes_pos_norm = (nodes_pos_norm - np.median(nodes_pos_norm))

    B = ConvexHull(nodes_pos_norm)
    # tmp = B.vertices
    outmost_nodes = np.transpose([nodes_pos_norm[B.vertices, 0], nodes_pos_norm[B.vertices, 1]])

    rotated_nodes = cw_rotate(outmost_nodes, ang=30)  ## genarate virtual nodes

    new_nodes = np.vstack((nodes_pos_norm, rotated_nodes))

    clean_nodes_edges = Convex_delaunay_graph_generate(new_nodes)

    return clean_nodes_edges, nodes_pos_norm



def graph_show(nodes_edges, coordinate, figsize_x = 10, figsize_y = 10, markersize=15.):
    
    ## draw nodes
    x_cs = []
    y_cs = []
    for i in range(len(coordinate)):
        x_c = coordinate[i][0]
        y_c = coordinate[i][1]
        x_cs.append(x_c)
        y_cs.append(y_c)

    ## draw edges
    src_des_x = []
    src_des_y = []
    for i in range(len(nodes_edges)):
        src = nodes_edges[i][0]
        des = nodes_edges[i][1]
        src_x = coordinate[src][0]
        src_y = coordinate[src][1]
        des_x = coordinate[des][0]
        des_y = coordinate[des][1]
        src_des_x.append([src_x,des_x])
        src_des_y.append([src_y,des_y])
    
    plt.figure(figsize = (figsize_x, figsize_y))  
    plt.plot(x_cs,y_cs, 'o', markersize = markersize, color='b',zorder=1)
    for i in range(len(nodes_edges)):
        plt.plot(src_des_x[i], src_des_y[i], color='c',zorder=0)
    ax = plt.gca()
    ax.set_aspect(1)
    
    plt.show()



def different_graph_show(nodes_edges, coordinate_1, coordinate_2, figsize_x = 10, figsize_y = 10, markersize=15.):
  
    ## draw nodes
    x_cs_1 = []
    y_cs_1 = []
    for i in range(len(coordinate_1)):
        x_c_1 = coordinate_1[i][0]
        y_c_1 = coordinate_1[i][1]
        x_cs_1.append(x_c_1)
        y_cs_1.append(y_c_1)
    
    x_cs_2 = []
    y_cs_2 = []
    for i in range(len(coordinate_2)):
        x_c_2 = coordinate_2[i][0]
        y_c_2 = coordinate_2[i][1]
        x_cs_2.append(x_c_2)
        y_cs_2.append(y_c_2)

    ## draw edges
    src_des_x = []
    src_des_y = []
    for i in range(len(nodes_edges)):
        src = nodes_edges[i][0]
        des = nodes_edges[i][1]
        src_x = coordinate_1[src][0]
        src_y = coordinate_1[src][1]
        des_x = coordinate_1[des][0]
        des_y = coordinate_1[des][1]
        src_des_x.append([src_x,des_x])
        src_des_y.append([src_y,des_y])
    

    plt.figure(figsize = (figsize_x, figsize_y))  
    plt.plot(x_cs_1,y_cs_1, 'o', markersize = markersize, color='b',zorder=1)
    plt.plot(x_cs_2,y_cs_2, 'o', markersize = markersize, color='r',zorder=1)

    for i in range(len(nodes_edges)):
        plt.plot(src_des_x[i], src_des_y[i], color='r',zorder=0)
    ax = plt.gca()
    ax.set_aspect(1)
    
    plt.show()



def edge_show(nodes_edges, coordinate, figsize_x = 10, figsize_y = 10, markersize=15.):
  
    ## draw edges
    src_des_x = []
    src_des_y = []
    for i in range(len(nodes_edges)):
        src = nodes_edges[i][0]
        des = nodes_edges[i][1]
        src_x = coordinate[src][0]
        src_y = coordinate[src][1]
        des_x = coordinate[des][0]
        des_y = coordinate[des][1]
        src_des_x.append([src_x,des_x])
        src_des_y.append([src_y,des_y])
    

    plt.figure(figsize = (figsize_x, figsize_y))  
    for i in range(len(nodes_edges)):
        plt.plot(src_des_x[i], src_des_y[i], color='r',zorder=0)

    ax = plt.gca()
    ax.set_aspect(1)
    
    plt.show()



def node_show(coordinate, figsize_x = 10, figsize_y = 10, markersize=15.):
  
    ## draw nodes
    x_cs = []
    y_cs = []
    for i in range(len(coordinate)):
        x_c = coordinate[i][0]
        y_c = coordinate[i][1]
        x_cs.append(x_c)
        y_cs.append(y_c)
    

    plt.figure(figsize = (figsize_x, figsize_y))  
    plt.plot(x_cs,y_cs, 'o', markersize = markersize, color='b',zorder=1)
    ax = plt.gca()
    ax.set_aspect(1)
    
    plt.show()



def different_node_show(coordinate_1, coordinate_2, figsize_x = 10, figsize_y = 10, markersize=15.):
  
    ## draw nodes
    x_cs_1 = []
    y_cs_1 = []
    for i in range(len(coordinate_1)):
        x_c_1 = coordinate_1[i][0]
        y_c_1 = coordinate_1[i][1]
        x_cs_1.append(x_c_1)
        y_cs_1.append(y_c_1)

    x_cs_2 = []
    y_cs_2 = []
    for i in range(len(coordinate_2)):
        x_c_2 = coordinate_2[i][0]
        y_c_2 = coordinate_2[i][1]
        x_cs_2.append(x_c_2)
        y_cs_2.append(y_c_2)
    

    plt.figure(figsize = (figsize_x, figsize_y))  
    plt.plot(x_cs_1, y_cs_1, 'o', markersize = markersize, color='b',zorder=1)
    plt.plot(x_cs_2, y_cs_2, 'o', markersize = markersize, color='r',zorder=1)
    ax = plt.gca()
    ax.set_aspect(1)
    
    plt.show()


class Plot_Voronoi_Graph:
    def __init__(self, cell_areas, cellvert_xcoord, cellvert_ycoord, cell_center, nodes_edges):
        super(Plot_Voronoi_Graph, self).__init__()
        self.cell_areas = cell_areas
        self.vert_xcoord = cellvert_xcoord
        self.vert_ycoord = cellvert_ycoord
        self.cell_center = cell_center
        self.nodes_edges = nodes_edges

    # @staticmethod
    def plotVoronoiGraph(self, cell_scale_fact=1):
        
        patches = []
        for cell_no in range(len(self.cell_areas)):
            poly_vertices_x = np.multiply(self.vert_xcoord[cell_no], cell_scale_fact)
            poly_vertices_y = np.multiply(self.vert_ycoord[cell_no], cell_scale_fact)
            poly_vertices = np.transpose(np.array([poly_vertices_x, poly_vertices_y]))
            patches.append(Polygon(poly_vertices, closed=True))

        p = PatchCollection(patches, alpha=0.55)
        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ['r', 'b', "w"])
        cnorm = mcol.Normalize(vmin=0, vmax=1)
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        color_scale_vec = [1]
        colors = cpick.to_rgba(color_scale_vec)
        p.set_color(colors)
        p.set_edgecolor([0, 0, 0]) ## rgb value
        p.set_linewidth(2)
        figg, axx = plt.subplots(figsize=(8, 8))
        axx.add_collection(p)
        axx.autoscale()

        ## draw nodes
        x_cs = []
        y_cs = []
        for i in range(len(self.cell_center)):
            x_c = self.cell_center[i][0]
            y_c = self.cell_center[i][1]
            x_cs.append(x_c)
            y_cs.append(y_c)

        plt.plot(x_cs,y_cs, 'o', markersize = 5, color='b',zorder=1)

        ## draw edges
        src_des_x = []
        src_des_y = []
        for i in range(len(self.nodes_edges)):
            src = self.nodes_edges[i][0]
            des = self.nodes_edges[i][1]
            src_x = self.cell_center[src][0]
            src_y = self.cell_center[src][1]
            des_x = self.cell_center[des][0]
            des_y = self.cell_center[des][1]
            src_des_x.append([src_x,des_x])
            src_des_y.append([src_y,des_y])

        for i in range(len(self.nodes_edges)):
            plt.plot(src_des_x[i], src_des_y[i], color='r',zorder=0)
        
        plt.axis('on')
        plt.axis('equal')
        plt.show()
        return p


    # @staticmethod
    def plotVoronoi(self, cell_scale_fact=1):
        
        patches = []
        for cell_no in range(len(self.cell_areas)):
            poly_vertices_x = np.multiply(self.vert_xcoord[cell_no], cell_scale_fact)
            poly_vertices_y = np.multiply(self.vert_ycoord[cell_no], cell_scale_fact)
            poly_vertices = np.transpose(np.array([poly_vertices_x, poly_vertices_y]))
            patches.append(Polygon(poly_vertices, closed=True))

        p = PatchCollection(patches, alpha=0.55)
        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ['r', 'b', "w"])
        cnorm = mcol.Normalize(vmin=0, vmax=1)
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        color_scale_vec = [1]
        colors = cpick.to_rgba(color_scale_vec)
        p.set_color(colors)
        p.set_edgecolor([0, 0, 0]) ## rgb value
        p.set_linewidth(2)
        figg, axx = plt.subplots(figsize=(8, 8))
        axx.add_collection(p)
        axx.autoscale()

        ## draw nodes
        x_cs = []
        y_cs = []
        for i in range(len(self.cell_center)):
            x_c = self.cell_center[i][0]
            y_c = self.cell_center[i][1]
            x_cs.append(x_c)
            y_cs.append(y_c)

        plt.plot(x_cs,y_cs, 'o', markersize = 5, color='b',zorder=1)
        
        plt.axis('on')
        plt.axis('equal')
        plt.show()
        return p
    

    # @staticmethod
    def plotGraph(self, cell_scale_fact=1):

        p = plt.subplots(figsize=(8, 8))
        # axx.add_collection(p)
        # axx.autoscale()

        ## draw nodes
        x_cs = []
        y_cs = []
        for i in range(len(self.cell_center)):
            x_c = self.cell_center[i][0]
            y_c = self.cell_center[i][1]
            x_cs.append(x_c)
            y_cs.append(y_c)

        plt.plot(x_cs,y_cs, 'o', markersize = 5, color='b',zorder=1)

        ## draw edges
        src_des_x = []
        src_des_y = []
        for i in range(len(self.nodes_edges)):
            src = self.nodes_edges[i][0]
            des = self.nodes_edges[i][1]
            src_x = self.cell_center[src][0]
            src_y = self.cell_center[src][1]
            des_x = self.cell_center[des][0]
            des_y = self.cell_center[des][1]
            src_des_x.append([src_x,des_x])
            src_des_y.append([src_y,des_y])

        for i in range(len(self.nodes_edges)):
            plt.plot(src_des_x[i], src_des_y[i], color='r',zorder=0)
        
        plt.axis('on')
        plt.axis('equal')
        plt.show()
        return p



