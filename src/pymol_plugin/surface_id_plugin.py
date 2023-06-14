# -*- coding: utf-8 -*-
# Pablo Gainza Cirauqui 2016 LPDI IBI STI EPFL
# This pymol plugin for Masif just enables the load ply functions. 

# Pablo Gainza Cirauqui 2016 
# This pymol function loads dot files into pymol.

#Adopted/Extended by Jae H. Lee At Sanofi to load the hit/target surface patches


from pymol import cmd, stored
from pymol.cgo import *
import os.path
import numpy as np

# Simple ply loading class. 
# I created this class to avoid the need to install pymesh if the only goal is to load ply files. 
# Use this only for the pymol plugin. Currently only supports ascii ply files.
# Pablo Gainza LPDI EPFL 2019
class Simple_mesh:

    def __init__(self):
        self.vertices = []
        self.faces = []

    def load_mesh(self, filename):
        lines = open(filename, 'r').readlines()
        # Read header
        self.attribute_names = []
        self.num_verts = 0
        self.num_verts_patch = 0
        self.num_hit_patch = 0 
        self.patch_indices = []
        self.patch_nums = []        
        self.patch_vertices = []
        line_ix = 0
        while 'end_header' not in lines[line_ix]: 
            line = lines[line_ix]
            if line.startswith('element vertex'): 
                self.num_verts = int(line.split(' ')[2])
            if line.startswith('property float'):
                name = line.split(' ')[2].rstrip()
                if name not in ["px", "py", "pz"]:
                    self.attribute_names.append('vertex_'+name)
                # else:
                # self.attribute_names.append('patch_'+name)                    
            if line.startswith('element face'):
                self.num_faces= int(line.split(' ')[2])
            if line.startswith("element hit_patch"):
                self.num_hit_patch = int(line.split(' ')[2])
                print("# of hit_patch", self.num_hit_patch)
            if line.startswith("element contact_patch"):
                self.num_verts_patch = int(line.split(' ')[2])
                print("# of contact_patch", self.num_verts_patch)                
            if line.startswith("element patch"):
                idx, n = line.lstrip("element patch").split(' ')
                self.patch_indices.append(idx)
                self.patch_nums.append(int(n))
            line_ix += 1
        line_ix += 1
        header_lines = line_ix
        self.attributes = {}
        for at in self.attribute_names:
            self.attributes[at] = []
        self.vertices = []
        self.normals = []
        self.faces = []
        # Read vertex attributes.
        for i in range(header_lines, self.num_verts+header_lines):
            cur_line = lines[i].split(' ')
            vert_att = [float(x) for x in cur_line]
            # Organize by attributes
            for jj, att in enumerate(vert_att): 
                self.attributes[self.attribute_names[jj]].append(att)
            line_ix += 1
        # Set up vertices
        for jj in range(len(self.attributes['vertex_x'])):
            self.vertices = np.vstack([self.attributes['vertex_x'],\
                                    self.attributes['vertex_y'],\
                                    self.attributes['vertex_z']]).T
        # Read faces.
        face_line_start = line_ix
        for i in range(face_line_start, face_line_start+self.num_faces):
            try:
                fields = lines[i].split(' ')
            except:
                ipdb.set_trace()
            face = [int(x) for x in fields[1:]]
            self.faces.append(face)

        # Read hit patch vertices.
        patch_line_start = face_line_start+self.num_faces
        self.hit_patch = []
        for i in range(patch_line_start, patch_line_start+self.num_hit_patch):
            cur_line = lines[i].split(' ')
            vert_att = [float(x) for x in cur_line]
            self.hit_patch.append(vert_att)
        print("# of hit_patch read", len(self.hit_patch))            

        # Read contact vertices.
        patch_line_start = face_line_start+self.num_faces
        self.verts_patch = []
        for i in range(patch_line_start, patch_line_start+self.num_verts_patch):
            cur_line = lines[i].split(' ')
            vert_att = [float(x) for x in cur_line]
            self.verts_patch.append(vert_att)
        print("# of verts_patch read", len(self.verts_patch))

        # Read patch vertices.
        start = patch_line_start + self.num_verts_patch        
        for n in self.patch_nums:
            vs = []
            for i in range(start, start+n, 1):
                cur_line = lines[i].split(' ')
                vert_att = [float(x) for x in cur_line]
                vs.append(vert_att)
            self.patch_vertices.append(vs)
            start += n
        self.faces = np.array(self.faces)
        self.vertices = np.array(self.vertices)

        # Convert to numpy array all attributes.
        for key in self.attributes.keys():
            self.attributes[key] = np.array(self.attributes[key])

    def get_attribute_names(self):
        return list(self.attribute_names)

    def get_attribute(self, attribute_name):
        return np.copy(self.attributes[attribute_name])




colorDict = {'sky': [COLOR, 0.0, 0.76, 1.0 ],
        'sea': [COLOR, 0.0, 0.90, 0.5 ],
        'yellowtint': [COLOR, 0.88, 0.97, 0.02 ],
        'hotpink': [COLOR, 0.90, 0.40, 0.70 ],
        'greentint': [COLOR, 0.50, 0.90, 0.40 ],
        'blue': [COLOR, 0.0, 0.0, 1.0 ],
        'green': [COLOR, 0.0, 1.0, 0.0 ],
        'yellow': [COLOR, 1.0, 1.0, 0.0 ],
        'orange': [COLOR, 1.0, 0.5, 0.0],
        'red': [COLOR, 1.0, 0.0, 0.0],
        'black': [COLOR, 0.0, 0.0, 0.0],
        'white': [COLOR, 1.0, 1.0, 1.0],
        'gray': [COLOR, 0.9, 0.9, 0.9] }

def load_dots(filename, color="white", name='ply', dotSize=0.2, lineSize = 0.5, doStatistics=False):
    lines = open(filename).readlines()
    lines = [line.rstrip() for line in lines]
    lines = [line.split(',') for line in lines]
    verts = [[float(x[0]), float(x[1]), float(x[2])] for x in lines]

    normals = None

    if len(lines[0]) > 3:
        # normal is the last column - draw it  
        normals = [[float(x[3]), float(x[4]), float(x[5])] for x in lines]
     
    # Draw vertices 
    obj = []

    for v_ix in range(len(verts)):
        colorToAdd = colorDict[color]
        vert = verts[v_ix]
        # Vertices
        obj.extend(colorToAdd)
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
#    obj.append(END)
    name = "vert_"+filename
    group_names = name
    cmd.load_cgo(obj,name, 1.0)
    obj = []
    # Draw normals
    if normals is not None:
        colorToAdd = colorDict['white']
        obj.extend([BEGIN, LINES])
        obj.extend([LINEWIDTH, 2.0])
        colorToAdd = colorDict[color]
        obj.extend(colorToAdd)
        for v_ix in range(len(verts)): 
            vert1 = verts[v_ix]
            vert2 = np.array(verts[v_ix])+np.array(normals[v_ix])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
#        obj.append(END)
        name = "norm_"+filename
        group_names = name
        cmd.load_cgo(obj,name, 1.0)
    # Draw normals


            
               
            



# Pablo Gainza Cirauqui 2016 
# This pymol function loads ply files into pymol. 
from pymol import cmd, stored
import sys 
import os,math,re
from pymol.cgo import *
import os.path
import numpy as np

colorDict = {'sky': [COLOR, 0.0, 0.76, 1.0 ],
        'sea': [COLOR, 0.0, 0.90, 0.5 ],
        'yellowtint': [COLOR, 0.88, 0.97, 0.02 ],
        'hotpink': [COLOR, 0.90, 0.40, 0.70 ],
        'greentint': [COLOR, 0.50, 0.90, 0.40 ],
        'blue': [COLOR, 0.0, 0.0, 1.0 ],
        'green': [COLOR, 0.0, 1.0, 0.0 ],
        'yellow': [COLOR, 1.0, 1.0, 0.0 ],
        'orange': [COLOR, 1.0, 0.5, 0.0],
        'red': [COLOR, 1.0, 0.0, 0.0],
        'black': [COLOR, 0.0, 0.0, 0.0],
        'white': [COLOR, 1.0, 1.0, 1.0],
        'gray': [COLOR, 0.9, 0.9, 0.9] }

# Create a gradient color from color 1 to whitish, to color 2. val goes from 0 (color1) to 1 (color2).
def color_gradient(vals, color1, color2):
    c1 = Color("white")
    c2 = Color("orange")
    ix = np.floor(vals*100).astype(int)
    crange = list(c1.range_to(c2, 100))
    mycolor = []
    print(crange[0].get_rgb())
    for x in ix: 
        myc = crange[x].get_rgb()
        mycolor.append([COLOR, myc[0], myc[1], myc[2]]) 
    return mycolor

def iface_color(iface):
    # max value is 1, min values is 0
    hp = iface.copy()
    hp = hp*2 - 1
    mycolor = charge_color(-hp)
    return mycolor

# Returns the color of each vertex according to the charge. 
# The most purple colors are the most hydrophilic values, and the most 
# white colors are the most positive colors.
def hphob_color(hphob):
    # max value is 4.5, min values is -4.5
    hp = hphob.copy()
    # normalize
    hp = hp + 4.5 
    hp = hp/9.0
    #mycolor = [ [COLOR, 1.0, hp[i], 1.0]  for i in range(len(hp)) ]
    mycolor = [ [COLOR, 1.0, 1.0-hp[i], 1.0]  for i in range(len(hp)) ]
    return mycolor

# Returns the color of each vertex according to the charge. 
# The most red colors are the most negative values, and the most 
# blue colors are the most positive colors.
def charge_color(charges):
    # Assume a std deviation equal for all proteins.... 
    max_val = 1.0
    min_val = -1.0

    norm_charges = charges
    blue_charges = np.array(norm_charges)
    red_charges = np.array(norm_charges)
    blue_charges[blue_charges < 0] = 0
    red_charges[red_charges > 0] = 0
    red_charges = abs(red_charges) 
    red_charges[red_charges>max_val] = max_val
    blue_charges[blue_charges< min_val] = min_val
    red_charges = red_charges/max_val
    blue_charges = blue_charges/max_val
    #red_charges[red_charges>1.0] = 1.0
    #blue_charges[blue_charges>1.0] = 1.0
    green_color  = np.array([0.0]*len(charges))
    mycolor = [ [COLOR, 0.9999-blue_charges[i], 0.9999-(blue_charges[i]+red_charges[i]), \
                    0.9999-red_charges[i]]  for i in range(len(charges)) ]
    for i in range(len(mycolor)):
        for k in range(1,4):
            if mycolor[i][k] < 0:
                mycolor[i][k] = 0

    return mycolor

def load_ply(filename, color="white", name='ply', dotSize=0.2, lineSize = 0.5, doStatistics=False):
## Pymesh should be faster and supports binary ply files. However it is difficult to install with pymol... 
#        import pymesh
#        mesh = pymesh.load_mesh(filename)
    
    mesh = Simple_mesh()
    mesh.load_mesh(filename)

    ignore_normal = False 
    with_normal = False
    with_color = False
        
    group_names = ''

    verts = mesh.vertices
    try:
        charge = mesh.get_attribute("vertex_charge")
        color_array = charge_color(charge)
    except:
        print('Could not load vertex charges.')
        color_array = [colorDict['green']]*len(verts)
        color_array = [colorDict['red']]*len(verts)
    if 'vertex_nx' in mesh.get_attribute_names():
        nx = mesh.get_attribute('vertex_nx')
        ny = mesh.get_attribute('vertex_ny')
        nz = mesh.get_attribute('vertex_nz')
        normals = np.vstack([nx, ny, nz]).T
        print(normals.shape)

    suffix = os.path.basename(filename).replace("e2e_final_", "")

    # Draw vertices 
    obj = []
    color = 'green'
    color = 'red'
    for v_ix in range(len(verts)):
        vert = verts[v_ix]
        colorToAdd = color_array[v_ix]
        # Vertices
        obj.extend(colorToAdd)
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
    name = "vert_"+suffix
    cmd.load_cgo(obj,name, 1.0)
    group_names = group_names+' '+name
    
    obj =[]
    faces = mesh.faces
    # Draw surface charges.
    if 'vertex_charge' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        color_array_surf = color_array
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "pb_"+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []
        group_names = group_names+' '+name

    obj = []
    # Draw hydrophobicity
    if 'vertex_hphob' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        hphob = mesh.get_attribute('vertex_hphob')
        color_array_surf = hphob_color(hphob)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "hphobic_"+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []
        group_names = group_names+' '+name

    obj = []
    # Draw shape index
    if 'vertex_si' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        si = mesh.get_attribute('vertex_si')
        color_array_surf = charge_color(si)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "si_"+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []
        group_names = group_names+' '+name

    obj = []
    # Draw shape index
    if 'vertex_si' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        si = mesh.get_attribute('vertex_si')
        color_array_surf = charge_color(si)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "si_"+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []

    obj = []
    # Draw ddc
    if 'vertex_ddc' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        ddc = mesh.get_attribute('vertex_ddc')
        # Scale to -1.0->1.0
        ddc = ddc*1.4285
        color_array_surf = charge_color(ddc)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "ddc_"+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []
        group_names = group_names+' '+name

    obj = []

    # Draw iface
    if 'vertex_iface' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        iface = mesh.get_attribute('vertex_iface')
        color_array_surf = iface_color(iface)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "iface_"+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []
        group_names = group_names+' '+name

    obj = []
    # Draw hbond
    if 'vertex_hbond' in mesh.get_attribute_names() and 'vertex_nx' in mesh.get_attribute_names(): 
        hbond = mesh.get_attribute('vertex_hbond')
        color_array_surf = charge_color(hbond)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            #obj.extend([ALPHA, 0.6])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = 'hbond_'+suffix
        cmd.load_cgo(obj,name, 1.0)
        obj = []
        group_names = group_names+' '+name

    # Draw triangles (faces)
    for tri in faces: 
        pairs = [[tri[0],tri[1]], [tri[0],tri[2]], [tri[1],tri[2]]]
        colorToAdd = colorDict['black']
        for pair in pairs: 
            vert1 = verts[pair[0]]
            vert2 = verts[pair[1]]
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
    name = "mesh_"+suffix 
    cmd.load_cgo(obj,name, 1.0)
    group_names = group_names + ' ' +name

    # Draw normals
    if with_normal and not ignore_normal:
        for v_ix in range(len(verts)):
            colorToAdd = colorDict['white']
            vert1 = verts[v_ix]
            vert2 = [verts[v_ix][0]+nx[v_ix],\
                    verts[v_ix][1]+ny[v_ix],\
                    verts[v_ix][2]+nz[v_ix]]
            obj.extend([LINEWIDTH, 2.0])
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
        cmd.load_cgo(obj,"normal_"+suffix, 1.0)

    # Draw contact vertices 
    print("Draw contact vertices")
    color = 'green'    
    color = 'red'    
    if mesh.num_verts_patch != 0:
        obj = []
        vertices_patch = mesh.verts_patch
        for v_ix in range(len(vertices_patch)):
            vert = vertices_patch[v_ix]
            colorToAdd = color_array[v_ix]
            # Vertices
            obj.extend(colorToAdd)
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        print(vert)
        print(colorToAdd)
        print(SPHERE)
        print(dotSize)
        name = "contact_"+ suffix
        cmd.load_cgo(obj,name, 1.0)
        group_names = group_names+' '+name

    # Draw hit vertices 
    print("Draw hit vertices")
      
    if mesh.num_hit_patch != 0:
        obj = []
        vertices_patch = mesh.hit_patch
        for v_ix in range(len(vertices_patch)):
            vert = vertices_patch[v_ix]
            colorToAdd = color_array[v_ix]
            # Vertices
            obj.extend(colorToAdd)
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        print(vert)
        print(colorToAdd)
        print(SPHERE)
        print(dotSize)
        name = "hit_"+ suffix
        cmd.load_cgo(obj,name, 1.0)
        group_names = group_names+' '+name        

    # Draw patch vertices
    print("Draw patch vertices")
    color = 'green'    
    color = 'red'    
    for vertices_patch, idx in zip(mesh.patch_vertices, mesh.patch_indices):
        obj = []
        for v_ix in range(len(vertices_patch)):
            vert = vertices_patch[v_ix]
            colorToAdd = [COLOR, 0, 1, 0] # color_array[v_ix]
            # Vertices
            obj.extend(colorToAdd)
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        name = f"q{idx}_{suffix}"
        cmd.load_cgo(obj,name, 1.0)        
        group_names = group_names+' '+name

    print(group_names)
    cmd.group(suffix, group_names)

# Load the sillouete of an iface.
def load_giface(filename, color="white", name='giface', dotSize=0.2, lineSize = 1.0):
    mesh = pymesh.load_mesh(filename)
    if 'vertex_iface' not in mesh.get_attribute_names():
        return
    iface = mesh.get_attribute('vertex_iface')
    # Color an edge only if:
        # iface > 0 for its two edges
        # iface is zero for at least one of its edges.
    # Go through each face. 
    faces = mesh.faces
    verts = mesh.vertices
    obj = []
    visited = set()
    colorToAdd = colorDict['green']
    obj.extend([BEGIN, LINES])
    obj.extend([LINEWIDTH, 5.0])
    obj.extend(colorToAdd)
    for tri in faces: 
        pairs = [[tri[0],tri[1], tri[2]], [tri[0],tri[2], tri[1]], [tri[1],tri[2], tri[0]]]
        for pair in pairs: 
            if iface[pair[0]] > 0 and iface[pair[1]] > 0 and iface[pair[2]] == 0:
                vert1 = verts[pair[0]]
                vert2 = verts[pair[1]]

                obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
                obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
    obj.append(END)
    name = "giface_"+filename 
    cmd.load_cgo(obj,name, 1.0)
    colorToAdd = colorDict['green']

    obj = []
    obj.extend(colorToAdd)
    for tri in faces: 
        pairs = [[tri[0],tri[1], tri[2]], [tri[0],tri[2], tri[1]], [tri[1],tri[2], tri[0]]]
        for pair in pairs: 
            if iface[pair[0]] > 0 and iface[pair[1]] > 0 and iface[pair[2]] == 0:
                vert1 = verts[pair[0]]
                vert2 = verts[pair[1]]

                obj.extend([SPHERE, (vert1[0]), (vert1[1]), (vert1[2]), 0.4])
                obj.extend([SPHERE, (vert2[0]), (vert2[1]), (vert2[2]), 0.4])
    #obj.append(END)
    name = "giface_verts_"+filename 
    cmd.load_cgo(obj,name, 1.0)
 

## load_ply_ref
def load_ply_ref(filename, hits, color="white", name='ply', dotSize=0.2, lineSize = 0.5, doStatistics=False):
## Pymesh should be faster and supports binary ply files. However it is difficult to install with pymol... 
#        import pymesh
#        mesh = pymesh.load_mesh(filename)
    
    mesh = Simple_mesh()
    mesh.load_mesh(filename)
    print(type(hits))
    if type(hits) == list:
        hits_idx=tuple(hits)
        num_hits = len(hits_idx)
    elif type(hits) == tuple:
        hits_idx = range(hits_idx[0], hits_idx[1]+1)
        num_hits = len(hits_idx)
    elif type(hits) == str:
        if hits.startswith("["):
            hits = map(int, hits.strip('[]').split(','))
            hits_idx = tuple(hits)
            num_hits = len(hits_idx)
        elif hits.startswith("("):
            hits = map(int, hits.strip('()').split(','))
            hits_idx = tuple(hits)
            hits_idx = range(hits_idx[0], hits_idx[1]+1)
            num_hits = len(hits_idx)
        elif hits == "all":
            num_hits = len(mesh.patch_indices)
            hits_idx = range(1, num_hits+1)
        else:
            try:
                hits=int(hits)
                hits_idx = range(1, hits+1)
                num_hits = hits
            except :
                print("""Provide hit regions for visualization using one of the following: 
 all for all hits (e.g. all)
 an integer number for top n hits (e.g. 3) 
 alist of hits id number (e.g. [1, 3, 5]) 
 a two elements tuple for first and last id number (e.g. (3,5))""")
                cmd.quit()
  
    print("attributes: ", mesh.get_attribute_names())

    ignore_normal = False 
    with_normal = False
    with_color = False
        
    group_names = ''

    verts = mesh.vertices
    try:
        charge = mesh.get_attribute("vertex_charge")
        color_array = charge_color(charge)
    except:
        print('Could not load vertex charges.')
        color_array = [colorDict['yellowtint']]*len(verts)
    if 'vertex_nx' in mesh.get_attribute_names():
        nx = mesh.get_attribute('vertex_nx')
        ny = mesh.get_attribute('vertex_ny')
        nz = mesh.get_attribute('vertex_nz')
        normals = np.vstack([nx, ny, nz]).T
        print(normals.shape)

    #suffix = os.path.basename(filename).replace("e2e_final_002_", "")
    suffix = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    
    # load ref pdb 
    pdb_name=filename.replace("ply","pdb") 
    pdb_name=pdb_name.replace("_ref","")
    pdb_name=pdb_name.replace("_R.pdb","_RLH.pdb")
    print(pdb_name)
    name = "pdb_"+suffix
    if os.path.exists(pdb_name):
        cmd.load(pdb_name,name)
    group_names = group_names + ' ' +name

    # Draw mesh
    obj = []
    faces = mesh.faces
        
    for tri in faces: 
        pairs = [[tri[0],tri[1]], [tri[0],tri[2]], [tri[1],tri[2]]]
        colorToAdd = colorDict['gray']
        for pair in pairs: 
            vert1 = verts[pair[0]]
            vert2 = verts[pair[1]]
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
    name = "mesh_"+suffix 
    cmd.load_cgo(obj,name, 1.0)
    group_names = group_names + ' ' +name
    
    
    # Draw hit vertices 
    print("Draw hit vertices")
      
    if mesh.num_hit_patch != 0:
        obj = []
        vertices_patch = mesh.hit_patch
        for v_ix in range(len(vertices_patch)):
            vert = vertices_patch[v_ix]
            colorToAdd = color_array[v_ix]
            # Vertices
            obj.extend(colorToAdd)
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        name = "hit_"+ suffix
        cmd.load_cgo(obj,name, 1.0)
        group_names = group_names+' '+name     
    
    
    # Draw patch vertices    
    print("Draw patch vertices")
    color = 'green'    
    color = 'red'    
    
    suffix = suffix.replace("_ref", "")
    for num, (vertices_patch, idx) in enumerate(zip(mesh.patch_vertices, mesh.patch_indices)):
        #print("vertices_patch: ", vertices_patch)
        #print("idx: ", idx[1:])
        #print(f"num: {num+1}")
        #print("hits_idx: {hits_idx}")
        obj = []
        if num+1 in hits_idx:
            for v_ix in range(len(vertices_patch)):
                vert = vertices_patch[v_ix]
                colorToAdd = [COLOR, 0, 1, 0] # color_array[v_ix]
                # Vertices
                obj.extend(colorToAdd)
                obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
            name = f"q{idx}_{suffix}"
            cmd.load_cgo(obj,name, 1.0)  
            #cmd.png(name+".png")
            #group_names = group_names+' '+name
            hit_name = f"{idx[1:]}_{suffix}"
            load_ply_hit(os.path.join(dirname,hit_name), name)
    #for idx, hit_name in 
    cmd.group(suffix, group_names)

    
    return 

## load_ply_hit
def load_ply_hit(filename, patch_name, color="white", name='ply', dotSize=0.2, lineSize = 0.5, doStatistics=False):
## Pymesh should be faster and supports binary ply files. However it is difficult to install with pymol... 
#        import pymesh
#        mesh = pymesh.load_mesh(filename)
    
    mesh = Simple_mesh()
    mesh.load_mesh(filename)

    pdb_name=filename.replace("ply","pdb")

    ignore_normal = False 
    with_normal = False
    #with_color = False
        
    group_names = patch_name

    verts = mesh.vertices
    
    
    try:
        charge = mesh.get_attribute("vertex_charge")
        color_array = charge_color(charge)
    except:
        print('Could not load vertex charges.')
        color_array = [colorDict['green']]*len(verts)
    """
    if 'vertex_nx' in mesh.get_attribute_names():
        nx = mesh.get_attribute('vertex_nx')
        ny = mesh.get_attribute('vertex_ny')
        nz = mesh.get_attribute('vertex_nz')
        normals = np.vstack([nx, ny, nz]).T
        print(normals.shape)

    """
    #suffix = os.path.basename(filename).replace("e2e_final_002_", "")
    suffix = os.path.basename(filename)
    dirname = os.path.dirname(filename)

    # load pdb files
    name = "pdb_"+suffix
    if os.path.exists(pdb_name):
        cmd.load(pdb_name,name)    
    group_names = group_names + ' ' +name
    
    #load complex pdb files    
    complex_pdb_dirname = dirname + "/20201107_SAbDab/"

    complex_pdb_name = complex_pdb_dirname + suffix.split("_")[1][:4] +".pdb"
    complex_pdb_domain = suffix.split("_")[2][0] 
    print(suffix)
    complex_name = "complex_pdb_"+suffix   
    if os.path.exists(complex_pdb_name):
        cmd.load(complex_pdb_name,complex_name)
        if os.path.exists(pdb_name):
            cmd.align(f"{complex_name} and chain {complex_pdb_domain}", name)
    group_names = group_names + ' ' +complex_name
    
    obj = []
    faces = mesh.faces
        
    for tri in faces: 
        pairs = [[tri[0],tri[1]], [tri[0],tri[2]], [tri[1],tri[2]]]
        colorToAdd = colorDict['yellowtint']
        for pair in pairs: 
            vert1 = verts[pair[0]]
            vert2 = verts[pair[1]]
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
    name = "mesh_hit_"+suffix 
    cmd.load_cgo(obj,name, 1.0)
    group_names = group_names + ' ' +name

    
    # Draw hit vertices 
    print("Draw hit vertices")
      
    if mesh.num_hit_patch != 0:
        obj = []
        vertices_patch = mesh.hit_patch
        for v_ix in range(len(vertices_patch)):
            vert = vertices_patch[v_ix]
            colorToAdd = color_array[v_ix]
            # Vertices
            obj.extend(colorToAdd)
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        print(vert)
        print(colorToAdd)
        print(SPHERE)
        print(dotSize)
        name = "hit_"+ suffix
        cmd.load_cgo(obj,name, 1.0)
        group_names = group_names+' '+name   
    cmd.group(suffix, group_names)
    
    
    return 

cmd.extend('sidloadply', load_ply)
cmd.extend('sidloaddots', load_dots)
cmd.extend('sidloadgiface', load_giface)
cmd.extend('sidloadplyref', load_ply_ref)

