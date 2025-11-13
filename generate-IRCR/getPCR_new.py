import ase.io
import numpy as np
from pymatgen.core import Structure



#Storing Pauling electronegativity values in a dictionary
electronegativity = {'H':2.20, 'He':0, 'Li':0.98, 'Be':1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44,\
                     'F': 3.98, 'Ne':0, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61,'Si':1.90, 'P':2.19, 'S':2.58, 'Cl': 3.16, \
                     'Ar':0, 'K':0.82, 'Ca':1.00, 'Sc':1.36, 'Ti': 1.54, 'V':1.63, 'Cr':1.66, 'Mn':1.55,\
                     'Fe':1.83, 'Co':1.88,\
                     'Ni':1.91, 'Cu':1.90, 'Zn':1.65, 'Ga': 1.81, 'Ge':2.01, 'As':2.18, 'Se':2.55,\
                     'Br':2.96, 'Kr':3.00, \
                     'Rb':0.82, 'Sr':0.95, 'Y':1.22, 'Zr':1.33, 'Nb':1.6, 'Mo':2.16, 'Tc':1.9,\
                     'Ru':2.2, 'Rh':2.28, 'Pd':2.20, \
                     'Ag':1.93, 'Cd':1.69, 'In':1.78, 'Sn':1.96, 'Sb':2.05,\
                     'Te':2.1, 'I':2.66, 'Xe':2.60, \
                     'Cs':0.79, 'Ba':0.89, 'La':1.1, \
                     'Ce':1.12, 'Pr':1.13, 'Nd':1.14, 'Pm': 1.13,\
                     'Sm':1.17, 'Eu':1.2, 'Gd':1.2, 'Tb':1.1,'Dy':1.22, \
                     'Ho':1.23, 'Er':1.24, 'Tm':1.25, 'Yb':1.1, 'Lu':1.27, \
                     'Hf':1.3, 'Ta':1.5, 'W':2.36, 'Re':1.9, 'Os':2.2, 'Ir':2.2,\
                     'Pt':2.28, 'Au':2.54, 'Hg':2.0, \
                     'Tl':1.62, 'Pb':1.87, 'Bi':2.02, 'Po':2.0, 'At':2.2, 'Rn':2.2,\
                     'Ac':1.1, 'Pu':1.28, 'U':1.38, 'Th':1.3, 'Np':1.36, 'Pa':1.5}


# defining group
def Group(z):
    if (z == 1) or (z == 3) or (z == 11) or (z == 55):
        grp = int(1)
    if (z == 4) or (z == 12) or (z == 56):
        grp = int(2)
    if (z == 2) or (z == 36) or (z == 54):
        grp = int(18)
    if (z >= 5) and (z <= 10):
        grp = int(z+8)
    if (z >= 13) and (z <= 18):
        grp = int(z)
    if ((z > 18) and (z <= 35)):
        grp = int(z%18)
    if (z >= 37) and (z <= 53):
        grp = int(z%18)
    if (z >= 57) and (z <= 71):
        grp = int(3)
    if (z >= 72) and (z <= 86):
        grp = int(z%18 + 4)
    if (z >= 89) and (z <= 103):
        grp = int(3)
    return(grp)


# defining valence electron number
def Valence(group):
    if(group >= 1) and (group <= 12):
        valence = int(group)

    if (group >=13) and (group <=17):
        valence = int(group%10)

    if (group == 18):
        valence = int(0)

    return(valence)

# defining period
def Period(z):
    if (z <= 2):
        period = int(1)
    if (z >= 3) and (z <= 10):
        period = int(2)
    if (z >= 11) and (z <= 18):
        period = int(3)
    if (z >= 19) and (z <= 36):
        period = int(4)
    if (z >= 37) and (z <= 54):
        period = int(5)
    if (z >= 55) and (z <= 86):
        period = int(6)
    if (z >= 87) and (z <= 118):
        period = int(7)
    return(period)


class PCR :
    
    """ 
    
    Goal : Create point cloud representation (PCR) upto ternary materials 
    
    Here we will create necessary matrices to create PCR .
    
    The matrices are following :
     
            Matrix                    Size
          ----------                --------
     1. Element Matrix          :   (z_max,3)       
     2. Lattice Matrix          :   (2,3)
     3. Basis Matrix            :   (n_sites,3)
     4. Site Occupancy Matrix   :   (n_sites,3)
     
    Here z_max is the maximum atomic number and,
    n_sites is the maximum number of sites in unit cell present in the dataset.
     
    In order to construct PCR, we just need three ingredients :  
     - cif file or POSCAR file
     - value of z_max
     - value of n_sites.  
    
    
    """
    
    def __init__(self,file,z_max,n_sites) :
        
        self.file = file
        self.z_max = z_max
        self.n_sites = n_sites
        self.atoms = ase.io.read(self.file)
        self.structure = Structure.from_file(self.file)
        self.z_total = list(self.atoms.numbers)
    
    def get_z_unique(self):
        z_total = list(self.atoms.numbers)
        z = []
        for val in z_total :
            if val not in z :
                z.append(val)
        return z
        
    def get_element_unique(self):
        element_all = self.atoms.get_chemical_symbols()
        element = []
        for ele in element_all :
            if ele not in element :
                element.append(ele)
        return element    
    
    def get_element_matrix(self):
        element_matrix = np.zeros((self.z_max,3))
        z_total = self.atoms.numbers
        z_unique = self.get_z_unique()
                
        # we will consider upto ternary material
        for i in range(3):
            if len(z_unique) < 3 :
                z_unique.append(0)
            
        for i in range(len(z_unique)):
            if z_unique[i] != 0 :
                element_matrix[z_unique[i]-1,i]=1 #i-th column               
        return element_matrix
    
    
    def get_lattice_matrix(self):
        cell = self.atoms.cell.cellpar()
        lattice_matrix = cell.reshape(2,3)
        return lattice_matrix
    
    
    def get_basis_matrix(self):
        basis = self.structure.frac_coords
        basis = np.asarray(basis)
        padding_dimension = self.n_sites - basis.shape[0]
        zero_mat = np.zeros((padding_dimension,3))
        basis_matrix = np.concatenate([basis,zero_mat],axis=0)
        return basis_matrix
    
    
    def get_site_occupancy_matrix(self):
        site_occupancy_matrix = np.zeros((self.n_sites,3))
        z = list(self.atoms.get_atomic_numbers())
        z_unique = self.get_z_unique()
        for i in range(len(z)):
            col_index = z_unique.index(z[i])
            site_occupancy_matrix[i,col_index]=1
        return site_occupancy_matrix
    

    def get_property_matrix(self):
        element_unique = self.get_element_unique()
        z_unique = self.get_z_unique()
        period = [Period(val) for val in z_unique]
        group = [Group(val) for val in z_unique]
        envt = [electronegativity[ele] for ele in element_unique]
        valence = [Valence(val) for val in group]
        num_each_atom = [self.z_total.count(val) for val in z_unique]
        stoichiometry = [val/np.sum(num_each_atom) for val in num_each_atom]        

        # zero padding if elemental or binary material
        for i in range(3):
            if len(z_unique) < 3 :
                z_unique.append(0)
                period.append(0)
                group.append(0)
                envt.append(0)
                valence.append(0)
                stoichiometry.append(0)

        property_list = z_unique + stoichiometry + period + group + envt + valence + [0,0,0] + [0,0,0]
        property_list = np.asarray(property_list)
        property_matrix = property_list.reshape((8,3))
        return property_matrix

    def get_pcr(self):
        element_matrix = self.get_element_matrix()
        lattice_matrix = self.get_lattice_matrix()
        basis_matrix = self.get_basis_matrix()
        site_occupancy_matrix = self.get_site_occupancy_matrix()
        property_matrix = self.get_property_matrix()

        pcr = np.concatenate([element_matrix,lattice_matrix,basis_matrix,\
                             site_occupancy_matrix,property_matrix],
                             axis=0)
        return pcr
        
    

