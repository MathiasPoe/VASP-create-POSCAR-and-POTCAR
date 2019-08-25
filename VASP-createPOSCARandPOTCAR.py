"""
Load CONTCAR files from VASP in the format of a pandas DataFrame. Each atom is one entry in the DataFrame with the properties Index, Element-type, coordinated (x, y, z) and fixed coordinates (fix_x, fix_y, fix_z)
This DataFrame can be modified and changed (rotation, translation, merging, ...).
Write such an DataFrame to POSCAR and POTCAR files, which are needed to run jobs. For writing the POTCAR file the needed Pseudopotentials of all element types need to be in the same directory in the format "POTCAR_X", where X is the element type (for example Fe for iron).
"""

import numpy as np
import pandas as pd
import glob
import os


def loadCONTCAR(file):

    with open(file, 'r') as load_file:
        load_file_lines = load_file.readlines()

    scale_matrix = []
    for i in range(3):
        scale_matrix.append(np.array([float(item) * float(load_file_lines[1]) for item in load_file_lines[i + 2].split()]))
    scale_matrix = np.array(scale_matrix).T

    atoms = load_file_lines[5].split()
    num_atoms = [int(item) for item in load_file_lines[6].split()]
    atom_labels = []
    for a, b in zip(atoms, num_atoms):
        for _ in range(b):
            atom_labels.append(a)

    try:
        xyz_tmp = pd.read_csv(file, header=7)
        float(xyz_tmp.loc[0].item().split()[0])
    except ValueError:
        xyz_tmp = pd.read_csv(file, header=8)
    if len(xyz_tmp.loc[0].item().split()) == 3:
        fix_incl = False
    if len(xyz_tmp.loc[0].item().split()) == 6:
        fix_incl = True

    xyz = {'idx': [], 'atom': [], 'x': [], 'y': [], 'z': [], 'fix_x': [], 'fix_y': [], 'fix_z': []}

    for key in xyz_tmp:
        for n, x in enumerate(xyz_tmp[key]):
            if n < sum(num_atoms):
                temp_line = np.array([float(item) for item in x.split()[:3]])
                scaled_line = np.dot(scale_matrix, temp_line)
                xyz['x'].append(scaled_line[0])
                xyz['y'].append(scaled_line[1])
                xyz['z'].append(scaled_line[2])
                xyz['idx'].append(n)
                xyz['atom'].append(atom_labels[n])
                if fix_incl:
                    xyz['fix_x'].append(x.split()[3])
                    xyz['fix_y'].append(x.split()[4])
                    xyz['fix_z'].append(x.split()[5])
                else:
                    xyz['fix_x'].append('T')
                    xyz['fix_y'].append('T')
                    xyz['fix_z'].append('T')
    xyz = pd.DataFrame(xyz)
    xyz.set_index('idx', inplace=True)
    xyz.sort_values(['atom', 'z', 'x', 'y'], inplace=True)
    xyz.reset_index(inplace=True, drop=True)
    return xyz, scale_matrix


def write_poscar_potcar(file, xyz, scale_matrix):
    posfile = open(file + 'POSCARnew', 'w')
    potfile = open(file + 'POTCARnew', 'w')

    inv_scale_matrix = np.linalg.inv(scale_matrix)

    POTCARFILES = glob.glob('POTCAR*')
    POTS = {}
    for potc in POTCARFILES:
        pf = open(potc, 'r')
        POTS['_'.join(potc.split('_')[1:])] = pf.readlines()

    atoms = np.unique(xyz.atom)
    num_atoms = [len(xyz[xyz['atom'] == atom_type]) for atom_type in atoms]

    for at in atoms:
        try:
            for potline in POTS[at]:
                potfile.write(potline)
        except KeyError:
            print('No POTCAR file for atom {} available!'.format(at))
            os.remove(file + 'POTCARnew')
            print(file + 'POTCARnew deleted!')
            break

    posfile. write(file + '\n')
    posfile.write('   1.00000000000000\n')
    for row in scale_matrix.T:
        posfile.write('   {: 20.16f}  {:20.16f}  {: 20.16f}\n'.format(*(row)))
    posfile.write(('{:>4s} ' * len(atoms)).format(*atoms) + '\n')
    posfile.write((' {: >5d}' * len(num_atoms)).format(*num_atoms) + '\n')
    posfile.write('Selective dynamics\nDirect\n')

    for idx in xyz.index.values:
        write_list = [*np.dot(inv_scale_matrix, xyz.loc[idx, ['x', 'y', 'z']]), xyz.fix_x[idx], xyz.fix_y[idx], xyz.fix_z[idx]]
        posfile.write(' {: 18.16f} {: 18.16f} {: 18.16f}   {:s}   {:s}   {:s}\n'.format(*write_list))

    posfile.close()
    potfile.close()


#####################################
#  !!!do the analysis down here!!!  #
#####################################
