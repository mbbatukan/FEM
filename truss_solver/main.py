#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:35:09 2020
Revised on Tue Jan 11 13:51:10 2022

@author: Mehmet B Batukan
"""
import os
import truss_solver as truss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


working_dir = r""
file = "/example/input_data.xlsx"

df_input_data = pd.read_excel(working_dir + file, header=1)

node_coordinates = df_input_data.iloc[:, 0:3]
node_coordinates = node_coordinates.iloc[:node_coordinates['Node label'].last_valid_index()+1,:].to_numpy()

elem_num = df_input_data.iloc[:,4:8]
A = elem_num.iloc[:elem_num['Area'].last_valid_index()+1,3].to_numpy()
elem_num = elem_num.iloc[:elem_num['Element number'].last_valid_index()+1,0:3].to_numpy()

ext_force_data = df_input_data.iloc[:,9:11]
ext_force_data = ext_force_data.iloc[:ext_force_data['DOF'].last_valid_index()+1,0:2].to_numpy()

rest_dofs = df_input_data.iloc[:,12]
rest_dofs = rest_dofs.iloc[:rest_dofs.last_valid_index()+1].to_numpy().astype(int)

E = df_input_data.iloc[:,14]
E = E.iloc[:E.last_valid_index()+1].to_numpy()[0]

mag_factor = df_input_data.iloc[:,19]
mag_factor = mag_factor.iloc[:mag_factor.last_valid_index()+1].to_numpy()[0]

plotting_flags = df_input_data.iloc[:,16:18]
plotting_flags = plotting_flags.iloc[:plotting_flags['Plot'].last_valid_index()+1,0:2].set_index('Plot')

num_nod = len(node_coordinates)
num_ele = len(elem_num)

nod_coor = node_coordinates[:,1:3]
ele_nod = elem_num[:,1:3].astype(int)

ele_coor = np.array([])

for i in range(num_ele):
  ele_i, ele_j = ele_nod[i,0], ele_nod[i,1]
  ele_co = np.array([[nod_coor[ele_i-1,0]],[nod_coor[ele_j-1,0]],[nod_coor[ele_i-1,1]],[nod_coor[ele_j-1,1]]]).T
  ele_coor = np.append([ele_coor], [ele_co])

K = np.zeros([2*num_nod,2*num_nod])
L = np.zeros([num_ele,1])
theta = np.zeros([num_ele,1])

for i in range(num_ele):
  L[i], theta[i] = truss.PlaneTrussElementLength(ele_coor[4*i+0],ele_coor[4*i+1],ele_coor[4*i+2],ele_coor[4*i+3])

ele_stif = np.zeros([num_ele*4,num_ele*4])
stif_array = np.array([])

for i in range(num_ele):
  ele_stif = truss.PlaneTrussElementStiffness(E, A[i], L[i], theta[i])
  stif_array = np.append([stif_array], [ele_stif])
  
for i in range(num_ele):
  K = truss.PlaneTrussAssemble(K, np.array([stif_array[i*16+0:i*16+4],stif_array[i*16+4:i*16+8],stif_array[i*16+8:i*16+12],stif_array[i*16+12:i*16+16]]), ele_nod[i,0], ele_nod[i,1])

idx = list(range(num_nod * 2))

L_unkn = [v for i,v in enumerate(idx) if i not in frozenset((rest_dofs - 1))] 
L_zero = [v for i,v in enumerate(idx) if i not in frozenset((L_unkn))] 

BIG_NUM = 1e100

K_new = np.copy(K)
K_new[L_zero,L_zero] = K_new[L_zero,L_zero] * BIG_NUM
Disp = np.zeros([num_nod * 2,1])
Force = np.zeros(num_nod*2)
Force[list((ext_force_data[:,0] - 1).astype(int))] = ext_force_data[:,1]
Force_big = np.zeros((len(L_zero),1))
for i in range(len(L_zero)):
  Force_big[i] = Disp[L_zero[i]] * BIG_NUM * K[L_zero[i],L_zero[i]]
Force[L_zero] = Force_big[:,0]    

Disp = np.linalg.solve(K_new, Force)
Force = K @ Disp
U_total = np.zeros([num_nod * 2,1])
Disp_zero = U_total[L_zero].flatten()

Force_knwn = np.zeros(num_nod*2)
Force_knwn[list((ext_force_data[:,0] - 1).astype(int))] = ext_force_data[:,1]
Force_knwn = Force_knwn[L_unkn]

K_ff = K[:, L_unkn][L_unkn, :]
K_fs = K[:, L_zero][L_unkn, :]
K_sf = K[:, L_unkn][L_zero, :]
K_ss = K[:, L_zero][L_zero, :]

U = np.linalg.inv(K_ff) @ Force_knwn - np.linalg.inv(K_ff) @ K_fs @ Disp_zero
U_total[L_unkn, 0] = U
Force_total = K @ U_total

U_fac = mag_factor * U_total
U_act = np.reshape(nod_coor,(2*len(nod_coor),1),order='C')
U_def = U_fac + U_act
U_def2 = np.reshape(U_def,(len(nod_coor),2),order='C')

if not os.path.exists(working_dir + "/RESULTS"):
    os.makedirs(working_dir + "/RESULTS")

ele_coor2 = np.array([])
for i in range(num_ele):
  ele_i = ele_nod[i,0]
  ele_j = ele_nod[i,1]
  ele_co2 = np.array([[U_def2[ele_i-1,0]],[U_def2[ele_j-1,0]],[U_def2[ele_i-1,1]],[U_def2[ele_j-1,1]]]).T
  ele_coor2 = np.append([ele_coor2], [ele_co2])

if int(plotting_flags.loc['Node numbers']) == 1:
  plt.figure()
  for i in range(num_ele):
    plt.plot([ele_coor[i*4+0],ele_coor[i*4+1]],[ele_coor[i*4+2],ele_coor[i*4+3]], 'k' '-o', markersize=1, linewidth=0.2)
  plt.title('Truss - Node Numbers')
  plt.grid()
  plt.axis('equal')
  for i in range(num_nod):
    plt.annotate(i+1, np.array([[nod_coor[i,0]], [nod_coor[i,1]]]), ha='right', va='top', size=6)
  plt.savefig(working_dir + "/RESULTS/Node_numbers.svg", dpi=1200, format="svg")
  

if int(plotting_flags.loc['Element numbers']) == 1:
  plt.figure()
  for i in range(num_ele):
    plt.plot([ele_coor[i*4+0],ele_coor[i*4+1]],[ele_coor[i*4+2],ele_coor[i*4+3]], 'k' '-o', markersize=1, linewidth=0.2)
  plt.title('Truss - Element Numbers')
  plt.grid()
  plt.axis('equal')  
  for i in range(num_ele):
    plt.text(0.5 * (ele_coor[i*4+0] + ele_coor[i*4+1]), 0.5 * (ele_coor[i*4+2] + ele_coor[i*4+3]), i+1, ha='center', va='bottom', size=4)
  plt.savefig(working_dir + "/RESULTS/Element_numbers.svg", dpi=1200, format="svg")
  

if int(plotting_flags.loc['Deflected shape']) == 1:
  plt.figure()
  for i in range(num_ele):
    plt.plot([ele_coor[i*4+0],ele_coor[i*4+1]],[ele_coor[i*4+2],ele_coor[i*4+3]], 'k' '--')
    plt.plot([ele_coor2[i*4+0],ele_coor2[i*4+1]],[ele_coor2[i*4+2],ele_coor2[i*4+3]], 'r' '-')
  plt.title('Deflected Shape of Truss (x' + str(mag_factor) +')')
  plt.grid()
  plt.axis('equal')
  plt.savefig(working_dir + "/RESULTS/Deflected_shape.svg", dpi=1200, format="svg")


U_total_rs = np.reshape(U_total,(num_nod,2),order='C')
F_total_rs = np.reshape(Force_total,(num_nod,2),order='C')

ele_disp = np.array([])
for i in range(num_ele):
  ele_i = ele_nod[i,0]
  ele_j = ele_nod[i,1]
  ele_d = np.array([[U_total_rs[ele_i-1,0]],[U_total_rs[ele_i-1,1]],[U_total_rs[ele_j-1,0]],[U_total_rs[ele_j-1,1]]]).T
  ele_disp = np.append([ele_disp], [ele_d])
 
ele_stress = np.array([])
for i in range(num_ele):
    ele_str = truss.PlaneTrussElementStress(E / 1000, L[i], theta[i], np.array([ele_disp[i*4+0],ele_disp[i*4+1],ele_disp[i*4+2],ele_disp[i*4+3]]))
    ele_stress = np.append([ele_stress], [ele_str])

ele_force = np.array([])
for i in range(num_ele):
    ele_frc = truss.PlaneTrussElementForce(E, A[i], L[i], theta[i], np.array([ele_disp[i*4+0],ele_disp[i*4+1],ele_disp[i*4+2],ele_disp[i*4+3]]))
    ele_force = np.append([ele_force], [ele_frc])

cmap = cm.get_cmap('bwr')
values = ele_stress
norm = mcolors.TwoSlopeNorm(vmin=np.min(values),vcenter=0,vmax=np.max(values))

if int(plotting_flags.loc['Stresses']) == 1:
  plt.figure()
  for i in range(num_ele):
    plt.plot([ele_coor2[i*4+0],ele_coor2[i*4+1]],[ele_coor2[i*4+2],ele_coor2[i*4+3]], '--' 'k', alpha=0.7)
    plt.plot([ele_coor2[i*4+0],ele_coor2[i*4+1]],[ele_coor2[i*4+2],ele_coor2[i*4+3]], '-', color=cmap(norm(values[i])),alpha=0.9)
  plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))
  plt.title('Stresses in Truss Members')
  plt.grid()
  plt.axis('equal')
  plt.savefig(working_dir + "/RESULTS/Stesses.svg", dpi=1200, format="svg")

with open(working_dir + "/RESULTS/node_displacements.txt", 'w') as f:
  f.write('-'*80)
  f.write('\nNode # and Displacements (X,Y): (Unit = mm)\n')
  f.write('-'*80)
  f.write('\n')
  f.write('\n'.join("{} {}".format(x, y) for x, y in zip((list(range(1, num_nod+1))), 1000 * U_total_rs)))
  f.write('\n')
  f.write('-'*80)

with open(working_dir + "/RESULTS/nodal_forces.txt", 'w') as f:
  f.write('-'*80)
  f.write('\nNode # and Forces (X,Y): (Unit = kN)\n')
  f.write('-'*80)
  f.write('\n')
  f.write('\n'.join("{} {}".format(x, y) for x, y in zip((list(range(1, num_nod+1))), F_total_rs)))
  f.write('\n')
  f.write('-'*80)

