from pymol import cmd
import os, sys,glob
import numpy as np
sys.path.append('pymol_plugin')
from  loadply import *
import os

hits = ['2_1xiw_BE.1xiw_GH_4fqi_AB',        '14_6rvc_B.6rvc_E_4fqi_AB']


cmd.load('4fqi.pdb','4fqi')
cmd.select('d','not polymer')
cmd.remove('d')
load_ply(f"4fqi_AB_ref.ply",dotSize=0.35)


for i in cmd.get_names('all'):
    if i[0:2]=='q_' and '_'.join(i.replace('_ref.ply','').split('_')[1:]) not in hits:
        cmd.delete(i)
    
# load_ply(filename, color="white", name='ply', dotSize=0.2, lineSize = 0.5, doStatistics=False):
for i in hits:
    load_ply(f"{i}.ply",color='black',dotSize=0.35)

for ii,i in enumerate(hits):
    pid = i.split('_')[1]
    ag = i.split('_')[2].split('.')[0]
    ab = i.split('.')[1].split('_')[1]
    if len(ab)==1: ab = ab[0]+ab[0]
    if len(ag)==1: ag = ag[0]+ag[0]

    cmd.load(f"{i}.pdb",f"al{pid}_{ag}_{ab}_{ii}")
    cmd.load(f"../orig_pdbs/{pid}.cif",f"{pid}_{ag}_{ab}_{ii}")
    cmd.select('ref',f"al{pid}_{ag}_{ab}_{ii} & c. {ag[0]}+{ag[1]}")
    cmd.select('mob',f"{pid}_{ag}_{ab}_{ii} & c. {ag[0]}+{ag[1]}")
    cmd.align('mob','ref')
    cmd.select('d','not polymer')
    cmd.remove('d')
    cmd.remove(f"al{pid}_{ag}_{ab}_{ii}")
    cmd.select('d',f"{pid}_{ag}_{ab}_{ii} & c. {ab[0]}+{ab[1]}+{ag[0]}+{ag[1]}")
    cmd.select('dd',f"{pid}_{ag}_{ab}_{ii} & not d")
    cmd.remove('dd')
    cmd.delete('d')
    cmd.delete(f"al{pid}_{ag}_{ab}_{ii}")
    cmd.delete('dd')
    cmd.delete('ref')
    cmd.delete('mob')
    
cmd.zoom('all')
