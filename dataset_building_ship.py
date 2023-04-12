import scipy.io
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
import pickle

from mpl_toolkits import mplot3d
import os
import glob

#==================================================================================#
# Utility functions
#==================================================================================#

def file_names(xpath,ypath,zpath):

	os.chdir(xpath)
	xall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    xall_filenames.append(str(infile))

	os.chdir(ypath)
	yall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    yall_filenames.append(str(infile))

	os.chdir(zpath)
	zall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    zall_filenames.append(str(infile))

	# os.chdir(vpath)
	# vall_filenames=[]
	# for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	#     vall_filenames.append(str(infile))

	return xall_filenames,yall_filenames,zall_filenames
	#,vall_filenames

# Extract x,y,z columns from each file in each path and merge them to single file
# extact u from the u_vel x as well and append

# These functions return the u,v, and coordinates from each file
def read_vel(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[1]))

def read_coordinates(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[0]))

#==================================================================================#
#=========================== DATA PREPROCESSING ===================================#
#==================================================================================#
'''
The flow of data preprocessing is as follows :

Starccm+ ----> Full_dataset ----> Reduced_dataset ----> Different csv files ( domain,
boundary, initial ) ----> Use these csvs in the model building code.

'''

#==================================================================================#
#==================================================================================#
#==================================================================================#



#==================================================================================#
# Build a full dataset 

'''
This function builds the full dataset and returns a csv file at each time step
'''
#==================================================================================#

def build_dataset():
	finalpath=	r"E:\Vamsi_oe20s302\Original Ship Simulation\twice_velo\Full dataset"
	xpath=    	r"E:\Vamsi_oe20s302\Original Ship Simulation\twice_velo\uvel_dataset"
	ypath=    	r"E:\Vamsi_oe20s302\Original Ship Simulation\twice_velo\vvel_dataset"
	zpath=    	r"E:\Vamsi_oe20s302\Original Ship Simulation\twice_velo\wvel_dataset"
	

	xall_filenames,yall_filenames,zall_filenames=file_names(xpath,ypath,zpath)
	
	for i in range(len(xall_filenames)):

		print("Building "+str(i)+"th time_step file")
		name='/time_step'+str(i)+".csv"
		# reading each file name per timestep
		file_i=xpath+'/'+str(xall_filenames[i])
		file_j=ypath+'/'+str(yall_filenames[i])
		file_k=zpath+'/'+str(zall_filenames[i])

		# reading coordinates and converting them to numpy arrays
		x_co=read_coordinates(file_i).to_numpy()
		y_co=read_coordinates(file_j).to_numpy()
		z_co=read_coordinates(file_k).to_numpy()
		u=read_vel(file_i).to_numpy()
		v=read_vel(file_j).to_numpy()
		w=read_vel(file_k).to_numpy()

		t=np.repeat((0.0075*i),u.shape[0])
		t=np.reshape(t,(t.shape[0],1))
		
		data=np.concatenate([x_co,y_co,z_co,t,u,v,w],1)
		
		# Making a copy about x axis
		temp=np.copy(data)
		temp[:,1]=-1*temp[:,1]
		u_final=np.concatenate((data,temp),axis=0)
		
		data_frame=pd.DataFrame(u_final)
		data_frame.to_csv(finalpath + name,index=False)

# build_dataset()
#==================================================================================#
# Build a reduced dataset

'''
This function builds the reduced dataset and returns a csv file at each time step
'''
#==================================================================================#
def build_reduced_data(full_data_path,reduced_data_path):
	
	# Getting the names of files in full data 
	os.chdir(full_data_path)
	all_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    all_filenames.append(str(infile))

	# Building a reduced dataset from the full dataset
	for i in range(len(all_filenames)):
		
		print("Building"+str(i)+"th time_step file")
		test_name='/time_step'+str(i)+".csv"
		
		x=pd.read_csv(full_data_path+'/'+str(all_filenames[i]))
		x.sort_values(by=["0","1", "2"],axis=0,ascending=[True,True,True],inplace=True)
		x=x.to_numpy()

		# This reduces the computational domain  
		x_red=x[:,:][x[:,0]>-10]
		x_red=x_red[:,:][x_red[:,0]<-1]
		x_red=x_red[:,:][x_red[:,1]>0]
		x_red=x_red[:,:][x_red[:,1]<6]

		# making a copy of points about x axis
		temp=np.copy(x_red)
		temp[:,1]=-1*temp[:,1]
		u_final=np.concatenate((x_red,temp),axis=0)
		# converting the numpy array to dataframe
		reduced_data=pd.DataFrame(u_final)
		# saving the dataframe as a csv file
		reduced_data.to_csv(reduced_data_path+test_name,index=False)

# Run this to build a reduced dataset
full_data_path=r"E:\Vamsi_oe20s302\Original Ship Simulation\twice_velo\Full dataset"
reduced_data_path=r"E:\Vamsi_oe20s302\Original Ship Simulation\twice_velo\Reduced dataset"

build_reduced_data(full_data_path,reduced_data_path)

