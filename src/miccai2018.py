# This is the source for our paper titled "Evaluating surgical skills from kinematic data using convolutional neural networks" 
# The paper has been accepted at International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2018.
# If you find this code to be helpful in your research please do not hesitate to cite our paper. 

###################
## MICCAI - 2018 ##
###################

########################################################################################
## Evaluating surgical skills from kinematic data using convolutional neural networks ##
########################################################################################

import numpy as np
import random
import imageio
import time
from itertools import chain
from keras.models import Model
from keras.utils import np_utils
import keras
from keras import regularizers
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import collections
import re
import math
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import io 
from mpl_toolkits.mplot3d import Axes3D

def getExpertiseLevelOfSurgery(surgery_name):
	## function getMetaDataForSurgeries should be already called
	if surgeries_metadata.__contains__(surgery_name):
		return surgeries_metadata[surgery_name][0]
	return None 

def getMetaDataForSurgeries(surgery_type):
	surgeries_metadata = {}
	file = open(root_dir+surgery_type+'_kinematic/'+'meta_file_'+surgery_type+'.txt','r')
	for line in file: 
		line = line.strip() ## remove spaces
	
		if len(line)==0: ## if end of file
			break
	
		b = line.split()
		surgery_name = b[0] 
		expertise_level = b[1]
		b = b[2:]
		scores = [int(e) for e in b]
		surgeries_metadata[surgery_name]=(expertise_level,scores)
	return surgeries_metadata

def fit_encoder(y_train,y_test,y_val): 
	y_train_test_val = y_train+y_test+y_val
	encoder.fit(y_train_test_val)

def convertStringClassesToBinaryClasses(y_train,y_test,y_val):
	idx_y_test = len(y_train)
	idx_y_val = len(y_train)+len(y_test)
	y_train_test_val = y_train+y_test+y_val
	y_train_test_val = encoder.transform(y_train_test_val)
	y_train_test_val = np_utils.to_categorical(y_train_test_val)
	y_train = y_train_test_val[0:idx_y_test]
	y_test = y_train_test_val[idx_y_test:idx_y_val]
	y_val = y_train_test_val[idx_y_val:]
	return y_train,y_test,y_val

def get_trial_num(surgery_name,surgery_type):
	trial_num = surgery_name.replace(surgery_type+'_',"")[-1]
	return trial_num

def readFile(file_name,dtype,columns_to_use=None):
	X = np.loadtxt(file_name,dtype,usecols=columns_to_use)
	return X

def generateMaps(surgery_type):
	listOfSurgeries =[]
	y =[]
	path = root_dir+surgery_type+'_kinematic'+'/kinematics/AllGestures/'
	for subdir,dirs,files in os.walk(path):
		for file_name in files: 
			surgery = readFile(path+file_name,float,columns_to_use=dimensions_to_use)
			surgery_name = file_name[:-4]
			expertise_level = getExpertiseLevelOfSurgery(surgery_name)
			if expertise_level is None: 
				continue
			mapSurgeryDataBySurgeryName[surgery_name] = surgery
			mapExpertiseLevelBySurgeryName[surgery_name] = expertise_level
	return None


def write_csv_string_in_file(file_name,csv_string):
	file = open(path_to_results+ file_name + '.csv','w')
	file.write(csv_string)
	file.close()
	return True

# shuffles train and labels 
def shuffle(x_train,y_train):
	y_train = np.array(y_train)
	y_train = y_train.reshape(len(y_train),1)
	x_train = x_train.reshape(len(x_train),1)
	x_y_train = np.concatenate((x_train,y_train), axis=1)
	np.random.shuffle(x_y_train)
	return x_y_train[:,0] , x_y_train[:,1].tolist()

def validation(surgery_type = 'Suturing' , summary=False, reg =0.01, max_itr=20):
	# reg is the regularization parameter 
	# max_itr is the number of iterations to repeat the experiments
	counter = 0
	path = path_to_configurations+surgery_type +'/'+'unBalanced'+'/'+'GestureClassification'+'/'+'SuperTrialOut'
	results = "fold,iteration,macro,micro\n"
	for it in range(0,max_itr):
		for subdir,dirs,files in os.walk(path):
			# One configuration with two files Train.txt and Test.txt
			x_train = []
			y_train = []
			x_test = []
			y_test = []
			x_val = []
			y_val = []
			trial_added_to_val = None
			min_length_train = np.iinfo(np.int32).max # this is the minimum length of a training instance
			min_length_test = np.iinfo(np.int32).max # this is the minimum length of a test instance
			min_length_val = np.iinfo(np.int32).max # this is the minimum length of a val instance
			for file_name in files:
				data = readFile(subdir+'/'+file_name,str)
				surgeries_set = set()
				for gesture in data:
					surgery_name = find_pattern(gesture[0],surgery_type+'_.00.')
					surgeries_set.add(surgery_name)
	
				for surgery_name in surgeries_set:
					trial_num = get_trial_num(surgery_name,surgery_type)
					if file_name == 'Train.txt':
						if(trial_added_to_val is None):
							trial_added_to_val=trial_num
						
						if(trial_num==trial_added_to_val): 
							# we should add to validation set 
							min_length_val=min(len(mapSurgeryDataBySurgeryName[surgery_name]),min_length_val)
							x_val.append(mapSurgeryDataBySurgeryName[surgery_name])
							y_val.append(mapExpertiseLevelBySurgeryName[surgery_name])
						else: # we add to the train set 
							min_length_train = min(len(mapSurgeryDataBySurgeryName[surgery_name]),min_length_train)
							x_train.append(mapSurgeryDataBySurgeryName[surgery_name])
							y_train.append(mapExpertiseLevelBySurgeryName[surgery_name])
					else:
						# we are adding to the test set
						min_length_test = min(len(mapSurgeryDataBySurgeryName[surgery_name]),min_length_test)
						x_test.append(mapSurgeryDataBySurgeryName[surgery_name])
						y_test.append(mapExpertiseLevelBySurgeryName[surgery_name])
				# end of one file Train or Test 
			if(len(files)>0):
	
				x_train = np.array(x_train)
				x_test = np.array(x_test)
				x_val = np.array(x_val)
	
				print('train size:'+str(len(x_train)))
				print('val size:'+str(len(x_val)))
				print('test size:'+str(len(x_test)))
	
	
				fit_encoder(y_train,y_test,y_val)
	
				model = each_dim_build_model(input_shapes,summary=summary,reg =reg)
				
				fold = find_pattern(subdir,'SuperTrialOut'+'/.*_Out').replace('SuperTrialOut'+'/','').replace('_Out','')
				iteration = find_pattern(subdir, 'itr_.*').replace('itr_','')
				# we train on each training instance 
					
				y_test = fitModel(model,x_train,y_train,x_test,y_test,x_val,y_val)	
					
				model = load_model('model.h5')# reload the best model saved 

				# uncomment if you want to visualize the class activiation map as a gif 
				# generate_class_activation_map_for_all_surgeries(model,fold)
				
				# evaluate model and get results for confusion matrix 
				(macro,micro) = evaluateModel(model,x_test,y_test)
				results += fold+','+str(it)+','+str(macro)+','+str(micro)+'\n'
	
			# end of one configuration 
	matrix = confusion_matrix.as_matrix()
	macro = compute_macro(matrix)
	micro = compute_micro(matrix)
	results += 'total,total,'+str(macro)+','+str(micro)+'\n'
	results_file_name = 'results'
	return write_csv_string_in_file(results_file_name,results)


def find_pattern(word,pattern):
	return re.search(r''+pattern,word).group(0)

def compute_micro(matrix):
	return sum(matrix.diagonal()) / np.sum(matrix)

def compute_macro(matrix):
	res = matrix.diagonal()/np.sum(matrix,axis=1)
	return np.nansum(res)/float(nb_classes)

def fitModel(model,x_train,y_train,x_test,y_test,x_val,y_val):
	# x_test and y_test are used to monitor the overfitting / underfitting not for training 
	# minimum epoch loss on val set
	min_val_loss =  np.iinfo(np.int32).max 
	# train for many epochs as specified by nb_epochs
	for epoch in range(0,nb_epochs) : 
		# shuffle before every epoch training 
		x_train,y_train=shuffle(x_train,y_train)
		#convert string labels to binary forms
		y_train_binary,y_test_binary,y_val_binary = convertStringClassesToBinaryClasses(y_train,y_test,y_val)
		# train each sequence alone
		epoch_val_loss = 0
		for sequence,label in zip(x_train,y_train_binary):
			model.train_on_batch(split_input_for_training(sequence),label.reshape(1,nb_classes))
			
		epoch_val_loss = evaluate_for_epoch(model,x_val,y_val_binary)
		if(epoch_val_loss < min_val_loss): # this is to choose finally the model that yields the best results on the validation set 
			model.save('model.h5')
			min_val_loss= epoch_val_loss

	return y_test_binary

def evaluate_for_epoch(model,x_test,y_test):
	epoch_test_loss = 0 
	for test,label in zip(x_test,y_test):
		loss , acc = model.evaluate(split_input_for_training(test), label.reshape(1,nb_classes), verbose=0)
		epoch_test_loss += loss ############### change if monitor acc instead of loss
	return epoch_test_loss/len(x_test)

def evaluateModel(model,x_test,y_test):
	confusion_matrix_f = pd.DataFrame(np.zeros(shape = (nb_classes,nb_classes)), index = classes, columns = classes ) 

	for test,label in zip(x_test,y_test):
		loss , acc = model.evaluate(split_input_for_training(test), label.reshape(1,nb_classes), verbose=0)
		p = model.predict(split_input_for_training(test), batch_size = 1)
		predicted_integer_label = np.argmax(p).astype(int)
		predicted_label = encoder.inverse_transform(predicted_integer_label)
		correct_label = encoder.inverse_transform(np.argmax(label))
		confusion_matrix[correct_label][predicted_label]+=1.0
		confusion_matrix_f[correct_label][predicted_label]+=1.0

	matrix_f = confusion_matrix_f.as_matrix()
	macro = compute_macro(matrix_f)
	return (macro,compute_micro(matrix_f))

def create_video_feedback(time_series_original,original_binary_class,model,surgery_name,slave_manipulator='Left'):
	path_to_images = path_to_results+'feedback/example-video/images/'
	path_to_video = path_to_results+'feedback/example-video/video/'
	maximum_frames = time_series_original.shape[0]

	# save binary classes for reusing same model and encoder 
	np.save('binary_classes.npy', encoder.classes_)
	
	# generate an image for each frame 
	for i in range(0,maximum_frames): 
		class_activation_map(time_series_original,original_binary_class,model,path_to_images+surgery_name+'__'+str(f'{i:06}')+'.png',max_frame=i+1,slave_manipulator=slave_manipulator,angle=15*6,elev=75+180)
		# angle=15*6,elev=75+180

	# create video from images
	# os.system('ffmpeg -f image2 -r 1/5 -i '+path_to_images+surgery_name+'__%06d.png -vcodec mpeg4 -y '+path_to_video+surgery_name+'_feedback.mp4')
	os.system('ffmpeg -f image2 -framerate 30 -i '+path_to_images+surgery_name+'__%06d.png -s 640x480 '+ path_to_video+surgery_name+slave_manipulator+'.mp4')
	exit() 

def class_activation_map(time_series_original,original_binary_class,model,output_path,max_frame=0,angle=None,slave_manipulator=True,elev=None):
	if max_frame == 0: 
		max_frame=time_series_original.shape[0]

	w_k_c = model.layers[-1].get_weights()[0] # weights for each filter k for each class c 

	new_input_layer = model.inputs # same input of the original model

	new_outpu_layer = [model.get_layer("conv_final").output, model.layers[-1].output] # output is both the original as well as the before last layer 

	new_function = keras.backend.function(new_input_layer,new_outpu_layer)

	new_feed_forward = new_function

	[conv_out, predicted] = new_feed_forward(split_input_for_training(time_series_original))

	# print("original_label: "+str(encoder.inverse_transform(np.argmax(original_binary_class))))
	# print("original_shape: "+str(time_series_original.shape))
	# print("predicted_label:"+str(encoder.inverse_transform(np.argmax(predicted))))
	# print("predicted_shape:"+str(conv_out.shape))

	cas = np.zeros(dtype=np.float, shape = (conv_out.shape[1]))

	conv_out = conv_out[0,:,:]

	# print(np.argmax(original_binary_class))

	for k,w in enumerate(w_k_c[:,np.argmax(original_binary_class)]):
		cas += w * conv_out[:,k]

	minimum = np.min(cas)
	
	cas = cas - minimum

	cas = cas/max(cas)
	cas = cas * 100
	cas = cas.astype(int)

	if slave_manipulator=='Left':# it is the left slave manipulator we want to visualize 
		x_index = 38
		y_index = 39
		z_index = 40
	else: # it is the right slave manipulator we want to visualize 
		x_index = 57
		y_index = 58
		z_index = 59

	max_x_axis = max(time_series_original[:,x_index])
	max_y_axis = max(time_series_original[:,y_index])
	max_z_axis = max(time_series_original[:,z_index])
	min_x_axis = min(time_series_original[:,x_index])
	min_y_axis = min(time_series_original[:,y_index])
	min_z_axis = min(time_series_original[:,z_index])

	x = time_series_original[0:max_frame,x_index]
	y = time_series_original[0:max_frame,y_index]
	z = time_series_original[0:max_frame,z_index]

	cas = cas[0:max_frame]

	# just to have the maximum color 100 and another one with color 0 somwhere in (100,100,100) without appearing on screen
	x=np.concatenate((x,[100],[100]))
	y=np.concatenate((y,[100],[100]))
	z=np.concatenate((z,[100],[100]))
	# here we specify the colors 
	cas = np.concatenate((cas,[100],[0]))

	fig = plt.figure()
	plot3d = fig.add_subplot(111,projection='3d')
	pltmap = plot3d.scatter(x,y,z, c=cas, cmap='jet',s=3)
	ax = plt.gca()
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	ax.view_init(elev=elev, azim=angle)
	ax.set_xlim(min_x_axis,max_x_axis)
	ax.set_ylim(min_y_axis,max_y_axis)
	ax.set_zlim(min_z_axis,max_z_axis)
	# Get rid of the spines
	ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax.set_xlabel('X' )
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	# clbr = plt.colorbar(pltmap)
	# clbr.set_ticks([])
	plt.savefig(output_path)
	plt.close('all')


def generate_class_activation_map_for_all_surgeries(model,fold): 
	
	for surgery_name in mapSurgeryDataBySurgeryName.keys(): 
		# get the time series for this surgery 
		surgery = mapSurgeryDataBySurgeryName[surgery_name]
		# get skill level of the surgeon who performed thi surgery 
		skill = mapExpertiseLevelBySurgeryName[surgery_name]
		# convert string class to binary
		skill = convertStringClassesToBinaryClasses([skill],['N','I','E'],[])[0]
		# print class activation map for this surgery in the pdf file 
		# for every view angle 
		for angle in range(0,60):
			# name the output file
			# output_path = path_to_results+'feedback/'+surgery_type+'/'+str(fold)+'_out/'+surgery_name+'.png'
			output_path = path_to_results+'feedback/img_temp/img_'+str(f'{angle:03}')+'.png'
			# draw 
			class_activation_map(surgery,skill,model,output_path,angle=angle*6)

		# create the gif
		images=[] 
		for subdir,dirs,files in os.walk(path_to_results+'feedback/img_temp/'):
			files.sort()
			for file_name in files:
				images.append(imageio.imread(path_to_results+'feedback/img_temp/'+file_name))
			output_path = path_to_results+'feedback/'+surgery_type+'/'+str(fold)+'_out/'+surgery_name+'.gif'
			kargs = { 'duration': 0.25 }
			imageio.mimsave(output_path,images,'GIF',**kargs)
	return None 


# the sequence variable is the multivariate time series or in this case the surgical task
# we want to split the inputs in order to train  
def split_input_for_training(sequence):
	# get number of hands 
	num_hands= len(input_shapes)
	# get number of dimensions cluster for each hand 
	num_dim_clusters = len(input_shapes[0])
	# define the new input sequence 
	x = []
	# this is used to keep track of the assigned dimensions 
	last = 0
	# loop over each hand 
	for i in range(num_hands):
		# loop for each hand over the cluster of dimensions 
		for j in range(num_dim_clusters): 
			# assign new input same length but different dimensions each time 
			x.append(np.array([sequence[:,last:(last+input_shapes[i][j][1])]]))
			# remember last assigned 
			last= input_shapes[i][j][1]
	# return the new input 
	return x                              

def each_dim_build_model(input_shapes,summary=False, reg=0.00001): 
	# get number of hands 
	num_hands= len(input_shapes)
	# get number of dimensions cluster for each hand 
	num_dim_clusters = len(input_shapes[0])
	# first index for hand second for  dims
	x =[[None for a in range(0,num_dim_clusters)]for b in range(num_hands)] 
	# first conv layer on each dim cluster for each hand 
	conv1 = [[None for a in range(0,num_dim_clusters)]for b in range(num_hands)] 
	# merged layers for each hand 
	hand_layers =[None for a in range(num_hands)]
	# second conv layer on concatenated conv1 for each hand
	conv2 = [None for a in range(num_hands)] 
	# loop over each hand 
	for i in range(0,num_hands): 
		# loop for each hand over the dimension (or channels) clusters 
		for j in range(0,num_dim_clusters): 
			# input layer for each dimension cluster for each hand 
			x[i][j]=keras.layers.Input(input_shapes[i][j])
			# first conv layer over the clustered dimensions or channels in terms of keras 
			conv1[i][j] = keras.layers.Conv1D(8,kernel_size=3,strides=1,padding='same', activity_regularizer=regularizers.l2(reg))(x[i][j])
			conv1[i][j] = keras.layers.Activation('relu')(conv1[i][j])
		# concatenate convolutions of first layer over the channels dimension for each hand 
		hand_layers[i]=keras.layers.Concatenate(axis=-1)(conv1[i])
		# do a second convolution over features extracted from the first convolution over each hand 
		conv2[i] = keras.layers.Conv1D(16,kernel_size=3, strides=1, padding='same', activity_regularizer=regularizers.l2(reg))(hand_layers[i])
		conv2[i] = keras.layers.Activation('relu')(conv2[i])
	# concatenate the features of the two hands 
	final_input = keras.layers.Concatenate(axis=-1)(conv2) 
	# do a final convolution over the features concatenated for all hands 
	conv3 = keras.layers.Conv1D(32,kernel_size=3,strides=1,padding='same', activity_regularizer = regularizers.l2(reg))(final_input)
	conv3 = keras.layers.Activation('relu', name = "conv_final")(conv3)
	# do a globla average pooling of the final convolution 
	pooled = keras.layers.GlobalAveragePooling1D()(conv3)
	# add the final softmax classifier layer 
	out = keras.layers.Dense(nb_classes,activation='softmax')(pooled)
	# create the model and link input to output 
	model = Model(inputs=list(chain.from_iterable(x)),outputs=out)
	# show summary if specified 
	if summary==True : 
		model.summary()

	# choose the optimizer 
	optimizer = keras.optimizers.Adam()
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

#time 
start_time = time.time()

# Global parameters 
root_dir = os.getcwd()+'/../'+'/JIGSAWS/'
path_to_configurations = os.getcwd()+'/../'+'/JIGSAWS/Experimental_setup/'
path_to_results = os.getcwd()+'/../'+'/temp/'
nb_epochs = 1000
surgery_type = 'Suturing'
dimensions_to_use = range(0,76)
number_of_dimensions= len(dimensions_to_use)
input_shape = (None,number_of_dimensions) # input is used to specify the value of the second dimension (number of variables) 
input_shapes = [[(None,3),(None,9),(None,3),(None,3),(None,1)],[(None,3),(None,9),(None,3),(None,3),(None,1)],[(None,3),(None,9),(None,3),(None,3),(None,1)],[(None,3),(None,9),(None,3),(None,3),(None,1)]]
# for each manipulator   x,y,z  ,rot matrx, x'y'z' , a'b'g' , angle  , ... same for the second manipulator ...   

mapSurgeryDataBySurgeryName = collections.OrderedDict() # indexes surgery data (76 dimensions) by surgery name 
mapExpertiseLevelBySurgeryName = collections.OrderedDict() # indexes exerptise level by surgery name  
classes = ['N','I','E']
nb_classes = len(classes)
confusion_matrix = pd.DataFrame(np.zeros(shape = (nb_classes,nb_classes)), index = classes, columns = classes ) # matrix used to calculate the JIGSAWS evaluation
encoder = LabelEncoder() # used to transform labels into binary one hot vectors 

surgeries_metadata = getMetaDataForSurgeries(surgery_type)

generateMaps(surgery_type)
print('Number of different surgeries in total: '+str(len(mapSurgeryDataBySurgeryName)))

# comment then uncommment the commented lines if you want to load a pre-trained model 

validation(surgery_type,reg = 0.00001,summary=False,max_itr=40)

# model = load_model(path_to_results+'feedback/example-figure-in-paper/model.h5')

# encoder.classes_ = np.load(path_to_results+'feedback/example-figure-in-paper/binary_classes.npy')

# create_video_feedback(mapSurgeryDataBySurgeryName['Suturing_H004'],convertStringClassesToBinaryClasses(['N'],['N','I','E'],[])[0],model,'Suturing_H004',slave_manipulator='_Left_mod')

# uncomment to visualize the trajectory illustrated in the paper
# class_activation_map(mapSurgeryDataBySurgeryName['Suturing_H004'],convertStringClassesToBinaryClasses(['N'],['N','I','E'],[])[0],model,'Suturing_H004.pdf')

print(confusion_matrix)

print("--- %s seconds ---" % (time.time() - start_time))

print("End!")