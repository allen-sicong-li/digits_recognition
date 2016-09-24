import scipy.io as sio
import numpy as np
from PIL import Image
import copy

raw_data = sio.loadmat(r'digitsData.mat')

# a little code to visualize the picture of number

# a = np.array(raw_data['X'][0].T)
# pic = Image.fromarray(a)
# pic.show()

# input to training set and test set
training_set = np.array(raw_data['X'][0:400])
test_set = np.array(raw_data['X'][400:500])
training_y = np.array([[1]+[0]*9] * 400 + [[0]*10]*3600)
test_y = np.array([[1]+[0]*9] * 100 + [[0]*10]*900)


for i in range(1,10,1):
	training_set = np.vstack((training_set,np.array(raw_data['X'][i*500:i*500+400])))
	test_set = np.vstack((test_set,np.array(raw_data['X'][i*500+400:i*500+500])))

	for j in range(i*400,i*400+400,1):
		training_y[j][i] = 1

	for j in range(i*100,i*100+100,1):
		test_y[j][i] = 1

# initialize the matrix theta
epsilon = 0.5
theta1 = epsilon * np.random.uniform(-1,1,(25,401))
theta2 = epsilon * np.random.uniform(-1,1,(10,26))

# sigmoid function
def sigmoid(value):
	temp = 1 + np.exp((-1) * value)
	return 1/temp

# given the y with 0s and 1s and matrix with the value computed by
# sigmoid function, compute the entire cost.
def cost_func(matrix,y,theta1,theta2,pen):
	'the input matrix should be a 4000*10 matrix, a matrix with the sigmoid value. y should also be a 4000*10 matrix, all 0s and 1s.'
	
	# will compute the summation over all elements in (temp1+temp2)
	negY = 1-y
	negVal = np.log(1-matrix)
	temp1 = y*np.log(matrix)
	temp2 = negY*negVal

	sum1 = sum(sum(theta1**2))
	sum2 = sum(sum(theta2**2))
	temp3 = pen*(sum1+sum2)/(2*len(matrix))  # the regularization term
	return (-1)*sum(sum(temp1+temp2))/len(matrix) + temp3

# given theta1 and theta2, compute the final output (sigmoid value). (map from input to output)
def forward_prop(theta1,theta2,inputs):
	'theta1 is the first layer matrix. theta2 is the second layer matrix. inputs is the matrix of all m inputs'

	# add extra constant term
	ones = np.array([1]*len(inputs))
	inputs = np.insert(inputs, 0, values=ones, axis=1)
	inputs = inputs.T
	a2 = sigmoid(np.dot(theta1,inputs)) # a2: 25*4000
	a2 = np.insert(a2,0,values=ones,axis=0) # a2: 26*4000
	output = sigmoid(np.dot(theta2,a2)) # output: 10*4000
	return (a2,output)

# compute the partial derivative matrix of the second layer.
def partial_deri_2(a2,output,y,theta2,pen):
	'output should be the 10*4000 sigmoided values and y should be 10*4000 0s and 1s. a2 should be 26*4000 matrix.'
	temp = y-output
	temp2 = theta2*pen/len(output[0])  # regularization term
	return (-1)*np.dot(temp,a2.T)/len(output[0]) + temp2

# compute the partial derivative matrix of the first layer
def partial_deri_1(a2,output,y,inputs,theta2,theta1,pen):
	'the a2 should be a 26*4000 sigmoided value matrix. y should be 10*4000 0s and 1s. output should be 10*4000 sigmoided value.'
	
	#delete the extra const term
	theta2 = np.delete(theta2,0,axis = 1)
	a2 = np.delete(a2,0,axis = 0)
	temp1 = y - output
	temp2 = np.dot(theta2.T,temp1)
	temp3 = a2*(1-a2)*temp2
	ones = np.array([1]*len(inputs))
	inputs = np.insert(inputs, 0, values=ones, axis=1)
	temp4 = pen*theta1/len(inputs)
	return (-1)*np.dot(temp3,inputs)/len(inputs) + temp4

# using gradient descent to update theta1 and theta2
def gradient_descent(times,learn_rate,theta1,theta2,y,inputs,pen):
	a2, output = forward_prop(theta1,theta2,inputs)
	for i in range(times):
		
		theta1Change = partial_deri_1(a2,output,y.T,inputs,theta2,theta1,pen)
		theta2Change = partial_deri_2(a2,output,y.T,theta2,pen)
		theta1 = theta1 - learn_rate*theta1Change
		theta2 = theta2 - learn_rate*theta2Change
		a2, output = forward_prop(theta1,theta2,inputs)
		print('The current cost is:', cost_func(output.T,y,theta1,theta2,pen))

	return (theta1,theta2)

# set the penalty coefficient
pen = 5

# get the updated theta1 and theta2
new_theta1,new_theta2 = gradient_descent(10000,0.1,theta1,theta2,training_y,training_set,pen)

# use the trained theta1 and theta2 to test the test set.
test_a2, test_output = forward_prop(new_theta1,new_theta2,test_set)
a = [i.argmax() for i in test_output.T]

count = 0
for i in range(len(a)):
	if a[i] == int(i/100):
		count += 1

# compute the rate of correctness
print("The correctness on the test set is:",count/len(a))




# the following codes are used to check if the implementation of the computation of
# the two partial derivative matrix is correct.

# a2, output = forward_prop(theta1,theta2,training_set)
# part2 = partial_deri_2(a2,output,training_y.T,theta2,pen)
# print(part2)
# print('\n')

# part1 = partial_deri_1(a2,output,training_y.T,training_set,theta2,theta1,pen)
# print(part1)
# print('\n')

# epsilon2 = 0.00001

# part_de2 = []
# for i in range(len(theta2)):
# 	temp = []
# 	for j in range(len(theta2[0])):
# 		theta2temp = copy.deepcopy(theta2)
# 		theta2temp[i][j] += epsilon2
# 		a1, output = forward_prop(theta1,theta2temp,training_set)
# 		val1 = cost_func(output.T,training_y,theta1,theta2temp,pen)
# 		theta2temp[i][j] -= 2*epsilon2
# 		a1, output = forward_prop(theta1,theta2temp,training_set)
# 		val2 = cost_func(output.T,training_y,theta1,theta2temp,pen)
# 		der = (val1-val2)/(2*epsilon2)
# 		temp.append(der)
# 	part_de2.append(temp)
# part_de2 = np.array(part_de2)
# print(part_de2)
# print('\n')

# print(part_de2-part2)

# part_de1 = []
# for i in range(len(theta1)):
# 	temp = []
# 	for j in range(len(theta1[0])):
# 		theta1temp = copy.deepcopy(theta1)
# 		theta1temp[i][j] += epsilon2
# 		a1, output = forward_prop(theta1temp,theta2,training_set)
# 		val1 = cost_func(output.T,training_y,theta1temp,theta2,pen)
# 		theta1temp[i][j] -= 2*epsilon2
# 		a1, output = forward_prop(theta1temp,theta2,training_set)
# 		val2 = cost_func(output.T,training_y,theta1temp,theta2,pen)
# 		der = (val1-val2)/(2*epsilon2)
# 		temp.append(der)
# 	part_de1.append(temp)
# part_de1 = np.array(part_de1)
# print(part_de1)
# print('\n')

# print(part_de1-part1)
