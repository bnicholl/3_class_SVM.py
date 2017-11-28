#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:59:55 2017

@author: bennicholl
"""
# packages for analyses
import pandas as pd
import numpy as np

# packages for visuals and graphs 
import matplotlib.pyplot as plt
from sklearn.svm import SVC



"""just type 'df_' in order to see the data in spreadsheet form"""

"""upload our iris dataset"""
df_ = pd.read_csv('/Users/bennicholl/Desktop/iris.csv', names = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'],  engine =  'python')
"""the csv I'm upoading from prints the names as the 0'th value, so I just drop them here""" 
df_ = df_.drop([0])


class data():

        
    """intialize our 3 species dependant datasets, they will be put into a 3D array"""
    def __init__(self):
        """these are our 3 different species that we will be analyzing. here we are simply sperating each species,
        and then dropping the species name from the pandas dataframe so we can get rid off string values"""
        sertosa_df = df_[df_['Species'] == 'Iris-setosa']
        sertosa_df = sertosa_df.drop(['Species'], 1)
        
        versicolor_df = df_[df_['Species'] == 'Iris-versicolor']
        versicolor_df = versicolor_df.drop(['Species'], 1)
        
        virginica_df = df_[df_['Species'] == 'Iris-virginica']
        virginica_df = virginica_df.drop(['Species'], 1)
        
        """these 3 lines turn all 3 datasets into np arrays, and uses astype method to ensure they are floats"""
        self.sertosa_df = np.array(sertosa_df.values.astype(np.float)  )
        self.versicolor_df = np.array(versicolor_df.values.astype(np.float))
        self.virginica_df = np.array(virginica_df.values.astype(np.float)  ) 
        #return np.array([sertosa_df, versicolor_df, virginica_df])
    
        self.scatterplot()
        
    """create a scatter plot of the 3 different species"""
    def scatterplot(self):
        """creates empty plot"""
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        
        """these three lines put put each species sepal length on x1 axis, and sepal width on x2 axis"""
        """represented as blue diamonds"""
        self.ax1.scatter(self.sertosa_df.T[1], self.sertosa_df.T[2], s = 40, c='b', marker = 'D', label='first')
        """represented as red circles"""
        self.ax1.scatter(self.versicolor_df.T[1], self.versicolor_df.T[2], s = 40, c='r', marker = 'o', label='second')  
        """represented as black x's"""
        self.ax1.scatter(self.virginica_df.T[1], self.virginica_df.T[2], s = 40, c='k', marker = 'x', label='third')
        
        plt.legend(loc='upper left');
        plt.show()
        
        self.prepare_for_svm()
    """prepares data and classes"""
    def prepare_for_svm(self):          
        self.sertosa = [] 
        self.versicolor = []
        self.virginica = []
        
        i = 0
        """zip the sepal length and sepal width together, puting them in 2D vecotr spaces""" 
        """ex.  x1 = [5.1, 3.5].   where 5.1 = sepal length, and 3.5 = sepal width"""
        while i < len(self.sertosa_df.T[1]):
            self.sertosa.append([self.sertosa_df.T[1][i], self.sertosa_df.T[2][i]]) # zip([self.sertosa_df.T[1][i], self.sertosa_df.T[2][i]])  
            self.versicolor.append([self.versicolor_df.T[1][i], self.versicolor_df.T[2][i]]) # zip([self.versicolor_df.T[1][i], self.versicolor_df.T[2][i]]) 
            self.virginica.append([self.virginica_df.T[1][i], self.virginica_df.T[2][i]]) # zip([self.virginica_df.T[1][i], self.virginica_df.T[2][i]])
            i += 1
        """this class represents the species of sertosa"""    
        self.class1 = [0] * 50
        """this class represents the species of versicolor""" 
        self.class2 = [1] * 50
        """this class represents the species of virginica""" 
        self.class3 = [2] * 50
    
     
    """Run this method to get the 3 classes seperated via hyperplane"""
    """type 'linear' for linear function, 'poly' for polynomial function, or 'rbf' for rbf funtion"""
    def perform_svm(self, kernel = 'linear'):
        """creates svm with linear function"""
        lin_svm = SVC(kernel = kernel, C=1, random_state=0)
        """fits three of our species with there respective classes"""
        x = self.sertosa + self.versicolor + self.virginica
        z = self.class1 + self.class2 + self.class3
        """fit or train our data"""
        """to predict a class, type   'a.svm.predict([[2, 2]])'   into console"""
        lin_svm.fit(x,z)        
        
        """begin getting our contoured graph ready"""
        """get the max and min of our x1 and x2 directions"""
        min_x1, max_x1 = self.ax1.get_xlim() 
        min_x2, max_x2 = self.ax1.get_ylim()         

        X1 = np.linspace(min_x1, max_x1, 20)
        X2 = np.linspace(min_x2, max_x2, 20)
        
        """create a meshgird of our X1 and X2 coordinates"""
        x1mesh, x2mesh = np.meshgrid(X1, X2)
        
        """ravel method turns array of arrays into one long, single array"""
        """vstack turns two seperate arrays into 1""" 
        x1_x2 = np.vstack([x1mesh.ravel(), x2mesh.ravel()] )
        
        """transpose in order to turn the x1 and x2 arrays into x1,x2 pairs"""
        x1_x2 = x1_x2.T
        
        # I'm going to need to rehape this back into a the same dimension as the acutal plot
        """predict whether all of coordinate pairs are a 0, 1, or a 2"""
        self.class_values = lin_svm.predict(x1_x2)
        
        self.class_values = self.class_values.reshape(x1mesh.shape)
        """create empty plot"""
        self.figure = plt.figure()
        """gets current axis"""
        self.axes = self.figure.gca()
        """plot the class_values based on x1mesh and x2mesh"""
        self.axes.contour(x1mesh, x2mesh, self.class_values, levels = [0,1,2])
        """plot the scatters for each species"""
        self.axes.scatter(self.sertosa_df.T[1], self.sertosa_df.T[2], s = 40, c='b', marker = 'D', label='first')
        self.axes.scatter(self.versicolor_df.T[1], self.versicolor_df.T[2], s = 40, c='r', marker = 'o', label='first')
        self.axes.scatter(self.virginica_df.T[1], self.virginica_df.T[2], s = 40, c='k', marker = 'x', label='third')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 

    





       

        




