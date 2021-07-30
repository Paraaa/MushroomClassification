import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import dataframe_image as dfi
from seaborn.relational import lineplot


"""
This class is used to visualize data.
@author: Andrej Schwanke
"""
class Plotter:
    
    """
    Creates a directory to save plots to
    
    Optional parameters
    ----------
    save_plots: if set to true every plot made is saved in 
    path: this value specifies the path the plots are stored
    format: specifies the image format the plots are stored with
    
    @author: Andrej Schwanke
    """
    def __init__(self, save_plots=False, path="./plots/", format="svg"):
        sns.set_style("darkgrid")
        self.save_plots = save_plots
        self.path = path
        self.format = format
        


        if self.save_plots: 
            try: 
                os.mkdir(self.path) 
            except FileExistsError:
                print("Directory", self.path, " already exists")

      
    """
    Create pairplot of given data
    @author: Andrej Schwanke
    """
    def pairplot(self, data, title=None):
        plt.figure()
        plot = sns.pairplot(data)
        if title:
            plot.fig.subplots_adjust(top=0.9)
            plot.fig.suptitle(title)

        if self.save_plots: 
            self.save_plot(plot, "pairplot")

       
    """
    Create scatterplot of given data
    @author: Andrej Schwanke
    """
    def scatterplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.scatterplot(x,y,data=data,legend="full") 
        if title:
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "scatterplot")

    

    """
    Create lineplot of given data
    @author: Andrej Schwanke
    """
    def lineplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.lineplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "lineplot")


    """
    Create lineplot of given data
    @author: Andrej Schwanke
    """
    def catplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.catplot(x,y, data=data)
        if title: 
            plot.fig.subplots_adjust(top=0.9)
            plot.fig.suptitle(title)

        if self.save_plots:
            self.save_plot(plot, "catplot")


    """
    Create stripplot of given data
    @author: Andrej Schwanke
    """
    def stripplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.stripplot(x,y, data=data, jitter = 0.2)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "stripplot")


    """
    Create swarmplot of given data
    @author: Andrej Schwanke
    """
    def swarmplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.swarmplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "swarmplot")


    """
    Create violinplot of given data
    @author: Andrej Schwanke
    """
    def violinplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.violinplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "violinplot")


    """
    Create boxplot of given data
    @author: Andrej Schwanke
    """
    def boxplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.boxplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "boxplot")



    """
    Create boxenplot of given data
    @author: Andrej Schwanke
    """
    def boxenplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.boxenplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "boxenplot")


    """
    Create countplot of given data
    @author: Andrej Schwanke
    """
    def countplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.countplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "countplot")


    """
    Create barplot of given data
    @author: Andrej Schwanke
    """
    def barplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.barplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "barplot")



    """
    Create barplot of given data
    @author: Andrej Schwanke
    """
    def pointplot(self, data, x=None, y=None, title=None):
        plt.figure()
        plot = sns.pointplot(x,y, data=data)
        if title: 
            plot.set_title(title)

        if self.save_plots:
            self.save_plot(plot.get_figure(), "pointplot")

    """
    Create lmplot of given data
    (plots a linear model)
    @author: Andrej Schwanke
    """
    def lmplot(self, data, x=None, y=None, title=None): 
        plot = sns.lmplot(x, y, data=data)
        if title:
            plot.fig.subplots_adjust(top=0.9)
            plot.fig.suptitle(title)

        if self.save_plots: 
            self.save_plot(plot,"lmplot")


    """
    Creates plots of all features on the x axes 
    and the label on the y axes.
    It uses an lmplot to fit a linear model in the data.
    Therefor only columns of dtype: int64, int32 or float64
    can be used to generate the plots.
    @author: Andrej Schwanke
    """
    def regression_plot(self, data, label):        
        for column in data.columns:
            if((data[column].dtype == np.int64) or 
               (data[column].dtype == np.int32) or
               (data[column].dtype == np.float64)):
                self.lmplot(data, x=column, y = label, title=f"{label}:{column}")

    """
    Counts the given unique values in the dataframe.
    and plots them into a countplot.
    @author: Andrej Schwanke    
    """
    def count_unique_plot(self, data):
        for column in data.columns:
            self.countplot(data,x=column)

    """
    Creates a correlation heatmap based on pased data.
    @author: Andrej Schwanke
    """
    def correlation_heatmap(self, data, title=None):
        plt.figure()
        plot = sns.heatmap(data.corr(),annot=True)
        if title:
            plot.set_title(title)
        if self.save_plots:
            self.save_plot(plot.get_figure(), "correlation-heatmap")


    """
    Creates a boxplot for each column in the dataset
    @author: Andrej Schwanke
    """
    def show_outliers(self, data, title=None):
        for column in data.columns:
            self.boxplot(data, x=column)

    """
    This method provide a oppertunity to save a pandas dataframe
    as an image. It uses the dataframe_image modul to do so. 
    Currently it is only possible to export the dataframe as an .png
    file.
    This method can only be called if save_plots is enabled. 
    The optional parameter sample specifies the amount of sample data
    should be visualized in the image file. A high sample size is not
    recommended.
    @author: Andrej Schwanke
    """
    def export_table_as_png(self, data, sample=25):
        if self.save_plots:
            data_styled = data.head(sample).style.background_gradient()
            self.save_plot(data_styled, "table")

    """
    Saves a plot in a given format 
    @author: Andrej Schwanke
    """
    def save_plot(self, plot, plot_type):
        d = datetime.now()
        timeStamp = f"{d.day}_{d.month}_{d.year}-{d.hour}:{d.minute}:{d.second}" 
        filename  = f"{plot_type}-{timeStamp}"  
        path      = f"{self.path}{filename}.{self.format}" 

        if plot_type == "table":
            path = f"{self.path}{filename}.png"
            dfi.export(plot, path)
        else: 
            plot.savefig(path) 



    """
    Method to remove outliers with IQR
    Source: https://www.kaggle.com/abdullaahmed/house-price-prediction-linear-regression
    """
    def remove_outliers(df, x, lower_bound=25, upper_bound=75):
        q25, q75 = np.percentile(df[x], lower_bound), np.percentile(df[x], upper_bound)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = 1 ,  (q75 + cut_off)
        df = df[(df[x] < upper) & (df[x] > lower)]
        return df


    def show(self):
        plt.tight_layout()
        plt.show()

