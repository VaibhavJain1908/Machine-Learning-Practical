""" Predicting the Likelihood of E-Signing
    a loan Based on Financial History """
    
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

dataset = pd.read_csv("P39-Financial-Data.csv")

# EDA 
dataset.head()
dataset.describe()
dataset.columns

# Cleaning the data

""" Removing the nan """
dataset.isna().any() # No na's

# Histograms
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize = (15, 12))
plt.suptitle("Histograms of Numerical Columns", fontsize = 20)
for i in range(1, dataset2.shape[1]+ 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i - 1], bins = vals, color = '#3F5D7D')
plt.tight_layout(rect = [0, 0.03, 1, 0.95])

# Correlation with Response Variable (Note: Modelslike RF are not linear like these)
dataset2.corrwith(dataset.e_signed).plot.bar(
    figsize = (20, 10), title = "Correlation with Esigned",
    fontsize = 15, rot = 45, grid = True)

# Correlation Matrix
sn.set(style = "white", font_scale = 2)

""" compute the correlation matrix """
corr = dataset.corr()

""" gnerate a maskfor the upper triangle """
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

""" set up matplotlib figure """
f, ax = plt.subplots(figsize = (18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

""" generate a custom diverging colormap """
cmap = sn.diverging_palette(220, 10, as_cmap = True)

""" Draw the heatmap with the mask and correct aspect ratio """
sn.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,
           square = True, linewidths = .5, cbar_kws = {"shrink": .5})



