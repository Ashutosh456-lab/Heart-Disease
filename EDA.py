# Plotting Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import cufflinks as cf
# %matplotlib inline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
# Metrics for Classification technique
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Scaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import  RandomizedSearchCV, train_test_split

from sklearn.metrics import accuracy_score, recall_score

# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import pickle
#################################################################################################################
##################################   Exploratory Data Analysis(EDA)    ###########################################
#################################################################################################################


#Load Data

df= pd.read_excel("Heart_df.xlsx")
print(df)
print(df.info())
print(df.describe())
df = df.drop_duplicates(ignore_index=True)
print(df.info())
print(df.shape)

##################################   Univariate Analysis    ################################################

#categorical features:

class Univariate_EDA_Categorical:
    def __init__(self, df):
        self.df = df

    def feature_age(self):
        
        young = self.df[(self.df.age <= 35)]
        middle = self.df[(self.df.age > 35) & (self.df.age < 50)]
        elder = self.df[(self.df.age >= 50)]
        explode = [0, 0, 0.1]
        plt.figure(figsize=(10, 8))
        sns.set_context('notebook', font_scale=1.2)
        plt.pie([len(young), len(middle), len(elder)], 
                labels=['young ages', 'middle ages', 'elderly ages'], 
                explode=explode, 
                autopct='%1.1f%%')
        plt.tight_layout()
        
    def feature_sex(self):
        plt.figure(figsize=(10, 8))       
        target = [len(self.df[self.df['sex'] == 0]), len(self.df[self.df['sex'] == 1])]
        labels = ["Female", "Male"]
        plt.pie(x=target, labels=labels, autopct='%1.2f%%')
        plt.title("Gender Values Distribution")
        plt.show()
        
        
    def feature_chest_pain(self):
        plt.figure(figsize=(10, 8))
        value_counts = self.df['cp'].value_counts()
        categories = value_counts.index.tolist()
        counts = value_counts.tolist()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=categories, autopct='%1.1f%%', shadow=True)
        ax.axis('equal')
        plt.xlabel("Chest Pain Severity Type")
        plt.ylabel("Count")
        plt.title("Chest Pain Severity Pie Chart")
        plt.show()
        
        
    def fasting_blood_sugar(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10, 8))
        target = [len(self.df[self.df['fbs'] == 0]), len(self.df[self.df['fbs'] == 1])]
        labels = ["Less than 120 mg/dl", "Greater than 120 mg/dl"]
        plt.pie(x=target, labels=labels, autopct='%1.2f%%', explode=[0, 0.1])
        plt.title("FBS Values Distribution")
        plt.show()
        
        
    def resting_ecg(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10, 8))
        y = self.df['restecg'].value_counts()
        mylabels = ["left ventricular hypertrophy", "normal", "ST-T wave abnormality"]
        myexplode = [0, 0, 0.4]
        plt.pie(y, labels = mylabels, autopct='%1.2f%%', explode = myexplode, shadow = True)
        plt.show()     
                     
    def exr_agina(self):
        plt.figure(figsize=(10, 8))
        # Get the count of each category in the 'slope' column
        counts = self.df['slope'].value_counts()
        # Plot the pie chart
        plt.pie(counts, labels=counts.index, startangle=90, autopct='%1.1f%%', shadow=True)
        plt.axis('equal')  # Set aspect ratio to be equal so that pie is drawn as a circle
        plt.title("Slope of Peak Exercise")
        plt.show()
        
    def ca(self):
        plt.figure(figsize=(10, 8))
        # Get the count of each category in the 'ca' column
        counts = self.df['ca'].value_counts()
        # Plot the pie chart
        plt.pie(counts, labels=counts.index, startangle=90, autopct='%1.1f%%', shadow=True)
        plt.axis('equal')  # Set aspect ratio to be equal so that pie is drawn as a circle
        plt.title("Number of Major Vessels colored by flouropsy")
        plt.show()
        
    def thal(self):
        plt.figure(figsize=(10, 8))
        labels = self.df['thal'].value_counts().index
        values = self.df['thal'].value_counts().values
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title("Type of Defect count")
        plt.show()
        
    def slope(self):
        plt.figure(figsize=(10, 8))

        # Get the count of each category in the 'slope' column
        counts = self.df['slope'].value_counts()

        # Plot the pie chart
        plt.pie(counts, labels=counts.index, startangle=90, autopct='%1.1f%%', shadow=True)

        plt.axis('equal')  # Set aspect ratio to be equal so that pie is drawn as a circle
        plt.title("Slope of Peak Exercise")
        plt.show()
        

        
    def target(self):
        # Pie Chart
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(10, 8))
        target = [len(self.df[self.df['target'] == 0]), len(self.df[self.df['target'] == 1])]
        labels = ["Less Chances", "High Chances"]
        plt.pie(x=target, labels=labels, autopct='%1.2f%%', explode=[0, 0])
        plt.title("Chances of Heart Attack - Target")
        plt.show()
        

        
uni_eda=Univariate_EDA_Categorical(df)
uni_eda.feature_age()
uni_eda.feature_sex()
uni_eda.feature_chest_pain()
uni_eda.fasting_blood_sugar()
uni_eda.resting_ecg()
uni_eda.exr_agina()
uni_eda.ca()
uni_eda.thal()
uni_eda.slope()
uni_eda.target()

# Continusous Features:

class FeatureDistributionVisualization:
    def __init__(self, df):
        self.df = df
    
    def visualize(self, feature):
        sns.distplot(self.df[feature], color='red')
        plt.title(f"{feature} [ \u03BC : {self.df[feature].mean():.2f} | \u03C3 : {self.df[feature].std():.2f} ]")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()
visualizer = FeatureDistributionVisualization(df)
visualizer.visualize("oldpeak")
visualizer.visualize("chol")
visualizer.visualize("trestbps")
visualizer.visualize("thalach")
visualizer.visualize("age")


##################################   Bivariate Analysis    ################################################

class DataVisualization:
    def __init__(self, df, title, x_col, hue_col, palette, figsize=(10,5)):
        self.df = df
        self.title = title
        self.x_col = x_col
        self.hue_col = hue_col
        self.palette = palette
        self.figsize = figsize
        
    def plot_countplot(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = sns.countplot(x=self.x_col, hue=self.hue_col, data=self.df, palette=self.palette)
        ax.set_title(self.title, fontsize=13, weight='bold')

        totals = []
        for i in ax.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in ax.patches:
            ax.text(i.get_x() + .03, i.get_height() - 5,
                    str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14,
                    weight='bold')
        plt.tight_layout()
        plt.show()

def main():
    
    chest_pain_distribution = DataVisualization(df, "Chest Pain Distribution according to Target", 'cp', 'target', 'Set2', (10,5))
    chest_pain_distribution.plot_countplot()

    sex_distribution = DataVisualization(df, "Sex Distribution according to Target", 'sex', 'target', 'Set2', (10,5))
    sex_distribution.plot_countplot()

    sex_distribution = DataVisualization(df, "fasting_blood_sugar Distribution according to Target", 'fbs', 'target', 'Set2', (10,5))
    sex_distribution.plot_countplot()

    sex_distribution = DataVisualization(df, "Resting Electrocardiographic Results Distribution according to Target", 'restecg', 'target', 'Set2', (10,5))
    sex_distribution.plot_countplot()

    sex_distribution = DataVisualization(df, "Exercise Induced Angina Distribution according to Target", 'exang', 'target', 'Set2', (10,5))
    sex_distribution.plot_countplot()
    

if __name__ == '__main__':
    main()


##################################   Multivariate Analysis    ################################################

class Relplot:
    def __init__(self, data, x_col, y_col, hue_col, size_col, style_col):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.hue_col = hue_col
        self.size_col = size_col
        self.style_col = style_col
        self.title = 'Relationship Plot'
        self.xlabel = x_col
        self.ylabel = y_col
        
    def plot(self):
        sns.set_style('darkgrid')
        sns.relplot(x=self.x_col, y=self.y_col, hue=self.hue_col, size=self.size_col, style=self.style_col, data=self.data)
        plt.title(self.title, fontsize=15, weight='bold')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()


df=df
relplot1 = Relplot(df, x_col='chol', y_col='thalach', hue_col='sex', size_col='target', style_col='sex')
relplot1.plot()

relplot2 = Relplot(df, x_col='cp', y_col='thalach', hue_col='sex', size_col='target', style_col='sex')
relplot2.plot()

relplot3 = Relplot(df, x_col='trestbps', y_col='thalach', hue_col='sex', size_col='target', style_col='sex')
relplot3.plot()

relplot4 = Relplot(df, x_col='cp', y_col='age', hue_col='sex', size_col='target', style_col='sex')
relplot4.plot()

##################################    Pair Plot     ####################################################

# define continuous variable & plot
continous_features = ['age', 'chol', 'thalach', 'oldpeak','trestbps']  
sns.pairplot(df[continous_features + ['target']], hue='target')

"""
        1.oldpeak having a linear separation relation between disease and non-disease.
        2.thalach having a mild separation relation between disease and non-disease.
        3.Other features don’t form any clear separation

        """





def plot_box_plots(dataframe, features):
    fig, ax = plt.subplots(nrows=1, ncols=len(features), figsize=(20, 8))   
    for i, feature in enumerate(features):
        ax[i].boxplot(dataframe[feature])
        ax[i].set_title(feature)     
    plt.show()
# list of features to plot
features = ['trestbps', 'chol', 'thalach','oldpeak','age']
# plot the box plots
plot_box_plots(df, features)

# Boxplot Viualtaion scomments (*********)
# Quantative Analysis

class OutlierDetector:
    def __init__(self, dataframe, columns):
        self.dataframe = dataframe
        self.columns = columns
        
    def detect(self):
        outliers = {}
        for col in self.columns:
            feature_outliers = []
            mean = np.mean(self.dataframe[col])
            std = np.std(self.dataframe[col])
            for i in range(len(self.dataframe)):
                z_score = (self.dataframe[col][i] - mean) / std
                if z_score >= 3 or z_score <= -3:
                    feature_outliers.append(i)
            outliers[col] = feature_outliers
        return outliers
columns=['trestbps', 'chol', 'thalach','oldpeak']

outlier_detector = OutlierDetector(df, columns)
outliers = outlier_detector.detect()
outliers


#################################################################################################################
##################################    Feature Selection     #####################################################
#################################################################################################################

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
df.corr()



def calculate_vif(df, target_col=None):
    """
    Calculates the variance inflation factor (VIF) for all the features in a given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame for which to calculate the VIF.
    target_col (str): The name of the column that should be excluded from the VIF calculation.
    
    Returns:
    vif_df (pandas.DataFrame): A DataFrame containing the VIF for each feature.
    """
    if target_col is not None:
        df = df.drop(target_col, axis=1)
    
    vif_list = []
    
    for i in range(df.shape[1]):
        vif = variance_inflation_factor(df.values, i)
        vif_list.append(vif)
    
    vif_df = pd.DataFrame({
        'Features': df.columns,
        'VIF': vif_list
    })
    
    return vif_df


vif_df = calculate_vif(df, target_col='target')

print(vif_df)





#################################################################################################################
###############################          Train Test Split          #############################################
#################################################################################################################

x= df.drop('target', axis=1)
y=df['target']


X_train,X_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.2,stratify=y)

with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

