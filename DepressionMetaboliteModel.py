# Metabolomic Modeling of Depression in Pre- and Post-Menopausal Women
# Rosalie Wang, Mackenzie Enns, Noor Al-Rajab
# IBEHS 4QZ3 - Course Analysis Project
# 5 December 2024

#-----------------------------------------------------------------#

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.cross_decomposition import PLSRegression
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score


#-----------------------------------------------------------------#


# Load the meta and metabolite data from their CSV files
    #Loading from Rosie's computer
metaDataPath = r'C:\Users\rosie\OneDrive\Desktop\academics\IBEHS 4QZ3\Analysis Project\MetaData.csv'
metaboliteDataPath = r'C:\Users\rosie\OneDrive\Desktop\academics\IBEHS 4QZ3\Analysis Project\MetaboliteData.csv'

metaData = pd.read_csv(metaDataPath)
metaboliteData = pd.read_csv(metaboliteDataPath,header=None)

    #Loading from Mackie's computer
#metaDataPath = r'C:\Users\macke\OneDrive\Documents\MACKENZIE SCHOOL STUFF\Fall 2024\IBEHS 4QZ3\Project\MetaData.csv'
#metaboliteDataPath = r'C:\Users\macke\OneDrive\Documents\MACKENZIE SCHOOL STUFF\Fall 2024\IBEHS 4QZ3\Project\MetaboliteData.csv'



# Determine which columns correspond to each metabolite type (MP, ML, LC)
allSampleIDs = metaboliteData.iloc[0,1:]
MPCols = [index for index, value in enumerate(allSampleIDs) if '_MP_' in value]
MLCols = [index for index, value in enumerate(allSampleIDs) if '_ML_' in value]
LCCols = [index for index, value in enumerate(allSampleIDs) if '_LC_' in value]

# Add 1 to each column to align indices with metabolite data (accounts for metabolite name column)
MPColsAligned = [ind + 1 for ind in MPCols]
MLColsAligned = [ind + 1 for ind in MLCols]
LCColsAligned = [ind + 1 for ind in LCCols]

# Split up metaboliteData according to metabolite type using these column indices
MPData = metaboliteData.iloc[:,MPColsAligned]
MLData = metaboliteData.iloc[:,MLColsAligned]
LCData = metaboliteData.iloc[:,LCColsAligned]



# Subsetting a dataframe with iloc retains the indices in the original, so must reset
    # Reset rows
MPDataReset = MPData.reset_index(drop=True)
MLDataReset = MLData.reset_index(drop=True)
LCDataReset = LCData.reset_index(drop=True)
    # Reset cols
MPDataReset.columns = range(MPDataReset.shape[1])
MLDataReset.columns = range(MLDataReset.shape[1])
LCDataReset.columns = range(LCDataReset.shape[1])

# Remove rows with NaN values across all columns
MPDataCleaned = MPDataReset.dropna(axis=0,how='all')
MLDataCleaned = MLDataReset.dropna(axis=0,how='all')
LCDataCleaned = LCDataReset.dropna(axis=0,how='all')



# Determine columns in cleaned data corresponding to the 4 subject groups: Pre+Dep, Post+Dep, Pre+Con, Post+Con

    # MP
MPFactors = MPDataCleaned.iloc[1,:]
PreDepMPCols = [index for index, value in enumerate(MPFactors) if 'Pre' in value and 'Dep' in value]
PostDepMPCols = [index for index, value in enumerate(MPFactors) if 'Post' in value and 'Dep' in value]
PreConMPCols = [index for index, value in enumerate(MPFactors) if 'Pre' in value and 'Con' in value]
PostConMPCols = [index for index, value in enumerate(MPFactors) if 'Post' in value and 'Con' in value]

    # ML
MLFactors = MLDataCleaned.iloc[1,:]
PreDepMLCols = [index for index, value in enumerate(MLFactors) if 'Pre' in value and 'Dep' in value]
PostDepMLCols = [index for index, value in enumerate(MLFactors) if 'Post' in value and 'Dep' in value]
PreConMLCols = [index for index, value in enumerate(MLFactors) if 'Pre' in value and 'Con' in value]
PostConMLCols = [index for index, value in enumerate(MLFactors) if 'Post' in value and 'Con' in value]

    # LC
LCFactors = LCDataCleaned.iloc[1,:]
PreDepLCCols = [index for index, value in enumerate(LCFactors) if 'Pre' in value and 'Dep' in value]
PostDepLCCols = [index for index, value in enumerate(LCFactors) if 'Post' in value and 'Dep' in value]
PreConLCCols = [index for index, value in enumerate(LCFactors) if 'Pre' in value and 'Con' in value]
PostConLCCols = [index for index, value in enumerate(LCFactors) if 'Post' in value and 'Con' in value]



# Split up the cleaned data for each metabolite type according to subject group using these columns

    # MP
PreDepMP = MPDataCleaned.iloc[:,PreDepMPCols]
PostDepMP = MPDataCleaned.iloc[:,PostDepMPCols]
PreConMP = MPDataCleaned.iloc[:,PreConMPCols]
PostConMP = MPDataCleaned.iloc[:,PostConMPCols]

    # ML
PreDepML = MLDataCleaned.iloc[:,PreDepMLCols]
PostDepML = MLDataCleaned.iloc[:,PostDepMLCols]
PreConML = MLDataCleaned.iloc[:,PreConMLCols]
PostConML = MLDataCleaned.iloc[:,PostConMLCols]

    # LC
PreDepLC = LCDataCleaned.iloc[:,PreDepLCCols]
PostDepLC = LCDataCleaned.iloc[:,PostDepLCCols]
PreConLC = LCDataCleaned.iloc[:,PreConLCCols]
PostConLC = LCDataCleaned.iloc[:,PostConLCCols]

# Check that each metabolite type has same columns for each group 
    # Check PreDep
if PreDepMPCols == PreDepMLCols == PreDepLCCols:
    print("PreDep all identical")
else:
    print("PreDep not all identical")
    # Check PostDep
if PostDepMPCols == PostDepMLCols == PostDepLCCols:
    print("PostDep all identical")
else:
    print("PostDep not all identical")
    # Check PreCon
if PreConMPCols == PreConMLCols == PreConLCCols:
    print("PreCon all identical")
else:
    print("PreCon not all identical")
    # Check PostCon
if PostConMPCols == PostConMLCols == PostConLCCols:
    print("PostCon all identical")
else:
    print("PostCon not all identical")



# Rejoin all metabolite types according to subject group by stacking vertically, keeping the 
# first two rows for the first dataframe onlt
PreDep = pd.concat([PreDepMP,PreDepML.iloc[2:,:],PreDepLC.iloc[2:,:]], axis=0, ignore_index=True)
PostDep = pd.concat([PostDepMP,PostDepML.iloc[2:,:],PostDepLC.iloc[2:,:]], axis=0, ignore_index=True)
PreCon = pd.concat([PreConMP,PreConML.iloc[2:,:],PreConLC.iloc[2:,:]], axis=0, ignore_index=True)
PostCon = pd.concat([PostConMP,PostConML.iloc[2:,:],PostConLC.iloc[2:,:]], axis=0, ignore_index=True)

# Reset the column indices to start from 0
PreDep.columns = range(PreDep.shape[1])
PostDep.columns = range(PostDep.shape[1])
PreCon.columns = range(PreCon.shape[1])
PostCon.columns = range(PostCon.shape[1])

# Transpose each dataframe so that rows represent samples and columns represent features
PreDep = PreDep.T
PostDep = PostDep.T
PreCon = PreCon.T
PostCon = PostCon.T

# Split each dataframe into x and y
    # PreDep
xPreDep = PreDep.iloc[:,2:]
yPreDep = PreDep.iloc[:,1]
    # PostDep
xPostDep = PostDep.iloc[:,2:]
yPostDep = PostDep.iloc[:,1]  
    # PreCon
xPreCon = PreCon.iloc[:,2:]
yPreCon = PreCon.iloc[:,1]
    # PostDep
xPostCon = PostCon.iloc[:,2:]
yPostCon = PostCon.iloc[:,1]



# Split each dataframe into testing and training sets
xTrainPreDep, xTestPreDep, yTrainPreDep, yTestPreDep = train_test_split(xPreDep, yPreDep, test_size=0.2, random_state=42)
xTrainPostDep, xTestPostDep, yTrainPostDep, yTestPostDep = train_test_split(xPostDep, yPostDep, test_size=0.2, random_state=42)
xTrainPreCon, xTestPreCon, yTrainPreCon, yTestPreCon = train_test_split(xPreCon, yPreCon, test_size=0.2, random_state=42)
xTrainPostCon, xTestPostCon, yTrainPostCon, yTestPostCon = train_test_split(xPostCon, yPostCon, test_size=0.2, random_state=42)

# Combine data frames by stacking vertically to produce single x and y dataframes for training and testing
xTrain = pd.concat([xTrainPreDep,xTrainPostDep,xTrainPreCon,xTrainPostCon], axis=0, ignore_index=True)
xTest = pd.concat([xTestPreDep,xTestPostDep,xTestPreCon,xTestPostCon], axis=0, ignore_index=True)
yTrain = pd.concat([yTrainPreDep,yTrainPostDep,yTrainPreCon,yTrainPostCon], axis=0, ignore_index=True)
yTest = pd.concat([yTestPreDep,yTestPostDep,yTestPreCon,yTestPostCon], axis=0, ignore_index=True)

# Impute missing values using mean across all samples for that feature
    # Create imputer object that imputes according to feature mean
myImputer = SimpleImputer(strategy='mean')
    # Replace missing values currently represented by '----' with NaN
xTrain.replace('----', np.nan, inplace=True)
xTest.replace('----', np.nan, inplace=True)
    # Impute missing values
xTrainImputed = myImputer.fit_transform(xTrain)
xTestImputed = myImputer.fit_transform(xTest)
    # Convert back to pandas dataframe
xTrainImputed = pd.DataFrame(xTrainImputed)
xTestImputed = pd.DataFrame(xTestImputed)

# Combine the imputed xTrain and xTest matrices vertically
xTrainTestImputed = pd.concat([xTrainImputed,xTestImputed], axis=0, ignore_index=True)



# Standardize
scaler = StandardScaler()
xScaled = scaler.fit_transform(xTrainTestImputed)

# Convert back to pandas dataframe
xScaled = pd.DataFrame(xScaled)

# Split up the  scaled values back into training and testing
nTrainRows = xTrainImputed.shape[0]
xTrainScaled = xScaled.iloc[0:nTrainRows,:]
xTestScaled = xScaled.iloc[nTrainRows:,:]



# Apply One-Hot Encoding to y matrices
    # y matrices are currently pandas series. Convert to pandas dataframes for use in OneHotEncoder
yTrain = pd.DataFrame(yTrain)
yTest = pd.DataFrame(yTest)
    # Apply one-hot encoding to the y matrices
myOneHotEncoder = OneHotEncoder(sparse_output=False)
yTrainHot = myOneHotEncoder.fit_transform(yTrain)
yTestHot = myOneHotEncoder.fit_transform(yTest)
    # OneHotEncoder outputs a numpy ndarray. Convert to pandas dataframes
yTrainHot = pd.DataFrame(yTrainHot)
yTestHot = pd.DataFrame(yTestHot)


#-----------------------------------------------------------------#


# Perform PLS
    # Initialize the PLS model with a chosen number of components
pls = PLSRegression(n_components=6)
    # Fit the PLS model
pls.fit(xTrainScaled, yTrainHot)
    # Predict probabilities on the test data
yPredPLSContinuous = pls.predict(xTestScaled)
    # Convert probabilities to discrete class predictions
yPredPLSLabels = myOneHotEncoder.inverse_transform(yPredPLSContinuous)

# Evaluate Results
print("PLS Accuracy:", accuracy_score(yTest, yPredPLSLabels))
print(classification_report(yTest, yPredPLSLabels))



# Perform Random Forest Clustering
    # Initialize Random Forest model
randomForest = ensemble.RandomForestClassifier()
    # Fit the Random Forest model
randomForest.fit(xTrainScaled, yTrain)
    # Predict class of test data
yPredRandomForest = randomForest.predict(xTestScaled)

# Evaluate Results
print(" Random Forest Accuracy:", accuracy_score(yTest, yPredRandomForest))
print(classification_report(yTest, yPredRandomForest))



# Perform Logistic Regression
    # Initialize Logistic Regression model
logisticReg = LogisticRegression()
    # Fit the Logistic Regression model
logisticReg.fit(xTrainScaled, yTrain)
    # Predict class of test data
yPredLogistic = logisticReg.predict(xTestScaled)

# Evaluate Results
print("Logistic Regression Accuracy:", accuracy_score(yTest, yPredLogistic))
print(classification_report(yTest, yPredLogistic))


#-----------------------------------------------------------------#


# Statistical Evaluation Of Models Using Precision-Recall Curves & Average Precision Scores For Each Subject Category


# Classes & mappings based on the four studied categories
classes = ['PreDep', 'PostDep', 'PreCon', 'PostCon']

# Function to calculate PR Curves & plots for each model
def PR_Curves(yTest, yPred, model_name, classes):
    plt.figure(figsize=(10, 7))
    average_precision = []
    class_weights = np.sum(yTest, axis=0) / np.sum(yTest)  # Calculate class weights since some classes are larger than others

    # Loop through each class and calculate PR curve respectively
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(yTest[:, i], yPred[:, i])
        ap = average_precision_score(yTest[:, i], yPred[:, i])
        average_precision.append(ap)
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')

    # Weighted Average Precision-Recall AUC
    weighted_ap = np.average(average_precision, weights=class_weights)

    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='grey', label='Random Guess')
    plt.title(f'{model_name} - Precision-Recall Curves (Weighted AP = {weighted_ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    # Output Results
    print(f"\n{model_name} - Class-Specific AP Scores and Weighted AUC of PR Curve:")
    for class_name, ap, weight in zip(classes, average_precision, class_weights):
        print(f"  {class_name}: AP = {ap:.2f} (Weight = {weight:.2f})")
    print(f"  Weighted Average AP of PR Curve: {weighted_ap:.2f}\n")

    return average_precision, weighted_ap



# Calculate PR curves and AP scores using function 'PR_Curves'
print("PLS Model:")
yPredPLSContinuous = pls.predict(xTestScaled)  # Predicted probabilities for PLS
pls_ap, pls_weighted_ap = PR_Curves(yTestHot.values, yPredPLSContinuous, "PLS Regression", classes)

print("Random Forest Model:")
yPredRFProba = randomForest.predict_proba(xTestScaled)  # Predicted probabilities for Random Forest
rf_ap, rf_weighted_ap = PR_Curves(yTestHot.values, yPredRFProba, "Random Forest", classes)

print("Logistic Regression Model:")
yPredLogProba = logisticReg.predict_proba(xTestScaled)  # Predicted probabilities for Logistic Regression
log_ap, log_weighted_ap = PR_Curves(yTestHot.values, yPredLogProba, "Logistic Regression", classes)



# Summary table of AP scores for all models
print("\nSummary of Class-Specific Average Precision Scores for Each Model:")
summary_table = {
    "Class": classes,
    "PLS AP": pls_ap,
    "Random Forest AP": rf_ap,
    "Logistic Regression AP": log_ap,
}

for i in range(len(classes)):
    print(f"{summary_table['Class'][i]:<15} | PLS: {summary_table['PLS AP'][i]:.2f} | "
          f"Random Forest: {summary_table['Random Forest AP'][i]:.2f} | "
          f"Logistic Regression: {summary_table['Logistic Regression AP'][i]:.2f}")

print("\nWeighted Macro-Averaged AP of PR Curves for Each Model:")
print(f"  PLS Regression: {pls_weighted_ap:.2f}")
print(f"  Random Forest: {rf_weighted_ap:.2f}")
print(f"  Logistic Regression: {log_weighted_ap:.2f}")
