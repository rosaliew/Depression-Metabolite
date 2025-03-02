import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn import ensemble
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import matplotlib
import matplotlib.pyplot as plt


# Load the meta and metabolite data from their CSV files
metaDataPath = r'C:\Users\macke\OneDrive\Documents\MACKENZIE SCHOOL STUFF\Fall 2024\IBEHS 4QZ3\Project\MetaData.csv'
metaboliteDataPath = r'C:\Users\macke\OneDrive\Documents\MACKENZIE SCHOOL STUFF\Fall 2024\IBEHS 4QZ3\Project\MetaboliteData.csv'
metaData = pd.read_csv(metaDataPath)
metaboliteData = pd.read_csv(metaboliteDataPath,header=None)

# Determine which columns correspond to each metabolite type (MP, ML, LC)
allSampleIDs = metaboliteData.iloc[0,1:]
MPCols = [index for index, value in enumerate(allSampleIDs) if '_MP_' in value]
MLCols = [index for index, value in enumerate(allSampleIDs) if '_ML_' in value]
LCCols = [index for index, value in enumerate(allSampleIDs) if '_LC_' in value]

# Optional: Check that all metabolite groups have the same columns
"""
if MPCols == MLCols == LCCols:
    print("Metabolite groups all identical")
else:
    print("Metabolite groups not all identical")
"""
# ^ This shows that they do not have the same columns, therefore cannot be combined

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

# Optional: Check that each metabolite type has same columns for each group 
"""
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
"""
# ^ This shows that they all have the same columns, so can be combined

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

# Combine the subject group matrices back into one by joining horizontally
fullMat = pd.concat([PreDep,PostDep,PreCon,PostCon],axis=1,ignore_index=True)

# Transpose to arrange subjects (i.e. observations) along rows and metabolites (i.e. features) along columns
fullMat = fullMat.T

# Split into x and y
xFullMat = fullMat.iloc[:,2:]
yFullMat = fullMat.iloc[:,1]

# Create a dictionary for replacing the provided labels in yFullMat
subjectLabels = ["Postmenopause Control", "Postmenopause Depression", "Premenopause Control", "Premenopause Depression"]
replacement_dict = {
    "Sample source:Plasma | Condition:Postmenopause | Symptom:Control": subjectLabels[0],
    "Sample source:Plasma | Condition:Postmenopause | Symptom:Depression": subjectLabels[1],
    "Sample source:Plasma | Condition:Premenopause | Symptom:Control": subjectLabels[2],
    "Sample source:Plasma | Condition:Premenopause | Symptom:Depression": subjectLabels[3]
}

# Replace labels in yFullMat
yFullMat = yFullMat.replace(replacement_dict)

# Impute missing values using mean across all samples for that feature
    # Create imputer object that imputes according to feature mean
myImputer = SimpleImputer(strategy='mean')
    # Replace missing values currently represented by '----' with NaN
xFullMat = xFullMat.copy()
xFullMat.replace('----', np.nan, inplace=True)
    # Impute missing values
xFullMatImputed = myImputer.fit_transform(xFullMat)
    # Convert back to pandas dataframe
xFullMatImputed = pd.DataFrame(xFullMatImputed)

# Standardize
scaler = StandardScaler()
xFullMatStandardized = scaler.fit_transform(xFullMatImputed)

# Split into training and testing
xTrain, xTest, yTrain, yTest = train_test_split(xFullMatStandardized, yFullMat, test_size=0.2, random_state=42, stratify=yFullMat)

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

# Perform PLS
    # Initialize the PLS model with a chosen number of components
n_comps = 6
pls = PLSRegression(n_components=n_comps)
    # Fit the PLS model
pls.fit(xTrain, yTrainHot)
    # Predict probabilities on the test data
yPredPLSContinuous = pls.predict(xTest)
    # Convert probabilities to discrete class predictions
yPredPLSLabels = myOneHotEncoder.inverse_transform(yPredPLSContinuous)

def calculate_vip(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, _ = w.shape  # Number of features
    
    # SSY for each component
    ssy = np.sum((t @ q.T) ** 2, axis=1)  # Ensure ssy matches the number of components
    
    vip_scores = np.zeros((p,))
    n_components = pls_model.n_components  # Use the correct number of components
    
    for j in range(p):
        weight_sum = np.sum([(w[j, k]**2) * ssy[k] for k in range(n_components)])
        vip_scores[j] = np.sqrt(p * weight_sum / np.sum(ssy))
    
    return vip_scores

# Calculate VIP Scores
vip_scores = calculate_vip(pls, xTrain)

# Normalize VIP Scores
normVIP_scores = []
sumVIP_scores = sum(abs(vip_scores))
for score in vip_scores:
    normVIP_scores.append(score/sumVIP_scores)

# Threshold the relative importances to show only those with greatest impact
pieThreshold = 0.0145
metabolite_importances = [value for value in normVIP_scores if value > pieThreshold]
metabolite_inds = [i for i, value in enumerate(normVIP_scores) if value > pieThreshold]

# Determine the relative importance of 'other' and how many
other_importance = sum(normVIP_scores) - sum(metabolite_importances)
nOthers = sum(1 for score in normVIP_scores if score < pieThreshold)

# Define values for pie chart
metabolite_importances_series = pd.Series(metabolite_importances)
pie_importances = pd.concat([metabolite_importances_series, pd.Series([other_importance])], ignore_index=True)

# Obtain metabolite labels
metaboliteLabels = metaboliteData.iloc[2:,0]

# Define labels for pie chart
pieLabels = metaboliteLabels.iloc[metabolite_inds]
pieLabels = pd.concat([pieLabels, pd.Series([f'{nOthers} Other Metabolites'])], ignore_index=True)

# Plot pie chart 
fig,ax = plt.subplots()
ax.pie(pie_importances, labels=None, autopct='%1.1f%%', startangle=140)
ax.legend(pieLabels,loc="upper right")
plt.title(f'PLS-DA Metabolite Importances')
plt.show()

# Calculate VIP scores for each class individually (class_idx)
def calculate_class_vip(pls_model, X, class_idx):

    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_[class_idx, :].reshape(-1, 1)  # Select the row corresponding to the class
    p, _ = w.shape  # Number of features
    
    # SSY for the specific class
    ssy_class = np.sum((t @ q) ** 2, axis=1)
    
    vip_scores_class = np.zeros((p,))
    n_components = pls_model.n_components
    
    for j in range(p):
        weight_sum = np.sum([(w[j, k]**2) * ssy_class[k] for k in range(n_components)])
        vip_scores_class[j] = np.sqrt(p * weight_sum / np.sum(ssy_class))
    
    return vip_scores_class

# Generate pie plots for each class
for class_idx in range(yTrainHot.shape[1]):
    vip_scores_class = calculate_class_vip(pls, xTrain, class_idx)
    
    # Normalize VIP scores
    normVIP_scores_class = vip_scores_class / sum(abs(vip_scores_class))
    
    # Apply threshold
    metabolite_importances_class = [value for value in normVIP_scores_class if value > pieThreshold]
    metabolite_inds_class = [i for i, value in enumerate(normVIP_scores_class) if value > pieThreshold]
    
    # Compute 'other' importance and how many
    other_importance_class = 1.0 - sum(metabolite_importances_class)
    nOthers_class = sum(1 for score in normVIP_scores_class if score < pieThreshold)
    
    # Prepare pie chart data
    metabolite_importances_series_class = pd.Series(metabolite_importances_class)
    pie_importances_class = pd.concat([metabolite_importances_series_class, pd.Series([other_importance_class])], ignore_index=True)
    pieLabels_class = metaboliteLabels.iloc[metabolite_inds_class]
    pieLabels_class = pd.concat([pieLabels_class, pd.Series([f'{nOthers_class} Other Metabolites'])], ignore_index=True)
    
    # Plot pie chart
    fig,ax = plt.subplots()
    ax.pie(pie_importances_class, labels=None, autopct='%1.1f%%', startangle=140)
    ax.legend(pieLabels_class,loc="upper right")
    plt.title(f'PLS-DA Metabolite Importances for {subjectLabels[class_idx]} Group')
    plt.show()

