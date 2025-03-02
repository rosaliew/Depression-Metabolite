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

# Check that all metabolite groups have the same columns 
if MPCols == MLCols == LCCols:
    print("Metabolite groups all identical")
else:
    print("Metabolite groups not all identical")
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

# Combine the subject group matrices back into one by joining horizontally
fullMat = pd.concat([PreDep,PostDep,PreCon,PostCon],axis=1,ignore_index=True)

# Transpose to arrange subjects (i.e. observations) along rows and metabolites (i.e. features) along columns
fullMat = fullMat.T

# Split into x and y
xFullMat = fullMat.iloc[:,2:]
yFullMat = fullMat.iloc[:,1]

# Create a dictionary for replacing the provided labels in yFullMat
replacement_dict = {
    "Sample source:Plasma | Condition:Postmenopause | Symptom:Control": "Postmenopause Control",
    "Sample source:Plasma | Condition:Postmenopause | Symptom:Depression": "Postmenopause Depression",
    "Sample source:Plasma | Condition:Premenopause | Symptom:Control": "Premenopause Control",
    "Sample source:Plasma | Condition:Premenopause | Symptom:Depression": "Premenopause Depression"
}

# Replace labels in yFullMat
yFullMat = yFullMat.replace(replacement_dict)

print(yFullMat)

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
pls = PLSRegression(n_components=6)
    # Fit the PLS model
pls.fit(xTrain, yTrainHot)
    # Predict probabilities on the test data
yPredPLSContinuous = pls.predict(xTest)
    # Convert probabilities to discrete class predictions
yPredPLSLabels = myOneHotEncoder.inverse_transform(yPredPLSContinuous)

# Evaluate Results
print("PLS-DA Accuracy:", accuracy_score(yTest, yPredPLSLabels))
print(classification_report(yTest, yPredPLSLabels))

# Plot performance-recall curves for each class and the micro-average
plt.figure(1)
for i, class_name in enumerate(myOneHotEncoder.categories_[0]):
    precisionPLS, recallPLS, _ = precision_recall_curve(yTestHot.iloc[:, i], yPredPLSContinuous[:, i])
    # Add the precision-recall curve for each class to the plot
    plt.plot(recallPLS, precisionPLS, label=f"Class {class_name}")
    # Compute micro-averaged precision and recall
precision_microPLS, recall_microPLS, _ = precision_recall_curve(yTestHot.to_numpy().ravel(), yPredPLSContinuous.ravel())
    # Compute the average precision score for the micro-average
average_precision_microPLS = average_precision_score(yTestHot, yPredPLSContinuous, average="micro")
    # Add the micro-averaged precision-recall curve to the plot
plt.plot(recall_microPLS, precision_microPLS, label=f"Micro-average (AP={average_precision_microPLS:.2f})", color="black")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PLS-DA Model Precision-Recall Curve")
plt.legend()
plt.grid()

# Plot the confusion matrix
conf_matrixPLS = confusion_matrix(yTest, yPredPLSLabels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrixPLS, display_labels=myOneHotEncoder.categories_[0])
disp.plot(cmap=plt.cm.Blues)
plt.title("PLS-DA Model Confusion Matrix")

# Perform Random Forest Clustering
    # Initialize Random Forest model
randomForest = ensemble.RandomForestClassifier()
    # Fit the Random Forest model
randomForest.fit(xTrain, yTrain)
    # Predict class of test data
yPredRandomForest = randomForest.predict(xTest)

# Evaluate Results
print(" Random Forest Accuracy:", accuracy_score(yTest, yPredRandomForest))
print(classification_report(yTest, yPredRandomForest))

# Plot performance-recall curves for each class and the micro-average
    # Predict probabilities of test data
yPredProbRandomForest = randomForest.predict_proba(xTest)
plt.figure(3)
for i, class_name in enumerate(myOneHotEncoder.categories_[0]):
    precisionRF, recallRF, _ = precision_recall_curve(yTestHot.iloc[:, i], yPredProbRandomForest[:, i])
    # Add the precision-recall curve for each class to the plot
    plt.plot(recallRF, precisionRF, label=f"Class {class_name}")
    # Compute micro-averaged precision and recall
precision_microRF, recall_microRF, _ = precision_recall_curve(yTestHot.to_numpy().ravel(), yPredProbRandomForest.ravel())
    # Compute the average precision score for the micro-average
average_precision_microRF = average_precision_score(yTestHot, yPredProbRandomForest, average="micro")
    # Add the micro-averaged precision-recall curve to the plot
plt.plot(recall_microRF, precision_microRF, label=f"Micro-average (AP={average_precision_microRF:.2f})", color="black")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest Model Precision-Recall Curve")
plt.legend()
plt.grid()

# Plot the confusion matrix
conf_matrixRF = confusion_matrix(yTest, yPredRandomForest)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrixRF, display_labels=myOneHotEncoder.categories_[0])
disp.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Model Confusion Matrix")

# Perform Logistic Regression
    # Initialize Logistic Regression model
logisticReg = LogisticRegression()
    # Fit the Logistic Regression model
logisticReg.fit(xTrain, yTrain)
    # Predict class of test data
yPredLogistic = logisticReg.predict(xTest)

# Evaluate Results
print("Logistic Regression Accuracy:", accuracy_score(yTest, yPredLogistic))
print(classification_report(yTest, yPredLogistic))

# Plot performance-recall curves for each class and the micro-average
    # Predict probabilities of test data
yPredProbLogistic = logisticReg.predict_proba(xTest)
plt.figure(5)
for i, class_name in enumerate(myOneHotEncoder.categories_[0]):
    precisionLR, recallLR, _ = precision_recall_curve(yTestHot.iloc[:, i], yPredProbLogistic[:, i])
    # Add the precision-recall curve for each class to the plot
    plt.plot(recallLR, precisionLR, label=f"Class {class_name}")
    # Compute micro-averaged precision and recall
precision_microLR, recall_microLR, _ = precision_recall_curve(yTestHot.to_numpy().ravel(), yPredProbLogistic.ravel())
    # Compute the average precision score for the micro-average
average_precision_microLR = average_precision_score(yTestHot, yPredProbLogistic, average="micro")
    # Add the micro-averaged precision-recall curve to the plot
plt.plot(recall_microLR, precision_microLR, label=f"Micro-average (AP={average_precision_microLR:.2f})", color="black")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic Regression Model Precision-Recall Curve")
plt.legend()
plt.grid()

# Plot the confusion matrix
conf_matrixLR = confusion_matrix(yTest, yPredLogistic)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrixLR, display_labels=myOneHotEncoder.categories_[0])
disp.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression Model Confusion Matrix")
plt.show()