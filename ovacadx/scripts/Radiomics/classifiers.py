import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from utils import filter_df, get_dataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 2 output classes for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def check_disjoint(df1,df2):
    IDs = set(df1['ID'])
    IDs2 = set(df2['ID'])
    if not IDs.isdisjoint(IDs2):
        raise ValueError("2 DF's are not disjoint, they share " + str(len(IDs.intersection(IDs2))) + " samples.")

import joblib
import os


def save_models(models, directory, feature_names, PCA_transform):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if 'svm' in models.keys():
        joblib.dump(models['svm'], os.path.join(directory, 'svm_model.pkl'))
    if 'rf' in models.keys():
        joblib.dump(models['rf'], os.path.join(directory, 'rf_model.pkl'))
    if 'lr' in models.keys():
        joblib.dump(models['lr'], os.path.join(directory, 'lr_model.pkl'))
    if 'nn' in models.keys():
        torch.save(models['nn'].state_dict(), os.path.join(directory, 'nn_model.pth'))
    joblib.dump(PCA_transform,os.path.join(directory,"pca_transform.pkl"))
    joblib.dump(feature_names,os.path.join(directory,"feature_names.pkl"))


def load_models(directory):
    """# Usage example:
    models = {
        'svm': svm,
        'rf': rf,
        'log_reg': log_reg,
        'neural_net': neural_net
    }
    save_models(models, 'models_directory')"""
    loaded_models = {}


    PCA_transform = joblib.load(os.path.join(directory, 'pca_transform.pkl'))
    pre_pca_feature_names = joblib.load(os.path.join(directory, 'feature_names.pkl'))
    if PCA_transform is not None:
        input_size = PCA_transform.components_.shape[0]
    else:
        input_size = len(pre_pca_feature_names) - 2 #Subtract 2 (IDs and classes)


    if os.path.exists(os.path.join(directory, 'svm_model.pkl')):
        loaded_models['svm'] = joblib.load(os.path.join(directory, 'svm_model.pkl'))
    if os.path.exists(os.path.join(directory, 'rf_model.pkl')):
        loaded_models['rf'] = joblib.load(os.path.join(directory, 'rf_model.pkl'))
    if os.path.exists(os.path.join(directory, 'lr_model.pkl')):
        loaded_models['lr'] = joblib.load(os.path.join(directory, 'lr_model.pkl'))
    if os.path.exists(os.path.join(directory, 'nn_model.pth')):
        loaded_models['nn'] = NeuralNetwork(input_size)
        loaded_models['nn'].load_state_dict(torch.load(os.path.join(directory, 'nn_model.pth')))
        loaded_models['nn'].eval()  # Set the model to evaluation mode
    return loaded_models, pre_pca_feature_names, PCA_transform

def test_classifiers(models,test_df,k):
    results = [] #
    ID_col = test_df['ID']
    label_col = "label"
    ID_col_name = "tumor_id" #compativle with cris
    X_test = test_df.drop(columns=['ID', label_col]).to_numpy()
    y_test = test_df[label_col].to_numpy()

    # Initialize dictionaries to store logits DataFrames
    logits_dfs = {
        'SVM': pd.DataFrame(),
        'RF': pd.DataFrame(),
        'LR': pd.DataFrame(),
        'NN': pd.DataFrame()
    }

    # SVM
    if "svm" in models.keys():
        svm = models['svm']
        y_pred_svm = svm.predict(X_test)
        y_pred_proba_svm = svm.predict_proba(X_test)[:, 1]
        acc_svm = accuracy_score(y_test, y_pred_svm)
        auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
        results.append({'Classifier': 'SVM', 'ACC': acc_svm, 'AUC': auc_svm})
        logits_dfs['SVM'] = pd.DataFrame({
            ID_col_name: ID_col,
            str(k): y_pred_proba_svm
        })

    # Random Forest
    if "rf" in models.keys():
        rf = models['rf']
        y_pred_rf = rf.predict(X_test)
        y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
        acc_rf = accuracy_score(y_test, y_pred_rf)
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
        results.append({'Classifier': 'Random Forest', 'ACC': acc_rf, 'AUC': auc_rf})
        logits_dfs['RF'] = pd.DataFrame({
            ID_col_name: ID_col,
            str(k): y_pred_proba_rf
        })

    # Logistic Regression
    if "lr" in models.keys():
        lr = models['lr']
        y_pred_log_reg = lr.predict(X_test)
        y_pred_proba_log_reg = lr.predict_proba(X_test)[:, 1]
        acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
        auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg)
        results.append({'Classifier': 'Logistic Regression', 'ACC': acc_log_reg, 'AUC': auc_log_reg})
        logits_dfs['LR'] = pd.DataFrame({
            ID_col_name: ID_col,
            str(k): y_pred_proba_log_reg
        })

    # Neural Network
    if "nn" in models.keys():
        neural_net = models['nn']
        neural_net.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_proba_nn = torch.softmax(neural_net(X_test_tensor), dim=1)[:, 1].detach().numpy()
        y_pred_nn = np.argmax(neural_net(X_test_tensor).detach().numpy(), axis=1)
        acc_nn = accuracy_score(y_test, y_pred_nn)
        auc_nn = roc_auc_score(y_test, y_pred_proba_nn)
        results.append({'Classifier': 'Neural Network', 'ACC': acc_nn, 'AUC': auc_nn})
        logits_dfs['NN'] = pd.DataFrame({
            ID_col_name: ID_col,
            str(k): y_pred_proba_nn
        })

    results_df = pd.DataFrame(results)
    return logits_dfs #results_df

def train_classifiers(df, mod,plot=False,verbose=False, model_types = ["lr","nn","rf","svm"]):
    ''' Given a train_val df and a value for k, this function will train classifiers and report their scores on the validation sets'''
    results = []

    train_df = filter_df(df, mod.train_samples)
    val_df = filter_df(df, mod.val_samples)
    label_col = 'label'
    X_train = train_df.drop(columns=['ID', label_col]).to_numpy()
    y_train = train_df[label_col].to_numpy()
    X_test = val_df.drop(columns=['ID', label_col]).to_numpy()
    y_test = val_df[label_col].to_numpy()

    # SVM
    if "svm" in model_types:
        svm = SVC(probability=True)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        y_pred_proba_svm = svm.predict_proba(X_test)[:, 1]
        acc_svm = accuracy_score(y_test, y_pred_svm)
        auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
        results.append({'Classifier': 'SVM', 'ACC': acc_svm, 'AUC': auc_svm})

    # Random Forest
    if "rf" in model_types:
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train,y_train) #np.random.choice([0, 1], size=len(y_train))
        y_pred_rf = rf.predict(X_test)
        y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
        acc_rf = accuracy_score(y_test, y_pred_rf)
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
        results.append({'Classifier': 'Random Forest', 'ACC': acc_rf, 'AUC': auc_rf})

    # Logistic Regression
    if "lr" in model_types:
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        y_pred_log_reg = log_reg.predict(X_test)
        y_pred_proba_log_reg = log_reg.predict_proba(X_test)[:, 1]
        acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
        auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg)
        results.append({'Classifier': 'Logistic Regression', 'ACC': acc_log_reg, 'AUC': auc_log_reg})

    # Neural Network using PyTorch
    if "nn" in model_types:
        input_size = X_train.shape[1]
        neural_net = NeuralNetwork(input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(neural_net.parameters(), lr=0.001)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        neural_net.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = neural_net(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        neural_net.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_proba_nn = torch.softmax(neural_net(X_test_tensor), dim=1)[:, 1].detach().numpy()
        y_pred_nn = np.argmax(neural_net(X_test_tensor).detach().numpy(), axis=1)
        acc_nn = accuracy_score(y_test, y_pred_nn)
        auc_nn = roc_auc_score(y_test, y_pred_proba_nn)
        results.append({'Classifier': 'Neural Network', 'ACC': acc_nn, 'AUC': auc_nn})

    # Create a DataFrame from the list of results
    results_df = pd.DataFrame(results)


    # Assuming you have the classifiers already trained and predictions available

    if plot:
        classifiers = [svm, rf, log_reg, neural_net]  # Add classifiers as needed
        # Create a list to store AUC values and accuracy values
        auc_values = []
        accuracy_values = []
    # Plot ROC curves for each classifier
        plt.figure(figsize=(10, 8))

        for clf in classifiers:
            if clf == neural_net:
                y_scores = y_pred_proba_nn  # Use neural network probabilities
            else:
                y_scores = clf.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)
            plt.plot(fpr, tpr, lw=2, label=f'{clf.__class__.__name__} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.show()
    elif verbose:
        print(results_df)
    available_models = {
        'svm': svm,
        'rf': rf,
        'lr': log_reg,
        'nn': neural_net
    }
    models = {key: available_models[key] for key in model_types if key in available_models} #Filter so that only model_types is included in models
    return results_df,models #Validation results, models dict
