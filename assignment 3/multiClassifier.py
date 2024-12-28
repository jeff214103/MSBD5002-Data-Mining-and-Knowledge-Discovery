import numpy as np
import os
import itertools
import contextlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import dataloader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from model import MultiClassClassifier, multiClassifierHiddenUnitIterator, reset_weights
import matplotlib.pyplot as plt

from datetime import datetime

DEBUG = False

EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.1

K_FOLD = 5

MODEL_PATH = "model"

#Multi-class dataset
MULTI_CLASS_PATH = "datasets/multi-class/"

#Multi class dataset
DIGITS = os.path.join(MULTI_CLASS_PATH, "digits.npz")


def train(data_loader, model, loss_fn, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for x_loader, y_loader in data_loader:
        x_loader, y_loader = x_loader.to(device), y_loader.to(device)
        optimizer.zero_grad()

        output = model(x_loader).squeeze(dim=-1)
        
        loss = loss_fn(output, y_loader)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += y_loader.size(0)
        correct += (predicted == y_loader).sum().item()    

    accuracy = 100 * correct / total

    return running_loss, accuracy

def test(data_loader, model, loss_fn, device):
    running_loss = 0.0
    correct = 0
    total = 0
    model.eval()

    y_true = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], device=device)

    with torch.no_grad():
        for x_loader, y_loader in data_loader:
            x_loader, y_loader = x_loader.to(device), y_loader.to(device)

            output = model(x_loader).squeeze(dim=-1)
            
            loss = loss_fn(output, y_loader)

            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += y_loader.size(0)
            correct += (predicted == y_loader).sum().item()

            y_true = torch.cat((y_true, y_loader), 0)
            y_pred = torch.cat((y_pred, predicted), 0)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = 100 * correct / total

    return running_loss, accuracy, y_true, y_pred

def plot_confusion_matrix(cm, base_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    output_path = os.path.join(base_path,name)
    if (not os.path.exists(output_path)):
        os.path.os.mkdir(output_path)
    plt.savefig(os.path.join(output_path,"Layer_1_Hidden_Unit_{layer_1}_Layer_2_Hidden_Unit_{layer_2}_Confunsion_Matrix.png".format(layer_1=hidden_unit_1, layer_2=hidden_unit_2)))

if __name__=="__main__":

    now = datetime.now()
    current_time = now.strftime("%H%M%S_%d-%b-%Y")

    base_path = "./output"
    if (not os.path.exists(base_path)):
        os.path.os.mkdir(base_path)
    base_path = os.path.join(base_path,"multiClassificationOutput")
    if (not os.path.exists(base_path)):
        os.path.os.mkdir(base_path)
    base_path = os.path.join(base_path, "lr_{}_EP_{}".format(LEARNING_RATE,EPOCHS))
    if (not os.path.exists(base_path)):
        os.path.os.mkdir(base_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    data = np.load(DIGITS)
    name = "digits"
    x_train = data["train_X"]
    y_train = data["train_y"]
    x_test = data["test_X"]
    y_test = data["test_y"]

    # Training data
    train_data = dataloader(torch.FloatTensor(x_train), torch.FloatTensor(y_train).long())

    # Final Testing data
    test_data = dataloader(torch.FloatTensor(x_test), torch.FloatTensor(y_test).long())
    

    records = []
    best_profile = {"test_accuracy":0}

    for hidden_unit_1, hidden_unit_2 in multiClassifierHiddenUnitIterator():
        
        print("Multi Classifier (Layer 1 {} Unit, Layer 2 {} Unit) Running...".format(hidden_unit_1, hidden_unit_2))

        #define model, optimziaer and loss function
        model = MultiClassClassifier(input_shape=x_train.shape[1], hidden_layer_1=hidden_unit_1, hidden_layer_2=hidden_unit_2)
        model.to(device)

        #stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)

        #Cross Entropy
        loss_fn = nn.CrossEntropyLoss()

        #store best model
        hidden_unit_best_accuracy = 0
        hidden_unit_best_accuracy_loss = 0

        #Cross validation model training, 80% training and 20% testing
        kfold = KFold(n_splits=K_FOLD, shuffle=True)

        fold_accuracy = {}

        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
            
            print("{} Fold Training...".format(fold))

            fold_best_accuracy = 0

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_subsampler)
            valid_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=test_subsampler)

            model.apply(reset_weights)

            history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

            #Multi classifier
            for epoch in range(1, EPOCHS+1):
                train_loss, train_acc = train(data_loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
                valid_loss, valid_acc, _ , _ = test( data_loader=valid_loader,model=model, loss_fn=loss_fn, device=device)

                if (valid_acc > fold_best_accuracy):
                    fold_best_accuracy = valid_acc

                if (valid_acc > hidden_unit_best_accuracy):
                    hidden_unit_best_accuracy = valid_acc
                    hidden_unit_best_accuracy_loss = valid_loss
                    torch.save(model.state_dict(), MODEL_PATH)
                if (DEBUG):
                    print("[Train] Fold {fold} - Epoch {cur}/{total} complete: Accuracy = {val:.4f} Loss = {loss:.4f}".format(fold=fold,cur=epoch, total=EPOCHS,val=train_acc,loss=train_loss))
                    print("[Test] Fold {fold} - Epoch {cur}/{total} complete: Accuracy = {val:.4f} Loss = {loss:.4f}".format(fold=fold,cur=epoch, total=EPOCHS,val=valid_acc,loss=valid_loss))

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['valid_loss'].append(valid_loss)
                history['valid_acc'].append(valid_acc)
            
            fold_accuracy[fold] = fold_best_accuracy
            
            if (DEBUG):
                fig, axs = plt.subplots(2)
                axs[0].plot(history['train_loss'])
                axs[0].plot(history['valid_loss'])
                axs[0].set_title('Loss vs Epochs')
                axs[0].set_xlabel('Epochs')
                axs[0].set_ylabel('Loss')

                axs[1].plot(history['train_acc'])
                axs[1].plot(history['valid_acc'])
                axs[1].set_title('Accuracy vs Epochs')
                axs[1].set_xlabel('Epochs')
                axs[1].set_ylabel('Accuracy')
                fig.suptitle("{} Dataset - (Layer 1 Hidden Unit = {layer_1}, Layer 2 Hidden Unit = {layer_2}) - Fold {fold}".format(name, layer_1=hidden_unit_1, layer_2=hidden_unit_2, fold=fold))
                plt.tight_layout()
            
                output_path = os.path.join(base_path,name)
                if (not os.path.exists(output_path)):
                    os.path.os.mkdir(output_path)
                output_path = os.path.join(output_path,"Layer_1_Hidden_Unit_{layer_1}_Layer_2_Hidden_Unit_{layer_2}".format(layer_1=hidden_unit_1, layer_2=hidden_unit_2))
                if (not os.path.exists(output_path)):
                    os.path.os.mkdir(output_path)
                output_path = os.path.join(output_path,"fold")
                if (not os.path.exists(output_path)):
                    os.path.os.mkdir(output_path)
                plt.savefig(os.path.join(output_path,"Dataset_Fold_{}.png".format(fold)))
        
        #Load the best model in training set among k-fold training model
        model.load_state_dict(torch.load(MODEL_PATH))

        #Final test with original testing data of this dataset
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        final_loss, final_acc, final_truth, final_predict = test(data_loader=test_loader,model=model, loss_fn=loss_fn, device=device)

        cf_matrix = confusion_matrix(final_truth, final_predict)
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(cf_matrix, base_path=base_path, title="{} Dataset - (Layer 1 Hidden Unit = {layer_1}, Layer 2 Hidden Unit = {layer_2}) - Confusion Matrix ".format(name, layer_1=hidden_unit_1, layer_2=hidden_unit_2))

        profile = {"hidden_unit": (0,0), "train_valid_accuracy": 0,"train_valid_loss": 0, "test_accuracy": 0,"test_loss":0, "fold_accuracy":fold_accuracy}

        #Setup history profile
        profile["hidden_unit"] = (hidden_unit_1, hidden_unit_2)
        profile["train_valid_accuracy"] = hidden_unit_best_accuracy
        profile["train_valid_loss"] = hidden_unit_best_accuracy_loss
        profile["test_accuracy"] = final_acc
        profile["test_loss"] = final_loss

        records.append(profile)

        if (final_acc > best_profile["test_accuracy"]):
            best_profile = profile
    

    output_path = os.path.join(base_path,name)
    if (not os.path.exists(output_path)):
        os.path.os.mkdir(output_path)
    
    with open(os.path.join(output_path,"output.txt"), "w") as o:
        with contextlib.redirect_stdout(o):
            for record in records:
                print("========================================================================")
                print("[Final Best Train (By Validation)] Model (Layer 1 Hidden Unit = {layer_1}, Layer 2 Hidden Unit = {layer_2}): Loss= {loss:.4f} Best Accuracy = {val:.4f}".format(layer_1=record["hidden_unit"][0], layer_2=record["hidden_unit"][1],val=record["train_valid_accuracy"],loss=record["train_valid_loss"]))
                print("[Final Best Test (By Test Set)] Model (Layer 1 Hidden Unit = {layer_1}, Layer 2 Hidden Unit = {layer_2}): Loss= {loss:.4f}  Best Accuracy = {val:.4f}".format(layer_1=record["hidden_unit"][0], layer_2=record["hidden_unit"][1],val=record["test_accuracy"],loss=record["test_loss"]))
                print("Cross Validation Details: ")
                sum = 0.0
                for key, value in record["fold_accuracy"].items():
                    print(f'Fold {key}: {value} %')
                    sum += value
                print(f'Average: {sum/len(record["fold_accuracy"].items())} %')

            print("========================================================================")
            print("{} Best Multi Classification Model ".format(name))
            print("Best Hidden Unit: Layer 1 Hidden Unit = {layer_1}, Layer 2 Hidden Unit = {layer_2}".format(layer_1=best_profile["hidden_unit"][0],layer_2=best_profile["hidden_unit"][1]))
            print("Best Train (By Validation) Accuracy: {}".format(best_profile["train_valid_accuracy"]))
            print("Best Test (By Test Set) Accuracy: {}".format(best_profile["test_accuracy"]))
            print("========================================================================")
        
