import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Receives a results file path and returns it as a dictionary
def return_dict(df):
    index = df.columns.values[0] # indexing using the first column of any giver dataframe
    dict = df.set_index(index).T.to_dict('records') # returns a list of dicts
    dict = dict[0] # so we need to get the only dict that matters, the first and only
    return dict

# This method creates a graph comparing every class's accuracy using it's respective results file
def graph_by_class(file_path,window_size,epochs):
    df = pd.read_csv(file_path)
    class_result = return_dict(df)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    keys = [(k) for k,v in sorted(class_result.items())]
    y_pos = np.arange(len(np.array(keys)))
    ax.barh(y_pos, np.array([(v) for k,v in sorted(class_result.items())]), align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(keys))
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy by Class')
    ax.grid(True)

    plt.xlim(0.92, 1.00005)
    if not os.path.exists('results/graphs'):
        os.makedirs('results/graphs')

    plt.savefig('results/graphs/accuracy_by_class_'+str(window_size)+"-"+str(epochs)+'.png', bbox_inches='tight', pad_inches=1)
    del keys
    plt.close('all')

# Creates one graph representing data stored in the "total_accuracy.csv" file.
def graph_by_window_size(file_path):
    df = pd.read_csv(file_path, index_col=False)
    #df = df.groupby(['epochs'])
    df = df[['batch_size', 'epochs', 'acc']]
    fig,ax = plt.subplots()
    x = np.arange(len(df.batch_size.unique()))
    bar_width = 0.1
    b1 = ax.bar(x, df.loc[df['epochs'] == 1, 'acc'],
            width=bar_width, label = '1')
    # Same thing, but offset the x by the width of the bar.
    b2 = ax.bar(x + bar_width, df.loc[df['epochs'] == 2, 'acc'],
            width=bar_width, label = '2')
    b3 = ax.bar(x + 2*bar_width, df.loc[df['epochs'] == 3, 'acc'],
            width=bar_width, label = '3')
    
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(df.batch_size.unique())
    ax.ticklabel_format(axis= 'y',useOffset=False)
    plt.ylim(0.8, 1)
    ax.grid(True)
    ax.legend(title="epochs")
    plt.title("Batch Size x Accuracy")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.show()
    return
    if not os.path.exists('results/graphs'):
        os.makedirs('results/graphs')

    plt.savefig('results/graphs/total_accuracy.png', bbox_inches='tight', pad_inches=1)
    plt.close('all')

def graph_class(file_path):
    #definições
    columns_names = ['-PAD-', 'V', 'PREP+ADV', 'CUR', 'PREP', 'PROADJ', 'PU', 'PREP+ART',
        'PRO-KS', 'N', 'KC', 'PCP', 'PREP+PROPESS', 'PROPESS', 'NUM', 'IN',
       'ADV-KS', 'PREP+PRO-KS', 'PREP+PROSUB', 'KS', 'NPROP', 'ART', 'ADJ',
       'ADV', 'PDEN', 'PROSUB', 'PREP+PROADJ']
    batch_size = [120, 120, 120, 90, 60, 60, 60, 90, 90]
    epochs = [1, 2, 3, 3, 3, 2, 1, 1, 2]
    
    df = pd.read_csv(file_path, index_col=False)
    print(len(df.columns))
    df = df.drop(columns=['batch_size', 'epochs', 'loss', 'fake_acc', 'acc'])
    #print(df.head(2))
    for index in df.index:
        #prep for plot
        df2 = df[df.index == index]
        y_pos = [i for i in range(0,len(df2.columns))]
        df2.loc[10] = y_pos
        df1 = df2.transpose()
        df1 = df1.rename(columns={index: "a", 10: "b"})
        df1['b'] = df1['b'].astype(str)

        #actual plot
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.barh(df1.b, df1.a, align='center')
        ax.set_yticks(df1.b)
        ax.set_yticklabels(columns_names)
        ax.set_xlabel('Accuracy')
        title = 'Accuracy by class with batch size = '+ str(batch_size[index]) + ' and ' + str(epochs[index]) + ' epochs'
        ax.set_title(str(title))
        title = 'acc_class_bs'+ str(batch_size[index])+'_epochs'+str(epochs[index])
        print(title)
        plt.savefig('results/graphs/'+title+'.png', bbox_inches='tight', pad_inches=1)
        plt.show()
    return

#graph_by_window_size("results.csv")
graph_class("results.csv")

# graph_by_class("results/7-1.csv",3,1)