import collections
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_cmap():
    
    # Custom color map
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "#33cccc"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    return cmap

def plot_performance(scores, title = 'Reward', label='Running average - 50 episodes', window = 50):

    plt.figure(figsize=(11,6))
    sns.set_context("talk")

    # Get moving average
    cumsum = np.cumsum(scores)
    moving_average = [scores[0]]

    for i in range(1,len(cumsum)):
        moving_average.append((cumsum[i]-cumsum[max(0,i-window)])/len(cumsum[max(0,i-window):i]))

    plt.plot(scores, 'k-', label='Original', alpha=0.25)
    plt.plot(moving_average, '#33cccc', label=label)
    plt.ylabel('Sum of Rewards')
    plt.xlabel('Episode')
    plt.grid(linestyle=':')
    plt.fill_between(range(len(moving_average)), -20, moving_average, color='#33cccc', alpha=0.05)
    plt.legend(loc='lower right')
    plt.title(title, fontsize=24 )
    plt.show();

def plot_state_values(qtable, idx_to_cord):

    # Custom color map
    cmap = get_cmap()

    direction_dict = {3:"←", 2:"↓", 1:"→", 0:"↑"}

    df = pd.DataFrame(qtable)
    df['state_value'] = df.max(axis=1)
    df = df.reset_index()


    df["x"] = df['index'].apply(lambda x: idx_to_cord[x][0])
    df["y"] = df['index'].apply(lambda x: idx_to_cord[x][1])

    df['best_action'] = df[[0,1,2,3]].apply(lambda x: np.argmax(x), axis=1).apply(lambda x: direction_dict[x])
    df = df[df['state_value']!=0]

    df_state_values = df.pivot(columns='x', index='y', values='state_value').sort_index(ascending=False)
    df_best_action = df.pivot(columns='x', index='y', values='best_action').sort_index(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18,6))
    fig.suptitle('State Values for Optimal Policy', fontsize=24)
    sns.heatmap(df_state_values, annot=True, linewidths=.5, cbar=False, cmap=cmap, fmt = '.2f', ax=axes[0], annot_kws={"fontsize":11})
    sns.heatmap(df_state_values, annot=df_best_action, linewidths=.5, cbar=False, cmap=cmap, fmt = '', ax=axes[1], annot_kws={"fontsize":15})

    plt.show()

def plot_frequencies(state_history, training_state_history, idx_to_cord):

    # Custom color map
    cmap = get_cmap()

    # Traning
    training_counter = collections.Counter(state_history)
    df_training = pd.DataFrame()
    df_training["state"] = training_counter.keys()
    df_training['freq'] = training_counter.values()
    df_training["x"] = df_training['state'].apply(lambda x: idx_to_cord[x][0])
    df_training["y"] = df_training['state'].apply(lambda x: idx_to_cord[x][1])
    df_freqs_training = df_training.pivot(columns='x', index='y', values='freq').sort_index(ascending=False)

    # Testing
    testing_counter = collections.Counter(training_state_history)
    df_testing = pd.DataFrame()
    df_testing["state"] = testing_counter.keys()
    df_testing['freq'] = testing_counter.values()
    df_testing["x"] = df_testing['state'].apply(lambda x: idx_to_cord[x][0])
    df_testing["y"] = df_testing['state'].apply(lambda x: idx_to_cord[x][1])

    df_testing = df_training.merge(df_testing[['state', 'freq']], how='outer', on='state', suffixes=('', '_testing')).fillna(0)
    df_freqs_testing = df_testing.pivot(columns='x', index='y', values='freq_testing').sort_index(ascending=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18,6))
    fig.suptitle('Frequency of States', fontsize=24)
    sns.heatmap(df_freqs_training, annot=True, linewidths=.5, cbar=False, cmap=cmap, fmt = 'g', ax=axes[0], annot_kws={"fontsize":11})
    sns.heatmap(df_freqs_testing, annot=True, linewidths=.5, cbar=False, cmap=cmap, fmt = 'g', ax=axes[1], annot_kws={"fontsize":11})

    plt.show()
