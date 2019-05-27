import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import string


##Various algorithms log order per column
a2c_columns={"episode":0,"score":1,"total_loss":2,"policy_loss":3,"value_loss":4}
per_ddpg_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"critic_loss":6}
dqn_columns = {"episode":0,"episode_step":1,"total_step":2,"total_score":3,"unused":4,"total_loss":5,"avg_q_value":6,"track_name":7,"race position":8,"max_speed":9,"avg_speed":10}
#dqn_columns = {"episode":0,"episode_step":1,"total_step":2,"total_score":3,"epsilon":4,"loss":5,"avg_q_value":6} ##This is not working with Torcs_dqn_5400
sac_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"qf_1_loss":6,"qf_2_loss":7,"vf_loss":8,"alpha_loss":9,"track_name":10,"race_position":11,"max_speed":12,"avg_speed":13,"max_rolling_total_score":14}
t3d_columns ={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor:loss":5,"critic1_loss":6,"critic2_loss":7,"track_name":8,"race_position":9,"max_speed":10,"avg_speed":11}

algo_column_list={
    "a2c":a2c_columns,
    "per-ddpg":per_ddpg_columns,
    "dqn": dqn_columns,
    "sac": sac_columns,
    "SAC": sac_columns,
    "SACLSTM": sac_columns,
    "t3d": t3d_columns
}



def get_column_indice(algo,column_name):
    columns = algo_column_list[algo]
    column_indice=columns[column_name]
    return column_indice

def get_column_values(algo,column_name):
    columns = algo_column_list[algo]
    column_indice=columns[column_name]
    return column_indice

def get_color(color_index):
    if color_index==0:
        return "red"
    elif color_index==1:
        return "blue"
    elif color_index==2:
        return "green"
    else:
        return "magenta"



def plot_multi_algo_single_feature(files, x_column, y_column, smooth_factor=100):
    color_index = 0

    # find min_episode for plots
    # load logs
    all_logs = []
    for logfilename_name in files:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_episode = np.min([len(logs) for logs in all_logs])

    for file in files:
        # Find algorithm type
        algo = file.split("_")[1]
        print("algo:" + algo)

        # Get column indices
        x_indice = get_column_indice(algo, x_column)
        y_indice = get_column_indice(algo, y_column)
        print(
            "x_column:" + x_column + " indice:" + str(x_indice) + " y_column:" + y_column + " indice:" + str(y_indice))

        # Get column values
        x_values, y_values = read_log_file(file, x_indice, y_indice)

        # Moving average y_values
        y_values = smoother(y_values, smooth_factor)

        #Cut exceeding episodes
        y_values = y_values[:min_episode]
        x_values = x_values[:len(y_values)] ##smoothing reduces the array length a bit


        # Get line color and advance to new color
        color = get_color(color_index)
        color_index += 1

        # Plot
        #plt.plot(y_values, color=color, label=algo)
        plt.plot(x_values,y_values, color=color, label=algo)

        plt.xlabel(x_column)
        plt.ylabel(y_column)
    plt.legend()
    plt.show()

def plot_algo_per_track(files, x_column, y_column,tracks, smooth_factor=100,graph_title=""):
    color_index = 0

    # find min_iteration for plots
    # load logs
    all_logs = []
    for logfilename_name in files:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_iteration = np.min([len(logs) for logs in all_logs])


    for track in tracks:

        for file in files:
            # Find algorithm type
            algo = file.split("_")[1]
            print("algo:" + algo)

            # Get column indices
            x_indice = get_column_indice(algo, x_column)
            y_indice = get_column_indice(algo, y_column)
            print(
                "x_column:" + x_column + " indice:" + str(x_indice) + " y_column:" + y_column + " indice:" + str(y_indice))

            # Get column values
            df = read_log_file_to_df(file)

            #df["track_name"].unique()

            # Only relevant tracks
            if len(tracks)>0:
                df = df[df["track_name"]==track]

            x_values = df[x_column]
            y_values = df[y_column]

            # Moving average y_values
            y_values = smoother(y_values, smooth_factor)

            #Cut exceeding episodes
            y_values = y_values[:min_iteration]
            x_values = x_values[:len(y_values)] ##smoothing reduces the array length a bit

            # Get line color and advance to new color
            color = get_color(color_index)
            color_index += 1

            # Plot
            #plt.plot(y_values, color=color, label=algo)
            #plt.plot(x_values,y_values, color=color, label=algo+"-"+track)
            plt.plot(x_values, y_values, label=algo + "-" + track)
            plt.savefig('graphs/test' + algo+".jpg", bbox_inches='tight')

            plt.xlabel(x_column)
            plt.ylabel(y_column)

    plt.legend()
    plt.title(graph_title)
    persist_figure(plt,"tracks")
    plt.show()


def smoother(array, ws):
    """ Return smoothed array by the mean filter """
    return np.array([sum(array[i:i+ws])/ws for i in range(len(array) - ws)])


def read_log_file(filename, x_indice, y_indice):
    x_values=[]
    y_values=[]

    file = open(filename, "r")

    #skip first 2 lines
    file.readline()
    file.readline()

    #read lines and get values
    for line in file:
        tokens = line.split(";")
        x_value=int(tokens[x_indice])
        y_value = float(tokens[y_indice])
        x_values.append(x_value)
        y_values.append(y_value)

    return x_values,y_values

def read_log_file_to_df(filename):
    # Find algorithm type
    algo = filename.split("_")[1]

    file = open(filename, "r")

    columns=algo_column_list[algo].keys()
    column_names= list(columns)
    #read file to dataframe skipping the initial 2 lines
    df=pd.read_csv(file, names=column_names,sep = ';', skiprows=2)

    return df


def persist_figure(plt,title):
    rnd_filename= ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    plt.savefig('graphs/' + title+"_"+rnd_filename+".jpg", bbox_inches='tight')


def plot_same_algo_different_runs(logfilename_name_pairs, texts=[[""] * 3], smooth_factor=100):

    # find min_iteration for plots
    # load logs
    all_logs = []
    for logfilename_name,name in logfilename_name_pairs:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_iteration = np.min([len(logs) for logs in all_logs])


    for i, (title, xlabel, ylabel,y_axis_title) in enumerate(texts):

        #load logs
        all_logs=[]
        for logfilename_name,name in logfilename_name_pairs:
            logs = read_log_file_to_df(logfilename_name)
            all_logs.append(logs)

        #cut lenght to the min epiosode of logs
        min_episode = np.min([len(logs) for logs in all_logs])
        all_trimmed_logs =[]
        for logs in all_logs:
            all_trimmed_logs.append(logs.head(min_episode))

        smoothed_logs = np.stack([smoother(logs[ylabel], smooth_factor) for logs in all_trimmed_logs])

        std_logs = np.std(smoothed_logs, axis=0)
        mean_logs = np.mean(smoothed_logs, axis=0)
        max_logs = np.max(smoothed_logs, axis=0)
        min_logs = np.min(smoothed_logs, axis=0)

        plt.xlabel(xlabel)
        plt.ylabel(y_axis_title)
        plt.title(name)

        #Cut missing steps due to smoothing
        x_values = logs[xlabel][:len(mean_logs)]

        plt.plot(x_values,mean_logs, label=title)

        plt.legend()
        plt.fill_between(np.arange(len(mean_logs)),
                         np.minimum(mean_logs+std_logs, max_logs),
                         np.minimum(mean_logs-std_logs, min_logs),
                         alpha=0.4)

    persist_figure(plt,y_axis_title+"_"+name);
    plt.show()


##TESTS
if __name__ == "__main__":


    ##PLOT 1 Loss graphs min,max,mean,std of the runs with smoothing
    ## with multiple features ( i.e. Max Speed , Average speed)
    plot_texts = [
        [
            "total_step",#x_column_legend
            "total_step",#x_column value
            "total_loss",## y_column value
            "Loss",##y_axis title
        ],
    ]
    plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM")], texts=plot_texts,smooth_factor=20)
    #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN")],texts=plot_texts, smooth_factor=20)



    ##PLOT 2 Multiple runs of same algorithm plots min,max,mean,std of the runs with smoothing
    ## with multiple features ( i.e. Max Speed , Average speed)
    plot_texts = [
        [
            "Max Speed",#x_column_legend
            "total_step", #x_column value
            "max_speed", ## y_column value
            "speed" ##y_axis title
        ],
        [
            "Average Speed",#x_column_legend
            "total_step",#x_column value
            "avg_speed",## y_column value
            "speed",##y_axis title

        ]
    ]

    plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM"), ],texts=plot_texts, smooth_factor=50)
    #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN"), ],texts=plot_texts, smooth_factor=50)


    ##PLOT 3 - Rewards
    ## with multiple features ( i.e. Max Speed , Average speed)
    plot_texts = [
        [
            "episode_step",#x_column_legend
            "total_step", #x_column value
            "total_score", ## y_column value
            "Reward" ##y_axis title
        ]
    ]

    plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM")],texts=plot_texts, smooth_factor=100)
    #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN") ],texts=plot_texts, smooth_factor=200)

    ##PLOT 3.5 - Max Cumulative Rewards
    ## with multiple features ( i.e. Max Speed , Average speed)
    plot_texts = [
        [
            "episode_step",#x_column_legend
            "total_step", #x_column value
            "max_rolling_total_score", ## y_column value
            "Max_Rolling_Reward" ##y_axis title
        ]
    ]

    plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM")],texts=plot_texts, smooth_factor=100)
    #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN") ],texts=plot_texts, smooth_factor=200)




    ##PLOT 4 Compare different algos against same feature (i.e. max_reward)
    #x_column="episode"
    x_column = "total_step"
    y_column="total_score"
    log_filenames=["releases/TORCS_SACLSTM_512256128_L1_EP4000_N1_G99_trimmed.log", "releases/Torcs_sac_f622afd.log","releases/Torcs_dqn_a11ccc2.log"]
    #plot_multi_algo_single_feature(log_filenames, x_column=x_column, y_column=y_column, smooth_factor=1000)


    ##PLOT 5 Compare algos against tracks
    x_column = "total_step"
    y_column="max_speed"
    #log_filenames=["releases/TORCS_SAC_N4_G99_2000.log", "releases/TORCS_SACLSTM_512256128_L1_EP2000_N1_G99.log"]
    log_filenames = ["releases/TORCS_SAC_N4_G99_2000_trimmed.log", "releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log"]
    tracks_all=['e-track-2','alpine-1','g-track-1']
    plot_algo_per_track(log_filenames, tracks=tracks_all, x_column=x_column, y_column=y_column, smooth_factor=10,graph_title="Max Speed per track and algo")


    ##PLOT 6 Compare algos against tracks
    x_column = "total_step"
    y_column="total_score"
    #log_filenames=["releases/TORCS_SAC_N4_G99_2000.log", "releases/TORCS_SACLSTM_512256128_L1_EP2000_N1_G99.log"]
    log_filenames = ["releases/TORCS_SAC_N4_G99_2000_trimmed.log", "releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log"]
    tracks_all = ['e-track-2', 'alpine-1', 'g-track-1']
    plot_algo_per_track(log_filenames, tracks=tracks_all, x_column=x_column, y_column=y_column, smooth_factor=20,graph_title="Reward per track and algo")








