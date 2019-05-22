import matplotlib.pyplot as plt


##Various algorithms log order per column
a2c_columns={"episode":0,"score":1,"total_loss":2,"policy_loss":3,"value_loss":4}
per_ddpg_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"critic_loss":6}
dqn_columns = {"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"avg_q_value":5,"track_name":6,"race position":7,"max_speed":8,"avg_speed":9}
sac_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"qf_1_loss":6,"qf_2;loss":7,"vf_loss":8,"alpha_loss":9,"track_name":10,"race_position":11,"max_speed":12,"avg_speed":13}


algo_column_list={
    "a2c":a2c_columns,
    "per-ddpg":per_ddpg_columns,
    "dqn": dqn_columns,
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


def plot_algo_features(files,x_column,y_columns):
    #TODO
    pass


def plot_algos(files,x_column,y_column):

    color_index=0

    for file in files:
        #Find algorithm type
        algo=file.split("_")[1]
        print("algo:"+algo)

        #Get column indices
        x_indice = get_column_indice(algo,x_column)
        y_indice = get_column_indice(algo,y_column)
        print("x_column:" + x_column + " indice:" + str(x_indice) + " y_column:" + y_column + " indice:" + str(y_indice))

        #Get column values
        x_values,y_values = read_log_file(file,x_indice,y_indice)

        #Get line color and advance to new color
        color = get_color(color_index)
        color_index+=1

        #Plot
        plt.plot(x_values, y_values,color=color, label=algo)

        plt.xlabel(x_column)
        plt.ylabel(y_column)
    plt.legend()
    plt.show()


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

##TESTS
if __name__ == "__main__":

    x_column="episode"
    y_column="total_score"
    #log_filenames=["logs\Torcs_dqn_xyzw12345.txt", "logs\Torcs_per-ddpg_f18c5ea.txt"]
    log_filenames = ["logs\Torcs_per-ddpg_f18c5ea.txt"]

    plot_algos(log_filenames, x_column=x_column, y_column=y_column)