# Patterns in angles

import numpy as np

import data_analysis_best_angles as data
import matplotlib.pyplot as plt

import pickle


def get_best_angles(file_name):

    '''Takes the filename, runs the data analysis modules on it, and returns the best objective angles for gamma and beta
    '''

    with open(file_name,'rb') as handle:
        data_to_go = pickle.load(handle)
        handle.close()


    best_objective_angles_function,best_success_angles_function,success_prob_values_function,objective_values_function,exact_values_function = data.list_to_arrays(data_to_go)


    best_obj_gammas_function,best_obj_betas_function = data.best_angles(best_objective_angles_function)

    return best_obj_gammas_function,best_obj_betas_function


def angle_patterns_graph(angles):
    p = len(angles[0])

    i_array = np.arange(1,p+1,1)
    print(i_array)
    fig, ax = plt.subplots()
    #ax.plot(i, (betas % np.pi).transpose())
    ax.plot(i_array, angles.transpose())
    ax.grid()
    plt.show()

###########





gammas,betas = get_best_angles('unweighted_erdos_graph_N_6_p_6.pkl')  #### Enter in filename of pickle of optimum angles


print("gammas before",gammas)
print("after",gammas % np.pi )   # Play with symmetry !


angle_patterns_graph(gammas % np.pi)
