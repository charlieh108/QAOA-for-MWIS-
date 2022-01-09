# Script that takes the data collected by the angle finder functions
# data in the form

##  [[Graph 1] [Graph 2] ...]

#     [ Graph seed, [list_of_best], [all_results]]

# [  [ best_obj], [best success] ]

# [ angle seed , objective value for seed, success prob, [best angles]]


import numpy as np


def get_values_from_best_x(list_of_best_array):
    list_of_best_x = list_of_best_array[0]

    objective_value = list_of_best_x[1]
    success_prob = list_of_best_x[2]
    angles = list_of_best_x[3]

    return objective_value,success_prob,angles

def list_to_arrays(list_of_graphs):
    '''
    Takes the full list and returns
    list of angles that give the best objective values
    list of angles that give the best success values

    list of best success probs

    list of best objective values
    list of exact solution values
    '''

    exact_values = []

    best_objective_angles = []
    best_success_angles = []
    objective_values = []
    success_prob_values = []



    for graph in list_of_graphs:
        graph_seed = graph[0]
        exact_value = graph[1]
        exact_values.append(exact_value)
        list_of_best = graph[2]
        best_objective_values = list_of_best[0]
        best_success_values = list_of_best[1]

        obj_objective_value,obj_success_prob,obj_angles = get_values_from_best_x(best_objective_values)
        suc_objective_value,suc_success_prob,suc_angles = get_values_from_best_x(best_success_values)


        best_objective_angles.append(obj_angles)
        best_success_angles.append(suc_angles)

        success_prob_values.append(suc_success_prob)
        objective_values.append(obj_objective_value)

    return best_objective_angles,best_success_angles,success_prob_values,objective_values,exact_values


def best_angles(list_of_angles):

    '''
    takes the list of angles and returns an array of gammas and array of betas

    '''
    p = int(len(list_of_angles[0])/2)
    list_of_gammas =[]
    list_of_betas = []

    for angle_result in list_of_angles:
        gammas = angle_result[:p] # gammas first
        betas = angle_result[p:] # betas second

        list_of_gammas.append(gammas)
        list_of_betas.append(betas)

    array_of_gammas = np.array(list_of_gammas)
    array_of_betas = np.array(list_of_betas)

    return array_of_gammas,array_of_betas
