#from model_modified import generate_model, train_model, test_model
import copy
def lesion(model, neurons_lesion): 

    # initialize lesioned model
    model_lesion = copy.deepcopy(model)
    
    #W_in, W_rc, W_fb, W_out = model['W_in'], model['W_rc'], model['W_fb'], model['W_out']
    if len(neurons_lesion) > 0:
        model_lesion['W_in'][neurons_lesion, :] = 0
        model_lesion['W_rc'][neurons_lesion, :] = 0
        model_lesion['W_rc'][:, neurons_lesion] = 0
        model_lesion['W_fb'][neurons_lesion, :] = 0
        #model_lesion['W_out'][:, neurons_lesion] = 0

    return model_lesion