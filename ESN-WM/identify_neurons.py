import numpy as np 

def identify_neurons(W_out, num_neurons = 10): 

    most_significant_neurons = {}
    # Find the indices of the neurons with the largest norms, 
    # and then compare with neurons with the largest absolute value along each output dimension
    norms_all = np.linalg.norm(W_out, ord=None, axis=0, keepdims=False)
    print(norms_all.shape, max(norms_all), min(norms_all)) 
    assert len(norms_all) == 1000 # should compute 1 norm value for all 1000 neurons
    if W_out.shape[0] == 3:
        assert norms_all[0] == np.sqrt(W_out[0][0]**2 + W_out[1][0]**2 + W_out[2][0]**2)
    
    most_significant_neurons['most significant, all outputs'] = np.array([])
    most_significant_neurons['least significant, all outputs'] = np.array([])
    most_significant_neurons['random, all outputs'] = np.array([])
    for i in range(W_out.shape[0]): 
        most_significant_neurons[f'most significant, output {i}'] = np.array([])
        most_significant_neurons[f'least significant, output {i}'] = np.array([])
        most_significant_neurons[f'random, output {i}'] = np.array([])


    if num_neurons > 0: 
        most_significant_neurons['most significant, all outputs'] = np.argsort(norms_all)[-num_neurons:]
        most_significant_neurons['least significant, all outputs'] = np.argsort(norms_all)[:num_neurons]
        most_significant_neurons['random, all outputs'] = np.random.choice(np.argsort(norms_all), size=num_neurons, replace=False)
        for i in range(W_out.shape[0]): 
            norms = np.abs(W_out[i, :])
            most_significant_neurons[f'most significant, output {i}'] = np.argsort(norms)[-num_neurons:]
            most_significant_neurons[f'least significant, output {i}'] = np.argsort(norms)[:num_neurons]
            most_significant_neurons[f'random, output {i}'] = np.random.choice(np.argsort(norms), size=num_neurons, replace=False)

    

    return most_significant_neurons