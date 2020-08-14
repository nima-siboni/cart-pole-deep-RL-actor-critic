import numpy as np

def averager(input_array, stride_len):
    '''
    just an averager over strides
    '''

    n_array = np.shape(input_array)[0]

    line_id = 0

    averaged = []
    averaged = np.array(averaged)
    
    while (line_id < n_array - stride_len):

        averaged = np.append(averaged, np.mean(input_array[line_id: line_id + stride_len, :], axis=0))

        line_id += stride_len

    averaged = np.reshape(averaged, (-1, 2))
    return averaged


input_array = np.loadtxt('steps_vs_iteration_zero_epsilon.dat')

np.savetxt('averaged_steps_vs_iteration_zero_epsilon.dat', averager(input_array=input_array, stride_len=60))
