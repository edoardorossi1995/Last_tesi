#def sub2ind(array_shape, rows, cols):
#    if isinstance(rows, int) and isinstance(cols, int):
#        return rows * array_shape[1] + cols
#    else:
#        return [row * array_shape[1] + col for row, col in zip(rows, cols)]
    
import numpy as np

def ind2sub(array_shape, linear_index):
    #num_dims = len(array_shape)
    subscripts = []
    for dim_size in array_shape[::-1]:
        subscripts.insert(0, linear_index % dim_size)
        linear_index //= dim_size
    return tuple(subscripts)

def sub2ind3d(array_shape, coords):
   
    if len(array_shape) != len(coords):
        raise ValueError("La lunghezza di array_shape e coords devono essere uguali")

    if any(coord < 0 or coord >= shape for coord, shape in zip(coords, array_shape)):
        raise ValueError("Coordinate non valide per l'array")

    index = 0
    for i, coord in enumerate(coords):
        index += coord * np.prod(array_shape[i + 1:])
    return index

# from nn [a1, a2] to action F, VR, VS
def take_action(x_list, f_min, f_max):

    if x_list[0] < 0 or x_list[0] > 1:
        raise ValueError("Le azioni devono essere comprese tra 0 e 1 \n Errore su x_list[0]: ", x_list[0])
    if x_list[1] < 0 or x_list[1] > 1:
        raise ValueError("Le azioni devono essere comprese tra 0 e 1 \n Errore su x_list[1]: ", x_list[1])

    F = (1 - x_list[0]) * f_min + x_list[0] * f_max
    if x_list[1] == 0.5:
        VR = 0
        VS = 0
    else:
        if x_list[1] < 0.5:
            VR = 2 * (0.5 - x_list[1])
            VS = 0
        else: 
            VR = 0
            VS = 2 * (x_list[1] - 0.5)
    return F, VR, VS
            

def ind2sub3d(ind, shape):

    rows = shape[0]
    cols = shape[1]
    slices = shape[2]
    row = ind // (cols * slices)
    col = (ind // slices) % cols
    slice_idx = ind % slices
    return (row, col, slice_idx)