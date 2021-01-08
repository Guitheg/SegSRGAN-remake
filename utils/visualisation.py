import numpy as np
from matplotlib import pyplot as plt
from typing import Union

def visualiser(img_data : np.ndarray, axis : int, number : Union[float, int], show : bool = True, *args, **kwargs):
    """Visualisation d'une image 3d (qui provient d'un .nii)

    Args:
        img_data ([np.array]): [le tableau numpy qui provient de l'image nii]
        axis ([int]): [le numéro de l'axe (de 0 à 2)]
        number ([float]): [le pourcentage correspondant à la position de la tranche à afficher]
        show (bool, optional): [Si on veut que ça s'affichet par un plt.show()]. Defaults to True.

    Raises:
        Exception: [si l'axes n'est pas compris en 0 et 2]
    """
    if type(number) == float:
        x,y,z = img_data.shape
    else : 
        x,y,z = 1,1,1
    if axis==0:
        plt.imshow(img_data[int(x*number),:,:], *args, **kwargs)
    elif axis==1:
        plt.imshow(img_data[:,int(y*number),:], *args, **kwargs)
    elif axis==2:
        plt.imshow(img_data[:,:,int(z*number)], *args, **kwargs)
    else:
        raise Exception( "axis compris entre 0 et 2")
    if show == True:
        plt.show()
    
def visualiser_n_img(list_img : list, axis:int, number:float, *args, **kwargs):
    """Visualisation simultanée d'image 3d provenant de .nii

    Args:
        list_img (list<nd.array>): [list des images 3d .nii sous tableau numpy]
        axis ([int]): [le numéro de l'axe (de 0 à 2)]
        number ([float]): [le pourcentage correspondant à la position de la tranche à afficher]
    """

    fig = plt.figure(1)
    n_img = len(list_img)
    for i, img in enumerate(list_img):
        fig.add_subplot(1, n_img, i+1)
        visualiser(img, axis, number, show=False, *args, **kwargs)
    plt.show()