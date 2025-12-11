import numpy as np
from skimage import color

def mapa_deltaE_ciede2000(real, gerada):
    #normaliza para [0,1]
    if real.dtype == np.uint8:
        real = real.astype("float32") / 255.0
    if gerada.dtype == np.uint8:
        gerada = gerada.astype("float32") / 255.0
    
    lab_real = color.rgb2lab(real)
    lab_gerada = color.rgb2lab(gerada)

    deltaE = color.deltaE_ciede2000(lab_real, lab_gerada)
    
    return deltaE