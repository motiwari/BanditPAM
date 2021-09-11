import BanditPAM
import numpy as np
#BanditPAM.set_num_threads(1)
def multiply(a,b):
    print("Will compute ", a, " times ", b)
    lst_1 = np.array(a)
    lst_2 = np.array(b)
    c = lst_1 - lst_2
    #print("c====",c)
    c = np.linalg.norm(c)
    print("c====",c)
    return c

multiply(2,3)

