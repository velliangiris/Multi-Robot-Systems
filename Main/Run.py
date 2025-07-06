import IPSO.run
import GLS.run
import ACO_DWA.run
import LMBSWO.run
import MDHO_algorithm.run
import Proposed.run_main

def callmain(setup, n_r):
    Path_Length,Path_Smoothness,fitness = [], [], []
    if setup == 'Single Target':
        n_t = 1
    else: n_t = 2

    Proposed.run_main.main(n_r, n_t,Path_Length,Path_Smoothness,fitness)
    IPSO.run.main(n_r, n_t,Path_Length,Path_Smoothness,fitness)
    GLS.run.main(n_r, n_t,Path_Length,Path_Smoothness,fitness)
    ACO_DWA.run.main(n_r, n_t,Path_Length,Path_Smoothness,fitness)
    LMBSWO.run.main(n_r, n_t,Path_Length,Path_Smoothness,fitness)
    MDHO_algorithm.run.main(n_r, n_t,Path_Length,Path_Smoothness,fitness)


    return Path_Smoothness,Path_Length,fitness

