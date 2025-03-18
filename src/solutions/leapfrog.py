import numpy as np


def x_plus_1(x, deltat, v):
    return deltat*v + x

def v_plus_onehalf(a, deltat, m, v):
    return (a*deltat)/m + v



def harmonic_oscillator_leapfrog(ks, deltat, x_0, v_0, m):
    """
    Simulates the motion of a harmonic oscillator using the leapfrog integration method.

    Parameters:
    ks (list of float): List of spring constants for different simulations.
    deltat (float): Time step size.
    x_0 (float): Initial position of the oscillator.
    v_0 (float): Initial velocity of the oscillator.
    m (float): Mass of the oscillator (assumed to be 1 in this implementation).

    Returns:
    dict: A dictionary where each key is a spring constant k, and the value is a tuple 
          containing lists of positions and velocities over time.
    """
    data_per_k = dict()

    assert deltat > 0, f"deltat should be above 0, currently {deltat}"
    # loop over all different values for k, and calculate velocity and place
    for i, k in enumerate(ks):
        x = x_0

        # save initial datapoints
        all_xs = [x]
        all_vs = [v_0]

        # this is the case because m=1 
        a = -k*x
        max_deltax = 0
        v_half = v_0 + deltat/2 *a
        rangeje = int( 10//deltat)
        for t in range(rangeje):
            
            x_new = x_plus_1(x, deltat, v_half)
            if np.abs(x-x_new) > max_deltax:
                max_deltax = np.abs(x-x_new)
            x = x_new
            a = -k*x

            # calculate next step for velocity
            v_onehalf = v_plus_onehalf(a, deltat, m, v_half)
            v = (v_onehalf + v_half)/2

            # calculate velocity by aligning with x time-steps
            v = v_half + deltat/2*a
            all_xs.append(x)
            all_vs.append(v)

            v_half = v_onehalf
        data_per_k[k] = (all_xs, all_vs)

        print(f"max deltax for k: {k} = {max_deltax}")
    return data_per_k
