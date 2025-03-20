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
    for k in ks:
        x = x_0

        # save initial datapoints
        all_xs = [x]
        all_vs = [v_0]

        # this is the case because m=1 
        a = -k*x
        v_half = v_0 + deltat/2 *a
        v_half = 0.5*(v_0 + (v_0 + deltat*a))

        # calculate the number of steps until one period is completed
        rangeje = int(((2*np.pi)//np.sqrt(k) +1)/deltat)
        for t in range(rangeje):
            
            x_new = x_plus_1(x, deltat, v_half)
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

    return data_per_k


def extra_force(freq, deltat, iter, f_0=1):
    return f_0* np.sin(freq*deltat*iter)

def v_plus_onehalf_force(freq, deltat, iter, a, m, v):
    return (extra_force(freq, deltat, iter) +  a)* (deltat/m) + v

def harmonic_oscillator_extra_force(k, deltat, xs, v_0, m, freq=2, time=10):
    """
    Simulates the motion of a harmonic oscillator including an extra force using the leapfrog integration method.

    Parameters:
    k (float): spring constant. 
    deltat (float): Time step size.
    xs (List): Initial positions of the oscillator.
    v_0 (float): Initial velocity of the oscillator.
    m (float): Mass of the oscillator (assumed to be 1 in this implementation).

    Returns:
    List: a List for each x_0 containing lists of positions and velocities over time.
    """
    data_per_x0 = []

    assert deltat > 0, f"deltat should be above 0, currently {deltat}"
    # loop over all different values for k, and calculate velocity and place
    for x_0 in xs:
        x = x_0

        # save initial datapoints
        all_xs = [x]
        all_vs = [v_0]

        # this is the case because m=1 
        a = -k*x
        v_half = v_0 + deltat/2 *a

        # calculate the number of steps until time is reached
        rangeje = int(time//deltat)
        for i, t in enumerate(range(rangeje)):
            
            x_new = x_plus_1(x, deltat, v_half)
            x = x_new
            a = -k*x

            # calculate next step for velocity
            v_onehalf = v_plus_onehalf_force(freq,deltat, i, a, m, v_half)
            v = (v_onehalf + v_half)/2

            # calculate velocity by aligning with x time-steps
            v = v_half + deltat/2*a
            all_xs.append(x)
            all_vs.append(v)

            v_half = v_onehalf
        data_per_x0.append((all_xs, all_vs))

    return data_per_x0