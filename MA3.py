""" MA3.py

Student: Wilma Ojala
Mail: wilma.ojala.5821@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import functools
from numba import njit
import time

# Exc1
def approximate_pi(n):
    # n is the number of points
    x_in = []
    y_in = []

    x_out = []
    y_out = []

    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if (x**2 + y**2) <= 1:
            x_in.append(x)
            y_in.append(y)

        else:   
            x_out.append(x)
            y_out.append(y)

    nc = len(x_in)
    pi_approx = (4 * nc) / n

    print(f'Number of points: {n}\nApproximation of pi: {pi_approx}')

    #Plotting 
    plt.figure(figsize=(6,6))
    plt.scatter(x_in, y_in, color='red')
    plt.scatter(x_out, y_out, color='blue')
    plt.xlabel('x-values')
    plt.ylabel('y-values')
    plt.title('Approximation of π using Monte Carlo')
    plt.savefig(f'pi_approx_{n}.png')
    #plt.show()
    plt.close()
    
    return pi_approx 


# Exc2, approximation 
def sphere_volume(n, d): 
    # n is the number of points
    # d is the number of dimensions of the sphere 
    p_lst = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]

    def singel_p_sum(p):
        square_coords = map(lambda x : x**2, p)
        return functools.reduce(lambda a, b: a + b, square_coords)

    sum_square_coords = map(singel_p_sum, p_lst)

    in_sphere = list(filter(lambda s: s <= 1, sum_square_coords))

    nc = len(in_sphere)
    volume_approx = (nc / n) * (2**d)

    return volume_approx


#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    Vd = (m.pi ** (d/2)) / (m.gamma((d/2) + 1))

    return Vd


#Exc3

#Function to calculate time
def time_count(func, n, d):
    times_lst = []
    result_lst = []

    for i in range(3):
        tstart = pc()
        result = func(n, d)
        tstop = pc()
        
        time_mem = tstop - tstart 
        result_lst.append(result)
        times_lst.append(time_mem)

        print(f'Run: {i+1} - Approximated Volume: {result_lst[i]} Time: {times_lst[i]} sec')
  

# Numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    nc = 0
    for _ in range(n):
        sum_square = 0
        for _ in range(d):
            x = random.uniform(-1, 1)
            sum_square += x**2

        if sum_square <= 1:
            nc += 1

    volume_approx = (nc / n) * (2**d)

    return volume_approx



#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallell(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    points_per_pro = n // np
    n_lst = [points_per_pro] * np
    d_lst = [d] * np

    with future.ProcessPoolExecutor() as ex:
        results = list(ex.map(sphere_volume, n_lst, d_lst))

    return sum(results) / np

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)


    # Exc2
    n = 100000
    d = 2
    print(f"Approximated volume of {d} dimentional sphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    
    n = 100000
    d = 11
    print(f"Approximated volume of {d} dimentional sphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")


    # Exc3
    n = 1000000
    d = 11
    print(f"Exc3: Sequential time of {d} and {n}:")
    time_count(sphere_volume, n, d)
    print(f"Exc3: Numba time of {d} and {n}:")
    sphere_volume_numba(10, 11) # Värmer upp numba
    time_count(sphere_volume_numba, n, d)


    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")

    start = pc()
    sphere_volume_parallell(n, d)
    stop = pc()
    print(f"\nExc4: Parallel testing of {d} and {n}: {start-stop}")


if __name__ == '__main__':
	main()
