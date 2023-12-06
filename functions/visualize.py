import numpy as np
import matplotlib.pyplot as plt

def visualize(n,L,t_ind, num_scale = 100000, dot_scale=4, alpha=0.7):
    # num_samples = 10  # To be replaced with int(n,L)/normalization_factor
    norm_factor = np.trapz(n[-1,:], L)
    num_samples = int(num_scale*np.trapz(n[t_ind,:], L)/norm_factor)
    # print(num_samples)

    # assuming that L is uniform. The first entry is 0 and we dont want that,
    # so let's plot the "upper end" of each bin instead by adding dL to all the lengths
    dL = L[1]
    L_sampled = dL + np.random.choice(L, num_samples, replace=True, p=n[t_ind,:]/np.sum(n[t_ind,:]))
    num_circles = len(L_sampled)

    x_min=0
    y_min=0
    x_max=10
    y_max=10

    # Generate random x and y coordinates for circle centers
    x_coords = np.random.uniform(low=x_min, high=x_max, size=num_circles) 
    y_coords = np.random.uniform(low=y_min, high=y_max, size=num_circles)  

    # Create a plot
    
    fig, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, dot_scale*L_sampled, alpha=alpha)

    # Plot each circle
    
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('t='+str(t_ind))

    # Show the plot
    plt.show()


