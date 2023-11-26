import numpy as np
import matplotlib.pyplot as plt

def visualize(n,L,t_ind):
    num_samples = 30  # To be replaced with int(n,L)/normalization_factor

    # Ensure num_samples is not greater than the length of the arrays
    num_samples = min(num_samples, len(n[t_ind,:]))

    # Generate indices for evenly spaced samples
    indices = np.linspace(0, len(n[t_ind,:]) - 1, num_samples, dtype=int)

    L_sampled = L[indices]

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

    # Plot each circle
    for x, y, r in zip(x_coords, y_coords, L):
        circle = plt.Circle((x, y), r)
        ax.add_artist(circle)

    
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Show the plot
    plt.show()


