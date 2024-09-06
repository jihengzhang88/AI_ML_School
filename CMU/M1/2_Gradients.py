import numpy as np
import matplotlib.pyplot as plt

def L(w1, w2):
    return np.cos(4*w1 + w2/4 - 1) + w2*w2 + 2*w1*w1

def dLdw1(w1, w2):
    # YOUR CODE GOES HERE
    return -4*np.sin(4*w1 + w2/4 - 1) + 4*w1

def dLdw2(w1, w2):
    # YOUR CODE GOES HERE
    return -1/4*np.sin(4*w1 + w2/4 - 1) + 2*w2


"""
#ONLY work in Jupyter Notebook as it allows interactive widgets like sliders to be displayed in line
%matplotlib inline
from ipywidgets import interact, interactive, fixed, interact_manual, Layout, FloatSlider, Dropdown

def plot_gd(w1, w2, log_stepsize, log_steps):
    stepsize = 10 ** log_stepsize
    steps = int(10 ** log_steps)

    # Gradient Descent
    w1s = np.zeros(steps + 1)
    w2s = np.zeros(steps + 1)

    for i in range(steps):
        w1s[i], w2s[i] = w1, w2
        w1 = w1 - stepsize * dLdw1(w1s[i], w2s[i])
        w2 = w2 - stepsize * dLdw2(w1s[i], w2s[i])
    w1s[steps], w2s[steps] = w1, w2

    # Plotting
    vals = np.linspace(-1, 1, 50)
    x, y = np.meshgrid(vals, vals)
    z = L(x, y)

    plt.figure(figsize=(7, 5.8), dpi=120)
    plt.contour(x, y, z, colors="black", levels=np.linspace(-.5, 3, 6))
    plt.pcolormesh(x, y, z, shading="nearest", cmap="Blues")
    plt.colorbar()

    plt.plot(w1s, w2s, "g-", marker=".", markerfacecolor="black", markeredgecolor="None")
    plt.scatter(w1s[0], w2s[0], zorder=100, color="blue", marker="o", label=f"$w_0$ = [{w1s[0]:.1f}, {w2s[0]:.1f}]")
    plt.scatter(w1, w2, zorder=100, color="red", marker="x", label=f"$w^*$ = [{w1:.2f}, {w2:.2f}]")
    plt.legend(loc="upper left")

    plt.axis("equal")
    plt.box(False)
    plt.xlabel("$w_1$")
    plt.ylabel("$w_2$")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f"Step size = {stepsize:.0e}; {steps} steps")
    plt.show()


slider1 = FloatSlider(
    value=0,
    min=-1,
    max=1,
    step=.1,
    description='w1 guess',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=False,
    layout = Layout(width='550px')
)

slider2 = FloatSlider(
    value=0,
    min=-1,
    max=1,
    step=.1,
    description='w2 guess',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=False,
    layout = Layout(width='550px')
)

slider3 = FloatSlider(
    value=-1.5,
    min=-3,
    max=0,
    step=.5,
    description='step size',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=False,
    layout = Layout(width='550px')
)

slider4 = FloatSlider(
    value=2,
    min=0,
    max=3,
    step=.25,
    description='steps',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=False,
    layout = Layout(width='550px')
)


interactive_plot = interactive(
    plot_gd,
    w1 = slider1,
    w2 = slider2,
    log_stepsize = slider3,
    log_steps = slider4,
    )
output = interactive_plot.children[-1]
output.layout.height = '620px'

interactive_plot

"""