import networkx as nx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tabulate import tabulate

linestyles = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+',
              'x', 'D', 'd', '|', '_']
colormap = plt.cm.gist_ncar
colors = [colormap(i) for i in np.linspace(0, 0.9, len(linestyles))]


def plot_graph(weighted_adjacency_matrix, colors, size):
    # plot network to numpy image

    fig = plt.figure()
    M = np.around(weighted_adjacency_matrix, decimals=2)
    try:
        colors = np.around(colors, decimals=2)
    except:
        d = 7
    np.fill_diagonal(M, 0)
    G = nx.convert_matrix.from_numpy_array(M,
                                           parallel_edges=False,
                                           create_using=nx.DiGraph)
    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in G.edges(data=True)])
    pos = nx.circular_layout(G)
    node_colors = np.zeros((M.shape[0], 3))
    node_colors[:, 1] = colors[0:M.shape[0]]
    node_colors[:, 0] = 1 - colors[0:M.shape[0]]

    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.2)
    nx.draw(G, pos, node_color=node_colors)

    s, (width, height) = fig.canvas.print_to_buffer()
    # Option 2a: Convert to a NumPy array.
    graph_img = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    graph_img = graph_img[:, :, 0:3]
    graph_img = Image.fromarray(graph_img)
    graph_img = graph_img.resize(size)
    fig.clf()
    plt.close(fig)
    return np.array(graph_img)


def plot_histogram(histogram,
                   save_to=None,
                   show=False,
                   title="",
                   xlabel="",
                   ylabel="",
                   ):
    """ Plots a histogram using matplotlib

    Parameters
    ----------
    histogram: 1D numpy array,
        representing the weights for each bin.

    save_to: str, Optional
        file where the image should be saved (without extension)

    show: bool,
        If true then the image is shown.

    title: str,
        The title of the histogram.

    xlabel: str,
        label of the x axis in the plot

    ylabel: str,
        label of the y axis in the plot

    Returns
    -------
        numpy array representing the RGB image of the histogram.

    """

    y = histogram
    # Make a plot...
    fig = plt.figure()
    plot = fig.add_subplot(111)

    plot.hist(list(range(len(y))), weights=y, bins=len(y), color='#0504aa', alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    if not save_to is None:
        fig.savefig(save_to + ".pdf", bbox_inches='tight')

    if show:
        fig.show()
    fig.clf()
    plt.close(fig)
    return buf


def plot_function(x,
                  y,
                  size,
                  save_to=None,
                  show=False,
                  title="",
                  xlabel="",
                  ylabel="",
                  legends=None,

                  ):
    """ Plots a function using matplotlib

    Parameters
    ----------
    x: [1D numpy array],
        representing the x axis values. If non x axis values will automatically
        be generated.


    y: 1D numpy array,
    representing the x axis values

    save_to: str, Optional
        file where the image should be saved (without extension)

    show: bool,
        If true then the image is shown.

    title: str,
        The title of the histogram.

    xlabel: str,
        label of the x axis in the plot. 
    ylabel: str,
        label of the y axis in the plot

    Returns
    -------
        numpy array representing the RGB image of the histogram.

    """
    if x is None:
        x = list(range(0, len(y[0])))
    # Make a plot...
    fig = plt.figure()
    plot = fig.add_subplot(111)
    for i, yy in enumerate(y):
        plot.plot(x, yy, linestyles[i])

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not legends is None:
        plt.legend(legends)

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    if not save_to is None:
        fig.savefig(save_to + ".pdf", bbox_inches='tight')

    if show:
        fig.show()

    plot_img = Image.fromarray(buf)
    plot_img = plot_img.resize(size)
    fig.clf()
    plt.close(fig)
    return np.array(plot_img)


def html_table(table, save_to):
    """

    Parameters
    ----------
    table: 2D list
        2 dimensional array representing the table

    save_to:
        file where the table should be saved (without extension)

    Returns
    -------

    """
    html = tabulate(table, tablefmt='html')
    with open(save_to + ".html", "w") as text_file:
        text_file.write(html)
