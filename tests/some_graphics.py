
'''
def integerise(x):
    """
    Convert x to string without dot if it is an integer ('2.' -> '2').
    
    Parameters
    -------
    x: number 
    
    Returns
    -------
    string
        Integerised 'x'.
    """
    
    if np.isclose(x, round(x), atol=1e-3 ):
        return '{:.0f}'.format(x)
    else:
        return '{}'.format(x) # '{:.1f}'.format(x)
    

def import_msh(file_name):
    """
    Import file '.msh' (as explained in the FreeFEM++ documentation - https://doc.freefem.org/documentation/mesh-generation.html#format-of-mesh-data):
     - heading: number of vertices, number of triangles, number of edges on boundary
     - vertices: x, y, boundary label
     - triangles: index1, index2, index3, region label
     - edges: index1, index2, boundary label

    Parameters
    -------
    file_name : string
    
    Returns
    -------
    np.array
        A 3-n array of points on the boundary.
    np.array
        A 4-n array of triangles composing the mesh.
    np.array
        A 3-n array of edges composing the boundary.    
    """
    
    heading = np.loadtxt(file_name, max_rows=1, dtype=int)
    vertices_mesh = np.loadtxt(file_name, skiprows=1, max_rows=heading[0])
    triangles_mesh = np.loadtxt(file_name, skiprows=1+heading[0], max_rows=heading[1], dtype=int)
    edges_mesh = np.loadtxt(file_name, skiprows=1+heading[0]+heading[1], dtype=int)
    return vertices_mesh, triangles_mesh, edges_mesh

def extract_branches(vertices_mesh, edges_mesh, network_seeds):
    """
    Construct branches of the network from mesh edges.
    
    Parameters
    -------
    vertices_mesh : np.array
        A list of points composing the network.
    edges_mesh : np.array
        A list of edges composing the network.
    seed : list
        A list of indices in the list of points marking where the network starts.
    
    Returns
    -------
    np.array
        A list of 2-n arrays containing xy coordinates of points composing the branch.  
    """
    branches=[]
    while len(network_seeds)>0:
        seed_tmp=[network_seeds[0]]
        long_line = [edges_mesh[edges_mesh[:,1]==seed_tmp,0][0]]
        while len(seed_tmp)==1:
            long_line.append(seed_tmp[0])
            seed_tmp=edges_mesh[edges_mesh[:,0]==long_line[-1],1]
        network_seeds=np.append(network_seeds,seed_tmp)
        network_seeds=network_seeds[1:]
        branches.append(vertices_mesh[np.asarray(long_line)-1])
    return branches

def plot_tree(ax, file_name, height=2.0, width=2.0, **kwargs_tree_plot):
    """
    Plot a tree with optional arguments from **kwargs_tree and black box.    
    
    Parameters
    -------
    ax : Axes 
        Object to plot on.
    branches : list
        A list of branch arrays containing xy coordinates of the branch.
    height : float, default 2.0
        y limit to plot.
    kwargs_plots : dict, default {}
        Arguments to plot the tree.
    
    Returns
    -------
    none
    """
    options_tree_plot = {'color': '#0066CC', 'linewidth': 0.8}
    options_tree_plot.update(kwargs_tree_plot)
    
    box_indices = np.array([1, 2, 3, 4, 5, 1])
    
    [vertices_mesh, triangles_mesh, edges_mesh] = import_msh(file_name)
    branches = extract_branches(vertices_mesh, edges_mesh, network_seeds=[6])
    box = vertices_mesh[box_indices-1]
    
    # PLOT LINES
    for line in branches:
        ax.plot(line[:,0], line[:,1], **options_tree_plot)
    # PLOT BOX
    ax.plot(box[:,0], box[:,1], linewidth=options_tree_plot['linewidth']*2, color='0')
        
    ax.axis('off')
    # colouring background 
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)
    ax.set_facecolor('#def1ff')
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

def calculate_quantiles(data, q=0.25):
    """
    Calculate quantiles and width of a given data.
    
    Parameters
    -------
    data : np.array 
            
    Returns
    -------
    np.array
        A 1-7 array with:
        - quantiles of data of order 50%, q, 1-q
        - quantiles of abs(data) of order 50%, q, 1-q
        - width: (quant. of order q) - (quantiles of order 1-q)
        [q50, q32, q68, q50(abs), q32(abs), q68(abs), width]
        
    """
    q32 = np.quantile(data, q)
    q50 = np.quantile(data, 0.5)
    q68 = np.quantile(data, 1-q)
    q32_abs = np.quantile(abs(data), q)
    q50_abs = np.quantile(abs(data), 0.5)
    q68_abs = np.quantile(abs(data), 1-q)
    width = q68 - q32
    return np.array([q50, q32, q68, q50_abs, q32_abs, q68_abs, width])


def gather_results(eta_range, folder_name, stream_min=0, stream_max=1e5, step_min=0, step_max=1e5, q=0.25):
    """
    Gathering results from backward evolution with filtering set by stream/step min/max. Filtering done with less/greater-equal (<= or >=).

    Parameters
    -------
    eta_range : array
        An array of eta for which we gather data.
    folder_name : string
        A directory from which results will be gathered.
    stream_min : int, default=0
        Minimum stream index to be analysed.
    stream_max : int, default=1e5
        Maximum stream index to be analysed. 
    step_min : int, default=0
        Maximum step in backward evolution (on one branch) to be analysed.
    step_max : int, defult=1e5
        Maximum step in backward evolution (on one branch) to be analysed.
    
    Returns
    -------
    np.array
        A 3-len(eta_range)-7 array of results calculated in 'calculate_quantiles' including:
            - geometric deviation metric
            - angular deviation metric
            - local symmetry metric 
    ( Above len(eta_range)-7 subarrays compose of columns:  q50, q32, q68, q50(abs), q32(abs), q68(abs), width )
    """    

    results = np.empty((3, eta_range.shape[0], 7))
    
    for i, eta in enumerate(eta_range): 
        streams, steps_back = np.loadtxt('{}{}{}.txt/'.format(folder_name, file_backward, integerise(eta*10)), \
                                         usecols=(stream_column, step_column), delimiter=',', dtype='int', unpack=True)
        filter_data = (stream_min<=streams) & (streams<=stream_max) & (step_min<=steps_back) & (steps_back<=step_max)

        a1, a2, a3, geo, ang = np.loadtxt('{}{}{}.txt/'.format(folder_name, file_backward, integerise(eta*10)), \
                                          usecols=(a1_column, a2_column, a3_column, geo_dev_column, ang_dev_column), \
                                          delimiter=',', unpack=True)[:,filter_data]

        results[0, i,:] = calculate_quantiles(a2/a1**2, q)
        results[1, i,:] = calculate_quantiles(geo, q)
        results[2, i,:] = calculate_quantiles(ang, q)
    
    return results


def gather_results_bif(eta_range, folder_name):
    """
    Gathering results from bifurcation points in backward evolution.

    Parameters
    -------
    eta_range : array
        An array of eta for which we gather data.
    folder_name : string
        A directory from which results will be gathered.
    
    Returns
    -------
    np.array
        A 3-len(eta_range)x7 array of results in bifurcation points calculated in 'calculate_quantiles' including: 
            - length mismatch
            - a1 values
            - a3/a1 values
    ( Above len(eta_range)-7 subarrays compose of columns:  q50, q32, q68, q50(abs), q32(abs), q68(abs), width )
    
    Notes
    -------
    Gathering results from all bifurcation points seperately can be uncommented.
    """
    
    # bifs = np.loadtxt('{}{}{}.txt'.format(folder_name, file_bif, integerise(eta_original*10)), \
    #                   usecols=(0), delimiter=',', dtype='int')
    # a1_all_bif = np.empty((eta_range.shape[0],bifs.shape[0]))
    # a3_a1_all_bif = np.empty((eta_range.shape[0],bifs.shape[0]))
    # len_mis_all_bif = np.empty((eta_range.shape[0],bifs.shape[0]))

    results_bif = np.empty((3, eta_range.shape[0],7))
    for i, eta in enumerate(eta_range):
        a1, a2, a3, len_mis = np.loadtxt('{}{}{}.txt'.format(folder_name, file_bif, integerise(eta*10)), \
                                          usecols=(a1_column_bif, a2_column_bif, a3_column_bif, len_mis_column_bif), \
                                          delimiter=',', unpack=True)
        filter = a1>0
        results_bif[0, i,:] = calculate_quantiles(len_mis[filter])
        results_bif[1, i,:] = calculate_quantiles(a1[filter])
        results_bif[2, i,:] = calculate_quantiles(a3[filter]/a1[filter])
        
        # a1_all_bif[i,:] = a1
        # a3_a1_all_bif[i,:] = a3/a1
        # len_mis_all_bif[i,:] = len_mis
    
    return results_bif #, len_mis_all_bif, a1_all_bif, a3_a1_all_bif  


def plot_results(eta_range, eta_original, folder_name):
    """
    Calculate and plot results from the Backward Evolution Algorithm.
    Columns: 'Geometric measure', 'Angular measure', 'Local symmetry'
    Rows:  'Data', '|Data| (log)', 'Data width (log)'
    
    Parameters
    -------
    eta_range : array
        An array of eta to plot on X axis.
    eta_original : float
        The value of eta used in creation of the network (known or deducted) to mark on plots.
    folder_name : string
        A directory from which results will be gathered (with function 'gather_results')
        
    Returns
    -------
    plt.figure
        A figure containing plots with resutls.
    np.array
        An array 3-3 with axes of the figure.    
        
    Notes
    -------
    axvline(x=eta_original, ...) is always plotted as 0th line, so it can be popped by 'lines.pop(0)' and plotted from scratch outside the function.
    """
    
    cm2inch = 1/2.54
    golden = (1 + 5 ** 0.5) / 2
    results = gather_results(eta_range, folder_name) # [geo_results, ang_results, a2_norm_measure]
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(17.8*cm2inch,17.8/golden*cm2inch)) # 17.8/golden
    fig.subplots_adjust(wspace=0.4)
    # titles on top
    titles = ['Local symmetry', 'Overshoot', 'Angular deflection']
    # labels on the left
    axes[0, 0].set_ylabel('Data')# , labelpad=24) # labelpad: 35.5/10; 2nd line # 29.5/10 # 17.5/0
    axes[1, 0].set_ylabel('$|$Data$|$')# , labelpad=10)
    axes[2, 0].set_ylabel('Data width') # , labelpad=10)
    for i, measure_results in enumerate(results):
        # plot DATA
        axes[0, i].axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        axes[0, i].axhline(y=0, color='0.5', linewidth=0.5)
        axes[0, i].plot(eta_range,measure_results[:,0:3],label=['Median', 'Quantile 32\%', 'Quantile 68\%'])
        axes[0, i].fill_between(eta_range,measure_results[:,1],measure_results[:,2], alpha=0.3)
        # ticks, frame, title
        axes[0, i].spines['top'].set_visible(False)
        axes[0, i].spines['right'].set_visible(False)
        axes[0, i].spines['bottom'].set_visible(False)
        axes[0, i].xaxis.set_visible(False)
        axes[0, i].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        axes[0, i].set_title(titles[i], y=1.2) # (r'\textbf{'+titles[i]+'}', y=1.2)
        
        # plot DATA (log)
        axes[1, i].axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        axes[1, i].plot(eta_range,measure_results[:,3])
        # ticks, frame
        axes[1, i].set_yscale('log')
        axes[1, i].spines['top'].set_visible(False)
        axes[1, i].spines['right'].set_visible(False)
        axes[1, i].spines['bottom'].set_visible(False)
        axes[1, i].xaxis.set_visible(False)
        
        # plot WIDTH (log)
        axes[2, i].axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        axes[2, i].plot(eta_range, measure_results[:,6])
        axes[2, i].set_yscale('log')
        # ticks, frame
        axes[2, i].spines['top'].set_visible(False)
        axes[2, i].spines['right'].set_visible(False)
        # axes[2, i].set_xlabel('$\eta^*$', position=(1.08,0), labelpad=-18)
        axes[2, i].set_xlabel('$\eta^*$')
        axes[2, i].xaxis.set_label_coords(1.08, 0.08)
    fig.align_ylabels()

    return fig, axes


def plot_results_bif(eta_range, eta_original, folder_name, bif_type=1, figurewidth=11.4, figureheight_factor=1):
    """
    Calculate and plot results from bifurcation points. 
    Columns: 'Length mismatch', 'Bif. indicator'
    Rows: 'Data', 'Data width (log)'
    
    Parameters
    -------
    eta_range : array
        An array of eta to plot on X axis.
    eta_original : float
        The value of eta used in creation of the network (known or deducted) to mark on plots.
    folder_name : string
        A directory from which results will be gathered (with function 'gather_results_bif')
    bif_type : int (1 or 2), default=1
        Indicates which bifurcation indicator should be plotted ( a_1 if bif_type==1, a_3/a_1 if bif_type==2).
        It also indicates bif_value to be plotted with axhline : 0.8 for a_1 and -0.1 for a_3/a_1
    figurewidth : float, default=11.4
        Desired figure width in cm. Figure height is figurewidth/golden_ratio.
    Returns
    -------
    plt.figure
        A figure containing plots with results.
    np.array
        An array 2-2 with axes of the figure.
    
    Notes
    -------
    axvline(x=eta_original, ...) is always plotted as 0th line, so it can be popped by 'lines.pop(0)' and plotted from scratch
    """
    cm2inch = 1/2.54
    golden = (1 + 5 ** 0.5) / 2
    if bif_type==1:
        bif_indicator='$a_1$'
        bif_value=0.8
    elif bif_type==2:
        bif_indicator='$a_3/a_1$'
        bif_value=-0.1
    else:
        raise ValueError('Bifurcation type should be 1 or 2.')
    results_bif = gather_results_bif(eta_range, folder_name)[[0,bif_type]] # [len_mis_results_bif, a1_results_bif OR a3_a1_results_bif]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figurewidth*cm2inch,figureheight_factor*figurewidth/golden*cm2inch)) # 17.8/golden
    fig.subplots_adjust(wspace=0.4)
    titles = ['Length mismatch\n($l_0$)', 'Splitting indicator\n({})'.format(bif_indicator)]
    # FIRST ROW
    for i, ax in enumerate(axes[0,:]):
        ax.axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title(titles[i], y=1.2)
    # SECOND ROW
    for ax in axes[1,:]:
        ax.axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_xlabel('$\eta^*$', position=(1.08,0), labelpad=-18)
        ax.set_xlabel('$\eta^*$')
        ax.xaxis.set_label_coords(1.08, 0.08)
    
    # LENGTH MISMATCH QUANTILES
    axes[0, 0].axhline(y=0, color='0.2', linewidth=0.5)
    axes[0, 0].plot(eta_range, results_bif[0][:,0:3], label=['Median', 'Quantile 32\%', 'Quantile 68\%'])
    axes[0, 0].fill_between(eta_range, results_bif[0][:,1], results_bif[0][:,2], alpha=0.3)
    axes[0, 0].set_ylabel('Data')
    # axes[0, 0].set_yscale('log')
    # LENGTH MISMATCH WIDTH
    axes[1, 0].plot(eta_range, results_bif[0][:,6])
    # axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel('Data width')

    # BIFURCATION QUANTILES
    axes[0, 1].axhline(y=bif_value, color='0.5', linewidth=0.5)
    axes[0, 1].plot(eta_range, results_bif[1][:,0:3], label=['Median', 'Quantile 32\%', 'Quantile 68\%'])
    axes[0, 1].fill_between(eta_range, results_bif[1][:,1], results_bif[1][:,2], alpha=0.3)
    # BIFURCATION WIDTH
    axes[1, 1].plot(eta_range, results_bif[1][:,6])
    axes[1, 1].set_yscale('log')
    
    fig.align_ylabels()
    return fig, axes


def plot_results_bif_full(eta_range, eta_original, folder_name, figurewidth=11.4):
    """
    Calculate and plot results from bifurcation points. 
    Columns: 'Length mismatch', 'Bif. indicator (a1)', 'Bif. indicator (a3/a1)'
    Rows: 'Data', '|Data| (log)', 'Data width (log)'
    
    Parameters
    -------
    eta_range : array
        An array of eta to plot on X axis.
    eta_original : float
        The value of eta used in creation of the network (known or deducted) to mark on plots.
    folder_name : string
        A directory from which results will be gathered (with function 'gather_results_bif')
    figurewidth : float, default=11.4
        Desired figure width in cm. Figure height is figurewidth/golden_ratio.
    Returns
    -------
    plt.figure
        A figure containing plots with results.
    np.array
        An array 3-3 with axes of the figure.
    
    Notes
    -------
    axvline(x=eta_original, ...) is always plotted as 0th line, so it can be popped by 'lines.pop(0)' and plotted from scratch
    """
    cm2inch = 1/2.54
    golden = (1 + 5 ** 0.5) / 2

    results_bif = gather_results_bif(eta_range, folder_name) # [len_mis_results_bif, a1_results_bif OR a3_a1_results_bif]
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(figurewidth*cm2inch,figurewidth/golden*cm2inch)) # 17.8/golden
    fig.subplots_adjust(wspace=0.75)
    titles = ['Length mismatch\n($l_0$)', 'Splitting indicator\n($a_1$)', 'Splitting indicator\n($a_3/a_1$)']
    # FIRST ROW
    for i, ax in enumerate(axes[0,:]):
        ax.axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title(titles[i], y=1.2)
    # SECOND ROW
    for ax in axes[1,:]:
        ax.axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        
    # THIRD ROW
    for ax in axes[2,:]:
        ax.axvline(x=eta_original, color='0.5', linewidth=0.7, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('$\eta^*$')
        ax.xaxis.set_label_coords(1.08, 0.08)

    # LENGTH MISMATCH QUANTILES
    axes[0, 0].axhline(y=0, color='0.2', linewidth=0.5)
    axes[0, 0].plot(eta_range, results_bif[0][:,0:3])
    axes[0, 0].fill_between(eta_range, results_bif[0][:,1], results_bif[0][:,2], alpha=0.3)
    axes[0, 0].set_ylabel('Data')    
    # LENGTH MISMATCH WIDTH
    axes[1, 0].plot(eta_range, results_bif[0][:,3])
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel('$|$Data$|$')    
    # LENGTH MISMATCH WIDTH
    axes[2, 0].plot(eta_range, results_bif[0][:,6])
    axes[2, 0].set_yscale('log')
    axes[2, 0].set_ylabel('Data width')

    # BIFURCATION QUANTILES
    # axes[0, 1].axhline(y=bif_value, color='0.5', linewidth=0.5)
    axes[0, 1].plot(eta_range, results_bif[1][:,0:3], label=['Median', 'Quantile 32\%', 'Quantile 68\%'])
    axes[0, 1].fill_between(eta_range, results_bif[1][:,1], results_bif[1][:,2], alpha=0.3)
    # BIFURCATION LOG
    axes[1,1].yaxis.set_visible(False)
    axes[1,1].spines['left'].set_visible(False)
    axes[1,1].lines.pop(0)
    # axes[1, 1].plot(eta_range, results_bif[1][:,3])
    # axes[1, 1].set_yscale('log')
    # BIFURCATION WIDTH
    axes[2, 1].plot(eta_range, results_bif[1][:,6])
    axes[2, 1].set_yscale('log')
    
    # BIFURCATION QUANTILES
    # axes[0, 1].axhline(y=bif_value, color='0.5', linewidth=0.5)
    axes[0, 2].plot(eta_range, results_bif[2][:,0:3], label=['Median', 'Quantile 32\%', 'Quantile 68\%'])
    axes[0, 2].fill_between(eta_range, results_bif[2][:,1], results_bif[2][:,2], alpha=0.3)
    # BIFURCATION LOG
    axes[1,2].yaxis.set_visible(False)
    axes[1,2].spines['left'].set_visible(False)
    axes[1,2].lines.pop(0)
    # axes[1, 2].plot(eta_range, results_bif[2][:,3])
    # axes[1, 2].set_yscale('log')
    # BIFURCATION WIDTH
    axes[2, 2].plot(eta_range, results_bif[2][:,6])
    axes[2, 2].set_yscale('log')
    
    fig.align_ylabels()
    return fig, axes



file_backward = 'results_eta'
stream_column, step_column = 0, 1
a1_column, a2_column, a3_column, geo_dev_column, ang_dev_column = 2, 3, 4, 5, 6
file_bif = 'results_bif_eta'
a1_column_bif, a2_column_bif, a3_column_bif, len_mis_column_bif = 1, 2, 3, 4


plt.rcParams.update({
    'figure.dpi': 150,

    'text.usetex': True,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,

    'axes.linewidth': 0.5,
    'axes.titlesize': 12,

    'lines.linewidth': 1,
    'lines.markersize': 1,
    'lines.marker':'None',
    'lines.solid_capstyle':'round',
    
    'font.family': 'serif',
    'font.serif': ['Computer Modern'] # ['Times New Roman']
    })
    # 'font.size': 20,
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman']})
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm2inch = 1/2.54
golden = (1 + 5 ** 0.5) / 2

'''