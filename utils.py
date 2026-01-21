""" Various utility functions """
import numpy as np
import matplotlib.pyplot as plt
import string


## Broyden solver
def broyden_solver(f, x0, y0=None, tol=1E-7, maxcount=100, backtrack_c=0.5, noisy=True):
    """Similar to newton_solver, but solves f(x)=0 using approximate rather than exact Newton direction,
    obtaining approximate Jacobian J=f'(x) from Broyden updating (starting from exact Newton at f'(x0)).

    Backtracks only if error raised by evaluation of f, since improvement criterion no longer guaranteed
    to work for any amount of backtracking if Jacobian not exact.
    """

    x, y = x0, y0
    if y is None:
        y = f(x)

    # initialize J with Newton!
    J = obtain_J(f, x, y)
    for count in range(maxcount):
        if noisy:
            printit(count, x, y)

        if np.max(np.abs(y)) < tol:
            return x, y

        dx = np.linalg.solve(J, -y)

        # backtrack at most 29 times
        for bcount in range(30):
            # note: can't test for improvement with Broyden because maybe
            # the function doesn't improve locally in this direction, since
            # J isn't the exact Jacobian
            try:
                ynew = f(x + dx)
            except ValueError:
                if noisy:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                J = broyden_update(J, dx, ynew - y)
                y = ynew
                x += dx
                break
        else:
            raise ValueError('Too many backtracks, maybe bad initial guess?')
    else:
        raise ValueError(f'No convergence after {maxcount} iterations')


def obtain_J(f, x, y, h=1E-5):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((nx, ny))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx) - y) / h
    return J


def broyden_update(J, dx, dy):
    """Returns Broyden update to approximate Jacobian J, given that last change in inputs to function
    was dx and led to output change of dy."""
    return J + np.outer(((dy - J @ dx) / np.linalg.norm(dx) ** 2), dx)


def printit(it, x, y, **kwargs):
    """Convenience printing function for noisy iterations"""
    print(f'On iteration {it}')
    print(('x = %.3f' + ', %.3f' * (len(x) - 1)) % tuple(x))
    print(('y = %.3f' + ', %.3f' * (len(y) - 1)) % tuple(y))
    for kw, val in kwargs.items():
        print(f'{kw} = {val:.3f}')
    print('\n')

## Plotting functions

var_names = {
    'Y': 'Output',
    'Z': 'Post-tax income',
    'C': 'Consumption',
    'rb': 'Real interest rate',
    'w': 'Real wage',
    'N': 'Hours',
    'i': 'Nominal interest rate',
    'rbar': r'Taylor Rule intercept ($\overline{r}$)',
    'pibar': r'Taylor Rule target ($\overline{\pi}$)',
    'pi': 'Inflation',
    'piw': "Wage inflation",
    'A': 'Assets',
    'B': 'Liquid assets',
    'Bg': 'Real market value Govt. Debt',
    'G': 'Govt. consumption',
    'Gbar': r'Fiscal rule intercept ($\overline{G}$)',
    'T': 'Tax revenue',
    'Transfer': 'Transfer',
    'PD': "Primary deficit (G-T)",
    'FD': 'Fiscal deficit (Delta B)',
    'tau': 'Tax rate',
    'I': 'Investment',
    'ra': 'ra',
    'ra_e': 'Expected illiq. real rate',
    'rb_e': 'Expected liq. real rate',
    'Bg_Q_lb': 'Stock of Govt. bonds',
    'tax': 'Labor tax rate',
    'dYdN': 'Labor productivity',
    'mc' : 'Marginal cost'
}

## Helper functions


def rebase_irf(ss_old, ss_new, irfs, scale=1.0):
    """
    This takes irfs expressed as a difference with respect to ss_old 
    and expresses them as differences with respect to ss_new
    """
    irfs_new = dict()
    for k in irfs.keys():
        irfs_new[k] = (ss_old[k] + irfs[k] - ss_new[k])/scale 
    return irfs_new

def is_residual(name):
    if '_res' in name:
        return True
    elif '_mkt' in name:
        return True
    else:
        return False

# Helper functions for plotting
residual_names = {
    'asset_mkt': "Asset mkt clearing",
    'goods_mkt': "Goods mkt clearing",
    'bond_mkt': "Bond mkt clearing",
    'nkpc_res': "Price inflation",
    'wnkpc_res': "Wage inflation",
    'fisher_res': "Fisher equation",
    'ires': "Taylor",
    'euler': "Euler eq.",
    'B_res': "Govt. budget",
    'budget_constraint': "Budget constr.",
    'rpost_res': "Rpost res",
    'q_lb_res': "q_lb_res",
    'pshare_res': "pshare_res",
    'inv_res': 'inv_res',
    'val_res': 'val_res',
    'equity_res': 'equity_res',
    'G_res': 'Fiscal rule',
    'i_res': 'Taylor smoothing',
}

def plot_residuals(irf, residual_list, ncols=3):
    """
    Plot residuals after computing irfs
    """
    nrows = int(np.ceil(len(residual_list) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(14, 8))
    for i, k in enumerate(residual_list):
        plt.subplot(nrows, ncols, i+1)
        if k in irf.keys():
            plt.plot(irf[k])
        kname = residual_names[k]
        plt.title(kname, fontsize=18)
        plt.tight_layout()
    plt.show()

    
STOCKS = ['A', 'B', 'K', 'Bg', 'tot_wealth', 'Bg_Q_lb']
FLOWS = ['C', 'CHI', 'Y',  'I' ,'G', 'FD', 'PD', 'Gbar', 'T', 'Transfer']
RATES = ['pi', 'piw', 'rb', 'r', 'i', 'rbar', 'pibar', 'ra', 'ra_e', 'rb_e']

def plot_irfs(to_plot, plot_cases, save_name=None, variables=None, ncols=4):    
    """
    Plot IRFs across different models/parametrizations
    """
    cmap = ['red', 'black', '#636363']
    lmap = ['-', '--', '-.']
    vars_to_plot = ['Bg', 'B', 'G', 'T', 'Transfer', 'A', 'i', 'rb', 'rbar', 'Y', 'C', 'pi', 'pibar', 'N', 'Z', 'PD', 'Gbar']
    if variables:
        vars_to_plot = variables
    titles = [var_names[item] for item in vars_to_plot]     # 'vars_to_plot' should containt names in list 'var_names'

    linewidth = 3.0
    axessize = 16
    legendsize = 12
    titlesize = 18
    labelspace = 0.2
    labelsize = 12
    #ncols = 4
    figsize = (14, 8)

    nrows  = int(np.ceil(len(vars_to_plot) / ncols))
    Tplot = 40
    Tminus = 1
    time_x = np.linspace(-Tminus, Tminus + Tplot, Tplot+1).tolist()

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize) 
    
    alphabet = list(string.ascii_lowercase)

    for i, k in enumerate(vars_to_plot): 
        plt.subplot(nrows, ncols, i+1)
        title = f"{alphabet[i]}. {titles[i]}"
        plt.title(title, fontsize=titlesize)
        for a, cou in enumerate(plot_cases): 
            if k not in to_plot[cou]: 
                to_plot[cou][k] = to_plot[cou]['B'] - to_plot[cou]['B']
            if k in STOCKS:
                plt.plot(time_x, np.append([0], 100 * to_plot[cou][k][0:Tplot]/(1*4) ), label=f"{cou}", lw=linewidth, color=cmap[a], linestyle=lmap[a])
            elif k in FLOWS:
                plt.plot(time_x, np.append([0], 100 * to_plot[cou][k][0:Tplot]/1), label=f"{cou}", lw=linewidth, color=cmap[a], linestyle=lmap[a])
            elif k in RATES:
                plt.plot(time_x, np.append([0], 100 * to_plot[cou][k][0:Tplot]*4 ), label=f"{cou}", lw=linewidth, color=cmap[a], linestyle=lmap[a])
            else:
                plt.plot(time_x, np.append([0], 100 * to_plot[cou][k][0:Tplot] ), label=f"{cou}", lw=linewidth, color=cmap[a], linestyle=lmap[a]) 

        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.axhline(y=0.0, color='k', linewidth=0.5, linestyle=':')
        #if (i % ncols) == 0: plt.ylabel(r'percent of $Y_{0}$',  fontsize=axessize)
        if k in RATES:
            plt.ylabel(r'pp deviation',  fontsize=axessize)
        elif k in STOCKS + FLOWS:
            plt.ylabel(r'percent of $Y_{0}$',  fontsize=axessize)
        if i == 0: plt.legend(labelspacing=labelspace, fontsize=legendsize, frameon=False) 
        plt.tight_layout()
        plt.xlabel('Quarter',  fontsize=axessize)
    if save_name:
        fig.savefig(save_name)
    plt.show()

