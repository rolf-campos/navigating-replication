import numpy as np
import sequence_jacobian as sj  # note: sequence_jacobian has scipy as a dependency


## Helper functions
def log(x):
    return x.apply(np.log)

def exp(x):
    return x.apply(np.exp)

## Household block
# The household block has an adjustment cost function like the one in Alves, Kaplan, Moll, and Violante
from hh import hh

def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
    b_grid = sj.grids.agrid(amax=bmax, n=nB)
    a_grid = sj.grids.agrid(amax=amax, n=nA)
    k_grid = sj.grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi

def income(e_grid, tax, w, N, Transfer):
    z_grid = (1 - tax) * w * N * e_grid + Transfer
    return z_grid

hh_ext = hh.add_hetinputs([income, make_grids])

## Production block
@sj.simple
def labor(Y, w, K, Z, alpha):
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = 1/(1-alpha) * w * N / Y   
    return N, mc

@sj.simple
def investment(Q, K, ra_e, N, mc, Z, delta, epsI, alpha):
    inv_res = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
    val_res = alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1) -\
        (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI)) +\
        K(+1) / K * Q(+1) - (1 + ra_e) * Q 
    return inv_res, val_res

production = sj.combine([labor, investment])                             # create combined block
production_solved = production.solved(unknowns={'Q': 1.0, 'K': 9.0},     # turn it into solved block
                                      targets=['inv_res', 'val_res'],
                                      solver='broyden_custom')

## Other blocks
@sj.solved(unknowns={'pi': (-0.8, 0.8)}, targets=['nkpc_res'], solver="brentq")
def pricing_solved(pi, mc, ra_e, Y, kappap, mup, pi_ss):
    nkpc_res = kappap * (mc - 1/mup) + Y(+1) / Y * log((1 + pi(+1)) / (1 + pi_ss)) / \
           (1 + ra_e) - log((1 + pi)  / (1 + pi_ss))
    return nkpc_res

@sj.solved(unknowns={'p': (0.05, 150)}, targets=['equity_res'], solver="brentq")
def arbitrage_solved(div, p, ra_e):
    # r = ex-ante return on illiquid assets
    equity_res = div(+1) + p(+1) - p * (1 + ra_e)
    return equity_res

@sj.simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw

@sj.simple
def union(piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta, pi_ss):
    wnkpc_res = kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * UCE / muw) + beta * \
            log((1 + piw(+1)) / (1 + pi_ss)) - log((1 + piw) / (1 + pi_ss))
    return wnkpc_res

@sj.simple
def mkt_clearing(p, A, B, Bg, C, I, G, CHI, psip, Y):
    asset_mkt = A + B - p - Bg
    bond_mkt = B - Bg
    goods_mkt = C + I + G + CHI + psip - Y
    # psip   = price adjustment cost, 
    # CHI    = portfolio adjustment cost
    return asset_mkt, bond_mkt, goods_mkt

@sj.simple
def finance(p, div):
    # ra = ex-post return on illiquid assets
    ra = (div + p) / p(-1) - 1
    return ra 

@sj.simple
def eq_fisher(i, pi, rb):
    # rb = ex-post return on liquid assets
    fisher_res = 1 + i(-1) - (1 + rb) * (1 + pi)
    return fisher_res

@sj.simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI, pi_ss, tau_d):
    psip = mup / (mup - 1) / 2 / kappap * log((1 + pi) / (1 + pi_ss)) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = (Y - w * N - I - psip) * (1-tau_d)
    T_d =  (Y - w * N - I - psip) * tau_d
    return psip, I, div, T_d

@sj.simple
def taylor(rbar, pi, phi, pibar):
    # Non-linear version of i = rbar + pibar + phi * (pi - pibar)
    i = (1+rbar) * (1+pibar) * ((1+pi)/(1+pibar))**phi - 1
    rb_e = (1+i) / (1+pi(+1)) - 1
    return i, rb_e

# Baseline monetary rule
@sj.solved(unknowns={'i': 0.01}, targets=['i_res'], solver="brentq")
def taylor_smooth(i, rbar, pi, phi, pibar, rho_i):
    # Non-linear version of i = rbar + pibar + phi * (pi - pibar)
    i_desired = (1+rbar) * (1+pibar) * ((1+pi)/(1+pibar))**phi - 1
    i_res = (1 + i(-1)) ** rho_i * (1 + i_desired) ** (1-rho_i) - (1 + i)  # i = rho_i * i(-1) + (1 - rho_i) * i_desired
    rb_e = (1+i) / (1+pi(+1)) - 1
    return rb_e, i_res

# Baseline fiscal rule: G follows a rule
@sj.solved(unknowns={'Bg': 0}, targets=['B_res'], solver="brentq")
def G_rule(rb, tax, w, N, Transfer, Bg, Gbar, Bgbar, phi_G, T_d):
    T = tax * w * N + T_d
    G = Gbar - phi_G * (Bg(-1) - Bgbar)
    B_res = (1 + rb) * Bg(-1) + (G + Transfer) - T - Bg
    FD = Bg - Bg(-1)  # fiscal deficit
    PD = Transfer + G - T  # primary deficit (does not include interest payments)
    return G, B_res, FD, PD, T

# Alternative monetary policy rule: response to the output gap
@sj.solved(unknowns={'i': 0.01}, targets=['i_res'], solver="brentq")
def taylor_smooth_og(i, rbar, pi, phi, pibar, rho_i, Y, phi_og):
    # Non-linear version of i = rbar + pibar + phi * (pi - pibar)
    i_desired = (1+rbar) * (1+pibar) * ((1+pi)/(1+pibar))**phi * (Y/Y.ss)**phi_og - 1
    i_res = (1 + i(-1)) ** rho_i * (1 + i_desired) ** (1-rho_i) - (1 + i)  # i = rho_i * i(-1) + (1 - rho_i) * i_desired
    rb_e = (1+i) / (1+pi(+1)) - 1
    return rb_e, i_res

# Alternative monetary policy rule: Orphanides-Williams rule
@sj.solved(unknowns={'i': (-0.5, 0.5)}, targets=['i_res'], solver="brentq")
def orphanides(i, pi, phi, pibar, rbar):
    lhs = 1 + i
    rhs = (1 + i(-1)) * ((1+pi)/(1+pibar))**phi
    i_res = lhs - rhs
    rb_e = (1+i) / (1+pi(+1)) - 1
    return i_res, rb_e

# Holden's rule
@sj.simple
def taylor_holden(rb, pi, phi, pibar):
    rb_e = rb(+1)
    i = (1 + rb_e) * (1 + pibar) * ((1+pi)/(1+pibar))**phi - 1
    return i, rb_e

# Alternative fiscal rule: transfers computed as residual to replicate a path for Bg_Q_lb
@sj.simple
def residual_transfer(rb, tax, w, N, Bg_Q_lb, Bgbar, G, T_d, Q_lb):
    T = tax * w * N + T_d
    Bg = Bg_Q_lb * Q_lb
    Transfer = - (1 + rb) * Bg(-1) - G + T + Bg
    B_res = (1 + rb) * Bg(-1) + (G + Transfer) - T - Bg
    FD = Bg - Bg(-1)  # fiscal deficit
    PD = Transfer + G - T  # primary deficit (does not include interest payments)
    return Transfer, B_res, FD, PD, T, Bg

# Alternative fiscal rule: tax rate computed as residual to replicate a path for Bg_Q_lb
@sj.simple
def residual_tax(rb, Transfer, w, N, Bgbar, G, T_d, Bg_Q_lb, Q_lb):
    Bg = Bg_Q_lb * Q_lb
    T = Transfer + (1 + rb) * Bg(-1) + G  - Bg
    tax = (T - T_d) / (w * N)
    B_res = (1 + rb) * Bg(-1) + (G + Transfer) - T - Bg
    FD = Bg - Bg(-1)  # fiscal deficit
    PD = Transfer + G - T  # primary deficit (does not include interest payments)
    return tax, B_res, FD, PD, T, Bg

# Long bonds
# Pricing equation for long bonds
@sj.solved(unknowns={'Q_lb': (0.1, 150)}, targets=['q_lb_res'], solver="brentq")
def q_lb(Q_lb, i, delta_lb):
    q_lb_res = Q_lb - (1 + delta_lb * Q_lb(+1)) / (1 + i)
    return q_lb_res

# Ex-post real return of long bonds (expressed as a residual)
@sj.simple
def rpost_lb(Q_lb, rb, delta_lb, pi):
    rpost_res = (1 + delta_lb * Q_lb)/Q_lb(-1)  / (1+pi) - (1 + rb)
    return rpost_res

@sj.simple
def outcomes(Q_lb, Bg):
    Bg_Q_lb = Bg / Q_lb
    return Bg_Q_lb

## Define the baseline version of the model
# Baseline model
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, G_rule, finance, wage, union, mkt_clearing, taylor_smooth, q_lb, rpost_lb, outcomes]
hank_lb = sj.create_model(blocks, name='Baseline Two-Asset HANK')

## Define alternative versions of the model
# Model with transfers calculated as a residual
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, residual_transfer, finance, wage, union, mkt_clearing, taylor_smooth, q_lb, rpost_lb]
hank_tr = sj.create_model(blocks, name='Two-Asset HANK with residual transfers')

# Model with the labor tax rate calculated as a residual
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, residual_tax, finance, wage, union, mkt_clearing, taylor_smooth, q_lb, rpost_lb]
hank_tx = sj.create_model(blocks, name='Two-Asset HANK with residual tax rate')

# Model without a Taylor rule that responds to the output gap
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, G_rule, finance, wage, union, mkt_clearing, taylor_smooth_og, q_lb, rpost_lb, outcomes]
hank_og = sj.create_model(blocks, name='Two-Asset HANK with a Taylor rule that responds to the OG')

# Model without a Taylor rule that responds to the output gap
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, G_rule, finance, wage, union, mkt_clearing, orphanides, q_lb, rpost_lb, outcomes]
hank_ow = sj.create_model(blocks, name='Two-Asset HANK with an Orphanides-Williams rule')

# Model without interest rate smoothing
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, G_rule, finance, wage, union, mkt_clearing, taylor, q_lb, rpost_lb, outcomes]
hank_ns = sj.create_model(blocks, name='Two-Asset HANK without interest rate smoothing')

# Model with Holden's rule
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, G_rule, finance, wage, union, mkt_clearing, taylor_holden, q_lb, rpost_lb, outcomes]
hank_hr = sj.create_model(blocks, name="Baseline Two-Asset HANK with Holden's rule")

# Model with short bonds
blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved, 
          dividend, G_rule, finance, wage, union, mkt_clearing, taylor_smooth, eq_fisher]
hank_sb = sj.create_model(blocks, name='Two-Asset HANK with short bonds')

# Model without extra profits
@sj.simple
def dividend_alt(Y, w, N, K, pi, mup, kappap, delta, epsI, pi_ss, tau_d):
    psip = mup / (mup - 1) / 2 / kappap * log((1 + pi) / (1 + pi_ss)) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    profit = Y - w * N - I - psip
    profit_ss = Y.ss - w.ss * N.ss - I.ss - psip.ss
    div = profit_ss * (1-tau_d)
    T_d =  profit * tau_d
    Transfer_extra = profit - T_d - div  # (profit - profit_ss) * (1-tau_d)
    return psip, I, div, T_d, Transfer_extra

def income_extra(e_grid, tax, w, N, Transfer, Transfer_extra):
    z_grid = (1 - tax) * w * N * e_grid + Transfer + Transfer_extra
    return z_grid

hh_extra = hh.add_hetinputs([income_extra, make_grids])

blocks = [hh_extra, production_solved, pricing_solved, arbitrage_solved, 
          dividend_alt, G_rule, finance, wage, union, mkt_clearing, taylor_smooth, q_lb, rpost_lb, outcomes]
hank_np = sj.create_model(blocks, name='Two-Asset HANK without extra profits')
