clear 
set more off 
set matsize 11000

cd "X:\Dga_EI\Gen\PROYECTOS\MU-HANK\2024\Codes\replication\estimation"

global mainData="X:\Dga_EI\Gen\PROYECTOS\MU-HANK\2024\Codes\replication\estimation\data"
global mainResults="X:\Dga_EI\Gen\PROYECTOS\MU-HANK\2024\Codes\replication\output"

*********************** USE DATA***********************

import excel "$mainData\data_estimation.xlsx", sheet("data") firstrow clear


gen qdate = q(1967q1) + _n-1
format qdate %tq

tsset qdate, q

*===============================================================================
*============ Controls and variable of interes ========================
*===============================================================================
* Controls:

gen y=rstar_up
gen dy=d.y

*** CHANGE HERE IF YOU WANT TO PUT ANOTHER EXOGENOS VARIABLE
gen xxx=L1.debtgdp  


* Choose impulse response horizon
local hmax = 14
local p = 4
/* Generate LHS variables for the LPs */

* levels
forvalues h = 0/`hmax' {
	gen y_`h' = f`h'.y 	 
}
	 * differences
forvalues h = 0/`hmax' {
	gen yd`h' = f`h'.y - l.f`h'.y 
}
* Cumulative
forvalues h = 0/`hmax' {
	gen yc`h' = f`h'.y - l.y 
}


* LOCAL PROJECTION 
eststo clear
cap drop b_iv u_iv d_iv 
gen Quarters = _n-1 if _n<=`hmax'
gen Zero =  0     if _n<=`hmax'

gen b_iv=0
gen u_iv=0
gen d_iv=0
gen u_iv70=0 
gen d_iv70=0
qui forv h =0/`hmax' {

       reg y_`h' xxx L(1/`p').y   L(1/`p').xxx   L(1/`p').infla  L(1/`p').UR    L(1/`p').FFR , robust

replace b_iv = _b[xxx]                    if _n == `h'+1
replace u_iv70 = _b[xxx] + 1.0* _se[xxx] if _n == `h'+1 // 68% confidence interval
replace d_iv70 = _b[xxx] - 1.0* _se[xxx] if _n == `h'+1 // 68%
replace u_iv = _b[xxx] + 1.645* _se[xxx]  if _n == `h'+1
replace d_iv = _b[xxx] - 1.645* _se[xxx]  if _n == `h'+1

eststo
}

twoway ///
(rarea u_iv70 d_iv70 Quarters, fcolor(red%10) lcolor(gs13) lw(none) lpattern(solid)) ///
(rarea u_iv d_iv  Quarters,  ///
fcolor(red%15) lcolor(gs13) lw(none) lpattern(solid)) ///
(line b_iv Quarters, lcolor(red) lpattern(solid) lwidth(thick)) ///
(line Zero Quarters, lcolor(black)), legend(off) ///
ytitle("Percentage points", size(medsmall)) xtitle("Quarter", size(medsmall)) ///
graphregion(color(white)) plotregion(color(white)) xsc(r(0 12)) xlabel(0 4 8 12 )
graph export "$mainResults/LP_IRF.png", width(800) height(600) replace

