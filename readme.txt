--------------------------------------------------
GLIS - (GL)obal optimization via (I)nverse distance weighting
and radial basis function (S)urrogates.

(C) 2019-2021 by A. Bemporad
--------------------------------------------------

This package implements two algorithms in both MATLAB and PYTHON
for finding the global minimum of a function that is expensive to evaluate, 
possibly under constraints, using surrogate radial basis functions 
and inverse distance weighting:

*GLIS*: Global optimization of a function f that is possibly expensive 
to evaluate. The algorithm is based on the following paper:

[1] A. Bemporad, "Global optimization via inverse weighting and radial basis functions," 
Computational Optimization and Applications, vol. 77, pp. 571–595.

*GLISp*: Global optimization of a function f whose value cannot 
be evaluated but, given two points x,y, it is possible to query whether 
f(x) is better than f(y) (preference-based optimization). 
The algorithm is based on the following paper:
[2] A. Bemporad, D. Piga, "Active preference learning based on radial basis functions," 
Machine Learning, vol. 110, no. 2, pp. 417–448, 2021. Available on arXiv:1909.13049.

This software is distributed without any warranty. Please cite the above papers 
if you use this software.



Version tracking:
v3.0     (June 9, 2021) Unknown constraints introduced in GLIS and GLISP by Mengjia Zhu

v2.4     (January 12, 2021) Bugs fixed in PYTHON version of GLISP by Mengjia Zhu

v2.3     (December 20, 2020) Minor changes in MATLAB version.

v2.2     (January 22, 2020) Added support for Genetic Algorithm method 
         from MATLAB Global Optimization Toolbox. Minor change in GLISp 
         demo (MATLAB version only)
         
v2.1     (October 18, 2019) All files and functions renamed to GLIS/GLISp

v2.0.1   (September 29, 2019) Changed Python preference function to class

v2.0     (September 28, 2019) Added preference-based optimization functions. 
         (MATLAB version of preference-based Bayesian optimization by D. Piga)

v1.1.1   (September 2, 2019) Minor fix when using linprog to shrink bounds
         and LP solution is unbounded.

v1.1     (August 3, 2019) Python code largely optimized (by Marco Forgione)

v1.0.2   (July 6, 2019) Moved init and default functions to external files 
         idwgopt_init.py and idwgopt_default.py, respectively

v1.0.1   (July 4, 2019) Added option "shrink_range" to disable shrinking
         lb and ub to bounding box of feasible set. Fixed small bug in 
         computing initial best value and initial range of F. 

v1.0     (June 15, 2019) Initial version

