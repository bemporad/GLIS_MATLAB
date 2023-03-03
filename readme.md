
# GLIS / GLISp / C-GLISp

<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glis-logo.png" alt="drawing" width=40%/>

# Contents

* [Package description](#description)
    * [GLIS](#glis)
    * [GLISp](#glisp)
    * [C-GLIS(p)](#cglisp)
  
* [Python version](#python-version)

* [Basic usage](#basic-usage)
    * [Global optimization (GLIS)](#basic-glis)
    * [Preference-based optimization (GLISp)](#basic-glisp)

* [Advanced options](#advanced)
    * [Surrogate function](#surrogate)
    * [Acquisition function](#acquisition)
    * [RBF recalibration](#recalibration)
    * [Unknown constraints and satisfactory samples](#unknown)
    * [Objective function transformation](#transformation)
    * [Other options](#options)

* [Contributors](#contributors)

* [Citing GLIS](#bibliography)

* [License](#license)


<a name="description"></a>
## Package description 

**GLIS** is a package for finding the global (**GL**) minimum of a function that is expensive to evaluate, possibly under constraints, using inverse (**I**) distance weighting and surrogate (**S**) radial basis functions. Compared to Bayesian optimization, GLIS is very competitive and often computationally lighter.

The package implements two main algorithms, described here below.

<a name="glis"></a>
### GLIS

The GLIS algorithm solves the following constrained derivative-free global optimization problem

$$\min_x  f(x)$$

$$l \leq x\leq u$$

$$Ax \leq b,\ g(x)\leq 0$$

The approach is particularly useful when $f(x)$ is time-consuming to evaluate, as it attempts at minimizing the number of function evaluations by actively learning a surrogate of $f$. 

Finite bounds $l \leq x\leq u$ are required to limit the search within a bounded set, the remaining constraints are optional.

The algorithm is based on the following paper:

<a name="cite-Bem20"><a>
> [1] A. Bemporad, "[Global optimization via inverse weighting and radial basis functions](http://cse.lab.imtlucca.it/~bemporad/publications/papers/coap-glis.pdf)," *Computational Optimization and Applications*, vol. 77, pp. 571–595, 2020. [[bib entry](#ref1)]


<a name="glisp"></a>
### GLISp

GLISp solves global optimization problems in which the function $f$ cannot be evaluated but, given two samples $x$, $y$, it is possible to query whether $f(x)$ is better or worse than $f(y)$. More generally, one can only evaluate a *preference function* $\pi(x,y)$

<p align="center">
$\pi(x,y) = -1$ if $x$ is better than $y$
</p>

<p align="center">
$\pi(x,y) = 1$ if $x$ is worse than $y$
</p>

<p align="center">
$\pi(x,y) = 0$ if $x$ is as good as $y$,
</p>

and want to solve the following preference-based optimization problem:

<p align="center">
find $x^*$ such that $\pi(x^*,x)\leq 0$  $\ \forall x$
</p>

with $x^*,x$ satisfying the constraints $l \leq x\leq u$, 
and, if present, $Ax \leq b$, $g(x)\leq 0$.

GLISp is particularly useful to solve optimization problems that involve human assessments. In fact, there is no need to explicitly quantify an *objective function* $f$, which instead remains unexpressed in the head of the decision-maker determining the preferences. A typical example is solving a multi-objective optimization problem where the exact priority of each objective is not clear.

The algorithm is based on the following paper:

<a name="cite-BP21"><a>
> [2] A. Bemporad, D. Piga, "[Active preference learning based on radial basis functions](http://cse.lab.imtlucca.it/~bemporad/publications/papers/mlj_glisp.pdf)," *Machine Learning*, vol. 110, no. 2, pp. 417-448, 2021. [[bib entry](#ref2)]

<a name="cglisp"></a>
### C-GLISp

C-GLISp is an extension of GLIS and GLISp to handle *unknown* constraints on $x$, $x\in  X_u$, where the shape of the set $X_u$ is completely unknown and one can only query whether a certain $x\in X_u$ or $x\not\in X_u$.
A typical case is when $f(x)$ is the result of an experiment or simulation, parameterized by $x$, and one labels $x\not\in X_u$ if the experiment could not be executed. The algorithm also supports labeling samples $x$ as *satisfactory* or not, for example an experiment could be carried out but the outcome was not considered satisfactory. Both additional information (feasibility with respect to unknown constraints and satisfaction) are used by GLIS or GLISp to drive the search of the optimal solution.

The algorithm is based on the following paper:

<a name="cite-ZPB22"><a>
> [3] M. Zhu, D. Piga, A. Bemporad, "[C-GLISp: Preference-based global optimization under unknown constraints with applications to controller calibration](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeecst-c-glisp.pdf),” *IEEE Trans. Contr. Systems Technology*, vol. 30, no. 3, pp. 2176–2187, Sept. 2022. [[bib entry](#ref3)]

<a name="python-version"></a>
## Python version
A Python version of GLIS/GLISp is also available for download [here](https://github.com/bemporad/GLIS).


<a name="basic-usage"></a>
## Basic usage

<a name="basic-glis"></a>
### Global optimization (GLIS)

Minimize a function $f$ of a vector $x\in\mathbb{R}^n$
within the finite bounds *lb* $\leq x\leq$ *ub*:

The user can choose to solve the problem by feeding the function `f` into the GLIS solver, or solve the optimization problem step-by-step without explicitly passing the function handle `f` to GLIS.

~~~python
fprintf("Solve the problem by feeding the simulator/fun directly into the GLIS solver \n")
[xopt1, fopt1,prob_setup1] = solve_glis(fun,lb,ub,opts);

fprintf("Solve the problem incrementally (i.e., provide the function evaluation at each iteration) \n")
x_= initialize_glis(lb,ub,opts); % x_ is unscaled
for k = 1: maxevals
    f_val = fun(x_);
    [x_, prob_setup2] = update_glis(f_val);
end
xopt2 = prob_setup2.xbest;
fopt2 = prob_setup2.fbest;
~~~

Examples of numerical benchmark testing using GLIS can be found in the `examples` folder.

<a name="basic-glisp"></a>
### Preference-based global optimization (GLISp)

GLISp solves a preference-based optimization problem with preference function $\pi(x_1,x_2)$, $x_1,x_2\in\mathbb{R}^n$
within the finite bounds `lb` $\leq x\leq$ `ub`.

~~~python
fprintf("Solve the problem by feeding the  preference expression step directly into the GLISp solver  \n")
[xbest, out] = solve_glisp(pref, lb,ub,opts);


fprintf("Solve the problem incrementally (i.e., provide the preference at each iteration)  \n")
[xbest2, x2] = initialize_glisp(lb,ub,opts); 

for k = 1:maxevals-1
    pref_val = pref(x2,xbest2);
    [x2, out] = update_glisp(pref_val);
    xbest2 = out.xbest;
end
xbest2 = out.xbest;
out.X = out.X(1:end-1,:);
~~~

Examples of numerical benchmark testing using GLISp can be found in the `examples` folder.

Examples of synthetic preference functions can be found in the `examples` folder (e.g.,`glisp_function1.m`)

<a name="advanced"></a>
## Advanced options

<a name="surrogate"></a>
### Surrogate function

By default, GLIS/GLISp use the *inverse quadratic* RBF

$$rbf(x_1,x_2)=\frac{1}{1+(\epsilon||x_1-x_2||_2)^2}$$

with $\epsilon=1$ to construct the surrogate of the objective function. To use a different RBF, for example the Gaussian RBF

$$rbf(x_1,x_2)=e^{-(\epsilon||x_1-x_2||_2)^2}$$

use the following code:

~~~python
opts.rbf="gaussian";
~~~

The following RBFs are available in `rbf_fun.m`:

~~~python
"gaussian", "inverse_quadratic", "multiquadric", "thin_plate_spline", "linear", "inverse_multi_quadric"
~~~

In alternative to RBF functions, in GLIS we can use inverse distance weighting (IDW) surrogates:

~~~python
opts.rbf="idw";
~~~

Although IDW functions are simpler to evaluate, usually RBF surrogates perform better.


<a name="acquisition"></a>
### Acquisition function

GLIS acquires a new sample $x_k$ by solving the following nonlinear
programming problem

$$\min_x  a(x)=\hat f(x) -\alpha s^2(x) - \delta\Delta F z(x)$$

$$l \leq x\leq u$$

$$Ax \leq b,\ g(x)\leq 0$$

where $\hat f$ is the surrogate (RBF or IDW) function, $s^2(x)$ the IDW variance function, and $z(x)$ the IDW exploration function. GLIS attempts at finding a point $x_k$ where $f(x)$ is expected to have the lowest value ( $\min \hat f(x)$ ), getting $x_k$ where the surrogate is estimated to be most uncertain ( $\max s^2(x)$ ), and exploring new areas of the feasible space ( $\max z(x)$ ). The hyperparameters $\alpha$ and $\delta$ determine the tradeoff  ( $\Delta F$ is the current range of values of $f(x)$ collected so far and is used as a normalization factor).

GLIS uses Particle Swarm Optimization (PSO) to determine the minimizer $x_k$ of the acquisition problem, whose objective function $a(x)$ is cheap to evaluate.

By default, GLIS takes $\alpha=1$ and $\delta=\frac{1}{2}$. Increasing these values promotes *exploration* of the sample space, and particular increasing $\delta$ promotes *diversity* of the samples, indipendently on the function values $f(x)$ acquired, while increasing $\alpha$ promotes the informativeness of the samples and heavily depends on the constructed surrogate function $\hat f$.

To change the default values of the hyper-parameters $\alpha$ and $\delta$, use the following code:

~~~python
opts.alpha=0.5; opts.delta=0.1;
~~~

GLISp performs acquisition in a similar way than GLIS. The surrogate $\hat f$ is determined by determining the combination of RBF coefficients, through linear programming, that make the resulting $\hat f$ satisfy the collected preference constraints. The parameter $\alpha$ is ignored.

GLISp also supports, in alternative, the acquisition based on the maximimization of the *probability of improvement*, as defined in [[2]](#cite-BP21). This can be specified as follows:

~~~python
opts.acquisition_method=2; % 1 = IDW acquisition function, 2 = probability of improvement
~~~

By default, `opts.acquisition_method` = 1.


<a name="recalibration"><a>

### RBF recalibration

The performance of GLISp can be usually improved by recalibrating
the RBF parameter $\epsilon$. This is achieved by performing leave-one-out cross-validation on the available samples to find the scaling $\epsilon\leftarrow\theta\epsilon$ providing the surrogate function that best satisfies the given preference constraints:

~~~python
opts.RBFcalibrate = 1;
opts.thetas=thetas;
opts.RBFcalibrationSteps = steps;
~~~

where `steps` is an array of step indices at which recalibration must be performed, and `thetas` is the array of values of $\theta$ tested during recalibration.


<a name="unknown"><a>

### Unknown constraints and satisfactory samples

As detailed in [[3]](#cite-ZPB22), GLIS/GLISp can handle *unknown* constraints on $x$, where the shape of $X$ is unknown, and support labeling samples $x$ as *satisfactory* or not. Check the numerical benchmark under `example` folder for how to instruct the solver to collect such extra information during queries.

~~~python
% use GLIS
[xbest, fbest,out] = solve_glis(f,lb,ub,opts,eval_feas_,eval_sat_);

% use GLISp
[xbest,out]=solve_glisp(pref,lb,ub,opts,eval_feas_,eval_sat_);
~~~


<a name="transformation"><a>

### Objective function transformation
In GLIS, when the objective function has very large and very small values, it is possible to fit the surrogate of a nonlinear transformation of the objective rather the raw objective values. For example, in the *camel-six-humps* example we want to build the surrogate $\hat f(x)\approx \log(f(x+2))$ rather than $\hat f(x)\approx f(x)$. In GLIS, you can specify the transformation function as an optional argument:

~~~python
opts.obj_transform = @(f) log(f+2.);
~~~
 
<a name="options"><a>

### Other options

Further options in executing GLIS/GLISp are detailed below:

`svdtol` tolerance used in SVD decomposition when fitting the RBF function.

`shrink_range` flag, if True the given bounds `bounds` are shrunk to the bounding box of the feasible constrained set $X=\{x: Ax\leq b, g(x)\leq 0\}$.

`constraint_penalty` penalty used to penalize the violation of the constraints $Ax\leq b$, $g(x)\leq 0$ during acquisition.

`feasible_sampling` flag, if True all the initial samples satisfy the constraints $Ax\leq b$, $g(x)\leq 0$.

`scale_delta` flag, scale $\delta$ during the iterations, as described in [[3]](#cite-ZPB22).

`expected_max_evals` expected maximum number of queries (defaulted to `maxevals` when using `solve_glis.m`.

`display` verbosity level: 1  (default).

`PSOiters`, `PSOswarmsize`, `PSOminfunc`: parameters used by the PSO solver from the [`PSwarm.m`](http://www.norg.uminho.pt/aivaz/pswarm/software/PSwarmM_v2_1.zip) 
package used by GLIS.

`sepvalue` (GLISp only): amount of separation $\hat f(x_1)-\hat f(x_2)$ imposed in the surrogate function when imposing preference constraints.

`epsDeltaF` (GLISp only): lower bound on the range of the surrogate function.

                 
<a name="contributors"><a>
## Contributors

This package was coded by Alberto Bemporad and Mengjia Zhu. Marco Forgione and Dario Piga also contributed to the development of the package.


This software is distributed without any warranty. Please cite the above papers if you use this software.

<a name="bibliography"><a>
## Citing GLIS

<a name="ref1"></a>

```
@article{Bem20,
    author={A. Bemporad},
    title={Global optimization via inverse distance weighting and radial basis functions},
    journal={Computational Optimization and Applications},
    volume=77,
    pages={571--595},
    year=2020
}
```

<a name="ref2"></a>

```
@article{BP21,
    title={Active preference learning based on radial basis functions},
    author={A. Bemporad and D. Piga},
    journal={Machine Learning},
    volume=110,
    number=2,
    pages={417--448},
    year=2021
}
```

<a name="ref3"></a>

```
@article{ZPB22,
    author={M. Zhu and D. Piga and A. Bemporad},
    title={{C-GLISp}: Preference-Based Global Optimization under Unknown Constraints with Applications to Controller Calibration},
    journal={IEEE Transactions on Control Systems Technology},
    month=sep,
    volume=30,
    number=3,
    pages={2176--2187},
    year=2022
}
```

<a name="license"><a>
## License

Apache 2.0

(C) 2019-2023 A. Bemporad, M. Zhu