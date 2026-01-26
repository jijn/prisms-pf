## PRISMS-PF: Heat equation

We consider the 2D heat equation for a function $u(x, y, t)$ on a square domain $\Omega = [0,1] \times [0,1]$. The governing partial differential equation (PDE) is:

$$
\begin{equation}
    \frac{\partial u}{\partial t} - \nabla \cdot (D \nabla u) = 0
\end{equation}
$$

## Analytical solutions
In 2D, it is written as
$$
\begin{equation}
    \frac{\partial u}{\partial t} = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
\end{equation}
$$

where $D > 0$ is the constant thermal diffusivity. We assume the initial condition is given by the eigenfunctions of the Laplacian:
$$
\begin{equation}
    u(x, y, 0) = \phi(x, y)
\end{equation}
$$

Assuming a solution of the form $u(x, y, t) = X(x)Y(y)T(t)$, the PDE separates into:
$$
\begin{equation}
    \frac{1}{DT} \frac{dT}{dt} = \frac{1}{X} \frac{d^2X}{dx^2} + \frac{1}{Y} \frac{d^2Y}{dy^2} = -k^2
\end{equation}
$$
This leads to the temporal decay:
$$
\begin{equation}
    T(t) = e^{-D k_{n,m}^2 t} \quad \text{where} \quad k_{n,m}^2 = k_n^2 + k_m^2
\end{equation}
$$

### Boundary Condition 1: Dirichlet
The boundary conditions are $u=0$ on all edges:
$$
\begin{equation}
    u(0,y,t) = u(1,y,t) = u(x,0,t) = u(x,1,t) = 0
\end{equation}
$$
The eigenfunctions and the resulting analytical solution are:

- **Eigenfunctions:** $\phi_{n,m}(x,y) = \sin(n\pi x)\sin(m\pi y)$
- **Eigenvalues:** $k_{n,m}^2 = \pi^2(n^2 + m^2)$

The analytical solution is:
$$
\begin{equation}
    u(x, y, t) = \sum_{n=1}^{\infty} \sum_{m=1}^{\infty} A_{n,m} \sin(n\pi x) \sin(m\pi y) e^{-D\pi^2(n^2+m^2)t}
\end{equation}
$$
### Boundary Condition 2: Neumann (no-flux)
The boundary conditions are $\frac{\partial u}{\partial n} = 0$ on all edges:
$$
\begin{equation}
    u_x(0,y,t) = u_x(1,y,t) = u_y(x,0,t) = u_y(x,1,t) = 0
\end{equation}
$$
The eigenfunctions and the resulting analytical solution are:

- **Eigenfunctions:** $\phi_{n,m}(x,y) = \cos(n\pi x)\cos(m\pi y)$
- **Eigenvalues:** $k_{n,m}^2 = \pi^2(n^2 + m^2)$

The analytical solution is:
$$
\begin{equation}
    u(x, y, t) = A_{0,0} + \sum_{n=1}^{\infty} \sum_{m=1}^{\infty} A_{n,m} \cos(n\pi x) \cos(m\pi y) e^{-D\pi^2(n^2+m^2)t}
\end{equation}
$$
where the summation excludes the $n=m=0$ case, and $A_{0,0}$ represents the steady-state average temperature.

## Weak form
We multiply the PDE by a sufficiently smooth test function $w(\mathbf{x})$ and integrate over the entire domain $\Omega$:
$$
\begin{equation}
    \int_{\Omega} w \frac{\partial u}{\partial t} \, d\Omega - \int_{\Omega} w [\nabla \cdot (D \nabla u)] \, d\Omega = 0
\end{equation}
$$
Using Integration by Parts and the Divergence Theorem:
$$
\begin{equation}
    \int_{\Omega} w \frac{\partial u}{\partial t} \, d\Omega = \int_{\Omega} \nabla w \cdot (-D\nabla u) \, d\Omega + \int_{\partial \Omega} w ~\mathbf{n} \cdot (D \nabla u) \, d A
\end{equation}
$$
where $\mathbf{n}$ is the outward unit normal vector.

- **Dirichlet ($u=0$ on $\partial \Omega$):** We restrict the test function space such that $w = 0$ on $\partial \Omega$. The boundary integral vanishes.
- **Neumann ($\mathbf{n} \cdot \nabla u= 0$ on $\partial \Omega$):** The flux term $\mathbf{n} \cdot (D \nabla u)$ is zero. The boundary integral vanishes naturally.

## Time discretization: Forward Euler (explicit)
We discretize the time derivative by
$$
\begin{equation}
\frac{\partial u}{\partial t} = \frac{u^{n+1} - u^n}{\Delta t}
\end{equation}
$$
and use $u^{n}$ for the right hand side of the equation.

The weak form becomes
$$
\begin{equation}
    \int_{\Omega} w u^{n+1} \, d\Omega = \int_{\Omega} w u^{n} + \nabla w \cdot (-\Delta t D\nabla u^{n}) \, d\Omega
\end{equation}
$$

We have
$$
\begin{equation}
r_{u}= u^{n}
\end{equation}
$$

$$
\begin{equation}
r_{u x} = -\Delta t D \nabla u^{n}
\end{equation}
$$

The CFL condition for the timestep size is (2D case)
$$
\text{CFL} = \frac{D \Delta t}{\Delta x^2} \leq \frac{1}{4} \quad \Rightarrow \quad \Delta t \leq \frac{\Delta x^2}{4 D}
$$
Note that this condition is derived from finite difference discretization. For finite element, the basic structure of the CFL condition remain the same. But the constant 1/4 may differ, depending on element types. 

## Time discretization: Backward Euler (implicit)
We use $u^{n+1}$ for the right hand side of the equation.

The weak form becomes
$$
\begin{equation}
    \int_{\Omega} w ~ u^{n+1} + \nabla w \cdot (\Delta t D\nabla u^{n+1}) \, d\Omega = \int_{\Omega} w ~ u^{n} \, d\Omega
\end{equation}
$$

We have
$$
\begin{equation}
l_{u}= u^{n+1}
\end{equation}
$$

$$
\begin{equation}
l_{u x} = \Delta t D \nabla u^{n+1}
\end{equation}
$$

$$
\begin{equation}
r_{u}= u^{n}
\end{equation}
$$


The implicit time stepping scheme is not bound by any CFL-like condition on the timestep. We can choose the timestep adaptively by a given criteria. 


<!-- ## Time discretization: Crank-Nicolson

## Time discretization: BDF

## Time discretization: RK4 -->

