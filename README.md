# Deep Learning for Poroelasticity

Codes to solve the Poroelasticity problem (Biot's Consolidation Theory) with Artificial Neural Networks.

## Mathematical Model

The one-dimension model for poroelasticity is gave by the displacement equation

```math
-E\frac{\partial^2u}{\partial x^2} + \frac{\partial p}{\partial x} = U
```

and the pressure equation

```math
\frac{\partial}{\partial t}\left(\frac{\partial u}{\partial x}\right) - K\frac{\partial^2p}{\partial x^2} = P
```

where $E$ denotes the elastic modulus, $K$ represents hydraulic conductivity, $U$ signifies the density of the force applied to the body, and $P$ denotes the injection or extraction force of the fluid within the porous medium. The displacement is defined by \(u(x,t)\), and the pressure by $p(x,t)$, where $x$ is the spatial variable, and $t$ is the temporal variable. We consider a spatial domain $[0,L]$ and a temporal domain $[0,tf]$.

For the boundary conditions, we will assume free drainage without variation in displacement on the left boundary

```math
E\frac{\partial u(0,t)}{\partial x} = 0
```

```math
p(0,t) = 0
```

and stiffness without pressure variation on the right boundary

```math
u(L,t) = 0
```

```math
K\frac{\partial p(L,t)}{\partial x} = 0
```

## Physics-Informed Neural Networks (PINN)

