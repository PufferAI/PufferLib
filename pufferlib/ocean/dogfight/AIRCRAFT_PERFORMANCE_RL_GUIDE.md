# Aircraft Performance Approximation for High-Performance RL Environments

## A Comprehensive Guide for WW2 Dogfighting Simulation

**Purpose**: This document provides the mathematical foundations, equations, and implementation strategies for approximating aircraft performance in a headless reinforcement learning training environment. The goal is to achieve very high steps-per-second (SPS) while maintaining physically plausible flight dynamics suitable for WW2 dogfighting scenarios.

**Target Audience**: Claude agents developing a WW2 dogfighting RL environment using PufferLib or similar frameworks.

---

## Table of Contents

1. [Philosophy: Fidelity vs. Performance Trade-offs](#1-philosophy-fidelity-vs-performance-trade-offs)
2. [Coordinate Systems and Reference Frames](#2-coordinate-systems-and-reference-frames)
3. [Equations of Motion](#3-equations-of-motion)
4. [Aerodynamic Force Models](#4-aerodynamic-force-models)
5. [The Drag Polar](#5-the-drag-polar)
6. [Lift Coefficient Modeling](#6-lift-coefficient-modeling)
7. [Propulsion: Piston Engine and Propeller Models](#7-propulsion-piston-engine-and-propeller-models)
8. [Atmospheric Model](#8-atmospheric-model)
9. [Performance Calculations](#9-performance-calculations)
10. [Implementation Strategies for High SPS](#10-implementation-strategies-for-high-sps)
11. [WW2 Aircraft Reference Data](#11-ww2-aircraft-reference-data)
12. [Validation and Sanity Checks](#12-validation-and-sanity-checks)
13. [Sources and References](#13-sources-and-references)

---

## 1. Philosophy: Fidelity vs. Performance Trade-offs

### The Core Challenge

Full 6-DOF flight dynamics models (like JSBSim or Stevens & Lewis F-16 models) are computationally expensive. For RL training requiring millions of environment steps, we need simplifications that:

1. **Preserve emergent behavior**: Aircraft should fly like aircraft—stall at high AoA, turn radius should increase with speed, climb rate should decrease with altitude
2. **Enable fast vectorized computation**: All operations should be expressible as numpy/JAX/PyTorch tensor operations
3. **Avoid lookup tables where possible**: Analytical approximations are faster than interpolation
4. **Capture the essence of dogfighting**: Energy management, turn performance, climb/dive dynamics

### Recommended Model Hierarchy

| Model Type | DOF | Use Case | Typical SPS (single env) |
|------------|-----|----------|--------------------------|
| Full 6-DOF with stability derivatives | 12+ states | Flight sim, detailed control | 1,000-10,000 |
| Point-mass 3-DOF (vertical plane) | 6 states | Trajectory optimization | 50,000-200,000 |
| **Point-mass 3-DOF (3D)** | **6-9 states** | **RL dogfighting (recommended)** | **100,000-500,000** |
| Energy-state approximation | 3-4 states | Strategic AI | 1,000,000+ |

**Recommendation**: Use a 3-DOF point-mass model with instantaneous bank angle changes for maximum performance while retaining meaningful dogfight dynamics.

---

## 2. Coordinate Systems and Reference Frames

### Earth-Fixed Frame (Inertial, Flat Earth Approximation)
- Origin at some reference point on ground
- **x**: North (or arbitrary horizontal)
- **y**: East (or perpendicular horizontal)  
- **z**: Down (positive toward Earth center)

For WW2 dogfighting in a local area, flat Earth is entirely acceptable.

### Body-Fixed Frame
- Origin at aircraft CG
- **x_b**: Forward along fuselage
- **y_b**: Right wing
- **z_b**: Down through belly

### Wind/Velocity Frame
- **x_w**: Along velocity vector
- **z_w**: Perpendicular, in vertical plane containing velocity

### Key Angles
```
α (alpha) = Angle of Attack = angle between x_b and velocity vector (in vertical plane)
β (beta) = Sideslip angle = angle between velocity and x_b-z_b plane
γ (gamma) = Flight path angle = angle between velocity and horizontal
ψ (psi) = Heading angle = horizontal direction of velocity
φ (phi) = Bank/roll angle
θ (theta) = Pitch angle
```

For 3-DOF point-mass: we typically track (V, γ, ψ, x, y, h) where bank angle φ is a control input that changes instantaneously.

---

## 3. Equations of Motion

### 3-DOF Point-Mass Model (Recommended for RL)

This model treats the aircraft as a point mass with forces applied. It captures the essential performance characteristics without modeling rotational dynamics.

**State Vector**: `[V, γ, ψ, x, y, h]` or `[V, γ, ψ, x, y, h, m]` if tracking fuel

**Control Inputs**: `[T (thrust/throttle), n (load factor) or φ (bank), α (angle of attack)]`

#### Kinematic Equations
```python
dx/dt = V * cos(γ) * cos(ψ)
dy/dt = V * cos(γ) * sin(ψ)
dh/dt = V * sin(γ)  # Note: h positive up, so dh/dt = -dz/dt
```

#### Dynamic Equations (Forces)

In the wind-axes frame, summing forces parallel and perpendicular to velocity:

```python
# Along velocity (tangent to flight path)
m * dV/dt = T * cos(α) - D - W * sin(γ)

# Perpendicular to velocity, in vertical plane
m * V * dγ/dt = L * cos(φ) + T * sin(α) - W * cos(γ)

# Perpendicular to velocity, horizontal (turning)
m * V * cos(γ) * dψ/dt = L * sin(φ)
```

Where:
- `T` = Thrust
- `D` = Drag  
- `L` = Lift
- `W = m * g` = Weight
- `φ` = Bank angle
- `α` = Angle of attack (typically small, so cos(α) ≈ 1, sin(α) ≈ α)

#### Simplified Form (Small α Approximation)

For most flight conditions where α < 15°:

```python
dV/dt = (T - D) / m - g * sin(γ)
dγ/dt = (L * cos(φ) - W * cos(γ)) / (m * V)
dψ/dt = (L * sin(φ)) / (m * V * cos(γ))
```

#### Load Factor Formulation (Often More Convenient)

Define load factor `n = L / W`:

```python
dV/dt = (T - D) / m - g * sin(γ)
dγ/dt = (g / V) * (n * cos(φ) - cos(γ))
dψ/dt = (g * n * sin(φ)) / (V * cos(γ))
```

For a **coordinated turn** (no sideslip), the relationship is:
```python
n = 1 / cos(φ)  # for level turn
```

So for a 60° bank, n = 2 ("2g turn").

---

## 4. Aerodynamic Force Models

### Dynamic Pressure

The fundamental scaling quantity:
```python
q = 0.5 * ρ * V²
```
Where `ρ` is air density (kg/m³) and `V` is true airspeed (m/s).

### Lift Force
```python
L = q * S * C_L
```
Where:
- `S` = Wing reference area (m²)
- `C_L` = Lift coefficient (dimensionless)

### Drag Force
```python
D = q * S * C_D
```
Where `C_D` = Drag coefficient (dimensionless)

### Converting to Accelerations
```python
# Specific forces (acceleration per unit mass)
L/m = q * S * C_L / m = (q/W) * g * S * C_L = (ρ * V² * S * C_L) / (2 * m)

# Wing loading W/S is a key aircraft parameter
# Let W/S = wing loading in N/m² or lb/ft²
L/W = (q * C_L) / (W/S)
```

---

## 5. The Drag Polar

### The Parabolic Drag Polar (Primary Model)

The most important equation for aircraft performance:

```python
C_D = C_D0 + K * C_L²
```

Where:
- `C_D0` = Zero-lift drag coefficient (parasite drag)
- `K` = Induced drag factor = 1 / (π * AR * e)
- `AR` = Aspect Ratio = b² / S (wingspan squared over wing area)
- `e` = Oswald efficiency factor (typically 0.7-0.85 for WW2 fighters)

### Computing K from Aircraft Geometry

```python
AR = b² / S  # Aspect ratio
e = 0.78  # Typical for straight-wing WW2 fighter (Oswald efficiency)
K = 1 / (π * AR * e)
```

**Empirical Formulas for Oswald Efficiency**:

For straight wings (Raymer approximation):
```python
e ≈ 1.78 * (1 - 0.045 * AR^0.68) - 0.64  # Raymer
e ≈ 0.7 + 0.1 * (AR - 6) / 4  # Linear approximation for AR 4-10
```

For WW2 aircraft, using `e = 0.75-0.85` is reasonable.

### Typical WW2 Fighter Values

| Parameter | P-51D Mustang | Spitfire Mk IX | Bf 109G | Fw 190A |
|-----------|---------------|----------------|---------|---------|
| C_D0 | 0.0163-0.020 | 0.020-0.021 | 0.024-0.028 | 0.021-0.024 |
| AR | 5.86 | 6.48 | 6.07 | 5.74 |
| e (estimated) | 0.80 | 0.85 | 0.78 | 0.78 |
| K | 0.0686 | 0.058 | 0.069 | 0.071 |

### Generalized Drag Polar (More Accurate)

```python
C_D = C_D_min + K * (C_L - C_L_min_drag)²
```

Where `C_L_min_drag` is typically 0.1-0.2 for cambered airfoils. For simplicity in RL, the standard parabolic form is usually sufficient.

### Mach Number Effects (Compressibility)

For transonic flight (M > 0.6), drag rises significantly. Simple approximation:

```python
if M < M_crit:
    C_D0_M = C_D0
elif M < 1.0:
    # Transonic drag rise
    C_D0_M = C_D0 * (1 + 10 * (M - M_crit)²)
else:
    # Supersonic (unlikely for WW2)
    C_D0_M = C_D0 * (1 + 0.5)

# M_crit typically 0.7-0.75 for WW2 aircraft
```

For WW2 fighters operating below M=0.6 in normal combat, Mach effects can often be ignored.

---

## 6. Lift Coefficient Modeling

### Linear Region (Pre-Stall)

In the linear region of the lift curve:

```python
C_L = C_Lα * (α - α_0)
```

Where:
- `C_Lα` = Lift curve slope (per radian)
- `α` = Angle of attack
- `α_0` = Zero-lift angle of attack (negative for cambered airfoils)

### Lift Curve Slope

**2D Airfoil (Thin Airfoil Theory)**:
```python
c_lα = 2 * π  # per radian ≈ 0.11 per degree
```

**3D Finite Wing (Lifting Line Theory)**:
```python
C_Lα = c_lα / (1 + c_lα / (π * AR))  # Approximation for elliptic wing
C_Lα = c_lα * AR / (AR + 2)  # Alternative approximation
```

For typical AR=6 WW2 fighter:
```python
C_Lα ≈ 2π * 6 / (6 + 2) ≈ 4.71 per radian ≈ 0.082 per degree
```

### Stall Modeling

**Critical**: For dogfighting, stall behavior matters! 

Simple piecewise model:
```python
def C_L(α, C_Lα, α_0, α_stall, C_L_max):
    α_effective = α - α_0
    if α < α_stall:
        return C_Lα * α_effective
    else:
        # Post-stall: lift drops off
        # Simple linear dropoff
        return C_L_max - 0.5 * (α - α_stall) * C_Lα
```

**Smooth stall model** (better for gradient-based methods):
```python
def C_L_smooth(α, C_Lα, α_0, α_stall, C_L_max):
    """Sigmoid-smoothed stall transition"""
    α_eff = α - α_0
    C_L_linear = C_Lα * α_eff
    
    # Smooth saturation using tanh
    k = 10  # Sharpness of stall transition
    C_L = C_L_max * np.tanh(C_L_linear / C_L_max)
    
    return C_L
```

**Typical values for WW2 fighters**:
- `α_stall` ≈ 14-18° (clean configuration)
- `C_L_max` ≈ 1.3-1.6 (clean), up to 2.0+ with flaps

### Relating Lift Coefficient to Load Factor

In a maneuver:
```python
C_L = (n * W) / (q * S) = (2 * n * W) / (ρ * V² * S)
```

So for a given load factor and speed, you can find the required C_L, and from that, the required α.

---

## 7. Propulsion: Piston Engine and Propeller Models

### Engine Power vs. Altitude

Piston engines lose power with altitude due to decreasing air density. For naturally aspirated engines:

```python
P(h) = P_SL * σ  # Where σ = ρ/ρ_SL (density ratio)
```

For supercharged engines (most WW2 fighters):
```python
if h < h_critical:
    P(h) = P_rated  # Full power up to critical altitude
else:
    P(h) = P_rated * (ρ(h) / ρ(h_critical))
```

**Critical altitude** is where the supercharger can no longer maintain sea-level manifold pressure. Typical values:
- Single-stage supercharger: 15,000-20,000 ft
- Two-stage supercharger: 25,000-30,000 ft (stepped)

### Propeller Model

For prop aircraft, thrust depends on power and propeller efficiency:

```python
T = η_p * P / V
```

Where `η_p` = propeller efficiency (typically 0.7-0.85 in cruise).

**Problem**: At V=0, this gives infinite thrust!

### Propeller Efficiency Model

Simple model for variable-pitch propeller:
```python
def prop_efficiency(V, P, D_prop, rho):
    """
    V: airspeed (m/s)
    P: shaft power (W)
    D_prop: propeller diameter (m)
    rho: air density (kg/m³)
    """
    # Advance ratio proxy
    if V < 1:
        V = 1  # Avoid division by zero
    
    # Maximum theoretical efficiency from momentum theory
    T_ideal = P / V
    disk_area = π * (D_prop/2)²
    v_induced = np.sqrt(T_ideal / (2 * rho * disk_area))
    η_ideal = 1 / (1 + v_induced / V)
    
    # Practical efficiency (80-90% of ideal)
    η_p = 0.85 * η_ideal
    
    # Clamp to reasonable range
    η_p = np.clip(η_p, 0, 0.88)
    
    return η_p
```

### Simplified Thrust Model (Recommended for RL)

Rather than modeling η_p complexly, use an empirical fit:

```python
def thrust_model(V, P_max, V_max, rho, rho_SL):
    """
    Simple thrust model for WW2 prop aircraft
    
    At low speed: T approaches static thrust
    At high speed: T = η * P / V with η ≈ 0.8
    """
    # Power available (with altitude correction)
    P_avail = P_max * (rho / rho_SL)  # Simplified; use supercharger model for better accuracy
    
    # Static thrust approximation (from momentum theory)
    # T_static ≈ (P² * rho * disk_area * 2)^(1/3)
    D_prop = 3.0  # meters, typical WW2 fighter
    disk_area = π * (D_prop/2)**2
    T_static = (P_avail**2 * 2 * rho * disk_area)**(1/3)
    
    # High-speed thrust
    η_cruise = 0.80
    T_cruise = η_cruise * P_avail / max(V, 1)
    
    # Blend between static and cruise
    # Smooth transition around V_transition
    V_transition = 50  # m/s
    blend = np.tanh(V / V_transition)
    
    T = T_static * (1 - blend) + T_cruise * blend
    
    return T
```

### Even Simpler: Polynomial Thrust Model

For maximum speed, fit thrust vs velocity from known aircraft data:

```python
def thrust_polynomial(V, T_static, T_max_speed, V_max):
    """
    T(V) = T_static - k*V² approximately, where aircraft reaches V_max when T = D
    """
    k = (T_static - T_max_speed) / V_max**2
    T = T_static - k * V**2
    return max(T, 0)
```

---

## 8. Atmospheric Model

### International Standard Atmosphere (ISA)

For troposphere (h < 11,000 m / 36,089 ft):

```python
# Constants
T_SL = 288.15  # K (15°C)
P_SL = 101325  # Pa
ρ_SL = 1.225   # kg/m³
g = 9.80665    # m/s²
R = 287.05     # J/(kg·K), specific gas constant for air
γ_air = 1.4    # Ratio of specific heats
λ = 0.0065     # Temperature lapse rate, K/m

def atmosphere_troposphere(h):
    """
    ISA atmospheric properties for h in meters (h < 11000 m)
    """
    T = T_SL - λ * h
    P = P_SL * (T / T_SL) ** (g / (R * λ))
    ρ = ρ_SL * (T / T_SL) ** (g / (R * λ) - 1)
    a = np.sqrt(γ_air * R * T)  # Speed of sound
    
    return T, P, ρ, a
```

**Numerical values**:
```python
# Exponents
g / (R * λ) = 9.80665 / (287.05 * 0.0065) ≈ 5.256
g / (R * λ) - 1 ≈ 4.256
```

So:
```python
T = 288.15 - 0.0065 * h
P = 101325 * (T / 288.15) ** 5.256
ρ = 1.225 * (T / 288.15) ** 4.256
```

### Density Ratio (Most Important for Performance)

```python
σ = ρ / ρ_SL = (T / T_SL) ** 4.256 = (1 - h/44330) ** 4.256
```

Quick approximation:
```python
σ ≈ np.exp(-h / 9000)  # Rough exponential fit, h in meters
```

### Altitude in Feet (Common in Aviation)

```python
def atmosphere_ISA_feet(h_ft):
    """h_ft in feet"""
    h_m = h_ft * 0.3048
    T = 288.15 - 0.0065 * h_m
    σ = (T / 288.15) ** 4.256
    ρ = 1.225 * σ
    a = np.sqrt(1.4 * 287.05 * T)
    return T, ρ, a, σ
```

---

## 9. Performance Calculations

### Maximum Level Flight Speed

At maximum speed, Thrust = Drag:
```python
T_max = D = q * S * C_D = 0.5 * ρ * V_max² * S * C_D0  # At high speed, induced drag small
```

Solving:
```python
V_max ≈ sqrt(2 * T_max / (ρ * S * C_D0))
```

### Stall Speed

At stall, L = W at C_L_max:
```python
V_stall = sqrt(2 * W / (ρ * S * C_L_max))
```

Or in terms of wing loading:
```python
V_stall = sqrt(2 * (W/S) / (ρ * C_L_max))
```

### Best Climb Speed and Rate

**Maximum rate of climb** occurs at the speed where excess power is maximum:
```python
P_excess = P_avail - P_required
P_required = D * V = 0.5 * ρ * V³ * S * C_D
```

For prop aircraft, best climb occurs roughly at:
```python
V_best_climb ≈ sqrt(2 * (W/S) / (ρ * sqrt(3 * C_D0 / K)))  # Minimum power speed
```

**Rate of climb**:
```python
RC = P_excess / W = (P_avail - D*V) / W
```

Or in terms of specific excess power:
```python
P_s = (T - D) * V / W = V * (T/W - D/W)
RC = P_s  # for small climb angles
```

### Turn Performance

**Turn radius**:
```python
R = V² / (g * sqrt(n² - 1))
```

For n >> 1:
```python
R ≈ V² / (g * n)
```

**Turn rate** (angular velocity):
```python
ω = V / R = g * sqrt(n² - 1) / V
```

**Maximum instantaneous turn rate** (limited by C_L_max):
```python
n_max_aero = q * S * C_L_max / W
ω_max = g * sqrt(n_max_aero² - 1) / V
```

**Maximum sustained turn rate** (limited by thrust = drag):
Must have T = D at the required C_L:
```python
# At sustained turn, T = D = q * S * (C_D0 + K * C_L²)
# And L = n * W = q * S * C_L
# Solve for n_sustained given T_avail
```

### Energy Management

**Specific energy** (energy height):
```python
E_s = h + V² / (2 * g)
```

**Specific excess power**:
```python
P_s = dE_s/dt = (T - D) * V / W
```

This is THE key parameter for dogfighting—aircraft with higher P_s at a given flight condition will "win" the energy game.

---

## 10. Implementation Strategies for High SPS

### Vectorization is Everything

Write all computations to operate on batched tensors:

```python
import numpy as np

def step_vectorized(state, action, aircraft_params):
    """
    state: (N, 6) array of [V, γ, ψ, x, y, h] for N environments
    action: (N, 3) array of [throttle, bank_cmd, pitch_cmd]
    """
    V, γ, ψ, x, y, h = state.T
    throttle, φ_cmd, α_cmd = action.T
    
    # Atmospheric properties (vectorized)
    T_atm = 288.15 - 0.0065 * h
    σ = (T_atm / 288.15) ** 4.256
    ρ = 1.225 * σ
    
    # Dynamic pressure
    q = 0.5 * ρ * V**2
    
    # Aerodynamic coefficients
    C_L = compute_CL_vectorized(α_cmd, aircraft_params)
    C_D = aircraft_params['CD0'] + aircraft_params['K'] * C_L**2
    
    # Forces
    L = q * aircraft_params['S'] * C_L
    D = q * aircraft_params['S'] * C_D
    T = compute_thrust_vectorized(V, throttle, σ, aircraft_params)
    W = aircraft_params['mass'] * 9.81
    
    # Equations of motion
    n = L / W
    dV_dt = (T - D) / aircraft_params['mass'] - 9.81 * np.sin(γ)
    dγ_dt = (9.81 / V) * (n * np.cos(φ_cmd) - np.cos(γ))
    dψ_dt = (9.81 * n * np.sin(φ_cmd)) / (V * np.cos(γ) + 1e-6)
    
    # Kinematics
    dx_dt = V * np.cos(γ) * np.cos(ψ)
    dy_dt = V * np.cos(γ) * np.sin(ψ)
    dh_dt = V * np.sin(γ)
    
    # Euler integration
    dt = 0.02  # 50 Hz
    new_state = state + dt * np.stack([dV_dt, dγ_dt, dψ_dt, dx_dt, dy_dt, dh_dt], axis=1)
    
    return new_state
```

### Avoid These Performance Killers

1. **Python loops over environments** - Always use vectorized operations
2. **Conditionals on per-environment basis** - Use `np.where` or `np.clip` instead
3. **Complex lookup tables** - Replace with polynomial/analytical approximations
4. **Trigonometric functions** - Cache sin/cos when possible; consider small-angle approximations
5. **Division by small numbers** - Add epsilon to avoid NaN/Inf

### JAX Implementation for GPU

```python
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2,))
def step_jax(state, action, aircraft_params):
    """JIT-compiled step function for maximum GPU performance"""
    V, γ, ψ, x, y, h = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4], state[..., 5]
    
    # ... same logic as numpy version ...
    
    return new_state

# Vectorize over batch dimension
step_batched = jax.vmap(step_jax, in_axes=(0, 0, None))
```

### Numerical Integration

For high SPS, use simple Euler integration with small timestep:

```python
dt = 0.02  # 50 Hz (20ms timestep)
state_new = state + dt * state_derivative
```

For better accuracy without much overhead, use semi-implicit Euler:
```python
# Update velocities first
V_new = V + dt * dV_dt
# Use new velocity for positions
x_new = x + dt * V_new * cos(γ) * cos(ψ)
```

RK4 is typically overkill for RL training and adds 4x computational cost.

### State Normalization

Normalize states for neural network input:
```python
state_normalized = (state - state_mean) / state_std

# Typical normalization values for WW2 dogfight:
# V: mean=150 m/s, std=50 m/s
# γ: mean=0, std=0.5 rad  
# ψ: mean=π, std=π (or use sin/cos representation)
# x, y: mean=0, std=5000 m
# h: mean=3000 m, std=2000 m
```

---

## 11. WW2 Aircraft Reference Data

### P-51D Mustang

```python
P51D = {
    'name': 'P-51D Mustang',
    'mass': 4175,          # kg (loaded)
    'S': 21.65,            # m² wing area
    'b': 11.28,            # m wingspan
    'AR': 5.86,            # aspect ratio
    'CD0': 0.0170,         # zero-lift drag
    'e': 0.80,             # Oswald efficiency
    'K': 0.068,            # induced drag factor
    'CL_max': 1.49,        # max lift coefficient (clean)
    'CL_alpha': 4.7,       # per radian
    'alpha_stall': 16,     # degrees
    'P_max': 1230000,      # W (1650 hp at WEP)
    'h_critical': 7620,    # m (25,000 ft) with 2-stage supercharger
    'V_max': 180,          # m/s (703 km/h at altitude)
    'V_stall': 46,         # m/s (100 mph clean)
    'RC_max': 17.5,        # m/s (3450 ft/min)
    'service_ceiling': 12770,  # m (41,900 ft)
    'n_limit': 8.0,        # structural g limit
}
```

### Supermarine Spitfire Mk IX

```python
SpitfireIX = {
    'name': 'Spitfire Mk IX',
    'mass': 3400,          # kg (loaded)
    'S': 22.48,            # m² wing area
    'b': 11.23,            # m wingspan (clipped)
    'AR': 5.61,            # aspect ratio
    'CD0': 0.0210,         # zero-lift drag
    'e': 0.85,             # Oswald efficiency (elliptical wing)
    'K': 0.067,            # induced drag factor
    'CL_max': 1.36,        # max lift coefficient
    'CL_alpha': 4.5,       # per radian
    'alpha_stall': 15,     # degrees
    'P_max': 1100000,      # W (1475 hp)
    'h_critical': 6100,    # m (20,000 ft)
    'V_max': 182,          # m/s (657 km/h)
    'V_stall': 42,         # m/s (82 kt)
    'RC_max': 21,          # m/s (4100 ft/min)
    'service_ceiling': 13100,  # m (43,000 ft)
    'n_limit': 8.0,
}
```

### Messerschmitt Bf 109G-6

```python
Bf109G = {
    'name': 'Bf 109G-6',
    'mass': 3100,          # kg (loaded)
    'S': 16.05,            # m² wing area
    'b': 9.92,             # m wingspan
    'AR': 6.13,            # aspect ratio
    'CD0': 0.0260,         # zero-lift drag (higher due to design)
    'e': 0.78,             # Oswald efficiency
    'K': 0.066,            # induced drag factor
    'CL_max': 1.52,        # max lift coefficient
    'CL_alpha': 4.8,       # per radian
    'alpha_stall': 17,     # degrees
    'P_max': 1050000,      # W (1410 hp)
    'h_critical': 5700,    # m (18,700 ft)
    'V_max': 170,          # m/s (621 km/h)
    'V_stall': 50,         # m/s (97 kt)
    'RC_max': 19,          # m/s (3750 ft/min)
    'service_ceiling': 11550,  # m (37,900 ft)
    'n_limit': 7.5,
}
```

### Focke-Wulf Fw 190A-8

```python
Fw190A = {
    'name': 'Fw 190A-8',
    'mass': 4400,          # kg (loaded)
    'S': 18.30,            # m² wing area
    'b': 10.51,            # m wingspan
    'AR': 6.04,            # aspect ratio
    'CD0': 0.0220,         # zero-lift drag
    'e': 0.78,             # Oswald efficiency
    'K': 0.068,            # induced drag factor
    'CL_max': 1.45,        # max lift coefficient
    'CL_alpha': 4.6,       # per radian
    'alpha_stall': 16,     # degrees
    'P_max': 1270000,      # W (1700 hp with MW 50)
    'h_critical': 6300,    # m (20,700 ft)
    'V_max': 171,          # m/s (615 km/h)
    'V_stall': 55,         # m/s (107 kt)
    'RC_max': 15,          # m/s (2950 ft/min)
    'service_ceiling': 10300,  # m (33,800 ft)
    'n_limit': 8.5,
}
```

### Mitsubishi A6M5 Zero

```python
A6M5 = {
    'name': 'A6M5 Zero',
    'mass': 2750,          # kg (loaded)
    'S': 22.44,            # m² wing area
    'b': 11.0,             # m wingspan
    'AR': 5.39,            # aspect ratio
    'CD0': 0.0230,         # zero-lift drag
    'e': 0.80,             # Oswald efficiency
    'K': 0.074,            # induced drag factor
    'CL_max': 1.40,        # max lift coefficient
    'CL_alpha': 4.3,       # per radian
    'alpha_stall': 15,     # degrees
    'P_max': 840000,       # W (1130 hp)
    'h_critical': 4500,    # m (14,800 ft)
    'V_max': 156,          # m/s (565 km/h)
    'V_stall': 40,         # m/s (78 kt)
    'RC_max': 16,          # m/s (3150 ft/min)
    'service_ceiling': 11740,  # m (38,520 ft)
    'n_limit': 7.0,        # Structural limit (lighter construction)
}
```

---

## 12. Validation and Sanity Checks

### Must-Pass Tests

Before deploying the environment, verify these behaviors:

1. **Level flight equilibrium**: At trim conditions, aircraft should maintain altitude
   ```python
   assert abs(dh_dt) < 0.1  # m/s at trim
   ```

2. **Stall speed matches data**: 
   ```python
   V_stall_computed = np.sqrt(2 * W / (ρ_SL * S * CL_max))
   assert abs(V_stall_computed - V_stall_data) / V_stall_data < 0.05
   ```

3. **Max speed matches data** (approximately):
   ```python
   # At altitude where V_max occurs
   V_max_computed = compute_max_speed(h_optimal)
   assert abs(V_max_computed - V_max_data) / V_max_data < 0.10
   ```

4. **Turn physics are correct**:
   ```python
   # 60° bank should give 2g and specific turn radius
   n = 1 / np.cos(np.radians(60))
   assert abs(n - 2.0) < 0.01
   R = V**2 / (g * np.sqrt(n**2 - 1))
   # At V=100 m/s, R ≈ 588 m
   ```

5. **Energy conservation** (with zero thrust/drag):
   ```python
   E_s = h + V**2 / (2*g)
   # With T=D=0, dE_s/dt should be zero
   ```

6. **Climb rate matches data**:
   ```python
   RC_computed = compute_max_RC(h=0)
   assert abs(RC_computed - RC_max_data) / RC_max_data < 0.15
   ```

### Behavioral Checks for Dogfighting

1. **Energy advantage matters**: Aircraft starting with more altitude/speed should have an advantage
2. **Turn fights favor appropriate aircraft**: Low wing-loading aircraft should out-turn heavy ones
3. **Boom-and-zoom works**: Fast diving attacks followed by climb-away should be viable
4. **Stall is dangerous**: Aircraft that stall should lose controllability temporarily
5. **Altitude matters**: Engine performance should degrade at high altitude

---

## 13. Sources and References

### Primary Textbooks

1. **Anderson, J.D.** - "Introduction to Flight" (McGraw-Hill)
   - Chapters on aircraft performance, drag polar, climb/turn performance
   - Standard reference for undergraduate aerodynamics

2. **Stevens, B.L. & Lewis, F.L.** - "Aircraft Control and Simulation" (Wiley, 3rd Ed. 2015)
   - Gold standard for flight dynamics modeling
   - F-16 model used in many research implementations
   - GitHub implementations: [isrlab/F16-Model-Matlab](https://github.com/isrlab/F16-Model-Matlab)

3. **Raymer, D.P.** - "Aircraft Design: A Conceptual Approach" (AIAA)
   - Empirical formulas for Oswald efficiency, drag estimation
   - Excellent for quick approximations

4. **Roskam, J.** - "Methods for Estimating Drag Polars of Subsonic Airplanes"
   - Detailed component buildup methods

5. **Stengel, R.E.** - "Flight Dynamics" (Princeton University Press)
   - Excellent lecture notes available at: https://stengel.mycpanel.princeton.edu/
   - MATLAB code for 6-DOF simulation: [FLIGHTv2](https://stengel.mycpanel.princeton.edu/FDcodeB.html)

### Open Source Flight Dynamics Models

1. **JSBSim** - https://github.com/JSBSim-Team/jsbsim
   - Industry-standard open-source FDM
   - Used in FlightGear, DARPA ACE program
   - XML-based aircraft configuration files
   - Has WW2 aircraft models (P-51, etc.)

2. **AeroBenchVV** - https://github.com/pheidlauf/AeroBenchVV
   - F-16 model for verification and validation
   - MATLAB implementation of Stevens & Lewis model

3. **F16 Flight Dynamics (Python/C++)** - https://github.com/EthanJamesLew/f16-flight-dynamics
   - Efficient C++ implementation with Python bindings
   - Good reference for high-performance implementation

### RL Environment Implementations

1. **LAG (Light Aircraft Game)** - https://github.com/liuqh16/LAG
   - JSBSim-based air combat environment
   - PPO/MAPPO implementations included
   - Good reference for reward shaping in dogfights

2. **BVR Gym** - https://arxiv.org/html/2403.17533
   - Beyond-visual-range air combat environment
   - Built on JSBSim with OpenAI Gym interface

3. **Tunnel** - https://arxiv.org/html/2505.01953v1
   - Lightweight F-16 RL environment
   - Focus on simplicity and accessibility

4. **DBRL** - https://github.com/mrwangyou/DBRL
   - Dogfighting benchmark for RL research

### High-Performance RL Infrastructure

1. **EnvPool** - https://github.com/sail-sg/envpool
   - C++-based parallel environment execution
   - 1M+ steps/second demonstrated
   - Good patterns for vectorized environments

2. **Isaac Gym** - https://github.com/isaac-sim/IsaacGymEnvs
   - GPU-accelerated physics simulation
   - Demonstrates full GPU pipeline for RL

3. **PufferLib** - https://github.com/PufferAI/PufferLib
   - Clean, fast RL training framework
   - Good target platform for implementation

### WW2 Aircraft Data

1. **WWII Aircraft Performance** - https://www.wwiiaircraftperformance.org/
   - Comprehensive primary source documents
   - Flight test data, performance charts

2. **CFD Evaluation of WW2 Fighters** - Lednicer (1995)
   - ResearchGate: "A CFD evaluation of three prominent World War II fighter aircraft"
   - Spitfire, P-51, Fw 190 aerodynamic comparison

3. **Aerodynamics of the Spitfire** - Royal Aeronautical Society
   - Detailed analysis of Spitfire aerodynamics
   - CD0 ≈ 0.020-0.021 for Spitfire

### Atmospheric Models

1. **International Standard Atmosphere** - ISO 2533:1975
   - Official ISA specification
   - Wikipedia summary is accurate and sufficient

2. **ICAO Standard Atmosphere** - ICAO Doc 7488-CD
   - Extended to 80 km altitude

---

## Quick Reference Card

### Core Equations (Copy-Paste Ready)

```python
# === ATMOSPHERE (ISA, troposphere) ===
T = 288.15 - 0.0065 * h  # Temperature [K], h in meters
σ = (T / 288.15) ** 4.256  # Density ratio
ρ = 1.225 * σ  # Density [kg/m³]
a = 20.05 * np.sqrt(T)  # Speed of sound [m/s]

# === AERODYNAMICS ===
q = 0.5 * ρ * V**2  # Dynamic pressure [Pa]
C_L = C_Lα * α  # Lift coefficient (linear region)
C_D = C_D0 + K * C_L**2  # Drag polar
L = q * S * C_L  # Lift [N]
D = q * S * C_D  # Drag [N]

# === PROPULSION (simple) ===
P = P_max * σ * throttle  # Power available [W]
T = η_p * P / V  # Thrust [N], η_p ≈ 0.80

# === PERFORMANCE ===
V_stall = np.sqrt(2 * W / (ρ * S * C_L_max))  # Stall speed
R_turn = V**2 / (g * np.sqrt(n**2 - 1))  # Turn radius
ω_turn = g * np.sqrt(n**2 - 1) / V  # Turn rate [rad/s]
RC = (P * η_p - D * V) / W  # Rate of climb [m/s]

# === EQUATIONS OF MOTION (3DOF point mass) ===
dV_dt = (T - D) / m - g * np.sin(γ)
dγ_dt = (g / V) * (n * np.cos(φ) - np.cos(γ))
dψ_dt = g * n * np.sin(φ) / (V * np.cos(γ))
dx_dt = V * np.cos(γ) * np.cos(ψ)
dy_dt = V * np.cos(γ) * np.sin(ψ)
dh_dt = V * np.sin(γ)
```

### Typical Values to Remember

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| C_D0 | 0.017-0.028 | Lower = cleaner aircraft |
| e (Oswald) | 0.75-0.85 | Elliptic wing ≈ 0.85 |
| K | 0.06-0.08 | K = 1/(π·AR·e) |
| C_L_max | 1.3-1.6 | Clean configuration |
| C_Lα | 4.5-5.0 /rad | 3D wing |
| α_stall | 14-18° | Depends on airfoil |
| η_propeller | 0.75-0.85 | Cruise conditions |
| σ at 20,000 ft | 0.53 | Density ratio |
| σ at 30,000 ft | 0.37 | Density ratio |

---

*Document prepared for Claude agents developing WW2 dogfighting RL environments. Focus on computational efficiency while maintaining physical plausibility.*
