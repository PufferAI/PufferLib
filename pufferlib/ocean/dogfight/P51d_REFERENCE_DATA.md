# P-51D Mustang Reference Data for RL Simulation Validation

## Document Purpose

This document provides authoritative reference data for the P-51D Mustang to enable validation of a simplified RL flight simulation. The goal is NOT perfect simulation fidelity, but rather reasonable agreement with historical performance data to ensure the RL environment conveys realistic WW2 fighter dynamics.

**Key Philosophy**: Run automated test scripts that "hijack" the policy (e.g., "maintain level flight at 100% throttle") and compare simulated performance against these reference values.

---

## 1. STANDARD TEST CONDITION

For all validation tests, use this standardized configuration:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Weight** | 9,000 lb (4,082 kg) | ~25% internal fuel (~45 gal remaining) |
| **Fuel Load** | 45 US gal | From 180 gal full internal |
| **Altitude** | Sea level (unless noted) | ISA conditions |
| **Configuration** | Clean | No external stores, gear up |
| **Power Setting** | As specified per test | |

**Why 9,000 lb?** This represents a combat-ready fighter after burning fuel en route to engagement:
- Empty weight: 7,635 lb
- Pilot + equipment: ~200 lb
- Ammo (full): ~330 lb
- Oil: ~90 lb
- 45 gal fuel: ~270 lb
- Misc: ~475 lb
- **Total: ~9,000 lb**

Historical reference: NAA report NA-46-130 uses 9,611 lb (full 180 gal internal fuel).

---

## 2. PHYSICAL DIMENSIONS

### 2.1 Overall Dimensions

| Parameter | Imperial | Metric |
|-----------|----------|--------|
| Length | 32.25 ft | 9.83 m |
| Wingspan | 37.0 ft | 11.28 m |
| Height (tail down) | 13.67 ft | 4.17 m |
| Wing Area | 233 ft² | 21.65 m² |

### 2.2 Wing Geometry

| Parameter | Value | Notes |
|-----------|-------|-------|
| Aspect Ratio | 5.86 | AR = b²/S = 37²/233 |
| Mean Aerodynamic Chord (MAC) | 6.63 ft | 2.02 m |
| Root Chord | 8.48 ft | 2.58 m |
| Tip Chord | 3.87 ft | 1.18 m |
| Taper Ratio | 0.456 | λ = c_tip/c_root |
| **Wing Incidence** | **+1° to +2°** | Root chord to fuselage datum |
| Washout (twist) | 2-3° | Tip relative to root (reduces tip stall) |
| Dihedral | ~5° | |

### 2.3 Control Surfaces

| Surface | Span | Chord (% wing) | Area |
|---------|------|----------------|------|
| Aileron | 8.5 ft each | ~25% | ~9 ft² each |
| Flap | ~5.5 ft each | ~25% | ~12 ft² each |

### 2.4 Tail Geometry

| Parameter | Horizontal Stabilizer | Vertical Stabilizer |
|-----------|----------------------|---------------------|
| Area | 45.4 ft² | 14.8 ft² |
| Span | 13.1 ft | 4.7 ft |
| Root Chord | 4.6 ft | 4.7 ft |
| Tip Chord | 2.3 ft | 1.6 ft |

---

## 3. AERODYNAMIC DATA

### 3.1 Airfoil

**Profile**: NAA/NACA 45-100 (laminar flow)
- Root: 15.1% thick, 1.6% camber at 50% chord
- Tip: 11.4% thick, 1.6% camber at 50% chord
- Max thickness at ~39% chord (laminar flow design)

### 3.2 Lift Characteristics

| Parameter | Value | Source/Notes |
|-----------|-------|--------------|
| **C_L_α (lift curve slope)** | 0.095-0.10 /deg | ~5.4-5.7 /rad; 3D wing |
| **α₀ (zero-lift angle)** | **-1.0° to -1.5°** | Due to 1.6% camber |
| **C_L_max (clean)** | 1.45 - 1.50 | Flaps up |
| **C_L_max (flaps 50°)** | 1.80 - 1.90 | Landing configuration |
| **α_stall (clean)** | 19.1° | From IL-2 data |
| **α_stall (flaps)** | 16.3° | Landing configuration |

**Lift Equation**:
```
C_L = C_L_α × (α - α₀)
C_L = 0.097 × (α + 1.2°)   [deg input]
C_L = 5.56 × (α + 0.021)   [rad input, α₀ in rad]
```

### 3.3 Drag Characteristics

| Parameter | Value | Source/Notes |
|-----------|-------|--------------|
| **C_D0 (zero-lift drag)** | 0.0163 | Published Wikipedia/museum data |
| **Oswald Efficiency (e)** | 0.75 - 0.80 | Typical for tapered wing |
| **K (induced drag factor)** | 0.072 | K = 1/(π × AR × e) = 1/(π × 5.86 × 0.75) |

**Drag Polar**:
```
C_D = C_D0 + K × C_L²
C_D = 0.0163 + 0.072 × C_L²
```

**Drag Area**: 3.80 ft² (0.35 m²)

### 3.4 Lift-to-Drag Ratio

| Condition | L/D | Notes |
|-----------|-----|-------|
| Maximum L/D | 14.6 | At optimal C_L |
| Cruise (~350 mph) | ~12-13 | |
| Combat maneuvering | 6-10 | Higher C_L, more induced drag |

**Optimal L/D occurs at**:
```
C_L_opt = sqrt(C_D0 / K) = sqrt(0.0163 / 0.072) = 0.476
L/D_max = 1 / (2 × sqrt(C_D0 × K)) = 1 / (2 × sqrt(0.0163 × 0.072)) = 14.6
```

---

## 4. PROPULSION DATA

### 4.1 Engine: Packard V-1650-7 (Merlin 66)

| Parameter | Value |
|-----------|-------|
| Type | Liquid-cooled V-12, 2-stage 2-speed supercharger |
| Displacement | 1,647 in³ (27 L) |
| Bore × Stroke | 5.4" × 6.0" |
| Compression Ratio | 6.0:1 |
| Propeller Gear Ratio | 0.479:1 |

### 4.2 Power Settings

| Rating | MAP (in Hg) | RPM | BHP (SL) | Time Limit |
|--------|-------------|-----|----------|------------|
| **Military** | 61" | 3,000 | 1,490 hp | 15 min |
| **WEP (67")** | 67" | 3,000 | 1,650 hp | 5 min |
| **WEP (150 grade)** | 75" | 3,000 | 1,860 hp | 5 min |
| Cruise | 46" | 2,700 | ~1,100 hp | Unlimited |

### 4.3 Power vs Altitude (Military Power, 61" Hg)

| Altitude (ft) | BHP | Notes |
|---------------|-----|-------|
| Sea Level | 1,490 | Low blower |
| 2,600 | 1,580 | Peak low blower |
| 6,400 | 1,400 | High blower |
| 9,700 | ~1,400 | Near optimal high blower |
| 15,000 | ~1,350 | |
| 25,000 | ~1,100 | |
| 35,000 | ~750 | |

**Simplified Altitude Correction** (for sim):
```python
def get_power(P_rated, altitude_ft, throttle=1.0):
    """Simplified Merlin power model"""
    sigma = get_density_ratio(altitude_ft)
    
    # Two-stage supercharger approximation
    if altitude_ft < 4000:
        # Low blower - slight power increase
        P_available = P_rated * min(1.0, sigma * 1.05)
    elif altitude_ft < 10000:
        # Low blower critical altitude region
        P_available = P_rated * 1.0
    elif altitude_ft < 25000:
        # High blower
        P_available = P_rated * (0.95 - 0.01 * (altitude_ft - 10000) / 1000)
    else:
        # Above critical altitude, power drops
        P_available = P_rated * sigma * 1.2
    
    return P_available * throttle
```

### 4.4 Propeller

| Parameter | Value |
|-----------|-------|
| Type | Hamilton Standard 24D50 constant-speed |
| Diameter | 11 ft 2 in (3.40 m) |
| Blades | 4 |
| Propeller RPM at 3000 engine RPM | 1,437 |
| **Propeller Efficiency (cruise)** | 0.80 - 0.85 |
| **Propeller Efficiency (climb)** | 0.75 - 0.80 |
| **Propeller Efficiency (static)** | 0.50 - 0.60 |

**Thrust Calculation**:
```python
def get_thrust(power_hp, velocity_fps, eta_prop=0.80):
    """
    T = η_p × P / V
    
    power_hp: shaft horsepower
    velocity_fps: true airspeed in ft/s
    eta_prop: propeller efficiency (0.50-0.85 depending on V)
    returns: thrust in lbf
    """
    power_ftlb_s = power_hp * 550  # Convert HP to ft⋅lb/s
    
    if velocity_fps < 50:  # Low speed / static
        # Use momentum theory approximation
        eta_prop = 0.55
        velocity_fps = max(velocity_fps, 30)  # Avoid division by zero
    
    thrust_lbf = eta_prop * power_ftlb_s / velocity_fps
    return thrust_lbf
```

---

## 5. PERFORMANCE VALIDATION TARGETS

### 5.1 Level Flight - Maximum Speed

**Test**: Set throttle to specified power, maintain level flight (γ = 0), record stabilized TAS.

| Condition | Power | Altitude | Target Speed | Tolerance |
|-----------|-------|----------|--------------|-----------|
| WEP | 67" Hg | Sea Level | 368 mph (592 km/h) | ±10 mph |
| Military | 61" Hg | Sea Level | 355 mph (571 km/h) | ±10 mph |
| WEP | 67" Hg | 11,300 ft | 414 mph (666 km/h) | ±10 mph |
| Military | 61" Hg | 13,300 ft | 412 mph (663 km/h) | ±10 mph |
| WEP | 67" Hg | 24,500 ft | 440 mph (708 km/h) | ±10 mph |
| Military | 61" Hg | 26,200 ft | 435 mph (700 km/h) | ±10 mph |

**Test Script Logic**:
```python
def test_max_speed(sim, altitude_ft, power_setting):
    """
    1. Initialize at altitude, moderate speed
    2. Set throttle to power_setting
    3. Command pitch to maintain level (gamma = 0)
    4. Run until speed stabilizes (d|V|/dt < 0.1 ft/s²)
    5. Record stabilized TAS
    """
    sim.reset(altitude=altitude_ft, speed_fps=400)
    sim.set_throttle(power_setting)
    
    for step in range(10000):  # Max 200 seconds at 50 Hz
        # Simple level-flight controller
        gamma = sim.get_flight_path_angle()
        pitch_cmd = -gamma * 2.0  # P controller
        sim.set_pitch_rate_cmd(pitch_cmd)
        
        sim.step()
        
        if abs(sim.get_acceleration()) < 0.1:
            return sim.get_TAS_mph()
    
    return sim.get_TAS_mph()  # Return final value
```

### 5.2 Stall Speed

**Test**: At constant altitude, gradually reduce power while maintaining level flight until stall.

| Weight (lb) | Configuration | Target Stall Speed | Notes |
|-------------|---------------|-------------------|-------|
| 9,071 | Clean | ~100 mph (161 km/h) IAS | |
| 9,071 | Flaps down | 95.4 mph (154 km/h) IAS | NAA data |
| 9,000 | Clean | 99 mph (159 km/h) IAS | Test weight |
| 10,000 | Clean | 104 mph (168 km/h) IAS | Heavier |

**Stall Speed Formula**:
```python
def calculate_stall_speed(weight_lb, rho_slugft3, wing_area_ft2, CL_max):
    """
    V_stall = sqrt(2W / (ρ × S × CL_max))
    
    At sea level (ρ = 0.002377 slug/ft³), S = 233 ft², CL_max = 1.48:
    V_stall = sqrt(2 × 9000 / (0.002377 × 233 × 1.48))
    V_stall = sqrt(18000 / 0.820) = sqrt(21951) = 148 ft/s = 101 mph
    """
    V_stall_fps = math.sqrt(2 * weight_lb / (rho_slugft3 * wing_area_ft2 * CL_max))
    return V_stall_fps * 0.6818  # Convert to mph
```

### 5.3 Rate of Climb

**Test**: Set throttle to specified power, maintain V_y (best climb speed ~165 mph IAS), record climb rate.

| Power | Altitude | Target ROC | Tolerance |
|-------|----------|------------|-----------|
| WEP (67") | Sea Level | 3,410 ft/min | ±200 ft/min |
| Military (61") | Sea Level | 3,030 ft/min | ±200 ft/min |
| WEP | 7,500 ft | 3,510 ft/min | ±200 ft/min |
| WEP | 21,200 ft | 2,680 ft/min | ±200 ft/min |
| Military | 9,700 ft | 3,170 ft/min | ±200 ft/min |
| Military | 23,200 ft | 2,300 ft/min | ±200 ft/min |

**Rate of Climb Formula** (steady):
```python
def calculate_ROC(thrust_lb, drag_lb, velocity_fps, weight_lb):
    """
    ROC = (T - D) × V / W  = V × sin(γ)
    
    Or: ROC = (P_available × η_p - P_required) / W
    """
    excess_thrust = thrust_lb - drag_lb
    sin_gamma = excess_thrust / weight_lb
    ROC_fps = velocity_fps * sin_gamma
    return ROC_fps * 60  # Convert to ft/min
```

**Test Script**:
```python
def test_climb_rate(sim, altitude_ft, power_setting, climb_speed_mph=165):
    """
    1. Initialize at altitude, at Vy
    2. Set throttle, maintain constant IAS
    3. Record stabilized climb rate
    """
    V_target_fps = climb_speed_mph * 1.467
    sim.reset(altitude=altitude_ft, speed_fps=V_target_fps)
    sim.set_throttle(power_setting)
    
    for step in range(5000):
        # Maintain constant airspeed in climb
        V_error = V_target_fps - sim.get_TAS_fps()
        pitch_cmd = -V_error * 0.1  # Speed-hold via pitch
        sim.set_pitch_rate_cmd(pitch_cmd)
        sim.step()
    
    return sim.get_climb_rate_fpm()
```

### 5.4 Level Turn Performance

**Test**: Maintain altitude and constant speed in coordinated turn, measure turn rate and radius.

| Speed (IAS) | Load Factor | Turn Rate | Turn Radius | Turn Time |
|-------------|-------------|-----------|-------------|-----------|
| 180 mph (290 km/h) | ~3.5g | 18°/s | ~800 ft | 20 sec |
| 250 mph | ~2.5g | 10°/s | ~1,500 ft | 36 sec |
| 350 mph | ~2.0g | 5°/s | ~3,500 ft | 72 sec |

**Turn Physics**:
```python
def calculate_turn(velocity_fps, load_factor, g=32.174):
    """
    In a level turn:
    L = n × W (lift = load factor × weight)
    L_vertical = W (to maintain altitude)
    L_horizontal = W × sqrt(n² - 1) (centripetal force)
    
    Turn radius: R = V² / (g × sqrt(n² - 1))
    Turn rate: ω = g × sqrt(n² - 1) / V [rad/s]
    """
    sqrt_term = math.sqrt(load_factor**2 - 1)
    
    radius_ft = velocity_fps**2 / (g * sqrt_term)
    turn_rate_rad_s = g * sqrt_term / velocity_fps
    turn_rate_deg_s = math.degrees(turn_rate_rad_s)
    turn_time_sec = 360 / turn_rate_deg_s
    
    return {
        'radius_ft': radius_ft,
        'turn_rate_deg_s': turn_rate_deg_s,
        'turn_time_sec': turn_time_sec
    }

# Example: 290 km/h (180 mph = 264 ft/s) at 3.5g
# sqrt(3.5² - 1) = sqrt(11.25) = 3.35
# R = 264² / (32.17 × 3.35) = 69696 / 107.8 = 647 ft
# ω = 32.17 × 3.35 / 264 = 0.408 rad/s = 23.4°/s
```

### 5.5 Level Flight at Zero AoA

**Critical Test**: What speed maintains level flight at α = 0°?

With wing incidence of ~+1.5°, the wing is at α_wing = +1.5° when fuselage is level.
At this AoA, considering α₀ ≈ -1.2°:
```
C_L = C_L_α × (α_wing - α₀)
C_L = 0.097 × (1.5 - (-1.2)) = 0.097 × 2.7° = 0.262
```

**Speed for Level Flight at 0° Fuselage Pitch (α_wing = +1.5°)**:
```python
# L = W (level flight)
# q × S × C_L = W
# 0.5 × ρ × V² × S × C_L = W
# V = sqrt(2W / (ρ × S × C_L))

W = 9000  # lb
rho = 0.002377  # slug/ft³ at sea level
S = 233  # ft²
C_L = 0.262

V = math.sqrt(2 * W / (rho * S * C_L))
# V = sqrt(18000 / 0.145) = sqrt(124138) = 352 ft/s = 240 mph
```

**At true 0° wing AoA** (fuselage pitched down 1.5°):
```
C_L = 0.097 × (0 - (-1.2)) = 0.097 × 1.2° = 0.116

V = sqrt(2 × 9000 / (0.002377 × 233 × 0.116))
V = sqrt(18000 / 0.064) = sqrt(281250) = 530 ft/s = 361 mph
```

**Summary Table - Level Flight AoA vs Speed**:

| Fuselage Pitch | Wing α | C_L | Speed (mph) | Speed (ft/s) |
|----------------|--------|-----|-------------|--------------|
| -1.5° | 0° | 0.116 | 361 | 530 |
| 0° | 1.5° | 0.262 | 240 | 352 |
| +2° | 3.5° | 0.456 | 182 | 267 |
| +5° | 6.5° | 0.747 | 142 | 208 |
| +10° | 11.5° | 1.233 | 111 | 163 |
| +15° | 16.5° (near stall) | 1.48 | 101 | 148 |

---

## 6. FLIGHT ENVELOPE LIMITS

### 6.1 Speed Limits

| Limit | Speed | Notes |
|-------|-------|-------|
| V_NE (never exceed) | 505 mph IAS | 812 km/h |
| V_max dive | 525-550 mph | Pilots reported exceeding redline |
| M_crit | 0.75-0.80 | Onset of compressibility |
| Max Mach achieved | ~0.85 | With structural risk |

### 6.2 G-Limits

| Condition | Limit | Notes |
|-----------|-------|-------|
| Design limit (positive) | +8g | At 8,000 lb |
| Design limit (negative) | -4g | |
| With external stores | +6.5g | Reduced for safety |
| Ultimate load | +12g | Structural failure |

### 6.3 Altitude Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Service ceiling | 41,900 ft | 100 ft/min ROC |
| Absolute ceiling | ~44,000 ft | |
| Combat ceiling | 36,900 ft | At 3000 RPM |

---

## 7. IMPLEMENTATION CONSTANTS

Copy-paste ready Python constants:

```python
# ============================================
# P-51D MUSTANG SIMULATION CONSTANTS
# Reference Weight: 9000 lb (combat weight)
# ============================================

# Physical Dimensions
WINGSPAN_FT = 37.0
WINGSPAN_M = 11.28
WING_AREA_FT2 = 233.0
WING_AREA_M2 = 21.65
MAC_FT = 6.63
ASPECT_RATIO = 5.86

# Mass Properties (test condition)
WEIGHT_LB = 9000.0
WEIGHT_KG = 4082.0
MASS_SLUG = WEIGHT_LB / 32.174  # 279.8 slug
MASS_KG = WEIGHT_KG

# Wing Geometry
WING_INCIDENCE_DEG = 1.5  # Root chord to fuselage
WING_INCIDENCE_RAD = 0.0262

# Aerodynamic Coefficients
CL_ALPHA_PER_DEG = 0.097      # 3D wing lift curve slope
CL_ALPHA_PER_RAD = 5.56
ALPHA_ZERO_LIFT_DEG = -1.2    # Zero-lift angle (cambered airfoil)
ALPHA_ZERO_LIFT_RAD = -0.021
CL_MAX_CLEAN = 1.48
CL_MAX_FLAPS = 1.85
CD0 = 0.0163                   # Zero-lift drag coefficient
OSWALD_E = 0.75               # Oswald efficiency factor
K_INDUCED = 1.0 / (3.14159 * ASPECT_RATIO * OSWALD_E)  # 0.072

# Stall
ALPHA_STALL_CLEAN_DEG = 19.1
ALPHA_STALL_FLAPS_DEG = 16.3

# Propulsion
ENGINE_POWER_WEP_HP = 1650    # 67" Hg, 3000 RPM
ENGINE_POWER_MIL_HP = 1490    # 61" Hg, 3000 RPM
ENGINE_POWER_CRUISE_HP = 1100 # 46" Hg, 2700 RPM
PROP_DIAMETER_FT = 11.167     # 11 ft 2 in
PROP_EFFICIENCY_CRUISE = 0.82
PROP_EFFICIENCY_CLIMB = 0.78
PROP_EFFICIENCY_STATIC = 0.55

# Limits
VNE_MPH = 505
VNE_FPS = 740
G_LIMIT_POS = 8.0
G_LIMIT_NEG = -4.0
SERVICE_CEILING_FT = 41900

# Atmosphere (sea level ISA)
RHO_SL_SLUGFT3 = 0.002377
RHO_SL_KGM3 = 1.225
TEMP_SL_K = 288.15
PRESSURE_SL_PA = 101325
LAPSE_RATE_K_PER_M = 0.0065

# Unit Conversions
FPS_TO_MPH = 0.6818
MPH_TO_FPS = 1.467
FPS_TO_KTS = 0.5925
KTS_TO_FPS = 1.688
FT_TO_M = 0.3048
M_TO_FT = 3.281
HP_TO_WATTS = 745.7
LBF_TO_N = 4.448
```

---

## 8. VALIDATION TEST SUITE

### 8.1 Test Categories

| Test | Priority | Tolerance | Notes |
|------|----------|-----------|-------|
| Stall speed | HIGH | ±5 mph | Fundamental lift validation |
| Max speed (SL) | HIGH | ±10 mph | Drag + power validation |
| Max speed (altitude) | MEDIUM | ±15 mph | Supercharger model |
| ROC (SL) | HIGH | ±200 ft/min | Excess power validation |
| ROC (altitude) | MEDIUM | ±300 ft/min | |
| Level turn time | MEDIUM | ±2 sec | Turn physics |
| Level flight speed @ α=0 | HIGH | ±15 mph | Incidence angle validation |

### 8.2 Example Test Script

```python
import numpy as np

class P51DValidationSuite:
    """Validation tests for P-51D flight model"""
    
    def __init__(self, sim_env):
        self.sim = sim_env
        self.results = {}
    
    def test_stall_speed(self):
        """Test: Gradually reduce speed until stall"""
        # Initialize at safe speed, level flight
        self.sim.reset(altitude_ft=5000, speed_fps=250, gamma_deg=0)
        self.sim.set_weight(9000)
        
        # Gradually reduce throttle while maintaining level
        for throttle in np.linspace(1.0, 0.0, 100):
            self.sim.set_throttle(throttle)
            
            # Run for 5 seconds to stabilize
            for _ in range(250):  # 50 Hz × 5 sec
                # Level flight controller
                gamma = self.sim.get_gamma()
                self.sim.command_pitch_rate(-gamma * 2.0)
                self.sim.step()
            
            # Check for stall
            if self.sim.get_alpha() > 18 or self.sim.is_stalled():
                stall_speed_mph = self.sim.get_speed_fps() * 0.6818
                break
        
        expected = 100  # mph
        actual = stall_speed_mph
        passed = abs(actual - expected) < 5
        
        self.results['stall_speed'] = {
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'tolerance': 5
        }
        return passed
    
    def test_max_speed_sea_level(self):
        """Test: Maximum speed at sea level with WEP"""
        self.sim.reset(altitude_ft=0, speed_fps=400, gamma_deg=0)
        self.sim.set_weight(9000)
        self.sim.set_throttle(1.0)  # WEP
        
        # Accelerate until stable
        prev_speed = 0
        for step in range(10000):
            # Maintain level flight
            gamma = self.sim.get_gamma()
            self.sim.command_pitch_rate(-gamma * 2.0)
            self.sim.step()
            
            # Check for stabilization
            current_speed = self.sim.get_speed_fps()
            if abs(current_speed - prev_speed) < 0.01:
                break
            prev_speed = current_speed
        
        max_speed_mph = self.sim.get_speed_fps() * 0.6818
        
        expected = 368  # mph at WEP, SL
        actual = max_speed_mph
        passed = abs(actual - expected) < 10
        
        self.results['max_speed_sl'] = {
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'tolerance': 10
        }
        return passed
    
    def test_climb_rate_sea_level(self):
        """Test: Rate of climb at sea level with WEP"""
        self.sim.reset(altitude_ft=1000, speed_fps=242, gamma_deg=10)  # ~165 mph
        self.sim.set_weight(9000)
        self.sim.set_throttle(1.0)
        
        target_speed_fps = 242  # 165 mph Vy
        
        # Climb at constant airspeed
        for step in range(5000):
            V_error = target_speed_fps - self.sim.get_speed_fps()
            self.sim.command_pitch_rate(-V_error * 0.05)
            self.sim.step()
        
        roc_fpm = self.sim.get_vertical_speed_fps() * 60
        
        expected = 3410  # ft/min at WEP, SL
        actual = roc_fpm
        passed = abs(actual - expected) < 200
        
        self.results['climb_rate_sl'] = {
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'tolerance': 200
        }
        return passed
    
    def test_turn_time(self):
        """Test: 360° turn time at 290 km/h (180 mph)"""
        V_fps = 264  # 180 mph
        self.sim.reset(altitude_ft=5000, speed_fps=V_fps, gamma_deg=0)
        self.sim.set_weight(9000)
        self.sim.set_throttle(1.0)
        
        # Bank to ~70° for ~3.5g turn
        bank_target_rad = np.radians(70)
        initial_heading = self.sim.get_heading()
        
        time_elapsed = 0
        while True:
            # Maintain bank angle
            bank_error = bank_target_rad - self.sim.get_bank()
            self.sim.command_roll_rate(bank_error * 2.0)
            
            # Maintain altitude
            # ... (pull back to compensate)
            
            self.sim.step()
            time_elapsed += self.sim.dt
            
            # Check for 360° turn completion
            heading_change = self.sim.get_heading() - initial_heading
            if heading_change >= 2 * np.pi:
                break
            
            if time_elapsed > 60:  # Timeout
                break
        
        expected = 20  # seconds
        actual = time_elapsed
        passed = abs(actual - expected) < 2
        
        self.results['turn_time'] = {
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'tolerance': 2
        }
        return passed
    
    def run_all_tests(self):
        """Run complete validation suite"""
        tests = [
            ('Stall Speed', self.test_stall_speed),
            ('Max Speed (SL)', self.test_max_speed_sea_level),
            ('Climb Rate (SL)', self.test_climb_rate_sea_level),
            ('Turn Time', self.test_turn_time),
        ]
        
        print("=" * 60)
        print("P-51D MUSTANG FLIGHT MODEL VALIDATION")
        print("=" * 60)
        
        all_passed = True
        for name, test_func in tests:
            try:
                passed = test_func()
                result = self.results.get(name.lower().replace(' ', '_').replace('(', '').replace(')', ''), {})
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{name:20s}: {status}")
                print(f"  Expected: {result.get('expected', 'N/A')}")
                print(f"  Actual:   {result.get('actual', 'N/A'):.1f}")
                print(f"  Tolerance: ±{result.get('tolerance', 'N/A')}")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"{name:20s}: ✗ ERROR - {e}")
                all_passed = False
        
        print("=" * 60)
        print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 60)
        
        return all_passed
```

---

## 9. QUICK REFERENCE CARD

### For Simulator Implementation

**Level Flight Speed by AoA (9000 lb, sea level)**:
| Wing AoA | Speed |
|----------|-------|
| 0° | 361 mph |
| 2° | 263 mph |
| 5° | 166 mph |
| 10° | 117 mph |
| 15° | 105 mph (near stall) |

**Key Performance Numbers (Military Power, 9000 lb)**:
- Max Speed (SL): 355 mph
- Max Speed (25,000 ft): 435 mph
- Stall Speed (clean): 100 mph
- ROC (SL): 3,000 ft/min
- Best Climb Speed: 165 mph IAS

**Equations to Implement**:
```
C_L = 5.56 × (α - (-0.021))        [rad]
C_D = 0.0163 + 0.072 × C_L²
L = 0.5 × ρ × V² × 233 × C_L       [lb]
D = 0.5 × ρ × V² × 233 × C_D       [lb]
T = η_p × (P × 550) / V            [lb]
```

---

## 10. SOURCES AND REFERENCES

1. **NAA Report NA-46-130**: Performance Calculations for P-51D Airplane
   - Source: wwiiaircraftperformance.org
   - Primary source for validated performance data

2. **IL-2 Great Battles**: P-51D-15 Specifications
   - Source: aergistal.github.io/il2/planes/p51d15.html
   - Extensively validated against historical records

3. **Virginia Tech Aerospace Archive**: P-51D Mustang Student Report
   - Source: archive.aoe.vt.edu/mason/Mason_f/P51DMustang.pdf
   - Aerodynamic data and wing geometry

4. **Mid America Flight Museum**: P-51 Mustang Specifications
   - Source: midamericaflightmuseum.com
   - CD0, drag area, L/D data

5. **WW2Aircraft.net Forums**: Technical discussions
   - CL_max, wing incidence, airfoil data
   - Model builder and pilot experiences

6. **Packard V-1650 Merlin Wikipedia**: Engine specifications
   - Power curves, supercharger data

7. **NACA Airfoil Theory**: Thin airfoil theory for α₀
   - Zero-lift angle calculations

---

## Document Version

- **Version**: 1.0
- **Date**: January 2026
- **Purpose**: RL Environment Validation Reference
- **Author**: Claude (Anthropic) + Research

---

*Note: This document is intended for simulation purposes. Actual aircraft performance varies with exact configuration, atmospheric conditions, and pilot technique. Tolerances are provided to account for these variations and simplifications inherent in the simulation model.*
