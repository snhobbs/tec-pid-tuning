from typing import Tuple
from dataclasses import dataclass, field
import numbers
from collections import deque
from scipy import signal
import numpy as np
from scipy.optimize import curve_fit


def delay_integrator_model(t, K, theta, y0):
    """Step response of a delay + integrator system."""
    y = np.piecewise(t,
                     [t < theta, t >= theta],
                     [lambda t: y0,
                      lambda t: y0 + K * (t - theta)])
    return y


def fit_delay_integrator(t_data, y_data, diff_thresh=0.1):
    """
    Fit a delay + integrator model to data.

    Parameters:
    - t_data: time array (1D)
    - y_data: output values (1D)
    - plot: if True, plots the result

    Returns:
    - popt: Fitted parameters [K, theta, y0]
    - pcov: Covariance matrix of the parameters
    """
    # Initial guesses: slope, delay, baseline
    K0 = (y_data[-1] - y_data[0]) / (t_data[-1] - t_data[0])
    theta0 = t_data[np.argmax(np.diff(y_data) > diff_thresh)]
    y0 = y_data[0]
    p0 = [K0, theta0, y0]

    # Fit the model
    popt, pcov = curve_fit(delay_integrator_model, t_data, y_data, p0=p0)
    return popt, pcov


def trapezoid_integral(p1, p2, h=1):
    """Compute area using the trapezoidal rule."""
    area = ((p1+p2)*h)/2
    return area


def simc_tuning(delay: float, slope: float, *, tc: float = 3):
    """
    SIMC tuning for an integrating process with delay:
    G(s) = K / s * e^(-theta * s)

    Parameters:
        delay (float): Process delay (theta)
        slope (float): Ramp slope (R)
        tc (float): Multiplier for desired closed-loop time constant tc

    Returns:
        kp (float): Proportional gain
        ti (float): Integral time
    """
    ti = 4 * (delay + tc*delay)
    kp = 1 / (slope * (delay + tc*delay))
    return kp, ti


def calculate_pi_coefficients(
        delay, gain, update_frequency=1, method="simc", **kwargs):
    """
    Calculate discrete-time PI gains from continuous-time tuning.

    Parameters:
        delay (float): Process delay
        gain (float): Integrator slope (K')
        update_frequency (float): Controller update frequency (Hz)
        method (str): Tuning method (only 'simc' supported)

    Returns:
        kp (float): Proportional gain
        ki (float): Discrete integrator gain
    """
    methods = ("simc",)
    if method == "simc":
        kp, ti = simc_tuning(delay=delay, slope=gain, **kwargs)
    else:
        raise ValueError(f"Method {method} not one of {methods}")

    ki = (kp / ti) / update_frequency
    return kp, ki


@dataclass
class Bang_Bang_Filter:
    """
    Discrete-time Bang bang controller with optional current limit.

    Attributes:
        hysteresis (float): Accepted temperature error
        sample_time (float): Time between updates
        set_point (float): Desired value
        ilim (float or tuple): Symmetric or asymmetric output limit
    """
    hysteresis: float
    sample_time: float = 1
    set_point: float = 0
    ilim: Tuple[float, float] = (0, 0)

    sample: float = 0  # Sample index
    control: float = 0

    def reset(self):
        """Reset controller state."""
        self.sample = 0

    def get_state(self):
        return {"sample": self.sample}

    def update(self, reading):
        """
        """
        self.sample += 1
        if (reading + self.hysteresis/2) <= self.set_point:
            self.control = max(self.ilim)
        elif reading >= (self.set_point + self.hysteresis/2):
            self.control = min(self.ilim)
        return self.control


@dataclass
class IIR_PI_Filter:
    """
    Discrete-time PI controller with optional current limit and anti-windup.

    Attributes:
        kp (float): Proportional gain
        ki (float): Integrator gain (per sample)
        sample_time (float): Time between updates
        set_point (float): Desired value
        ilim (float or tuple): Symmetric or asymmetric output limit
        antiwindup (bool): Enable conditional integration
    """
    kp: float
    ki: float
    sample_time: float = 1
    set_point: float = 0
    ilim: float | None | tuple[float, float] = ()
    antiwindup: bool = False

    sample: float = 0  # Sample index
    integral: float = 0
    primed: bool = False
    last_error: float = 0
    error: float = 0

    def reset(self):
        """Reset controller state."""
        self.integral = 0
        self.primed = False
        self.error = 0
        self.last_error = 0
        self.sample = 0

    def get_state(self):
        return {"sample": self.sample,
                "integral": self.integral,
                "primed": self.primed,
                "error": self.error}

    def update_(self, reading):
        """
        Core PI update logic (excluding output limiting).
        Implements conditional integrator priming.
        """
        error = reading - self.set_point
        self.last_error = self.error
        self.error = error

        kprime_length = 2  # Samples to setup filter

        control = 0
        # Give time to prime filter. Runs only the proportional component at first.
        if (not self.primed):
            if (self.sample >= kprime_length):
                self.primed = True

            # Run just the proportional controller when not primed
            control = -(error*self.kp)

        else:
            self.integral += trapezoid_integral(self.error,
                                                self.last_error, self.sample_time)
            control = -(error*self.kp + self.integral*self.ki)
        self.sample += 1
        return control

    def update(self, reading):
        """
        Update controller with optional output limiting and anti-windup.
        """
        last_integral = self.integral
        control = self.update_(reading)

        if self.ilim:
            n_ilim, p_ilim = (-self.ilim, self.ilim) if isinstance(self.ilim,
                                                                   numbers.Number) else (min(self.ilim), max(self.ilim))
            limit_exceeded = False
            if (control > p_ilim):
                control = p_ilim
                limit_exceeded = True

            elif (control < n_ilim):
                control = n_ilim
                limit_exceeded = True

            # if in current limit use the previous integrator value
            if self.antiwindup and limit_exceeded:
                self.integral = last_integral

        return control


@dataclass
class DelayIntegratorPlantModel:
    delay: float
    gain: float
    ambient: float
    rate: float = 1
    heat_leak: float = 0
    temperature: float = 0
    history: deque = field(default_factory=deque)

    def reset(self):
        self.history.clear()
        self.temperature = self.ambient

    def update(self, control):
        time_step = 1. / self.rate
        self.history.appendleft(control)
        while len(self.history) > self.delay * self.rate:
            setting = self.history[-1]  # front
            self.history.pop()
            temperature_forcing = linear_calculate_heat_leak(
                self.temperature, self.ambient, self.heat_leak
            )
            self.temperature += (setting + temperature_forcing) * \
                time_step * self.gain
        return self.temperature


def quadratic_calculate_heat_leak(temperature, ambient, factor):
    """
    Quadratic convection heat leak model.
    Acts like a control value in the direction of ambient temperature.
    """
    delta = ambient - temperature
    return factor * delta * abs(delta)  # preserves sign


def linear_calculate_heat_leak(temperature, ambient, factor):
    """
    Linear conduction heat leak model.
    Acts like a control value in the direction of ambient temperature.
    """
    return (ambient - temperature) * factor  # temp > ambient -> negative leak


def simulate_controlled_plant(plant, filt, npoints, forcing=None, reset=True):
    """
    Simulate a closed-loop control of a plant with optional external forcing.

    Parameters:
    - plant: the plant object with `.reset()` and `.update(control + forcing)` methods
    - filt: the PI or PID controller object with `.reset()` and `.update(temperature)` methods
    - npoints: number of timesteps to simulate
    - forcing: a callable f(i) or array of same length as npoints, or None

    Returns:
    - dict with time series: temperature, integral, control
    """
    if reset:
        plant.reset()
    filt.reset()

    temp = []
    control = []
    state = {key: [] for key in filt.get_state()}

    for i in range(npoints):
        forcing_val = 0
        if callable(forcing):
            forcing_val = forcing(i)
        elif isinstance(forcing, (list, np.ndarray)):
            forcing_val = forcing[i]

        control_setting = filt.update(plant.temperature)
        control.append(control_setting)
        for key, value in filt.get_state().items():
            state[key].append(value)

        # Apply both control and external forcing to the plant
        temp.append(plant.update(control_setting + forcing_val))

    state.update({
        "temp": np.array(temp),
        "control": np.array(control)})

    return state


def sim_delay_integrator(K=1.0, delay=2.0, duration=5.0, dt=0.01, input_value=1.0):
    """
    Simulate the response of a delay-integrator system to a step input.

    The system is modeled as:
        y(t) = ∫ K * u(t - delay) dt

    Parameters:
        K (float): Integrator gain (slope).
        delay (float): Time delay before integration begins.
        duration (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        input_value (float): Constant input after delay (step input).
        y0 (float): Initial output value.

    Returns:
        t (ndarray): Time array.
        y (ndarray): Output (integrated) signal.
    """
    t = np.arange(0, duration, dt)
    y = np.zeros_like(t)

    # Integrate after delay
    delay_idx = int(delay / dt)
    for i in range(delay_idx, len(t)):
        y[i] = y[i-1] + dt * input_value * K
    y = np.array(y) + 2
    return t, y
