# streamlit run phys_modeller.py
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import re

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GenAI Physics Modeler", page_icon="⚛️")

# --- Constants ---
PRICING = {
    # OpenAI Models (https://openai.com/api/pricing/)
    "gpt-4o": {"input": 2.50, "output": 10.00, "provider": "OpenAI"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "OpenAI"},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "provider": "OpenAI"},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "provider": "OpenAI"},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "provider": "OpenAI"},
    "o1": {"input": 15.00, "output": 60.00, "provider": "OpenAI"},
    "o1-mini": {"input": 1.10, "output": 4.40, "provider": "OpenAI"},
    "o1-pro": {"input": 150.00, "output": 600.00, "provider": "OpenAI"},
    "o3": {"input": 10.00, "output": 40.00, "provider": "OpenAI"},
    "o3-mini": {"input": 1.10, "output": 4.40, "provider": "OpenAI"},
    "o4-mini": {"input": 1.10, "output": 4.40, "provider": "OpenAI"},
    # xAI Grok Models (https://docs.x.ai/docs/models)
    "grok-4-1-fast-reasoning": {"input": 3.00, "output": 12.00, "provider": "xAI"},
    "grok-4-0709": {"input": 3.00, "output": 15.00, "provider": "xAI"},
    "grok-3": {"input": 3.00, "output": 15.00, "provider": "xAI"},
    "grok-3-fast": {"input": 5.00, "output": 25.00, "provider": "xAI"},
    "grok-3-mini": {"input": 0.30, "output": 0.50, "provider": "xAI"},
    "grok-3-mini-fast": {"input": 0.10, "output": 0.40, "provider": "xAI"},
    "grok-2-1212": {"input": 2.00, "output": 10.00, "provider": "xAI"},
    "grok-2-vision-1212": {"input": 2.00, "output": 10.00, "provider": "xAI"},
    # DeepSeek Models (https://api-docs.deepseek.com/quick_start/pricing/)
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28, "provider": "DeepSeek"},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87, "provider": "DeepSeek"},
}

# Model display names with descriptions
OPENAI_MODELS = {
    "GPT-4.1 (Latest)": "gpt-4.1",
    "GPT-4.1 Mini (Balanced)": "gpt-4.1-mini",
    "GPT-4.1 Nano (Fast & Cheap)": "gpt-4.1-nano",
    "GPT-4o (Multimodal)": "gpt-4o",
    "GPT-4o Mini (Budget)": "gpt-4o-mini",
    "o1 (Reasoning)": "o1",
    "o1 Mini (Reasoning Lite)": "o1-mini",
    "o1 Pro (Reasoning Pro)": "o1-pro",
    "o3 (Advanced Reasoning)": "o3",
    "o3 Mini (Reasoning Budget)": "o3-mini",
    "o4 Mini (Latest Reasoning)": "o4-mini",
}

XAI_MODELS = {
    "Grok 4.1 Fast Reasoning [Best Value]": "grok-4-1-fast-reasoning",
    "Grok 4 (Standard)": "grok-4-0709",
    "Grok 3 (Standard)": "grok-3",
    "Grok 3 Fast": "grok-3-fast",
    "Grok 3 Mini (Budget)": "grok-3-mini",
    "Grok 3 Mini Fast (Cheapest)": "grok-3-mini-fast",
    "Grok 2 (Legacy)": "grok-2-1212",
    "Grok 2 Vision (Legacy)": "grok-2-vision-1212",
}

DEEPSEEK_MODELS = {
    "DeepSeek V4 Flash (Fast & Cheap)": "deepseek-v4-flash",
    "DeepSeek V4 Pro (Higher Capability)": "deepseek-v4-pro",
}

SCENARIOS = {
    "Custom": "",
    "Rotating Sphere with Gas": "A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container.",
    "Rarefied Gas Spin-Up": "Simulate the transfer of conserved momentum from a rotating solid disk to gas molecules immediately adjacent to it in a vacuum. The gas should gradually spin up due to wall collisions.",
    "Solar System w/ Comet": "A solar system simulation with a static yellow sun, 3 orbiting planets at different distances/speeds, and a comet passing through on a hyperbolic trajectory.",
    "Damped Pendulum": "A 3D visualization of a simple pendulum with damping. Show the pendulum bob swinging in 3D space and trace its path color-coded by velocity.",
    "Lorenz Attractor": "Simulate the Lorenz attractor (chaotic system). Visualize the trajectory of a point over time in 3D space, leaving a trail.",
}

# Bromine release model constants
BROMINE_MOLAR_MASS = 0.1598  # kg/mol for Br2
BROMINE_KINETIC_DIAMETER = 4.3e-10  # meters
BROMINE_VAPOR_PRESSURE_TORR = 230.0
ROOM_TEMPERATURE_K = 298.0
G_STANDARD = 9.80665
GAS_CONSTANT = 8.314462618

# Dark theme color constants
DARK_BG = '#0e1117'
DARK_SECONDARY = '#1a1d24'
GRID_COLOR = '#333940'
TEXT_COLOR = '#fafafa'

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = SCENARIOS["Rotating Sphere with Gas"]
if "auto_play" not in st.session_state:
    st.session_state.auto_play = True
if "loop_animation" not in st.session_state:
    st.session_state.loop_animation = True
if "show_slider" not in st.session_state:
    st.session_state.show_slider = True
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "GenAI Physics Generator"
if "bromine_auto_play" not in st.session_state:
    st.session_state.bromine_auto_play = False
if "bromine_loop_animation" not in st.session_state:
    st.session_state.bromine_loop_animation = True

# --- Helpers ---
def get_secret(key_name):
    for key in [key_name, key_name.upper(), key_name.lower()]:
        if key in st.secrets:
            return st.secrets[key]
    return None

def check_password():
    stored_password = get_secret("app_password")
    if not stored_password:
        st.error("❌ Configuration Error: 'app_password' not found in Secrets.")
        st.stop()

    def password_entered():
        if st.session_state["password"] == stored_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter App Password", type="password", on_change=password_entered, key="password")
        return False
    if not st.session_state["password_correct"]:
        st.text_input("Enter App Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

def get_pricing_info(model_name):
    """Get pricing info for a model with formatted display."""
    if model_name in PRICING:
        info = PRICING[model_name]
        return {
            "input": info["input"],
            "output": info["output"],
            "provider": info["provider"],
            "model": model_name
        }
    return {
        "input": 5.0,
        "output": 15.0,
        "provider": "Unknown",
        "model": model_name
    }

def display_pricing_indicator(model_name):
    """Display a visual pricing indicator for the selected model."""
    info = get_pricing_info(model_name)
    provider = info["provider"]
    
    if provider == "OpenAI":
        provider_icon = "🟢"
        provider_color = "green"
    elif provider == "xAI":
        provider_icon = "🔵"
        provider_color = "blue"
    elif provider == "DeepSeek":
        provider_icon = "🟣"
        provider_color = "purple"
    else:
        provider_icon = "⚪"
        provider_color = "gray"

    bg_start = "#e8f5e9"
    bg_end = "#c8e6c9"
    border_color = "#4caf50"
    if provider == "xAI":
        bg_start = "#e3f2fd"
        bg_end = "#bbdefb"
        border_color = "#2196f3"
    elif provider == "DeepSeek":
        bg_start = "#f3e8ff"
        bg_end = "#ddd6fe"
        border_color = "#7c3aed"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {bg_start} 0%, {bg_end} 100%);
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        border-left: 4px solid {border_color};
    ">
        <div style="font-weight: bold; margin-bottom: 8px;">
            {provider_icon} {provider} Pricing
        </div>
        <div style="font-size: 0.85em;">
            <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                <span>📥 Input:</span>
                <span><strong>${info['input']:.2f}</strong> / 1M tokens</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                <span>📤 Output:</span>
                <span><strong>${info['output']:.2f}</strong> / 1M tokens</span>
            </div>
        </div>
        <div style="font-size: 0.75em; color: #666; margin-top: 8px;">
            Model: <code>{model_name}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Stronger Sandbox ---
def execute_safe_code(code_str, global_vars):
    blocked_modules = ['os', 'sys', 'subprocess', 'shutil', 'requests', 'socket',
                      'pickle', 'ctypes', 'multiprocessing', 'urllib', 'webbrowser']
    
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split('.')[0] in blocked_modules or name in blocked_modules:
            raise ImportError(f"Importing '{name}' is forbidden for security reasons.")
        return __import__(name, globals, locals, fromlist, level)

    if isinstance(__builtins__, dict):
        safe_builtins = __builtins__.copy()
    else:
        safe_builtins = __builtins__.__dict__.copy()
    
    for dangerous in ['eval', 'exec', 'open', 'compile']:
        safe_builtins.pop(dangerous, None)
    
    safe_builtins['__import__'] = restricted_import
    global_vars['__builtins__'] = safe_builtins
    
    try:
        exec(code_str, global_vars)
        return True, None
    except Exception as e:
        return False, str(e)

# --- Improved Code Cleaning ---
def clean_code(raw):
    lines = raw.splitlines()
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()

def rpm_to_omega(rpm):
    return 2 * np.pi * rpm / 60

def bromine_mean_speed(temperature_k):
    return np.sqrt(8 * GAS_CONSTANT * temperature_k / (np.pi * BROMINE_MOLAR_MASS))

def centrifugal_acceleration_g(rpm, radius_m):
    omega = rpm_to_omega(rpm)
    acceleration = omega ** 2 * radius_m
    return acceleration, acceleration / G_STANDARD

def co_rotating_density_ratio(rpm, inner_radius_m, outer_radius_m, temperature_k):
    omega = rpm_to_omega(rpm)
    delta_r2 = outer_radius_m ** 2 - inner_radius_m ** 2
    exponent = BROMINE_MOLAR_MASS * omega ** 2 * delta_r2 / (2 * GAS_CONSTANT * temperature_k)
    return exponent, np.exp(exponent)

def estimate_final_pressure_torr(vial_volume_ml, chamber_volume_ml, vapor_pressure_torr=BROMINE_VAPOR_PRESSURE_TORR):
    if chamber_volume_ml <= 0:
        return 0
    return vapor_pressure_torr * vial_volume_ml / chamber_volume_ml

def estimate_mean_free_path_m(pressure_torr, temperature_k):
    pressure_pa = pressure_torr * 133.322
    if pressure_pa <= 0:
        return np.inf
    numerator = GAS_CONSTANT * temperature_k / 6.02214076e23
    denominator = np.sqrt(2) * np.pi * BROMINE_KINETIC_DIAMETER ** 2 * pressure_pa
    return numerator / denominator

def reflect_particles_in_circle(positions, velocities, radius):
    radii = np.linalg.norm(positions, axis=1)
    outside = radii > radius
    if not np.any(outside):
        return positions, velocities

    normals = positions[outside] / radii[outside, None]
    positions[outside] = normals * (2 * radius - radii[outside])[:, None]
    normal_speed = np.sum(velocities[outside] * normals, axis=1)
    velocities[outside] = velocities[outside] - 2 * normal_speed[:, None] * normals
    return positions, velocities

def reflect_particles_in_sphere(positions, velocities, radius):
    radii = np.linalg.norm(positions, axis=1)
    outside = radii > radius
    if not np.any(outside):
        return positions, velocities

    normals = positions[outside] / radii[outside, None]
    positions[outside] = normals * (2 * radius - radii[outside])[:, None]
    normal_speed = np.sum(velocities[outside] * normals, axis=1)
    velocities[outside] = velocities[outside] - 2 * normal_speed[:, None] * normals
    return positions, velocities

def make_sphere_wireframe(radius, line_color="#6b7280"):
    theta = np.linspace(0, 2 * np.pi, 90)
    phi_values = np.linspace(0.2 * np.pi, 0.8 * np.pi, 5)
    meridian_values = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    x_lines = []
    y_lines = []
    z_lines = []

    for phi in phi_values:
        x_lines.extend((radius * np.sin(phi) * np.cos(theta)).tolist() + [None])
        y_lines.extend((radius * np.sin(phi) * np.sin(theta)).tolist() + [None])
        z_lines.extend((np.full_like(theta, radius * np.cos(phi))).tolist() + [None])

    phi = np.linspace(0, np.pi, 90)
    for meridian in meridian_values:
        x_lines.extend((radius * np.sin(phi) * np.cos(meridian)).tolist() + [None])
        y_lines.extend((radius * np.sin(phi) * np.sin(meridian)).tolist() + [None])
        z_lines.extend((radius * np.cos(phi)).tolist() + [None])

    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color=line_color, width=2),
        opacity=0.45,
        name="Static chamber",
    )

def simulate_bromine_release_2d(
    chamber_radius_m,
    rotor_radius_m,
    rpm,
    temperature_k,
    particle_count,
    frames_count,
    wall_accommodation,
    seed,
):
    rng = np.random.default_rng(seed)
    omega = rpm_to_omega(rpm)
    mean_speed = bromine_mean_speed(temperature_k)
    thermal_std = np.sqrt(GAS_CONSTANT * temperature_k / BROMINE_MOLAR_MASS)
    tangential_speed = omega * rotor_radius_m
    t_cross = chamber_radius_m / mean_speed
    t_end = max(0.004, 4 * t_cross)
    times = np.linspace(0, t_end, frames_count)
    dt = times[1] - times[0] if frames_count > 1 else t_end

    release_spread = max(0.002, 0.04 * chamber_radius_m)
    positions = rng.normal(0, release_spread, size=(particle_count, 2))
    positions[:, 0] += rotor_radius_m
    velocities = rng.normal(0, thermal_std, size=(particle_count, 2))
    velocities[:, 1] += tangential_speed

    chamber_theta = np.linspace(0, 2 * np.pi, 240)
    chamber_x = chamber_radius_m * np.cos(chamber_theta)
    chamber_y = chamber_radius_m * np.sin(chamber_theta)
    rotor_x = rotor_radius_m * np.cos(chamber_theta)
    rotor_y = rotor_radius_m * np.sin(chamber_theta)

    frames = []
    saved_positions = []
    angular_memory = []
    for frame_i, t in enumerate(times):
        if frame_i > 0:
            positions = positions + velocities * dt
            positions, velocities = reflect_particles_in_circle(positions, velocities, chamber_radius_m)

            hit_wall = np.linalg.norm(positions, axis=1) > 0.98 * chamber_radius_m
            if np.any(hit_wall) and wall_accommodation > 0:
                velocities[hit_wall] *= (1 - wall_accommodation)
                randomized = rng.normal(0, thermal_std, size=(hit_wall.sum(), 2))
                velocities[hit_wall] += wall_accommodation * randomized

        saved_positions.append(positions.copy())
        radii = np.linalg.norm(positions, axis=1)
        tangential = np.divide(
            positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0],
            np.maximum(radii, 1e-9),
        )
        angular_memory.append(float(np.mean(tangential)))

        plume_angle = omega * t
        vial_x = rotor_radius_m * np.cos(plume_angle)
        vial_y = rotor_radius_m * np.sin(plume_angle)
        tangent_x = -np.sin(plume_angle)
        tangent_y = np.cos(plume_angle)

        frames.append(go.Frame(
            name=f"frame_{frame_i}",
            data=[
                go.Scatter(x=chamber_x, y=chamber_y, mode="lines", line=dict(color="#6b7280", width=2), name="Static chamber"),
                go.Scatter(x=rotor_x, y=rotor_y, mode="lines", line=dict(color="#374151", width=1, dash="dot"), name="Rotor path"),
                go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode="markers",
                    marker=dict(size=5, color=radii, colorscale="Oranges", cmin=0, cmax=chamber_radius_m, opacity=0.78),
                    name="Bromine tracers",
                ),
                go.Scatter(x=[0, vial_x], y=[0, vial_y], mode="lines", line=dict(color="#60a5fa", width=3), name="Rotor arm"),
                go.Scatter(x=[vial_x], y=[vial_y], mode="markers", marker=dict(size=12, color="#f97316"), name="Opened vial"),
                go.Scatter(
                    x=[vial_x, vial_x + tangent_x * 0.25 * chamber_radius_m],
                    y=[vial_y, vial_y + tangent_y * 0.25 * chamber_radius_m],
                    mode="lines",
                    line=dict(color="#facc15", width=4),
                    name="Tangential release",
                ),
            ],
            layout=go.Layout(title_text=f"Bromine release, t = {t * 1000:.2f} ms"),
        ))

    initial_positions = saved_positions[0]
    initial_radii = np.linalg.norm(initial_positions, axis=1)
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="Bromine Vapor Release into a Static Chamber",
            xaxis=dict(range=[-chamber_radius_m * 1.08, chamber_radius_m * 1.08], title="x (m)", scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-chamber_radius_m * 1.08, chamber_radius_m * 1.08], title="y (m)"),
            showlegend=True,
            height=680,
            margin=dict(l=20, r=20, t=60, b=40),
        )
    )

    final_radii = np.linalg.norm(saved_positions[-1], axis=1)
    bins = np.linspace(0, chamber_radius_m, 12)
    initial_hist, _ = np.histogram(initial_radii, bins=bins)
    final_hist, _ = np.histogram(final_radii, bins=bins)
    annular_areas = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    initial_density = initial_hist / annular_areas
    final_density = final_hist / annular_areas
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    density_fig = go.Figure()
    density_fig.add_trace(go.Scatter(x=bin_centers, y=initial_density, mode="lines+markers", name="At release"))
    density_fig.add_trace(go.Scatter(x=bin_centers, y=final_density, mode="lines+markers", name="After expansion"))
    density_fig.update_layout(
        title="Radial Tracer Density",
        xaxis_title="Radius (m)",
        yaxis_title="Tracer density per m²",
        height=340,
        margin=dict(l=20, r=20, t=50, b=40),
    )

    acceleration, g_equiv = centrifugal_acceleration_g(rpm, rotor_radius_m)
    exponent, density_ratio = co_rotating_density_ratio(
        rpm,
        max(0.001, rotor_radius_m / 2),
        rotor_radius_m,
        temperature_k,
    )
    final_pressure = estimate_final_pressure_torr(10, 1000)
    mean_free_path = estimate_mean_free_path_m(final_pressure, temperature_k)

    metrics = {
        "omega": omega,
        "mean_speed": mean_speed,
        "thermal_std": thermal_std,
        "t_cross_ms": t_cross * 1000,
        "t_end_ms": t_end * 1000,
        "tangential_speed": tangential_speed,
        "acceleration": acceleration,
        "g_equiv": g_equiv,
        "density_exponent": exponent,
        "density_ratio": density_ratio,
        "final_pressure_torr": final_pressure,
        "mean_free_path_m": mean_free_path,
        "angular_memory_start": angular_memory[0],
        "angular_memory_end": angular_memory[-1],
    }
    return fig, density_fig, metrics

def simulate_bromine_release_3d(
    chamber_radius_m,
    rotor_radius_m,
    rpm,
    temperature_k,
    particle_count,
    frames_count,
    wall_accommodation,
    seed,
):
    rng = np.random.default_rng(seed)
    omega = rpm_to_omega(rpm)
    mean_speed = bromine_mean_speed(temperature_k)
    thermal_std = np.sqrt(GAS_CONSTANT * temperature_k / BROMINE_MOLAR_MASS)
    tangential_speed = omega * rotor_radius_m
    t_cross = chamber_radius_m / mean_speed
    t_end = max(0.004, 4 * t_cross)
    times = np.linspace(0, t_end, frames_count)
    dt = times[1] - times[0] if frames_count > 1 else t_end

    release_spread = max(0.002, 0.04 * chamber_radius_m)
    positions = rng.normal(0, release_spread, size=(particle_count, 3))
    positions[:, 0] += rotor_radius_m
    velocities = rng.normal(0, thermal_std, size=(particle_count, 3))
    velocities[:, 1] += tangential_speed

    rotor_theta = np.linspace(0, 2 * np.pi, 160)
    rotor_x = rotor_radius_m * np.cos(rotor_theta)
    rotor_y = rotor_radius_m * np.sin(rotor_theta)
    rotor_z = np.zeros_like(rotor_theta)
    chamber_wire = make_sphere_wireframe(chamber_radius_m)

    frames = []
    saved_positions = []
    angular_memory = []
    for frame_i, t in enumerate(times):
        if frame_i > 0:
            positions = positions + velocities * dt
            positions, velocities = reflect_particles_in_sphere(positions, velocities, chamber_radius_m)

            hit_wall = np.linalg.norm(positions, axis=1) > 0.98 * chamber_radius_m
            if np.any(hit_wall) and wall_accommodation > 0:
                velocities[hit_wall] *= (1 - wall_accommodation)
                randomized = rng.normal(0, thermal_std, size=(hit_wall.sum(), 3))
                velocities[hit_wall] += wall_accommodation * randomized

        saved_positions.append(positions.copy())
        radii = np.linalg.norm(positions[:, :2], axis=1)
        tangential = np.divide(
            positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0],
            np.maximum(radii, 1e-9),
        )
        angular_memory.append(float(np.mean(tangential)))

        plume_angle = omega * t
        vial_x = rotor_radius_m * np.cos(plume_angle)
        vial_y = rotor_radius_m * np.sin(plume_angle)
        vial_z = 0
        tangent_x = -np.sin(plume_angle)
        tangent_y = np.cos(plume_angle)
        tangent_len = 0.28 * chamber_radius_m
        particle_radii = np.linalg.norm(positions, axis=1)

        frames.append(go.Frame(
            name=f"frame_{frame_i}",
            data=[
                chamber_wire,
                go.Scatter3d(
                    x=rotor_x,
                    y=rotor_y,
                    z=rotor_z,
                    mode="lines",
                    line=dict(color="#374151", width=3, dash="dot"),
                    name="Rotor path",
                ),
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=particle_radii, colorscale="Oranges", cmin=0, cmax=chamber_radius_m, opacity=0.72),
                    name="Bromine tracers",
                ),
                go.Scatter3d(
                    x=[0, vial_x],
                    y=[0, vial_y],
                    z=[0, vial_z],
                    mode="lines",
                    line=dict(color="#60a5fa", width=6),
                    name="Rotor arm",
                ),
                go.Scatter3d(
                    x=[vial_x],
                    y=[vial_y],
                    z=[vial_z],
                    mode="markers",
                    marker=dict(size=7, color="#f97316"),
                    name="Opened vial",
                ),
                go.Scatter3d(
                    x=[vial_x, vial_x + tangent_x * tangent_len],
                    y=[vial_y, vial_y + tangent_y * tangent_len],
                    z=[vial_z, vial_z],
                    mode="lines",
                    line=dict(color="#facc15", width=7),
                    name="Tangential release",
                ),
            ],
            layout=go.Layout(title_text=f"3D bromine release, t = {t * 1000:.2f} ms"),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="3D Bromine Vapor Release into a Static Chamber",
            scene=dict(
                xaxis=dict(range=[-chamber_radius_m * 1.1, chamber_radius_m * 1.1], title="x (m)"),
                yaxis=dict(range=[-chamber_radius_m * 1.1, chamber_radius_m * 1.1], title="y (m)"),
                zaxis=dict(range=[-chamber_radius_m * 1.1, chamber_radius_m * 1.1], title="z (m)"),
                aspectmode="cube",
            ),
            showlegend=True,
            height=720,
            margin=dict(l=0, r=0, t=60, b=30),
        )
    )

    initial_radii = np.linalg.norm(saved_positions[0], axis=1)
    final_radii = np.linalg.norm(saved_positions[-1], axis=1)
    bins = np.linspace(0, chamber_radius_m, 12)
    initial_hist, _ = np.histogram(initial_radii, bins=bins)
    final_hist, _ = np.histogram(final_radii, bins=bins)
    shell_volumes = (4 / 3) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    initial_density = initial_hist / shell_volumes
    final_density = final_hist / shell_volumes
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    density_fig = go.Figure()
    density_fig.add_trace(go.Scatter(x=bin_centers, y=initial_density, mode="lines+markers", name="At release"))
    density_fig.add_trace(go.Scatter(x=bin_centers, y=final_density, mode="lines+markers", name="After expansion"))
    density_fig.update_layout(
        title="Radial Tracer Density",
        xaxis_title="Radius (m)",
        yaxis_title="Tracer density per m³",
        height=340,
        margin=dict(l=20, r=20, t=50, b=40),
    )

    acceleration, g_equiv = centrifugal_acceleration_g(rpm, rotor_radius_m)
    exponent, density_ratio = co_rotating_density_ratio(
        rpm,
        max(0.001, rotor_radius_m / 2),
        rotor_radius_m,
        temperature_k,
    )
    final_pressure = estimate_final_pressure_torr(10, 1000)
    mean_free_path = estimate_mean_free_path_m(final_pressure, temperature_k)

    metrics = {
        "omega": omega,
        "mean_speed": mean_speed,
        "thermal_std": thermal_std,
        "t_cross_ms": t_cross * 1000,
        "t_end_ms": t_end * 1000,
        "tangential_speed": tangential_speed,
        "acceleration": acceleration,
        "g_equiv": g_equiv,
        "density_exponent": exponent,
        "density_ratio": density_ratio,
        "final_pressure_torr": final_pressure,
        "mean_free_path_m": mean_free_path,
        "angular_memory_start": angular_memory[0],
        "angular_memory_end": angular_memory[-1],
    }
    return fig, density_fig, metrics

# --- LLM Logic ---
def get_system_prompt():
    return """
You are a perfect Python code generator for 3D physics visualizations using only `numpy` and `plotly.graph_objects`.

STRICT RULES (never break them):
- Output ONLY raw Python code. No markdown, no ``` fences, no explanations.
- Start with the imports: import numpy as np and import plotly.graph_objects as go
- Pre-calculate exactly 60-90 frames and attach them to fig.frames
- Each frame MUST have a unique 'name' attribute (e.g., name=f"frame_{i}")
- Define the final figure as `fig = go.Figure(...)`
- NEVER define updatemenus, sliders, or play buttons — the app adds them automatically
- ALWAYS set fixed axis ranges (e.g. scene.xaxis.range=[-10, 10]) — never use auto-scaling
- Use heavy vectorization with np functions — never write out long lists manually
- Keep code as short and efficient as possible
"""

def call_llm(messages, key, url, model):
    client = openai.OpenAI(api_key=key, base_url=url if url else None)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=4096,
    )
    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    p_tokens = usage.prompt_tokens if usage else None
    c_tokens = usage.completion_tokens if usage else None
    return content, p_tokens, c_tokens

def generate_simulation(prompt, key, url, model):
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt}
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    dummy_globals = {"np": np, "go": go}

    for attempt in range(1, 4):
        raw_code, p_tok, c_tok = call_llm(messages, key, url, model)
        
        if p_tok is not None:
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok

        code = clean_code(raw_code)
        dummy_globals.clear()
        dummy_globals.update({"np": np, "go": go})
        
        success, error = execute_safe_code(code, dummy_globals)
        fig_ok = 'fig' in dummy_globals and isinstance(dummy_globals['fig'], go.Figure)

        if success and fig_ok:
            rates = PRICING.get(model, {"input": 5.0, "output": 15.0})
            if total_prompt_tokens > 0:
                cost = (total_prompt_tokens / 1_000_000) * rates["input"] + \
                       (total_completion_tokens / 1_000_000) * rates["output"]
            else:
                est_input = len(get_system_prompt() + prompt) / 4
                est_output = len(raw_code) / 4
                cost = (est_input / 1e6) * rates["input"] + (est_output / 1e6) * rates["output"]
            
            return code, cost, total_prompt_tokens, total_completion_tokens

        issues = []
        if not success:
            issues.append(f"Execution error: {error}")
        if 'fig' not in dummy_globals:
            issues.append("Missing `fig` variable")
        elif not isinstance(dummy_globals['fig'], go.Figure):
            issues.append("`fig` is not a plotly Figure")
        issues.append("Follow every rule in the system prompt exactly")
        
        feedback = "Please fix these issues and output ONLY the corrected raw Python code:\n• " + "\n• ".join(issues)
        
        messages.append({"role": "assistant", "content": raw_code})
        messages.append({"role": "user", "content": feedback})

    raise ValueError("Failed to generate valid code after 3 attempts")

def update_prompt():
    sel = st.session_state.scenario_selector
    if sel != "Custom":
        st.session_state.prompt = SCENARIOS[sel]

def apply_dark_theme(fig):
    """Apply dark theme to the Plotly figure."""
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        scene=dict(
            bgcolor=DARK_BG,
            xaxis=dict(
                backgroundcolor=DARK_SECONDARY,
                gridcolor=GRID_COLOR,
                color=TEXT_COLOR,
                showbackground=True,
                zerolinecolor=GRID_COLOR
            ),
            yaxis=dict(
                backgroundcolor=DARK_SECONDARY,
                gridcolor=GRID_COLOR,
                color=TEXT_COLOR,
                showbackground=True,
                zerolinecolor=GRID_COLOR
            ),
            zaxis=dict(
                backgroundcolor=DARK_SECONDARY,
                gridcolor=GRID_COLOR,
                color=TEXT_COLOR,
                showbackground=True,
                zerolinecolor=GRID_COLOR
            ),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_COLOR)
        )
    )
    return fig

def add_animation_controls(fig, frame_dur, show_slider):
    """Add play/pause buttons, restart, and optional slider to the figure."""
    
    num_frames = len(fig.frames) if hasattr(fig, 'frames') and fig.frames else 0
    
    buttons = [
        dict(
            label="▶️ Play",
            method="animate",
            args=[None, dict(
                frame=dict(duration=frame_dur, redraw=True),
                fromcurrent=True,
                transition=dict(duration=0),
                mode="immediate"
            )]
        ),
        dict(
            label="⏸️ Pause",
            method="animate",
            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
        ),
        dict(
            label="⏮️ Restart",
            method="animate",
            args=[[fig.frames[0].name if num_frames > 0 and fig.frames[0].name else "frame_0"],
                  dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))]
        )
    ]
    
    updatemenus = [dict(
        type="buttons",
        showactive=False,
        y=1.08,
        x=0,
        xanchor="left",
        yanchor="top",
        pad=dict(t=0, r=10),
        buttons=buttons,
        direction="left",
        bgcolor='rgba(50,50,50,0.8)',
        font=dict(color=TEXT_COLOR)
    )]
    
    sliders = None
    if show_slider and num_frames > 0:
        steps = []
        for i, frame in enumerate(fig.frames):
            frame_name = frame.name if frame.name else f"frame_{i}"
            step = dict(
                args=[[frame_name],
                      dict(frame=dict(duration=0, redraw=True),
                           mode="immediate",
                           transition=dict(duration=0))],
                method="animate",
                label=str(i + 1)
            )
            steps.append(step)
        
        sliders = [dict(
            active=0,
            steps=steps,
            x=0.0,
            y=-0.02,
            len=1.0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=40, b=10),
            currentvalue=dict(
                prefix="Frame: ",
                visible=True,
                xanchor="center",
                font=dict(size=12, color=TEXT_COLOR)
            ),
            transition=dict(duration=0),
            ticklen=4,
            font=dict(color=TEXT_COLOR),
            bgcolor=DARK_SECONDARY,
            bordercolor=GRID_COLOR,
            tickcolor=TEXT_COLOR
        )]
    
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        height=800,
        margin=dict(l=0, r=0, t=40, b=60 if show_slider else 0),
        scene=dict(aspectmode='cube')
    )
    
    return fig, num_frames

def render_plotly_with_autoplay(fig, frame_dur, auto_play, loop_animation, num_frames, height=800):
    """Render Plotly figure with custom animation loop for reliable auto-play and looping."""
    
    # Get frame names for JavaScript
    frame_names = []
    if hasattr(fig, 'frames') and fig.frames:
        frame_names = [f.name if f.name else f"frame_{i}" for i, f in enumerate(fig.frames)]
    
    frame_names_json = json.dumps(frame_names)
    
    # Convert figure to HTML
    fig_html = fig.to_html(
        include_plotlyjs='cdn',
        full_html=False,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'responsive': True
        }
    )
    
    # Custom animation JavaScript with reliable looping
    animation_script = f"""
    <script>
    (function() {{
        const FRAME_DURATION = {frame_dur};
        const SHOULD_LOOP = {str(loop_animation).lower()};
        const SHOULD_AUTOPLAY = {str(auto_play).lower()};
        const TOTAL_FRAMES = {num_frames};
        const FRAME_NAMES = {frame_names_json};
        
        let plotDiv = null;
        let animationTimer = null;
        let currentFrame = 0;
        let isPlaying = false;
        let buttonsHooked = false;
        
        function getPlotDiv() {{
            if (plotDiv && plotDiv._fullLayout) return plotDiv;
            const divs = document.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
            for (let d of divs) {{
                if (d._fullLayout && d.data) {{
                    plotDiv = d;
                    return d;
                }}
            }}
            return null;
        }}
        
        function goToFrame(idx) {{
            const div = getPlotDiv();
            if (!div || FRAME_NAMES.length === 0 || idx < 0 || idx >= FRAME_NAMES.length) return;
            
            Plotly.animate(div, [FRAME_NAMES[idx]], {{
                mode: 'immediate',
                transition: {{ duration: 0 }},
                frame: {{ duration: 0, redraw: true }}
            }});
        }}
        
        function step() {{
            if (!isPlaying) return;
            
            currentFrame++;
            if (currentFrame >= TOTAL_FRAMES) {{
                if (SHOULD_LOOP) {{
                    currentFrame = 0;
                }} else {{
                    pause();
                    return;
                }}
            }}
            goToFrame(currentFrame);
        }}
        
        function play() {{
            if (isPlaying) return;
            if (TOTAL_FRAMES === 0) return;
            
            isPlaying = true;
            if (animationTimer) clearInterval(animationTimer);
            animationTimer = setInterval(step, FRAME_DURATION);
            console.log('Animation started, loop:', SHOULD_LOOP);
        }}
        
        function pause() {{
            isPlaying = false;
            if (animationTimer) {{
                clearInterval(animationTimer);
                animationTimer = null;
            }}
            console.log('Animation paused at frame:', currentFrame);
        }}
        
        function restart() {{
            pause();
            currentFrame = 0;
            goToFrame(0);
            console.log('Animation restarted');
        }}
        
        function hookButtons() {{
            if (buttonsHooked) return true;
            
            const buttons = document.querySelectorAll('.updatemenu-button');
            if (buttons.length < 3) return false;
            
            // Clone buttons to remove Plotly's event handlers
            buttons.forEach((btn, idx) => {{
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
            }});
            
            // Get the new buttons and attach our handlers
            const newButtons = document.querySelectorAll('.updatemenu-button');
            
            // Play button
            newButtons[0].addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                play();
            }});
            
            // Pause button
            newButtons[1].addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                pause();
            }});
            
            // Restart button
            newButtons[2].addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                restart();
            }});
            
            buttonsHooked = true;
            console.log('Buttons hooked successfully');
            return true;
        }}
        
        function initialize() {{
            const div = getPlotDiv();
            if (!div) {{
                setTimeout(initialize, 100);
                return;
            }}
            
            console.log('Plotly div found, initializing animation controller...');
            
            // Wait for buttons to render then hook them
            function waitForButtons() {{
                if (hookButtons()) {{
                    // Auto-play if enabled
                    if (SHOULD_AUTOPLAY && TOTAL_FRAMES > 0) {{
                        setTimeout(function() {{
                            play();
                        }}, 200);
                    }}
                }} else {{
                    setTimeout(waitForButtons, 100);
                }}
            }}
            
            setTimeout(waitForButtons, 300);
        }}
        
        // Start initialization when DOM is ready
        if (document.readyState === 'complete') {{
            setTimeout(initialize, 200);
        }} else {{
            window.addEventListener('load', function() {{
                setTimeout(initialize, 200);
            }});
        }}
    }})();
    </script>
    """
    
    # Dark theme CSS to ensure consistent background
    dark_css = f"""
    <style>
        .plotly-graph-div, .js-plotly-plot, .plot-container, .svg-container {{
            background-color: {DARK_BG} !important;
        }}
        .modebar {{
            background-color: rgba(0,0,0,0.5) !important;
        }}
        .modebar-btn path {{
            fill: {TEXT_COLOR} !important;
        }}
        .modebar-btn:hover path {{
            fill: #4fc3f7 !important;
        }}
    </style>
    """
    
    # Combine everything
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {dark_css}
    </head>
    <body style="margin: 0; padding: 0; background-color: {DARK_BG};">
        <div style="width: 100%; height: {height}px; background-color: {DARK_BG};">
            {fig_html}
        </div>
        {animation_script}
    </body>
    </html>
    """
    
    components.html(full_html, height=height + 50, scrolling=False)

def render_bromine_release_app():
    st.title("Bromine Vapor Release Model")
    st.caption("Deterministic 2D/3D model based on the static-chamber bromine release paper.")

    with st.sidebar:
        st.divider()
        st.subheader("Bromine Model Controls")
        visualization_mode = st.radio(
            "Visualization",
            ["2D Top-Down", "3D Particle Chamber"],
            horizontal=False,
        )
        chamber_radius_cm = st.slider("Chamber radius", 5.0, 20.0, 10.0, 0.5, help="Radius of the static evacuated chamber.")
        rotor_radius_cm = st.slider("Rotor/vial radius", 1.0, chamber_radius_cm * 0.95, min(10.0, chamber_radius_cm * 0.8), 0.5)
        rpm = st.slider("Centrifuge speed", 0, 15000, 10000, 250)
        temperature_k = st.slider("Temperature", 250, 350, 298, 1)
        particle_count = st.slider("Visible tracer particles", 80, 800, 320, 20)
        frames_count = st.slider("Animation frames", 60, 90, 75, 5)
        bromine_speed = st.slider("Animation speed", 10, 200, 50, 5)
        wall_accommodation = st.slider(
            "Wall thermalization",
            0.0,
            1.0,
            0.35,
            0.05,
            help="Higher values randomize particle velocity more strongly after chamber-wall encounters.",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.bromine_auto_play = st.checkbox(
                "Auto-play model",
                value=st.session_state.bromine_auto_play,
            )
        with col2:
            st.session_state.bromine_loop_animation = st.checkbox(
                "Loop model",
                value=st.session_state.bromine_loop_animation,
            )
        seed = st.number_input("Deterministic seed", min_value=1, max_value=9999, value=42, step=1)

    chamber_radius_m = chamber_radius_cm / 100
    rotor_radius_m = rotor_radius_cm / 100

    simulator = simulate_bromine_release_3d if visualization_mode == "3D Particle Chamber" else simulate_bromine_release_2d
    fig, density_fig, metrics = simulator(
        chamber_radius_m=chamber_radius_m,
        rotor_radius_m=rotor_radius_m,
        rpm=rpm,
        temperature_k=temperature_k,
        particle_count=particle_count,
        frames_count=frames_count,
        wall_accommodation=wall_accommodation,
        seed=seed,
    )

    st.subheader("Spin Rate as Equivalent Gravity")
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.metric("Equivalent gravity at vial", f"{metrics['g_equiv']:,.0f} g")
    with g_col2:
        st.metric("Centrifugal acceleration", f"{metrics['acceleration']:,.0f} m/s²")
    with g_col3:
        st.metric("Tangential release speed", f"{metrics['tangential_speed']:.1f} m/s")

    st.markdown(
        """
        <div style="
            border-left: 5px solid #f97316;
            background: rgba(249, 115, 22, 0.12);
            padding: 16px 18px;
            margin: 12px 0 18px 0;
            border-radius: 6px;
        ">
            <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 6px;">
                Core prediction: high equivalent-g does not prevent rapid gas equilibration.
            </div>
            <div style="font-size: 0.96rem;">
                The spinning vial gives the released bromine vapor initial tangential momentum, but the outer chamber is static.
                Once the gas is released into the evacuated chamber, thermal molecular motion and pressure-driven expansion
                carry it into the available volume on a chamber-crossing timescale.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Here, equivalent gravity means the local centrifugal pseudo-acceleration at the vial radius, "
        "`a = omega² r`, divided by standard Earth gravity. It is useful for intuition precisely because it highlights the misconception: "
        "even thousands of g do not act like a confining wall for gas molecules released into a static vacuum chamber."
    )

    st.subheader(f"{visualization_mode} Chamber Model")
    fig = apply_dark_theme(fig)
    frame_dur = int(1000 / bromine_speed)
    fig, num_frames = add_animation_controls(fig, frame_dur=frame_dur, show_slider=True)
    render_plotly_with_autoplay(
        fig,
        frame_dur,
        st.session_state.bromine_auto_play,
        st.session_state.bromine_loop_animation,
        num_frames,
        height=720,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        density_fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
        )
        st.plotly_chart(density_fig, use_container_width=True)

    with col2:
        st.subheader("Computed Values")
        st.metric("Mean thermal speed", f"{metrics['mean_speed']:.0f} m/s")
        st.metric("Chamber crossing estimate", f"{metrics['t_cross_ms']:.2f} ms")
        st.metric("Ideal co-rotating density ratio", f"{metrics['density_ratio']:.2f}x")
        st.caption(
            f"Density ratio compares r = {rotor_radius_m / 2:.3f} m to r = {rotor_radius_m:.3f} m "
            f"for a hypothetical fully co-rotating gas."
        )

    with st.expander("Paper Equations and Interpretation", expanded=True):
        st.markdown(
            f"""
            **Central claim**

            The centrifuge creates a large local pseudo-acceleration while the vial is rotating, but that does not stop molecular gas from rapidly occupying available evacuated volume after release. The post-release chamber is static, so the final sustained state is not a centrifugal barometric distribution.

            **Thermal crossing time**

            `t_cross ≈ L / v_mean`

            For this setup: `L = {chamber_radius_m:.3f} m`, `v_mean = {metrics['mean_speed']:.1f} m/s`, so `t_cross ≈ {metrics['t_cross_ms']:.2f} ms`.

            **Equivalent gravity from spin**

            `omega = 2*pi*RPM/60`

            `a(r) = omega² r`

            At `RPM = {rpm:,}` and `r = {rotor_radius_m:.3f} m`, this gives `{metrics['acceleration']:,.0f} m/s²`, or about `{metrics['g_equiv']:,.0f} g`.

            This number describes the rotating-frame pseudo-acceleration at the vial radius. It does not describe a real inward or outward chamber wall, and it does not by itself determine whether gas will fill the surrounding vacuum.

            **Ideal co-rotating density comparison**

            `n(r) / n(r0) = exp(M * omega² * (r² - r0²) / (2RT))`

            For `r0 = {rotor_radius_m / 2:.3f} m` and `r = {rotor_radius_m:.3f} m`, the exponent is `{metrics['density_exponent']:.3f}`, giving a ratio of `{metrics['density_ratio']:.2f}x`.

            This comparison is intentionally idealized. It asks what would happen if the gas and enclosure were fully co-rotating. The actual experiment releases gas into a static chamber, so this is a ceiling/contrast case, not the final post-release equilibrium.

            **Post-release pressure and mean free path**

            Using the paper's representative `10 mL` vial, `1000 mL` chamber, and `230 Torr` bromine vapor pressure:

            `P_final ≈ 230 Torr * 10 / 1000 = {metrics['final_pressure_torr']:.1f} Torr`

            The estimated mean free path is `{metrics['mean_free_path_m']:.2e} m`, which is much smaller than the chamber size. That supports the paper's distinction between rapid visual filling and slower bulk angular-momentum relaxation.
            """
        )

    with st.expander("Model Limits", expanded=False):
        st.markdown(
            """
            This is a visual deterministic tracer model, not a full CFD solver. It shows the paper's main qualitative idea: initial tangential momentum can curve the plume, while thermal speeds rapidly spread the gas across the static chamber.

            The real post-release gas may quickly become collisional at the representative final pressure, so detailed plume shape, viscosity, pressure waves, boundary layers, and optical absorbance are simplified here. The particle animation should therefore be read as a teaching model for the misconception tested by the paper: large equivalent-g is not the same thing as molecular gas confinement.
            """
        )

# --- Main App ---
def main_app():
    with st.sidebar:
        st.session_state.app_mode = st.radio(
            "App Mode",
            ["GenAI Physics Generator", "Bromine Release Paper Model"],
            index=0 if st.session_state.app_mode == "GenAI Physics Generator" else 1,
        )

    if st.session_state.app_mode == "Bromine Release Paper Model":
        render_bromine_release_app()
        return

    with st.sidebar:
        st.title("⚙️ Settings")
        
        provider = st.radio("Model Source", ["xAI (Grok)", "OpenAI", "DeepSeek"], label_visibility="collapsed")
        
        api_key = None
        base_url = None
        model_name = ""
        
        if provider == "xAI (Grok)":
            api_key = get_secret("xai_api_key")
            base_url = "https://api.x.ai/v1"
            choice = st.selectbox("Grok Model", list(XAI_MODELS.keys()))
            model_name = XAI_MODELS[choice]
        elif provider == "DeepSeek":
            api_key = get_secret("deepseek_api_key")
            base_url = "https://api.deepseek.com"
            choice = st.selectbox("DeepSeek Model", list(DEEPSEEK_MODELS.keys()))
            model_name = DEEPSEEK_MODELS[choice]
        else:
            api_key = get_secret("openai_api_key")
            choice = st.selectbox("OpenAI Model", list(OPENAI_MODELS.keys()))
            model_name = OPENAI_MODELS[choice]
            
        if not api_key:
            st.error(f"Missing API Key for {provider}")
        
        # Display pricing indicator for selected model
        display_pricing_indicator(model_name)

        st.divider()
        st.subheader("🎮 Animation Controls")
        
        speed = st.slider("Speed Factor", 10, 200, 50, 5)
        frame_dur = int(1000 / speed)
        st.caption(f"Frame duration: {frame_dur} ms")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_play = st.checkbox(
                "Auto-play",
                value=st.session_state.auto_play,
                help="Automatically start animation when loaded"
            )
        with col2:
            st.session_state.loop_animation = st.checkbox(
                "Loop",
                value=st.session_state.loop_animation,
                help="Continuously loop the animation"
            )
        
        st.session_state.show_slider = st.checkbox(
            "Show Frame Slider",
            value=st.session_state.show_slider,
            help="Display a slider to scrub through animation frames"
        )

        st.divider()
        with st.expander("💰 Cost Summary", expanded=True):
            total_cost = sum(item.get('cost', 0) for item in st.session_state.history)
            st.metric("Session Total", f"${total_cost:.5f}")
            
            if st.session_state.history:
                last = st.session_state.history[-1]
                st.markdown("---")
                st.markdown("**Last Generation:**")
                
                last_info = get_pricing_info(last['model'])
                provider_icon = "🟢" if last_info['provider'] == "OpenAI" else "🟣" if last_info['provider'] == "DeepSeek" else "🔵"
                
                st.markdown(f"{provider_icon} **{last['model']}**")
                st.markdown(f"💵 Cost: **${last['cost']:.5f}**")
                
                if 'prompt_tokens' in last and last['prompt_tokens']:
                    st.markdown(f"📥 Input: {last['prompt_tokens']:,} tokens")
                    st.markdown(f"📤 Output: {last['completion_tokens']:,} tokens")

        st.divider()
        st.subheader("📜 Recent History")
        for i, item in enumerate(reversed(st.session_state.history[:5])):
            hist_info = get_pricing_info(item['model'])
            hist_icon = "🟢" if hist_info['provider'] == "OpenAI" else "🟣" if hist_info['provider'] == "DeepSeek" else "🔵"
            if st.button(f"{hist_icon} #{len(st.session_state.history)-i}: {item['model'][:15]}...", key=f"hist_{i}"):
                st.session_state.current_code = item["code"]
                st.session_state.prompt = item["prompt"]
                st.rerun()
        
        st.divider()
        with st.expander("📊 Pricing Reference", expanded=False):
            st.markdown("**OpenAI Models** 🟢")
            for display_name, model_id in OPENAI_MODELS.items():
                info = PRICING.get(model_id, {})
                st.caption(f"`{model_id}`: ${info.get('input', 'N/A')}/{info.get('output', 'N/A')}")
            
            st.markdown("**xAI Grok Models** 🔵")
            for display_name, model_id in XAI_MODELS.items():
                info = PRICING.get(model_id, {})
                st.caption(f"`{model_id}`: ${info.get('input', 'N/A')}/{info.get('output', 'N/A')}")

            st.markdown("**DeepSeek Models** 🟣")
            for display_name, model_id in DEEPSEEK_MODELS.items():
                info = PRICING.get(model_id, {})
                st.caption(f"`{model_id}`: ${info.get('input', 'N/A')}/{info.get('output', 'N/A')}")
            
            st.caption("_Prices shown as $/1M tokens (input/output)_")

    st.title("⚛️ Generative Physics Modeler")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.selectbox("📚 Scenarios", list(SCENARIOS.keys()), key="scenario_selector", on_change=update_prompt)
    
    with c2:
        pass

    prompt = st.text_area("Physics Description", height=110, key="prompt")

    if st.button("🚀 Generate Simulation", type="primary", use_container_width=True, disabled=not api_key):
        with st.status(f"Generating with {model_name}...", expanded=True) as status:
            st.write("🧠 Asking the model...")
            try:
                code, cost, p_tokens, c_tokens = generate_simulation(prompt, api_key, base_url, model_name)
                
                st.session_state.current_code = code
                st.session_state.history.append({
                    "code": code,
                    "prompt": prompt,
                    "model": model_name,
                    "cost": cost,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens
                })
                
                status.update(label="Success!", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")
                status.update(label="Failed", state="error")

    # --- Display current simulation ---
    if st.session_state.get("current_code"):
        with st.expander("🔍 View Generated Code & Details", expanded=False):
            tab1, tab2 = st.tabs(["Python Code", "Cost Breakdown"])
            with tab1:
                st.code(st.session_state.current_code, language="python")
                st.download_button("📥 Download .py", st.session_state.current_code, "simulation.py")
            with tab2:
                if st.session_state.history:
                    last = st.session_state.history[-1]
                    last_info = get_pricing_info(last['model'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Cost", f"${last['cost']:.5f}")
                    with col2:
                        if last.get('prompt_tokens'):
                            st.metric("Input Tokens", f"{last['prompt_tokens']:,}")
                    with col3:
                        if last.get('completion_tokens'):
                            st.metric("Output Tokens", f"{last['completion_tokens']:,}")
                    
                    st.markdown("---")
                    provider_icon = "🟢" if last_info['provider'] == "OpenAI" else "🟣" if last_info['provider'] == "DeepSeek" else "🔵"
                    st.markdown(f"**Provider:** {provider_icon} {last_info['provider']}")
                    st.markdown(f"**Model:** `{last['model']}`")
                    st.markdown(f"**Rates:** ${last_info['input']:.2f} input / ${last_info['output']:.2f} output per 1M tokens")

        exec_globals = {"np": np, "go": go}
        success, error = execute_safe_code(st.session_state.current_code, exec_globals)
        
        if success and "fig" in exec_globals:
            fig = exec_globals["fig"]
            
            # Apply dark theme
            fig = apply_dark_theme(fig)
            
            # Add animation controls to the figure
            fig, num_frames = add_animation_controls(
                fig,
                frame_dur,
                st.session_state.show_slider
            )
            
            # Display animation info
            if num_frames > 0:
                status_col1, status_col2, status_col3 = st.columns([1, 1, 2])
                with status_col1:
                    st.metric("Total Frames", num_frames)
                with status_col2:
                    duration_sec = (num_frames * frame_dur) / 1000
                    st.metric("Duration", f"{duration_sec:.1f}s")
                with status_col3:
                    status_items = []
                    if st.session_state.auto_play:
                        status_items.append("🔄 Auto-play ON")
                    if st.session_state.loop_animation:
                        status_items.append("🔁 Loop ON")
                    if status_items:
                        st.info(" | ".join(status_items))
            
            # Render with custom animation controller
            render_plotly_with_autoplay(
                fig,
                frame_dur,
                st.session_state.auto_play,
                st.session_state.loop_animation,
                num_frames,
                height=800
            )
                
        elif not success:
            st.error(f"⚠️ Runtime error in generated code:\n{error}")
        else:
            st.error("⚠️ Generated code did not define a variable named `fig`.")

if __name__ == "__main__":
    if check_password():
        main_app()
