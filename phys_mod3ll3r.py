# streamlit run phys_modeller.py
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import re

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GenAI Physics Modeler", page_icon="⚛️")

# --- Constants ---
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-0709": {"input": 3.00, "output": 15.00},
    "grok-2-1212": {"input": 2.00, "output": 10.00},
    "grok-3": {"input": 3.00, "output": 15.00},
}

SCENARIOS = {
    "Custom": "",
    "Rotating Sphere with Gas": "A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container.",
    "Rarefied Gas Spin-Up": "Simulate the transfer of conserved momentum from a rotating solid disk to gas molecules immediately adjacent to it in a vacuum. The gas should gradually spin up due to wall collisions.",
    "Solar System w/ Comet": "A solar system simulation with a static yellow sun, 3 orbiting planets at different distances/speeds, and a comet passing through on a hyperbolic trajectory.",
    "Damped Pendulum": "A 3D visualization of a simple pendulum with damping. Show the pendulum bob swinging in 3D space and trace its path color-coded by velocity.",
    "Lorenz Attractor": "Simulate the Lorenz attractor (chaotic system). Visualize the trajectory of a point over time in 3D space, leaving a trail.",
}

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = SCENARIOS["Rotating Sphere with Gas"]

# --- Secrets & Password ---
def get_secret(key_name):
    for key in [key_name, key_name.upper(), key_name.lower()]:
        if key in st.secrets:
            return st.secrets[key]
    return None

def check_password():
    pwd = get_secret("app_password")
    if not pwd:
        st.error("❌ 'app_password' missing in secrets.toml")
        st.stop()

    def verify():
        if st.session_state["pwd_input"] == pwd:
            st.session_state["auth"] = True
            if "pwd_input" in st.session_state:
                del st.session_state["pwd_input"]
        else:
            st.session_state["auth"] = False

    if "auth" not in st.session_state:
        st.text_input("Password", type="password", key="pwd_input", on_change=verify)
        return False
    if not st.session_state["auth"]:
        st.text_input("Password", type="password", key="pwd_input", on_change=verify)
        st.error("Incorrect password")
        return False
    return True

# --- strong sandbx
    def execute_safe_code(code_str, globals_dict):
        blocked = ['os', 'sys', 'subprocess', 'shutil', 'requests', 'socket', 'pickle', 'ctypes', 'urllib', 'pathlib', 'webbrowser']
        
        def restricted_import(name, *args, **kwargs):
            if name.split('.')[0] in blocked:
                raise ImportError(f"Import blocked: {name}")
            return __import__(name, *args, **kwargs)

        # FIXED: Safe handling for both dict and module types
        if isinstance(__builtins__, dict):
            safe_builtins = __builtins__.copy()
        else:
            safe_builtins = getattr(__builtins__, "__dict__", {}).copy() or vars(__builtins__).copy()

        for dangerous in ['eval', 'exec', 'open', 'compile', '__import__']:
            safe_builtins.pop(dangerous, None)
        safe_builtins['__import__'] = restricted_import

        globals_dict['__builtins__'] = safe_builtins
        try:
            exec(code_str, globals_dict)
            return True, None
        except Exception as e:
            return False, str(e)

# --- Code Cleaning ---
def clean_code(raw):
    lines = raw.splitlines()
    start = 0
    if lines and lines[0].lstrip().startswith("```"):
        start = 1
    end = len(lines)
    if lines and lines[-1].strip() == "```":
        end -= 1
    return "\n".join(lines[start:end]).strip()

# --- LLM ---
def get_system_prompt():
    return """
You are a perfect Python code generator for 3D physics animations using only numpy and plotly.graph_objects.

RULES (never break):
- Output ONLY raw Python code. No ```, no explanations.
- Always start with: import numpy as np and import plotly.graph_objects as go
- Pre-compute exactly 60–90 frames and add them to fig.frames
- Final figure must be named `fig = go.Figure(...)`
- NEVER define updatemenus or sliders — the app handles playback
- ALWAYS set fixed axis ranges (e.g. xaxis.range=[-10, 10]) — never auto
- Use full numpy vectorization — never write long lists manually
- Keep code short and efficient
"""

def call_llm(messages, key, url, model):
    client = openai.OpenAI(api_key=key, base_url=url if url else None)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=4096,
    )
    return (resp.choices[0].message.content,
            getattr(resp.usage, "prompt_tokens", None),
            getattr(resp.usage, "completion_tokens", None))

def generate_simulation(prompt, key, url, model):
    messages = [{"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}]

    total_in = total_out = 0
    dummy = {"np": np, "go": go}

    for attempt in range(3):
        raw, p_tok, c_tok = call_llm(messages, key, url, model)
        if p_tok: total_in += p_tok
        if c_tok: total_out += c_tok

        code = clean_code(raw)
        dummy.clear(); dummy.update({"np": np, "go": go})
        success, err = execute_safe_code(code, dummy)
        fig_ok = "fig" in dummy and isinstance(dummy["fig"], go.Figure)

        if success and fig_ok:
            rates = PRICING.get(model, {"input": 5.0, "output": 15.0})
            cost = (total_in / 1_000_000) * rates["input"] + (total_out / 1_000_000) * rates["output"]
            return code, round(cost, 6)

        feedback = "Fix and return ONLY corrected code:\n" + "\n".join([
            f"• {err}" if not success else "",
            "• Missing `fig`" if "fig" not in dummy else "",
            "• `fig` not a go.Figure" if not fig_ok else "",
            "• Follow system prompt exactly"
        ]).strip()
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})

    raise RuntimeError("Failed after 3 attempts")

# --- Main App ---
def main_app():
    with st.sidebar:
        st.title("⚙️ Settings")
        provider = st.radio("Model", ["xAI (Grok)", "OpenAI"], label_visibility="collapsed")

        api_key = base_url = model_name = None
        if provider == "xAI (Grok)":
            api_key = get_secret("xai_api_key")
            base_url = "https://api.x.ai/v1"
            opts = {
                "Grok 4.1 Fast (Best Value)": "grok-4-1-fast-reasoning",
                "Grok 4": "grok-4-0709",
                "Grok 2": "grok-2-1212",
            }
            model_name = opts[st.selectbox("Model", list(opts.keys()))]
        else:
            api_key = get_secret("openai_api_key")
            model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])

        if not api_key:
            st.error("Missing API key")

        st.divider()
        st.subheader("🎮 Playback")
        speed = st.slider("Speed", 10, 200, 60, 5)
        frame_dur = int(1000 / speed)
        st.caption(f"Frame: {frame_dur} ms")

        st.divider()
        with st.expander("💰 Cost", expanded=True):
            total = sum(h.get("cost", 0) for h in st.session_state.history)
            st.write(f"**Session total:** ${total:.6f}")
            if st.session_state.history:
                last = st.session_state.history[-1]
                st.write(f"**Last run:** ${last['cost']:.6f} ({last['model']})")

        st.divider()
        st.subheader("📜 Recent")
        for i, h in enumerate(reversed(st.session_state.history[:5])):
            if st.button(f"Load #{len(st.session_state.history)-i}", key=f"hist_{i}"):
                st.session_state.current_code = h["code"]
                st.session_state.prompt = h["prompt"]
                st.rerun()

    st.title("⚛️ Generative Physics Modeler")

    col1, _ = st.columns([3, 1])
    with col1:
        st.selectbox("Scenarios", list(SCENARIOS.keys()), key="scenario_selector",
                     on_change=lambda: st.session_state.update(prompt=SCENARIOS[st.session_state.scenario_selector] if st.session_state.scenario_selector != "Custom" else ""))

    prompt = st.text_area("Physics Description", value=st.session_state.prompt, height=110, key="prompt_input")
    st.session_state.prompt = prompt

    if st.button("🚀 Generate Simulation", type="primary", use_container_width=True, disabled=not api_key):
        with st.status(f"Generating with {model_name}...") as status:
            st.write("Thinking...")
            try:
                code, cost = generate_simulation(prompt, api_key, base_url, model_name)
                st.session_state.current_code = code
                st.session_state.history.append({"code": code, "prompt": prompt, "model": model_name, "cost": cost})
                status.update(label="Success!", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    # === DISPLAY ===
    if st.session_state.get("current_code"):
        with st.expander("🔍 Code & Cost", expanded=False):
            t1, t2 = st.tabs(["Code", "Cost"])
            with t1:
                st.code(st.session_state.current_code, language="python")
                st.download_button("Download .py", st.session_state.current_code, "simulation.py")
            with t2:
                cost = st.session_state.history[-1]["cost"] if st.session_state.history else 0
                st.metric("Actual Cost", f"${cost:.6f}")

        globals_dict = {"np": np, "go": go}
        success, err = execute_safe_code(st.session_state.current_code, globals_dict)

        if success and "fig" in globals_dict:
            fig = globals_dict["fig"]

            # === INFINITE AUTO-LOOP MAGIC ===
            fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, {
                                    "frame": {"duration": frame_dur, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                    "mode": "immediate"
                                }],
                                "label": "▶️ Play",
                                "method": "animate",
                            },
                            {
                                "args": [[None], {"mode": "immediate"}],
                                "label": "⏸️ Pause",
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 10},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.0,
                        "xanchor": "left",
                        "y": 1.15,
                        "yanchor": "top",
                    }
                ],
                # This forces infinite looping + auto-start
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"prefix": "Frame: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [[f.name], {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i, f in enumerate(fig.frames)
                    ]
                }],
                height=800,
                margin=dict(l=0, r=0, t=40, b=0),
                scene=dict(aspectmode='cube'),
                # Auto-start animation on load
                **({"sliders[0].active": len(fig.frames) - 1} if fig.frames else {})
            )

            # Tiny ∞ indicator
            fig.add_annotation(
                text="∞ looping",
                xref="paper", yref="paper",
                x=0.01, y=1.10,
                showarrow=False,
                font=dict(size=11, color="#00ff88"),
                bgcolor="#00000088"
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        elif not success:
            st.error(f"Runtime error:\n{err}")
        else:
            st.error("Code did not define variable `fig`")

if __name__ == "__main__":
    if check_password():
        main_app()
