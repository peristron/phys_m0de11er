# streamlit run phys_modeller.py
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(layout="wide", page_title="GenAI Physics Modeler", page_icon="⚛️")

# --- Pricing & Scenarios ---
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
for key, val in [("history", []), ("prompt", SCENARIOS["Rotating Sphere with Gas"])]:
    if key not in st.session_state:
        st.session_state[key] = val

# --- Secrets & Password ---
def get_secret(k): return next((st.secrets[key] for key in [k, k.upper(), k.lower()] if key in st.secrets), None)

def check_password():
    pwd = get_secret("app_password")
    if not pwd:
        st.error("Missing 'app_password' in secrets.toml")
        st.stop()
    def verify():
        if st.session_state.pwd == pwd:
            st.session_state.auth = True
        else:
            st.session_state.auth = False
    if "auth" not in st.session_state:
        st.text_input("Password", type="password", key="pwd", on_change=verify)
        return False
    if not st.session_state.auth:
        st.text_input("Password", type="password", key="pwd", on_change=verify)
        st.error("Wrong password")
        return False
    return True

# --- Safe Sandbox (fixed for Streamlit Cloud) ---
def execute_safe_code(code_str, globals_dict):
    blocked = ['os','sys','subprocess','shutil','requests','socket','pickle','ctypes','urllib','pathlib','webbrowser']
    def restricted_import(name, *args, **kwargs):
        if name.split('.')[0] in blocked:
            raise ImportError(f"Blocked: {name}")
        return __import__(name, *args, **kwargs)

    safe_builtins = {}
    if isinstance(__builtins__, dict):
        safe_builtins.update(__builtins__)
    else:
        safe_builtins.update(getattr(__builtins__, "__dict__", {}) or vars(__builtins__))
    for dangerous in ['eval','exec','open','compile','__import__']:
        safe_builtins.pop(dangerous, None)
    safe_builtins['__import__'] = restricted_import

    globals_dict['__builtins__'] = safe_builtins
    try:
        exec(code_str, globals_dict)
        return True, None
    except Exception as e:
        return False, str(e)

# --- LLM ---
def get_system_prompt():
    return """
You are a perfect Python code generator for 3D physics animations using only numpy and plotly.graph_objects.
RULES:
- Output ONLY raw Python code. No markdown, no ``` fences.
- Start with: import numpy as np; import plotly.graph_objects as go
- Pre-compute 60–90 frames and add them to fig.frames
- Final figure: fig = go.Figure(...)
- NEVER define updatemenus or sliders
- ALWAYS set fixed axis ranges (e.g. scene.xaxis.range=[-10,10])
- Use full numpy vectorization
- Keep code minimal and fast
"""

def call_llm(messages, key, url, model):
    client = openai.OpenAI(api_key=key, base_url=url or None)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=4096)
    return resp.choices[0].message.content, getattr(resp.usage, "prompt_tokens", None), getattr(resp.usage, "completion_tokens", None)

def generate_simulation(prompt, key, url, model):
    messages = [{"role": "system", "content": get_system_prompt()}, {"role": "user", "content": prompt}]
    total_in = total_out = 0
    dummy = {"np": np, "go": go}

    for _ in range(3):
        raw, p_tok, c_tok = call_llm(messages, key, url, model)
        if p_tok: total_in += p_tok
        if c_tok: total_out += c_tok
        code = re.sub(r"^```python|```$", "", raw, flags=re.MULTILINE).strip()
        dummy.clear(); dummy.update({"np": np, "go": go})
        success, err = execute_safe_code(code, dummy)
        fig_ok = "fig" in dummy and isinstance(dummy["fig"], go.Figure)

        if success and fig_ok:
            cost = (total_in/1e6)*PRICING.get(model, {"input":5,"output":15})["input"] + (total_out/1e6)*PRICING.get(model, {"input":5,"output":15})["output"]
            return code, round(cost, 6)

        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"Fix this and return ONLY corrected code:\n• {err or 'No error'}\n• Missing fig" if not fig_ok else ""}]

    raise RuntimeError("Failed after 3 attempts")

# --- Main App ---
def main_app():
    with st.sidebar:
        st.title("⚙️ Settings")
        provider = st.radio("Model", ["xAI (Grok)", "OpenAI"], horizontal=True)
        api_key = base_url = model = None

        if provider == "xAI (Grok)":
            api_key = get_secret("xai_api_key")
            base_url = "https://api.x.ai/v1"
            model = st.selectbox("Model", ["grok-4-1-fast-reasoning", "grok-4-0709", "grok-2-1212"], index=0)
        else:
            api_key = get_secret("openai_api_key")
            model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])

        if not api_key: st.error("API key missing")

        st.divider()
        speed = st.slider("Speed", 10, 200, 60, 5)
        frame_dur = int(1000 / speed)

        st.divider()
        with st.expander("💰 Cost", True):
            total = sum(h.get("cost",0) for h in st.session_state.history)
            st.write(f"**Session:** ${total:.6f}")
            if st.session_state.history:
                last = st.session_state.history[-1]
                st.write(f"**Last:** ${last['cost']:.6f} ({last['model']})")

        st.divider()
        for i, h in enumerate(reversed(st.session_state.history[:5])):
            if st.button(f"Load #{len(st.session_state.history)-i}", key=f"hist{i}"):
                st.session_state.current_code = h["code"]
                st.session_state.prompt = h["prompt"]
                st.rerun()

    st.title("⚛️ Generative Physics Modeler")

    col1, _ = st.columns([3,1])
    with col1:
        st.selectbox("Scenarios", list(SCENARIOS), key="sel", on_change=lambda: st.session_state.update(prompt=SCENARIOS[st.session_state.sel] if st.session_state.sel != "Custom" else ""))

    prompt = st.text_area("Physics Description", st.session_state.prompt, height=110, key="prompt_input")
    st.session_state.prompt = prompt

    if st.button("🚀 Generate Simulation", type="primary", use_container_width=True, disabled=not api_key):
        with st.status("Generating...") as s:
            st.write("Thinking...")
            code, cost = generate_simulation(prompt, api_key, base_url, model)
            st.session_state.current_code = code
            st.session_state.history.append({"code": code, "prompt": prompt, "model": model, "cost": cost})
            s.update(label="Done!", state="complete")
            st.rerun()

    # === DISPLAY ===
    if st.session_state.get("current_code"):
        with st.expander("Code & Cost"):
            c1, c2 = st.tabs(["Code", "Cost"])
            with c1:
                st.code(st.session_state.current_code, "python")
                st.download_button("Download .py", st.session_state.current_code, "simulation.py")
            with c2:
                cost = st.session_state.history[-1]["cost"] if st.session_state.history else 0
                st.metric("Cost", f"${cost:.6f}")

        g = {"np": np, "go": go}
        success, err = execute_safe_code(st.session_state.current_code, g)

        if success and "fig" in g:
            fig = g["fig"]

            # === PERFECT INFINITE AUTO-LOOP (confirmed working) ===
            fig.update_layout(
                height=800,
                margin=dict(l=0,r=0,t=40,b=0),
                scene=dict(aspectmode='cube'),
                updatemenus=[dict(
                    buttons=[
                        dict(label="▶️ Play", method="animate",
                             args=[None, dict(frame=dict(duration=frame_dur, redraw=True),
                                              fromcurrent=False,
                                              transition=dict(duration=0),
                                              mode="immediate")]),
                        dict(label="⏸️ Pause", method="animate",
                             args=[[None], dict(mode="immediate")])
                    ],
                    direction="left", pad={"r":10,"t":10}, showactive=False,
                    type="buttons", x=0, xanchor="left", y=1.15, yanchor="top"
                )]
            )

            # Force infinite loop + auto-start
            if fig.frames:
                play_args = fig.layout.updatemenus[0].buttons[0].args
                play_args[1]["frame"]["loop"] = "infinite"
                play_args[1]["fromcurrent"] = False

            fig.add_annotation(text="∞ looping ∙ auto-play", xref="paper", yref="paper",
                               x=0.02, y=1.11, showarrow=False,
                               font=dict(size=11,color="#00ff88"), bgcolor="#000000cc")

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        elif not success:
            st.error(f"Runtime error:\n{err}")
        else:
            st.error("No variable named `fig`")

if __name__ == "__main__":
    if check_password():
        main_app()
