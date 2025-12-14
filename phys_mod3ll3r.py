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

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = SCENARIOS["Rotating Sphere with Gas"]

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

# --- Stronger Sandbox ---
def execute_safe_code(code_str, global_vars):
    # Restricted __import__
    blocked_modules = ['os', 'sys', 'subprocess', 'shutil', 'requests', 'socket',
                      'pickle', 'ctypes', 'multiprocessing', 'urllib', 'webbrowser']
    
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split('.')[0] in blocked_modules or name in blocked_modules:
            raise ImportError(f"Importing '{name}' is forbidden for security reasons.")
        return __import__(name, globals, locals, fromlist, level)

    # Safe builtins — remove the really dangerous ones
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
    # Skip opening fence
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    # Drop closing fence
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()

# --- LLM Logic ---
def get_system_prompt():
    return """
You are a perfect Python code generator for 3D physics visualizations using only `numpy` and `plotly.graph_objects`.

STRICT RULES (never break them):
- Output ONLY raw Python code. No markdown, no ``` fences, no explanations.
- Start with the imports: import numpy as np and import plotly.graph_objects as go
- Pre-calculate exactly 60-90 frames and attach them to fig.frames
- Define the final figure as `fig = go.Figure(...)`
- NEVER define updatemenus or play buttons — the app adds them automatically
- ALWAYS set fixed axis ranges (e.g. scene.xaxis.range=[-10, 10]) — never use auto-scaling
- Use heavy vectorization with np functions — never write out long lists manually
- Keep code as short and efficient as possible
"""

def call_llm(messages, key, url, model):
    client = openai.OpenAI(api_key=key, base_url=url if url else None)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,          # lower = much more reliable code
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
            # Calculate accurate cost
            rates = PRICING.get(model, {"input": 5.0, "output": 15.0})
            if total_prompt_tokens > 0:
                cost = (total_prompt_tokens / 1_000_000) * rates["input"] + \
                       (total_completion_tokens / 1_000_000) * rates["output"]
            else:
                # Fallback approximation (very rare)
                est_input = len(get_system_prompt() + prompt) / 4
                est_output = len(raw_code) / 4
                cost = (est_input / 1e6) * rates["input"] + (est_output / 1e6) * rates["output"]
            
            return code, cost

        # Auto-correction feedback
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

# --- Main App ---
def main_app():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        provider = st.radio("Model Source", ["xAI (Grok)", "OpenAI"], label_visibility="collapsed")
        
        api_key = None
        base_url = None
        model_name = ""
        
        if provider == "xAI (Grok)":
            api_key = get_secret("xai_api_key")
            base_url = "https://api.x.ai/v1"
            model_options = {
                "Grok 4.1 Fast (Reasoning) [Best Value]": "grok-4-1-fast-reasoning",
                "Grok 4 (Standard)": "grok-4-0709",
                "Grok 2 (Legacy)": "grok-2-1212",
                "Grok 3 (Legacy)": "grok-3"
            }
            choice = st.selectbox("Version", list(model_options.keys()))
            model_name = model_options[choice]
        else:
            api_key = get_secret("openai_api_key")
            model_name = st.selectbox("Version", ["gpt-4o", "gpt-4o-mini"])
            
        if not api_key:
            st.error(f"Missing API Key for {provider}")

        st.divider()
        st.subheader("🎮 Animation")
        speed = st.slider("Speed Factor", 10, 200, 50, 5)
        frame_dur = int(1000 / speed)
        st.caption(f"Frame duration: {frame_dur} ms")

        st.divider()
        with st.expander("💰 Cost Estimator", expanded=True):
            total_cost = sum(item.get('cost', 0) for item in st.session_state.history)
            st.write(f"**Session total:** ${total_cost:.5f}")
            if st.session_state.history:
                last = st.session_state.history[-1]
                st.write(f"**Last run:** ${last['cost']:.5f} ({last['model']})")

        st.divider()
        st.subheader("📜 Recent History")
        for i, item in enumerate(reversed(st.session_state.history[:5])):
            if st.button(f"↩️ Load #{len(st.session_state.history)-i}: {item['model']}", key=f"hist_{i}"):
                st.session_state.current_code = item["code"]
                st.session_state.prompt = item["prompt"]
                st.rerun()

    st.title("⚛️ Generative Physics Modeler")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.selectbox("📚 Scenarios", list(SCENARIOS.keys()), key="scenario_selector", on_change=update_prompt)
    
    with c2:
        pass  # spacing

    prompt = st.text_area("Physics Description", height=110, key="prompt")

    if st.button("🚀 Generate Simulation", type="primary", use_container_width=True, disabled=not api_key):
        with st.status(f"Generating with {model_name}...", expanded=True) as status:
            st.write("🧠 Asking the model...")
            try:
                code, cost = generate_simulation(prompt, api_key, base_url, model_name)
                
                st.session_state.current_code = code
                st.session_state.history.append({
                    "code": code,
                    "prompt": prompt,
                    "model": model_name,
                    "cost": cost
                })
                
                status.update(label="Success!", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")
                status.update(label="Failed", state="error")

    # --- Display current simulation ---
    if st.session_state.get("current_code"):
        with st.expander("🔍 View Generated Code & Details", expanded=False):
            tab1, tab2 = st.tabs(["Python Code", "Cost"])
            with tab1:
                st.code(st.session_state.current_code, language="python")
                st.download_button("📥 Download .py", st.session_state.current_code, "simulation.py")
            with tab2:
                if st.session_state.history:
                    last_cost = st.session_state.history[-1]["cost"]
                    st.metric("Actual Cost", f"${last_cost:.5f}")

        exec_globals = {"np": np, "go": go}
        success, error = execute_safe_code(st.session_state.current_code, exec_globals)
        
        if success and "fig" in exec_globals:
            fig = exec_globals["fig"]
            
            # Inject Play/Pause buttons + consistent styling
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.05, x=0, xanchor="left", yanchor="top",
                    buttons=[
                        dict(label="▶️ Play", method="animate", args=[None, dict(frame=dict(duration=frame_dur, redraw=True), fromcurrent=True)]),
                        dict(label="⏸️ Pause", method="animate", args=[[None], dict(mode="immediate")])
                    ]
                )],
                height=800,
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(aspectmode='cube')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif not success:
            st.error(f"⚠️ Runtime error in generated code:\n{error}")
        else:
            st.error("⚠️ Generated code did not define a variable named `fig`.")

if __name__ == "__main__":
    if check_password():
        main_app()
