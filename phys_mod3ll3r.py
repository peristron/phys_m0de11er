# streamlit run phys_modeller.py
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
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
if "auto_play" not in st.session_state:
    st.session_state.auto_play = True
if "loop_animation" not in st.session_state:
    st.session_state.loop_animation = True
if "show_slider" not in st.session_state:
    st.session_state.show_slider = True

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
    else:
        provider_icon = "⚪"
        provider_color = "gray"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {'#e8f5e9' if provider == 'OpenAI' else '#e3f2fd'} 0%, {'#c8e6c9' if provider == 'OpenAI' else '#bbdefb'} 100%);
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        border-left: 4px solid {'#4caf50' if provider == 'OpenAI' else '#2196f3'};
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

def add_animation_controls(fig, frame_dur, loop_animation, show_slider):
    """Add play/pause buttons, restart, and optional slider to the figure."""
    
    num_frames = len(fig.frames) if hasattr(fig, 'frames') and fig.frames else 0
    
    play_args = dict(
        frame=dict(duration=frame_dur, redraw=True),
        fromcurrent=True,
        transition=dict(duration=0),
        mode="immediate"
    )
    
    buttons = [
        dict(
            label="▶️ Play",
            method="animate",
            args=[None, play_args]
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
        direction="left"
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
                font=dict(size=12)
            ),
            transition=dict(duration=0),
            ticklen=4
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
    """Render Plotly figure as HTML with embedded auto-play and loop JavaScript."""
    
    # Convert figure to HTML (without full HTML wrapper)
    fig_html = fig.to_html(
        include_plotlyjs='cdn',
        full_html=False,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'responsive': True
        }
    )
    
    # JavaScript for auto-play and looping
    animation_script = f"""
    <script>
    (function() {{
        const FRAME_DURATION = {frame_dur};
        const SHOULD_LOOP = {str(loop_animation).lower()};
        const SHOULD_AUTOPLAY = {str(auto_play).lower()};
        const TOTAL_FRAMES = {num_frames};
        
        let plotDiv = null;
        let isAnimating = false;
        let loopEnabled = SHOULD_LOOP;
        
        function findPlotDiv() {{
            // Find the Plotly graph div
            const divs = document.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
            for (let div of divs) {{
                if (div._fullLayout) {{
                    return div;
                }}
            }}
            return null;
        }}
        
        function getButtons() {{
            return document.querySelectorAll('.updatemenu-button');
        }}
        
        function clickPlay() {{
            const buttons = getButtons();
            if (buttons.length > 0) {{
                buttons[0].click();
                isAnimating = true;
                return true;
            }}
            return false;
        }}
        
        function clickRestart() {{
            const buttons = getButtons();
            if (buttons.length >= 3) {{
                buttons[2].click();
                return true;
            }}
            return false;
        }}
        
        function setupLoopHandler() {{
            if (!plotDiv || !loopEnabled) return;
            
            // Remove any existing handler
            if (plotDiv._loopHandler) {{
                plotDiv.removeListener('plotly_animated', plotDiv._loopHandler);
            }}
            
            // Create new handler
            plotDiv._loopHandler = function() {{
                if (loopEnabled) {{
                    setTimeout(function() {{
                        clickRestart();
                        setTimeout(function() {{
                            clickPlay();
                        }}, 100);
                    }}, 50);
                }}
            }};
            
            plotDiv.on('plotly_animated', plotDiv._loopHandler);
        }}
        
        function initialize() {{
            plotDiv = findPlotDiv();
            
            if (!plotDiv) {{
                setTimeout(initialize, 100);
                return;
            }}
            
            // Wait for Plotly to fully initialize
            if (typeof plotDiv.on !== 'function') {{
                setTimeout(initialize, 100);
                return;
            }}
            
            // Setup loop handler
            if (SHOULD_LOOP) {{
                setupLoopHandler();
            }}
            
            // Auto-play
            if (SHOULD_AUTOPLAY) {{
                setTimeout(function() {{
                    clickPlay();
                }}, 300);
            }}
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
    
    # Combine HTML and script
    full_html = f"""
    <div style="width: 100%; height: {height}px;">
        {fig_html}
    </div>
    {animation_script}
    """
    
    # Render using components.html
    components.html(full_html, height=height + 50, scrolling=False)

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
            choice = st.selectbox("Grok Model", list(XAI_MODELS.keys()))
            model_name = XAI_MODELS[choice]
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
                provider_icon = "🟢" if last_info['provider'] == "OpenAI" else "🔵"
                
                st.markdown(f"{provider_icon} **{last['model']}**")
                st.markdown(f"💵 Cost: **${last['cost']:.5f}**")
                
                if 'prompt_tokens' in last and last['prompt_tokens']:
                    st.markdown(f"📥 Input: {last['prompt_tokens']:,} tokens")
                    st.markdown(f"📤 Output: {last['completion_tokens']:,} tokens")

        st.divider()
        st.subheader("📜 Recent History")
        for i, item in enumerate(reversed(st.session_state.history[:5])):
            hist_info = get_pricing_info(item['model'])
            hist_icon = "🟢" if hist_info['provider'] == "OpenAI" else "🔵"
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
                    provider_icon = "🟢" if last_info['provider'] == "OpenAI" else "🔵"
                    st.markdown(f"**Provider:** {provider_icon} {last_info['provider']}")
                    st.markdown(f"**Model:** `{last['model']}`")
                    st.markdown(f"**Rates:** ${last_info['input']:.2f} input / ${last_info['output']:.2f} output per 1M tokens")

        exec_globals = {"np": np, "go": go}
        success, error = execute_safe_code(st.session_state.current_code, exec_globals)
        
        if success and "fig" in exec_globals:
            fig = exec_globals["fig"]
            
            # Add animation controls to the figure
            fig, num_frames = add_animation_controls(
                fig,
                frame_dur,
                st.session_state.loop_animation,
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
            
            # Render the figure with auto-play and loop functionality
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
