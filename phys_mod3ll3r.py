# streamlit run phys_modeller.py
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(layout="wide", page_title="GenAI Physics Modeler", page_icon="⚛️")

PRICING = {"gpt-4o":{"input":2.5,"output":10},"gpt-4o-mini":{"input":0.15,"output":0.6},
           "grok-4-1-fast-reasoning":{"input":0.2,"output":0.5},"grok-4-0709":{"input":3,"output":15},
           "grok-2-1212":{"input":2,"output":10},"grok-3":{"input":3,"output":15}}

SCENARIOS = {"Custom":"","Rotating Sphere with Gas":"A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container.",
             "Rarefied Gas Spin-Up":"Simulate the transfer of conserved momentum from a rotating solid disk to gas molecules immediately adjacent to it in a vacuum. The gas should gradually spin up due to wall collisions.",
             "Solar System w/ Comet":"A solar system simulation with a static yellow sun, 3 orbiting planets at different distances/speeds, and a comet passing through on a hyperbolic trajectory.",
             "Damped Pendulum":"A 3D visualization of a simple pendulum with damping. Show the pendulum bob swinging in 3D space and trace its path color-coded by velocity.",
             "Lorenz Attractor":"Simulate the Lorenz attractor (chaotic system). Visualize the trajectory of a point over time in 3D space, leaving a trail."}

for k,v in [("history",[]),("prompt",SCENARIOS["Rotating Sphere with Gas"])]: st.session_state.setdefault(k,v)

def get_secret(k): return next((st.secrets[i] for i in [k,k.upper(),k.lower()] if i in st.secrets), None)

def check_password():
    pwd = get_secret("app_password") or st.stop()
    def v(): st.session_state.auth = st.session_state.pwd == pwd
    if "auth" not in st.session_state:
        st.text_input("Password", type="password", key="pwd", on_change=v); return False
    if not st.session_state.auth:
        st.text_input("Password", type="password", key="pwd", on_change=v); st.error("Wrong"); return False
    return True

def execute_safe_code(code, g):
    blocked = ['os','sys','subprocess','shutil','requests','socket','pickle','ctypes','urllib','pathlib']
    def ri(name,*a,**kw): raise ImportError() if name.split('.')[0] in blocked else __import__(name,*a,**kw)
    safe = __builtins__.copy() if isinstance(__builtins__, dict) else vars(__builtins__).copy()
    for x in ['eval','exec','open','compile','__import__']: safe.pop(x,None)
    safe['__import__'] = ri
    g['__builtins__'] = safe
    try: exec(code,g); return True,None
    except Exception as e: return False,str(e)

def get_system_prompt():
    return """You are a perfect Python code generator for 3D physics animations using only numpy and plotly.graph_objects.
RULES:
- Output ONLY raw Python code. No markdown.
- Start with: import numpy as np; import plotly.graph_objects as go
- Pre-compute 60–90 frames → fig.frames
- Final: fig = go.Figure(...)
- NEVER define updatemenus or sliders
- ALWAYS set fixed axis ranges
- Full numpy vectorization only"""

def generate_simulation(prompt,key,url,model):
    messages = [{"role":"system","content":get_system_prompt()},{"role":"user","content":prompt}]
    total_in = total_out = 0
    for _ in range(3):
        resp = openai.OpenAI(api_key=key,base_url=url or None).chat.completions.create(model=model,messages=messages,temperature=0.2,max_tokens=4096)
        raw = resp.choices[0].message.content
        p = resp.usage.prompt_tokens if resp.usage else 0
        c = resp.usage.completion_tokens if resp.usage else 0
        total_in += p; total_out += c
        code = re.sub(r"^```.*?```$", "", raw, flags=re.DOTALL).strip()
        g = {"np":np,"go":go}
        ok, err = execute_safe_code(code,g)
        if ok and "fig" in g and isinstance(g["fig"],go.Figure):
            rates = PRICING.get(model,{"input":5,"output":15})
            cost = round((total_in/1e6)*rates["input"] + (total_out/1e6)*rates["output"],6)
            return code,cost
        messages += [{"role":"assistant","content":raw},{"role":"user","content":f"Fix ONLY code:\n{err or 'Issue'}"}]
    raise RuntimeError("Failed after 3 tries")

def main_app():
    with st.sidebar:
        st.title("Settings")
        p = st.radio("Model",["xAI (Grok)","OpenAI"],horizontal=True)
        key = url = model = None
        if p=="xAI (Grok)":
            key=get_secret("xai_api_key"); url="https://api.x.ai/v1"
            model=st.selectbox("Model",["grok-4-1-fast-reasoning","grok-4-0709","grok-2-1212"],0)
        else:
            key=get_secret("openai_api_key")
            model=st.selectbox("Model",["gpt-4o","gpt-4o-mini"])
        if not key: st.error("API key missing")
        speed = st.slider("Speed",10,200,60,5)
        frame_dur = int(1000/speed)

        # ← COST CALCULATOR IS BACK
        total_cost = sum(h.get("cost",0) for h in st.session_state.history)
        st.metric("💰 Session Cost", f"${total_cost:.6f}")
        if st.session_state.history:
            last = st.session_state.history[-1]
            st.caption(f"Last run: ${last['cost']:.6f} ({last['model']})")

    st.title("GenAI Physics Modeler")
    st.selectbox("Scenarios",list(SCENARIOS),key="sel",
                 on_change=lambda: st.session_state.update(prompt=SCENARIOS[st.session_state.sel] if st.session_state.sel!="Custom" else ""))

    prompt = st.text_area("Physics Description",st.session_state.prompt,height=110,key="prompt_input")
    st.session_state.prompt = prompt

    if st.button("Generate",type="primary",use_container_width=True,disabled=not key):
        with st.status("Generating..."):
            code,cost = generate_simulation(prompt,key,url,model)
            st.session_state.current_code = code
            st.session_state.history.append({"code":code,"prompt":prompt,"model":model,"cost":cost})
            st.rerun()

    if st.session_state.get("current_code"):
        g = {"np":np,"go":go}
        ok,err = execute_safe_code(st.session_state.current_code,g)
        if ok and "fig" in g:
            fig: go.Figure = g["fig"]
            fig.update_layout(height=800,margin=dict(l=0,r=0,t=40,b=0),scene=dict(aspectmode='cube'),
                updatemenus=[dict(buttons=[
                    dict(label="Play",method="animate",args=[None,dict(frame=dict(duration=frame_dur,redraw=True),
                                          fromcurrent=False,mode="immediate")]),
                    dict(label="Pause",method="animate",args=[[None],dict(mode="immediate")])
                ],x=0,y=1.15,xanchor="left",yanchor="top",showactive=False)])
            if fig.frames:
                fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["loop"] = "infinite"
            fig.add_annotation(text="∞ looping ∙ auto-play",xref="paper",yref="paper",x=0.02,y=1.11,
                               showarrow=False,font=dict(size=11,color="#00ff88"),
                               bgcolor="rgba(0,0,0,0.8)",bordercolor="#00ff88",borderwidth=1,borderpad=4)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else:
            st.error(err or "No `fig`")

if __name__ == "__main__":
    if check_password():
        main_app()
