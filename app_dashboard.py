import streamlit as st
import json
import time

st.set_page_config(page_title="Llama-3 Triage Monitor", page_icon="🤖", layout="wide")
st.title("🤖 Llama-3 Triage Agent Live Monitor")
st.markdown("Watching the deterministic OpenEnv environment (`dashboard_logs.jsonl`) for live ReAct updates...")

placeholder = st.empty()

while True:
    try:
        with open("dashboard_logs.jsonl", "r") as f:
            lines = f.readlines()
            
        with placeholder.container():
            if not lines:
                st.info("Waiting for ticket environment to boot and Llama-3 to act...")
                
            for i, line in enumerate(lines):
                if not line.strip(): continue
                data = json.loads(line.strip())
                st.markdown(f"### Step {i+1}")
                st.info(f"**🧠 Reasoning:** {data.get('thought')}")
                st.success(f"**⚡ Action Taken:** `{data.get('action', {}).get('action_type')}` \n\n **Payload:** {data.get('action', {}).get('parameters')}")
                st.divider()
                
    except FileNotFoundError:
        placeholder.warning("dashboard_logs.jsonl not found. Waiting for `inference.py` to start the episode...")
        
    time.sleep(0.5)
