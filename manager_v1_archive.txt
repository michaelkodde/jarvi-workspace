import requests
import base64
from playwright.sync_api import sync_playwright
import os
from datetime import datetime
import subprocess
import time
import json
import logging

# --- GOLD STANDARD: LOAD CONFIGURATION ---
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

WORKSPACE = CONFIG['workspace_path']
MODEL_STRATEGIST = CONFIG['models']['strategist']
MODEL_ANALYST = CONFIG['models']['analyst']
SEARCH_BASE = CONFIG['search_engine']

# --- GOLD STANDARD: PERSISTENT LOGGING ---
log_dir = os.path.join(WORKSPACE, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"jarvi_{datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

TICKET_PATH = os.path.join(WORKSPACE, "00_inbox", "ticket_001_investigation.md")
SKILL_BASE_PATH = os.path.join(WORKSPACE, "brain", "skills", "market_analysis_base.md")
SKILL_CHEF_PATH = os.path.join(WORKSPACE, "brain", "skills", "domain_chef.md")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(WORKSPACE, "01_proposals", f"cookware_investigation_report_{timestamp}.md")

def heartbeat(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    logging.info(msg) # Saves to the log file automatically

# Load Context
heartbeat("Manager: Loading Ticket and Skills...")
with open(TICKET_PATH, 'r') as f: ticket = f.read()
with open(SKILL_BASE_PATH, 'r') as f: skill_base = f.read()
with open(SKILL_CHEF_PATH, 'r') as f: skill_chef = f.read()

browser = None
try:
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=False, channel="msedge")
    page = browser.new_page()
    
    # --- PHASE 1: STRATEGIST ---
    heartbeat("Phase 1: Strategist generating 6 organic search queries...")
    strategy_prompt = f'''
    You are an expert investigative researcher. 
    Read the following Ticket and Skills.
    TICKET: {ticket}
    BASE PROTOCOL: {skill_base}
    CHEF OVERLAY: {skill_chef}
    
    Generate exactly 6 highly specific search queries to investigate this topic organically. 
    Include queries for Reddit opinions, material science, and brand controversies.
    Output ONLY the 6 queries. One per line. No numbers, no quotes.
    '''
    
    strategy_response = requests.post('http://localhost:11434/api/generate', json={
        "model": MODEL_STRATEGIST, "prompt": strategy_prompt, "stream": False
    })
    raw_queries = strategy_response.json().get('response', '')
    all_queries = [q.strip().strip('"').strip('-') for q in raw_queries.split('\n') if len(q.strip()) > 10][:6]
    
    if len(all_queries) < 6:
        fallback = ["site:reddit.com cookware materials toxicity", "cookware heat conductivity comparison", "cookware brand greenwashing exposed", "PFAS free cookware independent lab testing", "best non-toxic cookware professional chef reddit", "laminated bamboo cookware safety"]
        all_queries = all_queries + [f for f in fallback if f not in all_queries]
        all_queries = all_queries[:6]

    heartbeat(f"Strategist generated: {all_queries}")

    # --- PHASE 2: HANDS & VISION (Strict 1-by-1 Queue) ---
    master_context = "" 
    
    for i, query in enumerate(all_queries):
        heartbeat(f"Phase 2.{i+1}: Searching -> {query}")
        page.goto(f"{SEARCH_BASE}{query.replace(' ', '+')}")
        page.wait_for_timeout(3000) 
        screenshot = page.screenshot()
        img_base64 = base64.b64encode(screenshot).decode('utf-8')
        
        heartbeat(f"  -> Vision Model analyzing screenshot {i+1}/6...")
        
        vision_prompt = f'''
        Extract all relevant facts, brand names, user complaints, and material science data from this search result screenshot.
        Ignore ads and sponsored links. Focus on independent reviews and controversies.
        Format as a bulleted list of raw facts.
        '''
        
        vision_response = requests.post('http://localhost:11434/api/chat', json={
            "model": MODEL_ANALYST, 
            "messages": [{"role": "user", "content": vision_prompt, "images": [img_base64]}], 
            "stream": False
        })
        
        extracted_facts = vision_response.json().get('message', {}).get('content', 'No data found.')
        master_context += f"\n--- FACTS FROM SEARCH {i+1} ({query}) ---\n{extracted_facts}\n"
        heartbeat(f"  -> Extracted facts saved to memory. Image discarded.")

    # --- PHASE 3: TEXT MODEL (Final Report) ---
    heartbeat("Phase 3: Text Model writing final Product Requirements Document...")
    
    final_prompt = f'''
    You are an Investigative AI Agent.
    TICKET: {ticket}
    BASE PROTOCOL: {skill_base}
    CHEF OVERLAY: {skill_chef}

    RAW INVESTIGATIVE DATA:
    {master_context}

    Synthesize this raw data into a final Product Requirements Document (PRD).
    Remember: YOU ARE THE ANALYST. DO NOT recommend a final product to buy. 
    List the exact material, size, and features required based on the data.
    Use precise culinary terminology.
    '''
    
    final_response = requests.post('http://localhost:11434/api/generate', json={
        "model": MODEL_STRATEGIST, 
        "prompt": final_prompt, 
        "stream": False
    })
    result = final_response.json().get('response', 'Error getting response.')
    
    # 4. Save & Auto-Open
    heartbeat("Manager: Saving report...")
    with open(OUTPUT_PATH, 'w') as f:
        f.write("# Cookware Investigation Report (Iterative 6x1 Queue)\n\n")
        f.write(result)
        
    heartbeat(f"Task Complete! Opening report...")
    subprocess.Popen(['notepad.exe', OUTPUT_PATH])

except Exception as e:
    heartbeat(f"Manager Error: {e}")
    logging.exception("Critical Error occurred") # Saves the full error traceback to the log file
finally:
    if browser: browser.close()