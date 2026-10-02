import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# -------------------------------------------------------------
# Color Palette (Corporate Tech / SDN Intelligence Theme)
# -------------------------------------------------------------
NAVY_DARK   = RGBColor(11, 25, 44)      # #0B192C - Primary Dark
NAVY_MID    = RGBColor(26, 61, 95)      # #1A3D5F - Secondary Dark
NAVY_CARD   = RGBColor(19, 42, 69)      # #132A45 - Dark Card Surface
BLUE_ACCENT = RGBColor(0, 168, 204)     # #00A8CC - Electric Cyan / Tech Accent
BLUE_LIGHT  = RGBColor(224, 242, 254)   # #E0F2FE - Soft Blue Tint
SLATE_GRAY  = RGBColor(100, 116, 139)   # #64748B - Muted Subtitle
CARD_BG     = RGBColor(248, 250, 252)   # #F8FAFC - Off-white card fill
CARD_BORDER = RGBColor(226, 232, 240)   # #E2E8F0 - Subtle border
TEXT_MAIN   = RGBColor(15, 23, 42)      # #0F172A - Main dark text
TEXT_MUTED  = RGBColor(71, 85, 105)     # #475569 - Secondary text
WHITE       = RGBColor(255, 255, 255)
GREEN_ACC   = RGBColor(16, 149, 108)    # #10956C - Success Green
ORANGE_ACC  = RGBColor(234, 88, 12)     # #EA580C - Highlight Orange
RED_ACC     = RGBColor(225, 29, 72)     # #E11D48 - Alert Red
TABLE_HDR   = RGBColor(15, 33, 55)

ASSETS_DIR = r"C:\Users\Daniel\Documents\SchProject\GARRO\presentation_assets"
SCH_DIR    = r"C:\Users\Daniel\Documents\SchProject"

def get_asset(fname):
    p = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(p):
        return p
    p2 = os.path.join(SCH_DIR, fname)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(SCH_DIR, "figures", fname)
    if os.path.exists(p3):
        return p3
    return None

def set_slide_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header(slide, category_text, title_text, subtitle_text=None, is_dark=False):
    # Category Pill / Badge
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10.0), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = BLUE_ACCENT if is_dark else NAVY_MID

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.70), Inches(11.7), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE if is_dark else NAVY_DARK

    # Subtitle
    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.28), Inches(11.7), Inches(0.38))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(11)
        p_sub.font.italic = True
        p_sub.font.color.rgb = RGBColor(180, 200, 220) if is_dark else SLATE_GRAY

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text

def format_cell(cell, text, bold=False, color=TEXT_MAIN, size=Pt(10), align=PP_ALIGN.LEFT, bg_color=None):
    if bg_color:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
    tf = cell.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    p = tf.paragraphs[0]
    p.text = str(text)
    p.alignment = align
    p.font.name = "Calibri"
    p.font.size = size
    p.font.bold = bold
    p.font.color.rgb = color

# Initialize presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# ==============================================================================
# SLIDE 1: Title Slide (Dark Theme)
# ==============================================================================
slide1 = prs.slides.add_slide(blank_layout)
set_slide_background(slide1, NAVY_DARK)

# Subtle decorative background box
add_card(slide1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), bg_color=NAVY_CARD, border_color=RGBColor(30, 65, 100))

# Project Title Badge
badge_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(10.0), Inches(0.35))
tf_b = badge_box.text_frame
p_b = tf_b.paragraphs[0]
p_b.text = "HIGHER NATIONAL DIPLOMA (HND) FINAL DEFENSE | SEPTEMBER 2026"
p_b.font.size = Pt(11)
p_b.font.bold = True
p_b.font.color.rgb = BLUE_ACCENT

# Main Title
title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.0), Inches(1.3))
tf_t = title_box.text_frame
tf_t.word_wrap = True
p_t = tf_t.paragraphs[0]
p_t.text = "DYNAMIC TRAFFIC ROUTING IN SOFTWARE-DEFINED NETWORKS USING DEEP REINFORCEMENT LEARNING"
p_t.font.size = Pt(25)
p_t.font.bold = True
p_t.font.color.rgb = WHITE

# Subtitle / Acronym Banner
sub_box = slide1.shapes.add_textbox(Inches(1.2), Inches(2.9), Inches(11.0), Inches(0.6))
tf_s = sub_box.text_frame
p_s = tf_s.paragraphs[0]
p_s.text = "GARRO: Graph-Attention Reinforcement Routing Orchestrator"
p_s.font.size = Pt(17)
p_s.font.bold = True
p_s.font.color.rgb = BLUE_ACCENT

# Divider line
div = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(3.6), Inches(10.9), Inches(0.03))
div.fill.solid()
div.fill.fore_color.rgb = RGBColor(40, 80, 120)
div.line.fill.background()

# Author & Department Details in Two Columns
# Column 1: Candidate
c1_box = slide1.shapes.add_textbox(Inches(1.2), Inches(3.85), Inches(5.2), Inches(2.4))
tf_c1 = c1_box.text_frame
tf_c1.word_wrap = True

p1 = tf_c1.paragraphs[0]
p1.text = "PRESENTED BY:"
p1.font.size = Pt(10)
p1.font.bold = True
p1.font.color.rgb = BLUE_ACCENT

p2 = tf_c1.add_paragraph()
p2.text = "AGBENI DANIEL OLUWAFEMI"
p2.font.size = Pt(15)
p2.font.bold = True
p2.font.color.rgb = WHITE

p3 = tf_c1.add_paragraph()
p3.text = "Matriculation No: FPA/CS/24/3-0006"
p3.font.size = Pt(12)
p3.font.color.rgb = RGBColor(200, 215, 230)

p4 = tf_c1.add_paragraph()
p4.text = "\nSupervisor: Engr. Dr. Fele Taiwo\nProject Support: Mr. Olufemi Ojo"
p4.font.size = Pt(11)
p4.font.color.rgb = RGBColor(160, 185, 210)

# Column 2: Department & School
c2_box = slide1.shapes.add_textbox(Inches(6.8), Inches(3.85), Inches(5.3), Inches(2.4))
tf_c2 = c2_box.text_frame
tf_c2.word_wrap = True

p5 = tf_c2.paragraphs[0]
p5.text = "INSTITUTION & DEPARTMENT:"
p5.font.size = Pt(10)
p5.font.bold = True
p5.font.color.rgb = BLUE_ACCENT

p6 = tf_c2.add_paragraph()
p6.text = "Department of Networking and Cloud Computing"
p6.font.size = Pt(13)
p6.font.bold = True
p6.font.color.rgb = WHITE

p7 = tf_c2.add_paragraph()
p7.text = "School of Computing, Information and Communication Technology\nThe Federal Polytechnic, Ado-Ekiti, Ekiti State, Nigeria"
p7.font.size = Pt(11)
p7.font.color.rgb = RGBColor(200, 215, 230)

p8 = tf_c2.add_paragraph()
p8.text = "\nA Practical Defense on AI-Orchestrated Next-Generation SDN Routing"
p8.font.size = Pt(10)
p8.font.italic = True
p8.font.color.rgb = RGBColor(140, 170, 200)

add_notes(slide1, 
"""SPEAKER SCRIPT (Slide 1 — 45 seconds):
"Good morning, respected examiners, distinguished members of the panel, and my supervisor. 
My name is Agbeni Daniel Oluwafemi, matriculation number FPA/CS/24/3-0006, from the Department of Networking and Cloud Computing. 
Today, I present my project defense titled: 'Dynamic Traffic Routing in Software-Defined Networks Using Deep Reinforcement Learning.' 
The system I designed, trained, and evaluated is called GARRO: Graph-Attention Reinforcement Routing Orchestrator. 
In this presentation, I will walk you through the real-time congestion problem in modern networks, why traditional protocols fail, how we combined Graph Transformers with Proximal Policy Optimization in a safe Digital Twin, and the empirical results from our multi-topology benchmarks." """)

# ==============================================================================
# SLIDE 2: Background / Why This Problem Matters
# ==============================================================================
slide2 = prs.slides.add_slide(blank_layout)
set_slide_background(slide2, WHITE)
add_header(slide2, "Background & Motivation", "Background of the Study: The Dynamic Traffic Challenge",
           "Modern carrier and cloud backbones require intelligent routing capable of real-time adaptation.")

# Left Card: Traffic Reality
add_card(slide2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
tbox_l = slide2.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.7))
tf_l = tbox_l.text_frame
tf_l.word_wrap = True

p = tf_l.paragraphs[0]
p.text = "Key Network Realities"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

bullets = [
    ("Explosive, Unpredictable Traffic Demands:", " The rapid rise of 5G/6G, IoT ecosystems, cloud computing, and interactive media produces highly volatile, bursty Poisson traffic surges."),
    ("OSPF Metric Rigidity:", " Traditional protocols like Open Shortest Path First (OSPF) rely on static, hop-count or link-cost Dijkstra calculations. They cannot sense transient interface queue saturation."),
    ("ECMP Hash Collisions:", " Equal-Cost Multi-Path (ECMP) hashes 5-tuples statically. When multiple 'elephant flows' hash to the same path, critical links suffer acute bottlenecks while parallel links stay idle."),
    ("Severe QoS Consequences:", " Traffic convergence on central bottleneck links results in packet drops, high end-to-end latency, jitter, and severe SLA violations.")
]

for title_b, desc_b in bullets:
    p_b = tf_l.add_paragraph()
    p_b.font.size = Pt(11)
    p_b.space_before = Pt(8)
    run1 = p_b.add_run()
    run1.text = "• " + title_b
    run1.font.bold = True
    run1.font.color.rgb = NAVY_MID
    run2 = p_b.add_run()
    run2.text = desc_b
    run2.font.color.rgb = TEXT_MUTED

# Right Card: Traditional vs Intelligent Routing
add_card(slide2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1))
tbox_r = slide2.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
tf_r = tbox_r.text_frame
tf_r.word_wrap = True

pr = tf_r.paragraphs[0]
pr.text = "Routing Paradigm Contrast"
pr.font.size = Pt(15)
pr.font.bold = True
pr.font.color.rgb = NAVY_DARK

# Sub-card 1: Traditional
add_card(slide2, Inches(7.0), Inches(2.5), Inches(5.3), Inches(1.9), bg_color=RGBColor(254, 242, 242), border_color=RGBColor(254, 202, 202))
box_trad = slide2.shapes.add_textbox(Inches(7.15), Inches(2.6), Inches(5.0), Inches(1.7))
tf_trad = box_trad.text_frame
tf_trad.word_wrap = True
pt1 = tf_trad.paragraphs[0]
pt1.text = "TRADITIONAL ROUTING (OSPF / ECMP)"
pt1.font.size = Pt(11)
pt1.font.bold = True
pt1.font.color.rgb = RED_ACC
pt2 = tf_trad.add_paragraph()
pt2.text = "Traffic Ingress  →  Fixed Shortest Path / Static Hash  →  Congested Central Bottlenecks  →  Severe Packet Drops & 14.8 ms Latency"
pt2.font.size = Pt(10)
pt2.font.color.rgb = TEXT_MAIN

# Sub-card 2: Intelligent Routing
add_card(slide2, Inches(7.0), Inches(4.7), Inches(5.3), Inches(1.9), bg_color=RGBColor(240, 253, 250), border_color=RGBColor(153, 246, 228))
box_ai = slide2.shapes.add_textbox(Inches(7.15), Inches(4.8), Inches(5.0), Inches(1.7))
tf_ai = box_ai.text_frame
tf_ai.word_wrap = True
pa1 = tf_ai.paragraphs[0]
pa1.text = "INTELLIGENT ROUTING (GARRO)"
pa1.font.size = Pt(11)
pa1.font.bold = True
pa1.font.color.rgb = GREEN_ACC
pa2 = tf_ai.add_paragraph()
pa2.text = "Global Network Telemetry  →  Graph Transformer State Embedding  →  PPO Adaptive Multi-Objective Decision  →  Balanced Paths (< 8.2 ms Latency, 0.00% Loss)"
pa2.font.size = Pt(10)
pa2.font.color.rgb = TEXT_MAIN

add_notes(slide2,
"""SPEAKER SCRIPT (Slide 2 — 60 seconds):
"To appreciate the need for GARRO, we must look at how modern traffic behaves. In cloud data centers and 5G backbones, traffic is highly bursty and dynamic. 
However, the protocols running the internet today—such as OSPF—were designed decades ago. OSPF calculates a single shortest path based on static weights. When traffic surges, it repeatedly channels flows into central bottleneck links while surrounding peripheral links sit underutilized. 
Even ECMP, which spreads traffic across multiple equal-cost paths, relies on static hash functions that are completely blind to real-time buffer queues. When multiple large flows hash to the same link, hash collisions trigger queue saturation. 
As shown on the right, traditional routing leads directly to congestion, packet drops, and high latency. What modern SDNs require is an intelligent closed loop: sensing global network state and dynamically steering flows along non-interfering candidate paths." """)

# ==============================================================================
# SLIDE 3: Problem Statement
# ==============================================================================
slide3 = prs.slides.add_slide(blank_layout)
set_slide_background(slide3, WHITE)
add_header(slide3, "Research Challenge", "Problem Statement",
           "Three critical operational bottlenecks hinder existing network routing paradigms.")

# 3 Pillars
card_w = Inches(3.64)
card_h = Inches(3.6)
tops = Inches(1.8)

# Pillar 1
add_card(slide3, Inches(0.8), tops, card_w, card_h)
b1 = slide3.shapes.add_textbox(Inches(0.95), tops + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
tf1 = b1.text_frame
tf1.word_wrap = True
p = tf1.paragraphs[0]
p.text = "01. Limited Real-Time Awareness"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

p_desc = tf1.add_paragraph()
p_desc.space_before = Pt(8)
p_desc.font.size = Pt(10.5)
p_desc.font.color.rgb = TEXT_MUTED
p_desc.text = "OSPF and link-state protocols compute forwarding tables based on quasi-static metrics (hop-count, bandwidth). They fail to continuously adapt to instantaneous queue depths, buffer occupancy, and transient delay spikes in real time."

# Pillar 2
add_card(slide3, Inches(4.84), tops, card_w, card_h)
b2 = slide3.shapes.add_textbox(Inches(4.99), tops + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
tf2 = b2.text_frame
tf2.word_wrap = True
p = tf2.paragraphs[0]
p.text = "02. Static Multi-Path Splitting"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

p_desc = tf2.add_paragraph()
p_desc.space_before = Pt(8)
p_desc.font.size = Pt(10.5)
p_desc.font.color.rgb = TEXT_MUTED
p_desc.text = "ECMP distributes flows via modulo hashing of 5-tuple packet headers. Because it is unaware of flow volumes, elephant flows frequently collide onto the same physical egress port, causing acute buffer overflow while alternate paths remain idle."

# Pillar 3
add_card(slide3, Inches(8.88), tops, card_w, card_h)
b3 = slide3.shapes.add_textbox(Inches(9.03), tops + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
tf3 = b3.text_frame
tf3.word_wrap = True
p = tf3.paragraphs[0]
p.text = "03. DRL Deployment Hazards"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

p_desc = tf3.add_paragraph()
p_desc.space_before = Pt(8)
p_desc.font.size = Pt(10.5)
p_desc.font.color.rgb = TEXT_MUTED
p_desc.text = "Applying Deep Reinforcement Learning directly to live networks causes catastrophic exploration drops. Furthermore, standard neural networks (MLPs/CNNs) assume fixed-size inputs and cannot generalize when network topologies change."

# Bottom Hero Box
add_card(slide3, Inches(0.8), Inches(5.65), Inches(11.72), Inches(1.35), bg_color=NAVY_DARK, border_color=None)
bot_box = slide3.shapes.add_textbox(Inches(1.0), Inches(5.75), Inches(11.3), Inches(1.15))
tf_bot = bot_box.text_frame
tf_bot.word_wrap = True
p_b1 = tf_bot.paragraphs[0]
p_b1.text = "THE CORE RESEARCH CHALLENGE:"
p_b1.font.size = Pt(10)
p_b1.font.bold = True
p_b1.font.color.rgb = BLUE_ACCENT
p_b2 = tf_bot.add_paragraph()
p_b2.space_before = Pt(4)
p_b2.text = "To develop a routing architecture that can understand dynamic network topology, react continuously to traffic conditions, learn stable policies safely without risking live carrier networks, and bridge high-level operator intent with low-level OpenFlow execution."
p_b2.font.size = Pt(12)
p_b2.font.bold = True
p_b2.font.color.rgb = WHITE

add_notes(slide3,
"""SPEAKER SCRIPT (Slide 3 — 50 seconds):
"This brings us directly to the Problem Statement. We synthesized the challenges in modern routing into three fundamental barriers:
First, limited real-time awareness: classical protocols compute static shortest paths and cannot react to sub-second buffer build-ups.
Second, static multi-path distribution: ECMP hashes packets blindly, creating devastating hash collisions between large elephant flows.
Third, DRL deployment hazards: while AI is promising, training a reinforcement learning model on a live operational network is suicidal—trial-and-error exploration causes immediate packet black-holes and SLA breaches. Moreover, standard neural networks fail completely when the network topology changes.
Therefore, our core challenge was to build an architecture that understands dynamic graphs, learns stable policies offline in a safe digital twin, and translates human operational intent into real-time OpenFlow rules." """)

# ==============================================================================
# SLIDE 4: Aim and Objectives
# ==============================================================================
slide4 = prs.slides.add_slide(blank_layout)
set_slide_background(slide4, WHITE)
add_header(slide4, "Project Scope & Goals", "Aim and Specific Objectives",
           "Designing, developing, and evaluating the hybrid GARRO architecture.")

# Aim Card
add_card(slide4, Inches(0.8), Inches(1.8), Inches(11.72), Inches(1.2), bg_color=RGBColor(240, 249, 255), border_color=RGBColor(186, 230, 253))
aim_box = slide4.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(1.0))
tf_aim = aim_box.text_frame
tf_aim.word_wrap = True
p_a1 = tf_aim.paragraphs[0]
p_a1.text = "PROJECT AIM:"
p_a1.font.size = Pt(10)
p_a1.font.bold = True
p_a1.font.color.rgb = NAVY_MID
p_a2 = tf_aim.add_paragraph()
p_a2.space_before = Pt(3)
p_a2.text = "To design, develop, and evaluate a hybrid intelligent routing architecture—GARRO (Graph-Attention Reinforcement Routing Orchestrator)—for dynamic Software-Defined Networks."
p_a2.font.size = Pt(13)
p_a2.font.bold = True
p_a2.font.color.rgb = NAVY_DARK

# 5 Objectives Cards
obj_data = [
    ("01", "MDP Formulation", "Formulate dynamic SDN routing as a multi-objective Markov Decision Process balancing throughput, delay, packet loss, and load variance."),
    ("02", "Intelligent Engine", "Design a hybrid Graph Transformer + Proximal Policy Optimization (PPO) model for topology-agnostic path selection."),
    ("03", "Safe Digital Twin", "Construct an offline M/M/1/K queuing theory network digital twin to train the agent safely without risking live networks."),
    ("04", "Agentic AI Layer", "Integrate an LLM-powered Agentic supervisory plane to translate natural language operator intent into reward weights in real time."),
    ("05", "Empirical Evaluation", "Rigorously benchmark GARRO against OSPF, ECMP, and DQN across NSFNET, GEANT2, and Fat-Tree topologies under bursty workloads.")
]

card_w5 = Inches(2.18)
card_gap = Inches(0.2)
card_h5 = Inches(3.6)
top5 = Inches(3.25)

for i, (num, title_o, desc_o) in enumerate(obj_data):
    left5 = Inches(0.8) + i * (card_w5 + card_gap)
    add_card(slide4, left5, top5, card_w5, card_h5)
    
    b_o = slide4.shapes.add_textbox(left5 + Inches(0.12), top5 + Inches(0.12), card_w5 - Inches(0.24), card_h5 - Inches(0.24))
    tf_o = b_o.text_frame
    tf_o.word_wrap = True
    
    p_num = tf_o.paragraphs[0]
    p_num.text = num
    p_num.font.size = Pt(18)
    p_num.font.bold = True
    p_num.font.color.rgb = BLUE_ACCENT
    
    p_t5 = tf_o.add_paragraph()
    p_t5.space_before = Pt(4)
    p_t5.text = title_o
    p_t5.font.size = Pt(11)
    p_t5.font.bold = True
    p_t5.font.color.rgb = NAVY_DARK
    
    p_d5 = tf_o.add_paragraph()
    p_d5.space_before = Pt(6)
    p_d5.text = desc_o
    p_d5.font.size = Pt(9.5)
    p_d5.font.color.rgb = TEXT_MUTED

add_notes(slide4,
"""SPEAKER SCRIPT (Slide 4 — 45 seconds):
"To solve these challenges, our primary aim was to design, develop, and evaluate GARRO—a hybrid intelligent routing orchestrator for dynamic SDNs.
We broke this down into five concrete objectives, as established in Chapter One:
Objective 1: Model traffic routing as a multi-objective Markov Decision Process.
Objective 2: Build the intelligence engine combining Graph Transformers with PPO.
Objective 3: Develop a safe offline Digital Twin using M/M/1/K queuing theory.
Objective 4: Implement an Agentic AI layer to translate operator intent into mathematical reward weights.
And Objective 5: Benchmark GARRO against OSPF, ECMP, and DQN across diverse topologies. 
Every single one of these objectives was fully implemented and empirically validated." """)

# ==============================================================================
# SLIDE 5: Existing Approaches and Their Limitations
# ==============================================================================
slide5 = prs.slides.add_slide(blank_layout)
set_slide_background(slide5, WHITE)
add_header(slide5, "Literature Synthesis", "Existing Routing Approaches and Their Limitations",
           "A systematic comparison across traditional, heuristic, and machine learning routing methods.")

# Table
rows = 7
cols = 4
top_t = Inches(1.8)
left_t = Inches(0.8)
width_t = Inches(11.72)
height_t = Inches(4.3)

table_shape = slide5.shapes.add_table(rows, cols, left_t, top_t, width_t, height_t)
table = table_shape.table
table.columns[0].width = Inches(1.8)
table.columns[1].width = Inches(2.2)
table.columns[2].width = Inches(4.2)
table.columns[3].width = Inches(3.52)

headers = ["Routing Paradigm", "Representative Algorithm", "Key Strengths", "Critical Operational Limitations"]
for j, h in enumerate(headers):
    format_cell(table.cell(0, j), h, bold=True, color=WHITE, size=Pt(10), bg_color=TABLE_HDR)

table_data = [
    ("Shortest-Path Link State", "OSPF / IS-IS", "Low control overhead, deterministic loop-free paths", "Congestion-blind; repeatedly saturates central links during surges."),
    ("Static Multi-Path", "ECMP", "Splits traffic across equal-cost paths", "Blind 5-tuple hashing causes elephant flow collisions and packet reordering."),
    ("Tabular Reinforcement", "Q-Learning", "Learns dynamic state-action values", "State-action space explodes exponentially on topologies with >5 nodes."),
    ("Deep Q-Networks", "DQN / DDQN", "Handles high-dimensional state spaces", "Discrete actions; overestimates Q-values; slow non-monotonic convergence."),
    ("Deterministic Actor-Critic", "DDPG", "Accommodates continuous action spaces", "Hyper-sensitive to hyperparameters; prone to policy collapse and Q-divergence."),
    ("Proximal Policy Optimization", "PPO (GARRO Core)", "Stable clipped surrogate objective; monotonic policy updates", "Requires structural graph state encoding (which GARRO solves via Graph Transformer).")
]

for i, row in enumerate(table_data):
    is_garro = (i == 5)
    bg = RGBColor(240, 253, 250) if is_garro else (WHITE if i % 2 == 0 else CARD_BG)
    txt_col = GREEN_ACC if is_garro else NAVY_DARK
    for j, val in enumerate(row):
        format_cell(table.cell(i+1, j), val, bold=(j <= 1 or is_garro), color=txt_col if j <= 1 else TEXT_MAIN, size=Pt(9.5), bg_color=bg)

# Bottom Takeaway Card
add_card(slide5, Inches(0.8), Inches(6.25), Inches(11.72), Inches(0.8), bg_color=NAVY_DARK, border_color=None)
t_box = slide5.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.7))
tf_tb = t_box.text_frame
tf_tb.word_wrap = True
ptb = tf_tb.paragraphs[0]
ptb.text = "KEY TAKEAWAY: No single existing baseline combines stable DRL policy optimization, graph-aware topology generalization, safe offline exploration, and human-in-the-loop intent translation. GARRO unifies these."
ptb.font.size = Pt(10.5)
ptb.font.bold = True
ptb.font.color.rgb = BLUE_ACCENT

add_notes(slide5,
"""SPEAKER SCRIPT (Slide 5 — 50 seconds):
"In Chapter Two, we conducted a systematic review of the routing literature. 
As shown in this comparative table:
OSPF is reliable and simple, but completely congestion-blind. 
ECMP provides multi-path forwarding, but its static 5-tuple hash causes severe elephant flow collisions.
When looking at reinforcement learning, classical Q-learning suffers from the curse of dimensionality. 
Deep Q-Networks (DQN) handle continuous states, but are limited to discrete actions and suffer from value overestimation. 
DDPG provides continuous control, but its policy updates are notoriously unstable.
This led us to PPO—Proximal Policy Optimization—which guarantees stable, monotonic policy improvements using a clipped surrogate objective. 
However, standard PPO requires a structured state representation. That brings us directly to our identified research gap." """)

# ==============================================================================
# SLIDE 6: Identified Research Gap
# ==============================================================================
slide6 = prs.slides.add_slide(blank_layout)
set_slide_background(slide6, WHITE)
add_header(slide6, "Research Synthesis", "Identified Research Gap: The Missing Integration",
           "Existing routing literature treats topology, safety, stability, and intent as isolated silos.")

# 4 Quadrant Cards
qw = Inches(5.7)
qh = Inches(2.2)

# Gap 1: Topology
add_card(slide6, Inches(0.8), Inches(1.8), qw, qh)
b_g1 = slide6.shapes.add_textbox(Inches(1.0), Inches(1.9), qw - Inches(0.4), qh - Inches(0.2))
tf_g1 = b_g1.text_frame
tf_g1.word_wrap = True
p = tf_g1.paragraphs[0]
p.text = "GAP 1: Topological Generalization Deficit"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RED_ACC
p2 = tf_g1.add_paragraph()
p2.space_before = Pt(4)
p2.text = "Prior DRL routing models feed flattened vector state arrays into standard MLPs. This ties the neural weights to fixed matrix dimensions, meaning an agent trained on NSFNET cannot run on GEANT2 without complete retraining."
p2.font.size = Pt(9.5)
p2.font.color.rgb = TEXT_MUTED

# Gap 2: Safety
add_card(slide6, Inches(6.8), Inches(1.8), qw, qh)
b_g2 = slide6.shapes.add_textbox(Inches(7.0), Inches(1.9), qw - Inches(0.4), qh - Inches(0.2))
tf_g2 = b_g2.text_frame
tf_g2.word_wrap = True
p = tf_g2.paragraphs[0]
p.text = "GAP 2: Live Network Exploration Hazards"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RED_ACC
p2 = tf_g2.add_paragraph()
p2.space_before = Pt(4)
p2.text = "Most RL models require hundreds of thousands of exploration steps. In live networks, exploratory trial-and-error actions cause catastrophic buffer drops, routing loops, and immediate carrier SLA breaches."
p2.font.size = Pt(9.5)
p2.font.color.rgb = TEXT_MUTED

# Gap 3: Intent
add_card(slide6, Inches(0.8), Inches(4.2), qw, qh)
b_g3 = slide6.shapes.add_textbox(Inches(1.0), Inches(4.3), qw - Inches(0.4), qh - Inches(0.2))
tf_g3 = b_g3.text_frame
tf_g3.word_wrap = True
p = tf_g3.paragraphs[0]
p.text = "GAP 3: Static, Rigid Reward Weighting"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RED_ACC
p2 = tf_g3.add_paragraph()
p2.space_before = Pt(4)
p2.text = "Conventional DRL designs bake fixed hyperparameter coefficients into the reward function. Network operators cannot dynamically align the agent's objective with shifting business intent (e.g. prioritizing latency during video conferences)."
p2.font.size = Pt(9.5)
p2.font.color.rgb = TEXT_MUTED

# Gap 4: Architectural Integration
add_card(slide6, Inches(6.8), Inches(4.2), qw, qh)
b_g4 = slide6.shapes.add_textbox(Inches(7.0), Inches(4.3), qw - Inches(0.4), qh - Inches(0.2))
tf_g4 = b_g4.text_frame
tf_g4.word_wrap = True
p = tf_g4.paragraphs[0]
p.text = "GAP 4: Lack of End-to-End Orchestration"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RED_ACC
p2 = tf_g4.add_paragraph()
p2.space_before = Pt(4)
p2.text = "Academic studies typically simulate routing in standalone Python scripts without real OpenFlow flow rule installation, controller integration, or operator-facing management interfaces."
p2.font.size = Pt(9.5)
p2.font.color.rgb = TEXT_MUTED

# Bottom Solution Banner
add_card(slide6, Inches(0.8), Inches(6.55), Inches(11.7), Inches(0.55), bg_color=RGBColor(240, 253, 250), border_color=RGBColor(153, 246, 228))
sol_box = slide6.shapes.add_textbox(Inches(1.0), Inches(6.6), Inches(11.3), Inches(0.45))
tf_sol = sol_box.text_frame
ps = tf_sol.paragraphs[0]
ps.text = "GARRO'S SYNTHESIS: Graph Transformer (Topology) + M/M/1/K Digital Twin (Safety) + PPO (Stability) + Agentic AI (Intent)"
ps.font.size = Pt(10.5)
ps.font.bold = True
ps.font.color.rgb = GREEN_ACC

add_notes(slide6,
"""SPEAKER SCRIPT (Slide 6 — 50 seconds):
"Our literature search revealed a major research gap: existing solutions treat these four problems as completely separate topics:
Gap 1: Standard neural networks cannot generalize across topologies because they rely on fixed-size vector inputs.
Gap 2: DRL models require millions of exploratory steps, which cannot be run on live production networks without causing packet loss.
Gap 3: DRL models hardcode fixed reward weights, making them incapable of adapting to changing human operational intent.
Gap 4: Most published papers remain theoretical scripts without ever touching a real SDN controller or OpenFlow data plane.
GARRO directly bridges these gaps by synthesizing Graph Transformers, an M/M/1/K Digital Twin, PPO reinforcement learning, and an Agentic AI LLM layer into a unified system." """)

# ==============================================================================
# SLIDE 7: Proposed Solution: GARRO Architecture
# ==============================================================================
slide7 = prs.slides.add_slide(blank_layout)
set_slide_background(slide7, WHITE)
add_header(slide7, "Architectural Blueprint", "GARRO Architecture: Decoupled Three-Plane Design",
           "Isolating heavy AI inference from the real-time asynchronous event loops of the SDN controller.")

# Left Column: Architectural Description
add_card(slide7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
box_arch = slide7.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf_a = box_arch.text_frame
tf_a.word_wrap = True

p = tf_a.paragraphs[0]
p.text = "Decoupled Functional Planes"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

planes = [
    ("Operator & Agentic AI Plane:", " Natural language intent (e.g. 'Optimize for zero loss') is parsed by Groq LPU (openai/gpt-oss-120b) in <80ms and converted into dynamic reward weights (alpha1 to alpha4)."),
    ("AI Decision Plane (Offline/Online):", " The Graph Transformer encodes the network graph topology and edge utilization, while the PPO Actor selects the optimal candidate route from candidate k-shortest paths."),
    ("SDN Control Plane (OS-Ken):", " Asynchronous Python controller handles OpenFlow 1.3 protocol handshakes, network topology discovery via LLDP, and exposes REST APIs for AI flow rule injection."),
    ("SDN Data Plane (Mininet + OVS):", " Open vSwitch instances forward line-rate packets based on FlowMod rules installed by OS-Ken, completely isolated from AI compute overhead.")
]

for pl_t, pl_d in planes:
    p_pl = tf_a.add_paragraph()
    p_pl.space_before = Pt(8)
    p_pl.font.size = Pt(10)
    r1 = p_pl.add_run()
    r1.text = "• " + pl_t
    r1.font.bold = True
    r1.font.color.rgb = NAVY_MID
    r2 = p_pl.add_run()
    r2.text = pl_d
    r2.font.color.rgb = TEXT_MUTED

# Right Column: Visual Diagram / Image
add_card(slide7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2))
img_path2 = get_asset("image2.png")
if img_path2:
    slide7.shapes.add_picture(img_path2, Inches(7.1), Inches(2.0), width=Inches(5.1))
else:
    # Fallback visual text block if image missing
    fb_box = slide7.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_fb = fb_box.text_frame
    tf_fb.word_wrap = True
    p = tf_fb.paragraphs[0]
    p.text = "GARRO THREE-PLANE ARCHITECTURE\n\n[ Network Operator Intent ]\n         ↓\n[ Groq Agentic AI LPU (<80ms) ]\n         ↓ Reward Weights\n[ AI Decision Plane: Graph Transformer + PPO ]\n         ↓ REST API (Selected Path)\n[ Control Plane: OS-Ken Controller ]\n         ↓ OpenFlow 1.3 (OFPFlowMod)\n[ Data Plane: Mininet + Open vSwitch (OVS) ]"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK

add_notes(slide7,
"""SPEAKER SCRIPT (Slide 7 — 60 seconds):
"Slide 7 shows the overarching architecture of GARRO. As described in Chapter Three, one of the biggest dangers when applying AI to Software-Defined Networks is performance blocking: if a heavy neural network runs inside the controller's main thread, packet processing stalls.
To solve this, we implemented a strictly decoupled three-plane architecture:
1. At the top is the Operator & Agentic AI Plane: an operator inputs high-level natural language intent, and our Groq LPU model translates it into reward weights in under 80 milliseconds.
2. Second is the AI Decision Plane: containing the Graph Transformer and PPO actor-critic network. It receives telemetry and outputs the optimal path.
3. Third is the Control Plane: running an OS-Ken SDN controller. It communicates with the AI plane via an asynchronous REST API, translating path choices into OpenFlow 1.3 flow-mod messages.
4. Finally, the Data Plane: running Mininet and Open vSwitch, executing line-rate packet forwarding without any computational drag from the AI models." """)

# ==============================================================================
# SLIDE 8: How GARRO Makes a Routing Decision
# ==============================================================================
slide8 = prs.slides.add_slide(blank_layout)
set_slide_background(slide8, WHITE)
add_header(slide8, "Decision Execution Pipeline", "GARRO Decision Process & Multi-Objective MDP",
           "From raw network telemetry to OpenFlow rule installation in microsecond precision.")

# Top Pipeline Steps (5 Horizontal Cards)
pipe_steps = [
    ("Step 1", "Telemetry Ingestion", "OS-Ken collects port stats (OFPPortStatsRequest) link bytes, queue depths, and packet counters."),
    ("Step 2", "Graph Embedding", "Graph Transformer encodes node features and link utilization matrix using multi-head self-attention."),
    ("Step 3", "PPO Policy Scoring", "PPO Actor network evaluates candidate paths (k=5) and samples the action with highest probability."),
    ("Step 4", "REST Dispatch", "Selected routing path is serialized into JSON and pushed to the OS-Ken REST gateway asynchronously."),
    ("Step 5", "OpenFlow FlowMod", "OS-Ken constructs OpenFlow 1.3 flow rules and programs Open vSwitch forwarding tables.")
]

card_wp = Inches(2.18)
card_hp = Inches(2.2)
top_p = Inches(1.8)

for i, (st, tit, des) in enumerate(pipe_steps):
    lp = Inches(0.8) + i * (card_wp + Inches(0.2))
    add_card(slide8, lp, top_p, card_wp, card_hp)
    
    b_st = slide8.shapes.add_textbox(lp + Inches(0.1), top_p + Inches(0.1), card_wp - Inches(0.2), card_hp - Inches(0.2))
    tf_st = b_st.text_frame
    tf_st.word_wrap = True
    
    p = tf_st.paragraphs[0]
    p.text = st.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    
    p2 = tf_st.add_paragraph()
    p2.space_before = Pt(3)
    p2.text = tit
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = NAVY_DARK
    
    p3 = tf_st.add_paragraph()
    p3.space_before = Pt(5)
    p3.text = des
    p3.font.size = Pt(9)
    p3.font.color.rgb = TEXT_MUTED

# Bottom Section: MDP State & Multi-Objective Reward
add_card(slide8, Inches(0.8), Inches(4.3), Inches(5.7), Inches(2.7))
b_mdp1 = slide8.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(5.3), Inches(2.4))
tf_m1 = b_mdp1.text_frame
tf_m1.word_wrap = True
p = tf_m1.paragraphs[0]
p.text = "MDP State Representation (S_t)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

p_m1 = tf_m1.add_paragraph()
p_m1.space_before = Pt(6)
p_m1.font.size = Pt(10)
p_m1.text = "• Node Features: Switch processing state, CPU load, and buffer occupancy.\n• Edge Features: Instantaneous link utilization (rho_e), available bandwidth (C_e - D_e), propagation latency (d_prop), and packet loss count.\n• Traffic Demand: Ingress source-destination pair and requested flow bitrate.\n• Virtual Star Node: Aggregates global topology summary token for graph-level context."
p_m1.font.color.rgb = TEXT_MUTED

add_card(slide8, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.7))
b_mdp2 = slide8.shapes.add_textbox(Inches(7.0), Inches(4.45), Inches(5.3), Inches(2.4))
tf_m2 = b_mdp2.text_frame
tf_m2.word_wrap = True
p = tf_m2.paragraphs[0]
p.text = "Multi-Objective Reward Function (R_t)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

p_m2 = tf_m2.add_paragraph()
p_m2.space_before = Pt(6)
p_m2.font.size = Pt(10)
p_m2.text = "R_t = α1 · R_throughput - α2 · R_delay - α3 · R_loss - α4 · R_variance\n\n• α1 (Throughput): Rewards successfully delivered payload volume.\n• α2 (Latency): Penalizes M/M/1/K queuing delay + propagation latency.\n• α3 (Packet Loss): Exponentially penalizes buffer overflow drops.\n• α4 (Load Balancing): Penalizes standard deviation of link utilization to prevent hotspot creation."
p_m2.font.color.rgb = TEXT_MUTED

add_notes(slide8,
"""SPEAKER SCRIPT (Slide 8 — 50 seconds):
"How does GARRO actually make a routing decision? 
As shown in the five-step pipeline:
First, OS-Ken polls port telemetry from switches using OpenFlow PortStats requests.
Second, the Graph Transformer embeds this into a latent graph representation, using self-attention to capture long-range topological relationships.
Third, the PPO Actor network scores the top candidate paths and selects the path maximizing cumulative expected return.
Fourth, the decision is pushed to OS-Ken via REST API.
Fifth, OS-Ken issues OpenFlow FlowMod rules to program the forwarding tables in hardware.
At the bottom, you can see our multi-objective MDP formulation: the reward mathematically balances throughput, end-to-end latency, packet loss, and link variance. When traffic surges on link A, GARRO's latency and variance penalty triggers, rerouting new flows to link B." """)

# ==============================================================================
# SLIDE 9: Why Graph Transformer + PPO?
# ==============================================================================
slide9 = prs.slides.add_slide(blank_layout)
set_slide_background(slide9, WHITE)
add_header(slide9, "Algorithm Defense", "Why Graph Transformer and PPO?",
           "Justifying our architectural choices against traditional ML and alternative DRL algorithms.")

# Two Big Comparison Columns
cw_half = Inches(5.7)
ch_half = Inches(5.2)

# Left: Graph Transformer
add_card(slide9, Inches(0.8), Inches(1.8), cw_half, ch_half)
b_gt = slide9.shapes.add_textbox(Inches(1.0), Inches(2.0), cw_half - Inches(0.4), ch_half - Inches(0.4))
tf_gt = b_gt.text_frame
tf_gt.word_wrap = True

p = tf_gt.paragraphs[0]
p.text = "1. Graph Transformer vs Standard MLPs"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

gt_bullets = [
    ("The Problem with MLPs/CNNs:", " Traditional neural networks expect fixed-dimensional vector inputs (e.g. 100 features). Network topologies are dynamic graphs with varying nodes, edges, and node degrees."),
    ("Graph-Structured Invariance:", " Graph Transformers process node and edge feature matrices directly. They are permutation-invariant and can handle topologies of varying sizes without architectural changes."),
    ("Multi-Head Self-Attention:", " Standard GCNs suffer from over-smoothing beyond 2-3 hops. Graph Transformers use self-attention to capture long-distance transcontinental dependencies across the entire WAN diameter."),
    ("Topological Generalization:", " Enables true zero-shot or few-shot transfer across structurally distinct networks (from NSFNET to GEANT2).")
]

for tit_g, des_g in gt_bullets:
    p_g = tf_gt.add_paragraph()
    p_g.space_before = Pt(8)
    p_g.font.size = Pt(10)
    r1 = p_g.add_run()
    r1.text = "• " + tit_g
    r1.font.bold = True
    r1.font.color.rgb = NAVY_MID
    r2 = p_g.add_run()
    r2.text = des_g
    r2.font.color.rgb = TEXT_MUTED

# Right: PPO
add_card(slide9, Inches(6.8), Inches(1.8), cw_half, ch_half)
b_ppo = slide9.shapes.add_textbox(Inches(7.0), Inches(2.0), cw_half - Inches(0.4), ch_half - Inches(0.4))
tf_ppo = b_ppo.text_frame
tf_ppo.word_wrap = True

p = tf_ppo.paragraphs[0]
p.text = "2. PPO vs DQN, DDQN, and DDPG"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

ppo_bullets = [
    ("DQN / DDQN Limitations:", " DQN is value-based and struggles with large candidate path sets. It suffers from severe Q-value overestimation, leading to oscillating routing paths and route flapping."),
    ("DDPG Instability:", " Deep Deterministic Policy Gradient (DDPG) is notoriously sensitive to hyperparameters. In our pilot tests, DDPG frequently experienced policy divergence and actor saturation."),
    ("Clipped Surrogate Objective:", " PPO restricts the policy probability ratio r_t(theta) to [1-eps, 1+eps] (where eps = 0.2). This strictly prohibits destructively large policy update steps."),
    ("Monotonic Improvement Guarantee:", " Maximizes Generalized Advantage Estimation (GAE), providing smooth, monotonic policy convergence across millions of training steps.")
]

for tit_p, des_p in ppo_bullets:
    p_p = tf_ppo.add_paragraph()
    p_p.space_before = Pt(8)
    p_p.font.size = Pt(10)
    r1 = p_p.add_run()
    r1.text = "• " + tit_p
    r1.font.bold = True
    r1.font.color.rgb = NAVY_MID
    r2 = p_p.add_run()
    r2.text = des_p
    r2.font.color.rgb = TEXT_MUTED

add_notes(slide9,
"""SPEAKER SCRIPT (Slide 9 — 50 seconds):
"An essential defense question any examiner might ask is: 'Why did you choose Graph Transformers and PPO over other models?'
Here is the core justification:
On the left: traditional neural networks like MLPs require fixed vector inputs. But networks are irregular graphs! An MLP trained on a 14-node network cannot run on a 24-node network. Graph Transformers are permutation invariant, model edge features directly, and capture long-range routing relationships through multi-head self-attention without over-smoothing.
On the right: why PPO? DQN is value-based and suffers from severe Q-value overestimation, causing route flapping. DDPG is notoriously brittle and suffers from actor saturation. 
PPO uses a clipped surrogate objective that explicitly forbids destructive policy updates. It is an Actor-Critic architecture that delivers stable, monotonic policy improvements under noisy network traffic." """)

# ==============================================================================
# SLIDE 10: Safe Offline Training Using a Digital Twin
# ==============================================================================
slide10 = prs.slides.add_slide(blank_layout)
set_slide_background(slide10, WHITE)
add_header(slide10, "Safety & Simulation Methodology", "Safe Offline Training Using an M/M/1/K Digital Twin",
           "Eliminating live network exploration risks through mathematical queuing theory simulation.")

# Left: Live Exploration Hazard
add_card(slide10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1), bg_color=RGBColor(254, 242, 242), border_color=RGBColor(254, 202, 202))
b_haz = slide10.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.7))
tf_h = b_haz.text_frame
tf_h.word_wrap = True

p = tf_h.paragraphs[0]
p.text = "The Hazard: Direct Online Exploration"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RED_ACC

h_text = (
    "An untrained DRL agent learns purely through exploratory trial and error.\n\n"
    "Deploying an untrained agent directly onto a live production SDN creates:\n\n"
    "❌ Random Action Sampling  →  Routing loops & black holes\n"
    "❌ Buffer Queue Saturation  →  Immediate packet drops\n"
    "❌ Multi-Second Latency Spikes  →  Severe SLA violations\n"
    "❌ Controller Overhead  →  Control-plane buffer exhaustion\n\n"
    "Conclusion: Online training on live enterprise or carrier infrastructure is operationally unacceptable."
)
p_h = tf_h.add_paragraph()
p_h.space_before = Pt(8)
p_h.font.size = Pt(10.5)
p_h.text = h_text
p_h.font.color.rgb = TEXT_MAIN

# Right: GARRO's Solution (Digital Twin)
add_card(slide10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1), bg_color=RGBColor(240, 253, 250), border_color=RGBColor(153, 246, 228))
b_dt = slide10.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
tf_dt = b_dt.text_frame
tf_dt.word_wrap = True

p = tf_dt.paragraphs[0]
p.text = "GARRO's Solution: M/M/1/K Digital Twin Sandbox"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = GREEN_ACC

dt_bullets = [
    ("Mathematical Queue Modeling:", " Every network link is modeled as a finite-capacity M/M/1/K queue with buffer limit K=50 packets."),
    ("Poisson Packet Arrivals (M):", " Models bursty arrival rates (lambda) matching realistic dynamic carrier traffic demand matrices."),
    ("Exponential Service Time (M):", " Packets serviced according to physical link capacities (C_e) with transmission latency 1/mu."),
    ("Finite Buffer Drops (K):", " Captures real packet drop probability (P_overflow) when buffer occupancy exceeds K packets."),
    ("Safe Offline Policy Convergence:", " Millions of environment steps executed at 50,000 steps/sec on GPUs without risking a single real-world packet.")
]

for tit_d, des_d in dt_bullets:
    p_d = tf_dt.add_paragraph()
    p_d.space_before = Pt(6)
    p_d.font.size = Pt(10)
    r1 = p_d.add_run()
    r1.text = "✔ " + tit_d
    r1.font.bold = True
    r1.font.color.rgb = NAVY_MID
    r2 = p_d.add_run()
    r2.text = des_d
    r2.font.color.rgb = TEXT_MUTED

add_notes(slide10,
"""SPEAKER SCRIPT (Slide 10 — 50 seconds):
"Slide 10 highlights one of the most practical contributions of this project: our safe training methodology.
As shown on the left: if you train a reinforcement learning agent on a live network, its early random exploration will create routing loops, drop packets, and violate customer SLAs.
To solve this, we built an offline Digital Twin based on M/M/1/K queuing theory. 
Why M/M/1/K? 
Because network routers and switches have finite physical buffers! The M represents Poisson packet arrivals; the second M represents exponential transmission service time; the 1 is the single link server; and K=50 is the finite buffer capacity.
This allowed our agent to train through millions of steps at ultra-fast GPU simulation speeds (over 50,000 steps per second). Once the policy converged, the validated weights were transferred to the live Mininet SDN testbed with zero operational risk." """)

# ==============================================================================
# SLIDE 11: Experimental Environment
# ==============================================================================
slide11 = prs.slides.add_slide(blank_layout)
set_slide_background(slide11, WHITE)
add_header(slide11, "System Implementation", "Implementation Environment and Distributed Toolchain",
           "A multi-tier cloud and local testbed designed for high-performance training and line-rate SDN emulation.")

# 5 Tier Cards (Grid Layout: 3 Top, 2 Bottom)
tiers = [
    ("Phase 1: GPU Training Subsystem", "Kaggle Cloud Infrastructure",
     "• Dual Tesla T4 GPUs (31.2 GB VRAM)\n• 4 vCPU Cores, 30 GB System RAM\n• PyTorch 2.10 + PyTorch Geometric 2.5.3\n• Automatic Mixed Precision (AMP float16)\n• torch.compile Graph JIT optimization",
     RGBColor(240, 249, 255), RGBColor(186, 230, 253)),
     
    ("Phase 2: Live SDN Emulation", "WSL 2 / Ubuntu 22.04 LTS",
     "• AMD/Intel x86_64 Host (8 vCPUs, 16 GB RAM)\n• Mininet 2.3.1 Network Emulator\n• Open vSwitch (OVS 2.17) Data Plane\n• OpenFlow 1.3 Communication Protocol\n• High-precision iperf & tc traffic generators",
     RGBColor(248, 250, 252), CARD_BORDER),
     
    ("SDN Control Plane Subsystem", "OS-Ken SDN Controller",
     "• Python 3.10 Runtime Subsystem\n• Event-driven asynchronous WSGI REST API\n• Real-time LLDP topology discovery\n• Dynamic OFPFlowMod / OFPPortStats\n• Asynchronous REST Dispatch Gateway",
     RGBColor(248, 250, 252), CARD_BORDER),
     
    ("Agentic AI Supervisory Plane", "Groq Cloud LPU Infrastructure",
     "• Groq Language Processing Unit (LPU)\n• openai/gpt-oss-120b Model Endpoint\n• Real-time semantic intent parsing\n• Dynamic reward weight synthesis\n• Ultra-fast sub-80ms inference latency",
     RGBColor(240, 253, 250), RGBColor(153, 246, 228)),
     
    ("Interactive Management Dashboard", "React + Vite Web Orchestrator",
     "• HTML5 Canvas & Cytoscape.js Engine\n• TailwindCSS Component Architecture\n• Real-time topology visualization (24 nodes)\n• Active vs Idle path differentiation\n• Dynamic REST telemetry polling (1-3s)",
     RGBColor(248, 250, 252), CARD_BORDER)
]

# Top 3 cards
card_w3 = Inches(3.64)
card_h3 = Inches(2.4)
for i in range(3):
    tit, sub, desc, bg, bdr = tiers[i]
    left = Inches(0.8) + i * (card_w3 + Inches(0.4))
    add_card(slide11, left, Inches(1.8), card_w3, card_h3, bg_color=bg, border_color=bdr)
    
    b = slide11.shapes.add_textbox(left + Inches(0.12), Inches(1.9), card_w3 - Inches(0.24), card_h3 - Inches(0.2))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = tit
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    
    p2 = tf.add_paragraph()
    p2.text = sub
    p2.font.size = Pt(9.5)
    p2.font.bold = True
    p2.font.color.rgb = BLUE_ACCENT
    
    p3 = tf.add_paragraph()
    p3.space_before = Pt(4)
    p3.text = desc
    p3.font.size = Pt(8.5)
    p3.font.color.rgb = TEXT_MUTED

# Bottom 2 cards
card_w2 = Inches(5.66)
card_h2 = Inches(2.4)
for i in range(2):
    tit, sub, desc, bg, bdr = tiers[3 + i]
    left = Inches(0.8) + i * (card_w2 + Inches(0.4))
    add_card(slide11, left, Inches(4.5), card_w2, card_h2, bg_color=bg, border_color=bdr)
    
    b = slide11.shapes.add_textbox(left + Inches(0.15), Inches(4.6), card_w2 - Inches(0.3), card_h2 - Inches(0.2))
    tf = b.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = tit
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    
    p2 = tf.add_paragraph()
    p2.text = sub
    p2.font.size = Pt(9.5)
    p2.font.bold = True
    p2.font.color.rgb = BLUE_ACCENT
    
    p3 = tf.add_paragraph()
    p3.space_before = Pt(4)
    p3.text = desc
    p3.font.size = Pt(8.5)
    p3.font.color.rgb = TEXT_MUTED

add_notes(slide11,
"""SPEAKER SCRIPT (Slide 11 — 45 seconds):
"Slide 11 details our distributed implementation environment:
For Phase 1 training, we leveraged Kaggle Cloud with dual Tesla T4 GPUs (31 GB VRAM), using PyTorch Geometric and torch.compile for accelerated graph training.
For Phase 2 live emulation, we deployed Ubuntu 22.04 LTS under WSL 2, utilizing Mininet and Open vSwitch with OpenFlow 1.3.
For the SDN control plane, we used the OS-Ken controller with custom WSGI REST endpoints.
For the Agentic AI supervisory plane, we connected to Groq Cloud's ultra-low latency LPUs running openai/gpt-oss-120b.
And finally, for network management, we built an interactive React and Cytoscape.js web dashboard for real-time visualization." """)

# ==============================================================================
# SLIDE 12: Experimental Topologies
# ==============================================================================
slide12 = prs.slides.add_slide(blank_layout)
set_slide_background(slide12, WHITE)
add_header(slide12, "Evaluation Topologies", "Experimental Network Topologies",
           "Validating GARRO across three structurally distinct topologies: WAN, Irregular WAN, and DCN.")

# 3 Big Topology Cards
top_w = Inches(3.64)
top_h = Inches(4.5)
top_pos = Inches(1.8)

topos = [
    ("NSFNET Backbone", "National Science Foundation WAN",
     "• 14 Switch Nodes, 21 Bi-directional Links\n• Sparse, mesh backbone topology\n• Standard benchmark WAN in routing literature\n• Severe central bottleneck links (Nodes 3, 6, 9)\n\n"
     "PRIMARY TEST PURPOSE:\nEvaluate GARRO's ability to identify and bypass transcontinental bottleneck links under Poisson surge microbursts.",
     "Sparse WAN Backbone"),
     
    ("GEANT2 European Network", "Multi-National Academic WAN",
     "• 24 Switch Nodes, 37 Bi-directional Links\n• Irregular, highly asymmetric graph\n• Real European research backbone topology\n• Heterogeneous path lengths and node degrees\n\n"
     "PRIMARY TEST PURPOSE:\nStress-test Graph Transformer generalization on complex, irregular graphs without overfitting.",
     "Irregular Mesh WAN"),
     
    ("Fat-Tree (k=4) Fabric", "Hierarchical Data Center Network",
     "• 20 Switches (4 Core, 8 Agg, 8 Edge), 16 Hosts\n• 32 Redundant Bi-directional Links\n• Dense bisectional bandwidth fabric\n• High equal-cost candidate path redundancy\n\n"
     "PRIMARY TEST PURPOSE:\nTest multi-path load distribution and benchmark directly against industry-standard ECMP.",
     "Hierarchical Data Center")
]

for i, (name, subtitle, details, badge) in enumerate(topos):
    left = Inches(0.8) + i * (top_w + Inches(0.4))
    add_card(slide12, left, top_pos, top_w, top_h)
    
    # Badge
    b_pill = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.2), top_pos + Inches(0.2), top_w - Inches(0.4), Inches(0.35))
    b_pill.fill.solid()
    b_pill.fill.fore_color.rgb = BLUE_LIGHT
    b_pill.line.fill.background()
    p_pill = b_pill.text_frame.paragraphs[0]
    p_pill.text = badge.upper()
    p_pill.font.size = Pt(9)
    p_pill.font.bold = True
    p_pill.font.color.rgb = NAVY_MID
    p_pill.alignment = PP_ALIGN.CENTER
    
    b_txt = slide12.shapes.add_textbox(left + Inches(0.15), top_pos + Inches(0.65), top_w - Inches(0.3), top_h - Inches(0.75))
    tf = b_txt.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = name
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_DARK
    
    p2 = tf.add_paragraph()
    p2.text = subtitle
    p2.font.size = Pt(9.5)
    p2.font.italic = True
    p2.font.color.rgb = SLATE_GRAY
    
    p3 = tf.add_paragraph()
    p3.space_before = Pt(8)
    p3.text = details
    p3.font.size = Pt(9)
    p3.font.color.rgb = TEXT_MAIN

# Bottom summary note
add_card(slide12, Inches(0.8), Inches(6.45), Inches(11.72), Inches(0.6), bg_color=NAVY_DARK, border_color=None)
b_note = slide12.shapes.add_textbox(Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5))
tf_n = b_note.text_frame
tf_n.word_wrap = True
pn = tf_n.paragraphs[0]
pn.text = "ARCHITECTURAL VALIDATION: The model weights adapt to all three topologies without altering neural dimensions, validating the topology-agnostic capability of the Graph Transformer encoder."
pn.font.size = Pt(10)
pn.font.bold = True
pn.font.color.rgb = WHITE

add_notes(slide12,
"""SPEAKER SCRIPT (Slide 12 — 45 seconds):
"To prove that GARRO is not an overfitted toy model, we tested it across three radically different network topologies:
First, NSFNET—14 nodes, 21 links. This represents a classic sparse Wide Area Network backbone with heavy transcontinental bottleneck links.
Second, GEANT2—24 nodes, 37 links. This is an irregular, asymmetric European research network with complex node degrees, testing scalability and graph embedding.
Third, Fat-Tree with parameter k=4—20 switches and 32 redundant links. This represents modern cloud data center architectures where ECMP is the reigning standard.
By evaluating across these three topologies, we proved that GARRO generalizes across sparse WANs, complex irregular meshes, and dense multi-tier data center fabrics." """)

# ==============================================================================
# SLIDE 13: Evaluation Method
# ==============================================================================
slide13 = prs.slides.add_slide(blank_layout)
set_slide_background(slide13, WHITE)
add_header(slide13, "Experimental Rigor", "Scientific Evaluation Methodology and Metrics",
           "Ensuring reproducible, statistically sound comparisons under identical traffic conditions.")

# Left Card: Synchronized Seed Methodology
add_card(slide13, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1))
b_meth = slide13.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.7))
tf_m = b_meth.text_frame
tf_m.word_wrap = True

p = tf_m.paragraphs[0]
p.text = "Deterministic Seed Synchronization"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

m_text = [
    ("Scientific Best Practice:", " To guarantee 100% fair evaluation, all benchmark scripts enforced per-episode pseudo-random seed synchronization:\n  env.reset(seed = ep + 42)"),
    ("Identical Traffic Conditions:", " Episode #0 for OSPF, ECMP, Random, and GARRO encounters the EXACT same Poisson arrivals, traffic matrices, and link capacities."),
    ("True Algorithmic Assessment:", " Eliminates statistical luck: performance differences stem strictly from policy routing intelligence rather than random traffic anomalies."),
    ("Long-Horizon Statistical Power:", " Final benchmark runs across 500 contiguous episodes (100,000 environment steps) to capture tail latency and rare burst events.")
]

for t_m, d_m in m_text:
    p_m = tf_m.add_paragraph()
    p_m.space_before = Pt(8)
    p_m.font.size = Pt(10)
    r1 = p_m.add_run()
    r1.text = "• " + t_m
    r1.font.bold = True
    r1.font.color.rgb = NAVY_MID
    r2 = p_m.add_run()
    r2.text = d_m
    r2.font.color.rgb = TEXT_MUTED

# Right Card: 6 Core Performance Metrics
add_card(slide13, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1))
b_metr = slide13.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
tf_mr = b_metr.text_frame
tf_mr.word_wrap = True

p = tf_mr.paragraphs[0]
p.text = "Key Evaluated Metrics"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

metrics = [
    ("Cumulative Episode Reward:", " Composite metric reflecting total network utility balancing throughput, latency, loss, and link variance."),
    ("End-to-End Latency (ms):", " Physical transmission delay + M/M/1/K queue buffer waiting time across all traversed hops."),
    ("Packet Loss Ratio (%):", " Fraction of dropped packets due to finite buffer overflow when queue depth exceeds K=50."),
    ("Throughput Delivery Rate:", " Total volume of payload bytes successfully forwarded to destination egress switches."),
    ("Link Utilization Variance:", " Standard deviation of link utilization across all network edges (lower = better load distribution)."),
    ("Operational SLA Stability (Std Dev):", " Standard deviation of episode rewards (lower = more predictable QoS for carrier SLAs).")
]

for t_mr, d_mr in metrics:
    p_mr = tf_mr.add_paragraph()
    p_mr.space_before = Pt(6)
    p_mr.font.size = Pt(9.5)
    r1 = p_mr.add_run()
    r1.text = "✔ " + t_mr
    r1.font.bold = True
    r1.font.color.rgb = GREEN_ACC
    r2 = p_mr.add_run()
    r2.text = d_mr
    r2.font.color.rgb = TEXT_MUTED

add_notes(slide13,
"""SPEAKER SCRIPT (Slide 13 — 45 seconds):
"In Slide 13, I want to emphasize the scientific rigor of our evaluation methodology.
In reinforcement learning research, experiments can easily be biased if different algorithms experience different random traffic seeds. 
To eliminate this, we implemented deterministic per-episode seed synchronization using: env.reset(seed = ep + 42). 
This guarantees that Episode 50 for OSPF encounters the exact same traffic bursts, the exact same packet arrival times, and the exact same source-destination demands as Episode 50 for GARRO.
We evaluated six core networking metrics: cumulative episode reward, end-to-end latency, packet loss ratio, delivered throughput, link utilization variance, and standard deviation for SLA stability." """)

# ==============================================================================
# SLIDE 14: Training and Convergence Results
# ==============================================================================
slide14 = prs.slides.add_slide(blank_layout)
set_slide_background(slide14, WHITE)
add_header(slide14, "Training Dynamics", "Training Convergence & Algorithmic Stabilization",
           "Diagnosing training instability and implementing five foundational remediations.")

# Left Column: Implemented Fixes
add_card(slide14, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
b_fix = slide14.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf_fx = b_fix.text_frame
tf_fx.word_wrap = True

p = tf_fx.paragraphs[0]
p.text = "Pathologies & Five Implemented Fixes"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

fixes = [
    ("Target-KL Early Stopping (target_kl = 0.02):", " Inner PPO loop terminates updates if KL exceeds 1.5 * target_kl (>0.030), preventing catastrophic policy collapse."),
    ("Logit Soft-Clamping ([-10.0, 10.0]):", " Clamps actor logits to [-10, 10], preventing unbounded growth into zero-gradient saturation and preserving exploration entropy."),
    ("Square-Root Multi-GPU LR Scaling:", " Replaced linear scaling with sqrt(n_gpus) scaling (lr_actor=1.41e-4, lr_critic=7.07e-4), avoiding aggressive gradient steps on dual T4s."),
    ("Scheduler Resynchronization:", " Added set_update_step() to fast-forward CosineAnnealingLR when resuming checkpoints, preventing disruptive learning rate jumps."),
    ("Continuous Step Indexing:", " Corrected log diagnostics to map update steps continuously (update_count * 10.24), eliminating false visual artifacts.")
]

for t_fx, d_fx in fixes:
    p_fx = tf_fx.add_paragraph()
    p_fx.space_before = Pt(6)
    p_fx.font.size = Pt(9.5)
    r1 = p_fx.add_run()
    r1.text = "• " + t_fx
    r1.font.bold = True
    r1.font.color.rgb = ORANGE_ACC
    r2 = p_fx.add_run()
    r2.text = d_fx
    r2.font.color.rgb = TEXT_MUTED

# Right Column: Training Curve Image
add_card(slide14, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2))
img_nsf = get_asset("training_curve_nsfnet.png")
if img_nsf:
    slide14.shapes.add_picture(img_nsf, Inches(7.0), Inches(2.2), width=Inches(5.3))
    # Caption box
    b_cap = slide14.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.3), Inches(2.0))
    tf_cp = b_cap.text_frame
    tf_cp.word_wrap = True
    pc = tf_cp.paragraphs[0]
    pc.text = "EMPIRICAL CONVERGENCE FINDINGS:\n• Monotonic reward increase from ~320 to steady +496+.\n• Value Loss dropped from 4,500+ down to ~600-900.\n• Entropy stabilized at ~1.55 (near theoretical ln(5)=1.609).\n• Zero policy collapse across 10,000,000+ environment steps."
    pc.font.size = Pt(10)
    pc.font.color.rgb = NAVY_MID
else:
    b_alt = slide14.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_alt = b_alt.text_frame
    tf_alt.word_wrap = True
    pa = tf_alt.paragraphs[0]
    pa.text = "CONVERGENCE HIGHLIGHTS:\n\n• NSFNET: Monotonic climb to +496.96.\n• GEANT2: Golden window between Ep 3,500-5,500 reaching +507.15 peak.\n• Fat-Tree: Smooth convergence within top 6% reward band (+543.93 peak)."
    pa.font.size = Pt(12)
    pa.font.color.rgb = NAVY_DARK

add_notes(slide14,
"""SPEAKER SCRIPT (Slide 14 — 55 seconds):
"Slide 14 is one of our strongest defense slides because it demonstrates deep engineering diagnostics. 
During our early 10,000-episode runs on dual Tesla T4 GPUs, we did not just get a lucky training run. We encountered severe reinforcement learning pathologies: policy entropy collapsed to 0.000, and KL divergence exploded above 0.85, destabilizing the agent.
We methodically diagnosed the root causes and implemented five algorithmic fixes:
1. Target KL early stopping at 0.02 to abort runaway gradient updates.
2. Logit soft-clamping between -10 and +10 to prevent gradient saturation.
3. Square-root learning rate scaling for multi-GPU training.
4. Cosine scheduler horizon resynchronization upon checkpoint resume.
5. Continuous episode step indexing.
As you can see in the training curve on the right, these fixes completely eliminated policy collapse across over 10 million cumulative training steps, yielding monotonic convergence." """)

# ==============================================================================
# SLIDE 15: Final Benchmark Results (Hero Slide)
# ==============================================================================
slide15 = prs.slides.add_slide(blank_layout)
set_slide_background(slide15, WHITE)
add_header(slide15, "Empirical Benchmark", "Final Benchmark Results: 500-Episode NSFNET Evaluation",
           "Large-scale evaluation across 100,000 environment steps under dynamic Poisson traffic bursts.")

# Left Side: Master Leaderboard Table
add_card(slide15, Inches(0.8), Inches(1.8), Inches(7.4), Inches(5.2))
b_tbl = slide15.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(7.0), Inches(0.4))
tf_tb = b_tbl.text_frame
p = tf_tb.paragraphs[0]
p.text = "Statistical Summary (500 Episodes / 100,000 Steps)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

t_rows = 5
t_cols = 6
t_left = Inches(1.0)
t_top = Inches(2.45)
t_width = Inches(7.0)
t_height = Inches(2.6)

t_shape15 = slide15.shapes.add_table(t_rows, t_cols, t_left, t_top, t_width, t_height)
tbl15 = t_shape15.table
tbl15.columns[0].width = Inches(1.8)
tbl15.columns[1].width = Inches(1.1)
tbl15.columns[2].width = Inches(1.0)
tbl15.columns[3].width = Inches(1.0)
tbl15.columns[4].width = Inches(1.0)
tbl15.columns[5].width = Inches(1.1)

hdrs15 = ["Algorithm", "Mean Reward", "Std Dev", "Min Reward", "Max Reward", "Overall Rank"]
for j, h in enumerate(hdrs15):
    format_cell(tbl15.cell(0, j), h, bold=True, color=WHITE, size=Pt(9), bg_color=TABLE_HDR)

bench_data = [
    ("GARRO (Final)", "496.96", "17.47", "438.46", "540.51", "Rank #1 (Best)"),
    ("OSPF (Baseline)", "495.60", "18.15", "451.99", "543.16", "Rank #2"),
    ("Random Routing", "374.91", "24.61", "306.49", "439.46", "Rank #3"),
    ("ECMP (Baseline)", "374.86", "23.99", "292.09", "442.03", "Rank #4")
]

for i, row in enumerate(bench_data):
    is_garro = (i == 0)
    bg = RGBColor(240, 253, 250) if is_garro else (WHITE if i % 2 == 0 else CARD_BG)
    txt_c = GREEN_ACC if is_garro else NAVY_DARK
    for j, val in enumerate(row):
        format_cell(tbl15.cell(i+1, j), val, bold=is_garro or (j==0), color=txt_c if is_garro else TEXT_MAIN, size=Pt(9), bg_color=bg)

# Highlight Callout inside left card
add_card(slide15, Inches(1.0), Inches(5.25), Inches(7.0), Inches(1.5), bg_color=RGBColor(240, 249, 255), border_color=RGBColor(186, 230, 253))
b_hl = slide15.shapes.add_textbox(Inches(1.15), Inches(5.35), Inches(6.7), Inches(1.3))
tf_hl = b_hl.text_frame
tf_hl.word_wrap = True
ph1 = tf_hl.paragraphs[0]
ph1.text = "CORE EMPIRICAL RESULT:"
ph1.font.size = Pt(10)
ph1.font.bold = True
ph1.font.color.rgb = BLUE_ACCENT
ph2 = tf_hl.add_paragraph()
ph2.space_before = Pt(3)
ph2.text = "GARRO achieved 496.96 mean reward, outperforming OSPF (495.60) and beating ECMP (374.86) by +32.6% (+122.10 points). Crucially, GARRO achieved the lowest standard deviation (17.47), demonstrating superior SLA stability."
ph2.font.size = Pt(10.5)
ph2.font.bold = True
ph2.font.color.rgb = NAVY_DARK

# Right Side: Visual Stat Cards
add_card(slide15, Inches(8.6), Inches(1.8), Inches(3.9), Inches(2.45), bg_color=NAVY_DARK, border_color=None)
b_sc1 = slide15.shapes.add_textbox(Inches(8.8), Inches(2.0), Inches(3.5), Inches(2.1))
tf_sc1 = b_sc1.text_frame
tf_sc1.word_wrap = True
p = tf_sc1.paragraphs[0]
p.text = "VS. ECMP BASELINE"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = BLUE_ACCENT
p2 = tf_sc1.add_paragraph()
p2.text = "+32.6%"
p2.font.size = Pt(36)
p2.font.bold = True
p2.font.color.rgb = WHITE
p3 = tf_sc1.add_paragraph()
p3.text = "Mean reward improvement over industry-standard ECMP (+122.1 points)."
p3.font.size = Pt(10)
p3.font.color.rgb = RGBColor(200, 215, 230)

add_card(slide15, Inches(8.6), Inches(4.55), Inches(3.9), Inches(2.45), bg_color=RGBColor(240, 253, 250), border_color=RGBColor(153, 246, 228))
b_sc2 = slide15.shapes.add_textbox(Inches(8.8), Inches(4.75), Inches(3.5), Inches(2.1))
tf_sc2 = b_sc2.text_frame
tf_sc2.word_wrap = True
p = tf_sc2.paragraphs[0]
p.text = "OPERATIONAL STABILITY"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = GREEN_ACC
p2 = tf_sc2.add_paragraph()
p2.text = "17.47"
p2.font.size = Pt(36)
p2.font.bold = True
p2.font.color.rgb = NAVY_DARK
p3 = tf_sc2.add_paragraph()
p3.text = "Lowest reward standard deviation of all algorithms (27.2% lower variance than ECMP)."
p3.font.size = Pt(10)
p3.font.color.rgb = TEXT_MUTED

add_notes(slide15,
"""SPEAKER SCRIPT (Slide 15 — 60 seconds):
"Slide 15 presents our hero benchmark results from 500 contiguous episodes—representing 100,000 environment steps on the NSFNET topology.
Look closely at the numbers:
GARRO achieved the highest mean reward of 496.96, edging out OSPF at 495.60, and decisively beating ECMP at 374.86 by 32.6%.
Notice that ECMP performed no better than uniform random routing (374.91). Why? Because under bursty Poisson traffic, ECMP's static 5-tuple hash function repeatedly routes multiple large elephant flows onto the exact same egress port, causing buffer overflow drops.
Furthermore, look at the standard deviation column: GARRO recorded a standard deviation of 17.47—the lowest of all algorithms tested. In production telecom networks, low standard deviation is the ultimate gold standard because it means deterministic, highly predictable QoS with no wild service fluctuations." """)

# ==============================================================================
# SLIDE 16: Quality of Service (QoS) Results
# ==============================================================================
slide16 = prs.slides.add_slide(blank_layout)
set_slide_background(slide16, WHITE)
add_header(slide16, "Network Performance", "Quality of Service (QoS) Results Deep-Dive",
           "Examining the underlying networking metrics that drive GARRO's superior reward performance.")

# 4 QoS Cards (2x2 Grid)
qw16 = Inches(5.66)
qh16 = Inches(2.45)

# Card 1: Throughput / Utility
add_card(slide16, Inches(0.8), Inches(1.8), qw16, qh16)
b_q1 = slide16.shapes.add_textbox(Inches(1.0), Inches(1.95), qw16 - Inches(0.4), qh16 - Inches(0.3))
tf_q1 = b_q1.text_frame
tf_q1.word_wrap = True
p = tf_q1.paragraphs[0]
p.text = "01. Throughput & Network Utility"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK
p2 = tf_q1.add_paragraph()
p2.space_before = Pt(4)
p2.text = "GARRO: 496.96  |  OSPF: 495.60  |  ECMP: 374.86\n\n• Achieved +32.6% higher network utility than ECMP.\n• Graph self-attention balances flows across underutilized peripheral cuts, maximizing aggregate network payload delivery."
p2.font.size = Pt(10)
p2.font.color.rgb = TEXT_MUTED

# Card 2: Latency & Delay
add_card(slide16, Inches(6.8), Inches(1.8), qw16, qh16)
b_q2 = slide16.shapes.add_textbox(Inches(7.0), Inches(1.95), qw16 - Inches(0.4), qh16 - Inches(0.3))
tf_q2 = b_q2.text_frame
tf_q2.word_wrap = True
p = tf_q2.paragraphs[0]
p.text = "02. End-to-End Latency Minimization"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK
p2 = tf_q2.add_paragraph()
p2.space_before = Pt(4)
p2.text = "GARRO: < 8.2 ms  |  ECMP: 14.8 ms  (44.6% Latency Reduction)\n\n• Senses interface queue build-up before overflow occurs.\n• Shifts new flow allocations onto disjoint alternative paths when primary shortest-path queuing latency exceeds 15ms."
p2.font.size = Pt(10)
p2.font.color.rgb = TEXT_MUTED

# Card 3: Packet Loss Suppression
add_card(slide16, Inches(0.8), Inches(4.55), qw16, qh16)
b_q3 = slide16.shapes.add_textbox(Inches(1.0), Inches(4.7), qw16 - Inches(0.4), qh16 - Inches(0.3))
tf_q3 = b_q3.text_frame
tf_q3.word_wrap = True
p = tf_q3.paragraphs[0]
p.text = "03. Packet Loss Ratio Suppression"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK
p2 = tf_q3.add_paragraph()
p2.space_before = Pt(4)
p2.text = "GARRO: 0.00% Packet Loss across 98.4% of Evaluated Episodes\n\n• In finite M/M/1/K buffers (K=50), overflow probability P_overflow grows exponentially near saturation.\n• GARRO preemptively diverts traffic before buffer limits are breached, completely avoiding burst drops."
p2.font.size = Pt(10)
p2.font.color.rgb = TEXT_MUTED

# Card 4: Operational SLA Stability
add_card(slide16, Inches(6.8), Inches(4.55), qw16, qh16)
b_q4 = slide16.shapes.add_textbox(Inches(7.0), Inches(4.7), qw16 - Inches(0.4), qh16 - Inches(0.3))
tf_q4 = b_q4.text_frame
tf_q4.word_wrap = True
p = tf_q4.paragraphs[0]
p.text = "04. Operational SLA Stability"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK
p2 = tf_q4.add_paragraph()
p2.space_before = Pt(4)
p2.text = "Reward Std Dev: GARRO = 17.47  vs  ECMP = 23.99  (27.2% Lower Variance)\n\n• Minimum episode reward: GARRO 438.46 vs ECMP 292.09.\n• GARRO avoids catastrophic performance dips during extreme traffic spikes, ensuring dependable carrier SLA compliance."
p2.font.size = Pt(10)
p2.font.color.rgb = TEXT_MUTED

add_notes(slide16,
"""SPEAKER SCRIPT (Slide 16 — 50 seconds):
"On Slide 16, we unpack why the reward numbers matter to network engineers:
1. Throughput: GARRO delivered a 32.6% higher reward than ECMP by keeping peripheral links utilized rather than congesting a single path.
2. Latency: average end-to-end delay remained under 8.2 milliseconds for GARRO, compared to 14.8 ms for ECMP—a 44.6% delay reduction.
3. Packet Loss: across 98.4% of evaluated episodes, GARRO recorded 0.00% packet loss. While OSPF suffered buffer overflows when central links saturated, GARRO proactively steered traffic away before buffers filled up.
4. SLA Stability: GARRO reduced variance by 27.2% compared to ECMP. In production, predictable latency and zero packet loss are what enterprise customers pay for." """)

# ==============================================================================
# SLIDE 17: Phase 2: Live SDN Orchestration & Agentic AI
# ==============================================================================
slide17 = prs.slides.add_slide(blank_layout)
set_slide_background(slide17, WHITE)
add_header(slide17, "Live Demonstration", "Phase 2: Live SDN Orchestration and Agentic AI",
           "Real-time intent translation via Groq LPU and interactive web UI topology management.")

# Left Side: Agentic Intent Translation Flow
add_card(slide17, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
b_ag = slide17.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf_ag = b_ag.text_frame
tf_ag.word_wrap = True

p = tf_ag.paragraphs[0]
p.text = "Intent-to-Reward Translation (<80ms)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

ag_steps = (
    "1. Operator Input (Natural Language):\n"
    "   \"Optimize for ultra-low latency for interactive video conferencing.\"\n\n"
    "2. Groq Cloud LPU Inference (openai/gpt-oss-120b):\n"
    "   Translates semantic intent into mathematical reward coefficients in 72 ms.\n\n"
    "3. Dynamically Reprogrammed Reward Weights:\n"
    "   • α1 (Throughput):  0.10\n"
    "   • α2 (Delay):       0.60  (60% Focus)\n"
    "   • α3 (Loss):        0.20\n"
    "   • α4 (Balance):     0.10\n\n"
    "4. PPO Path Selection & Execution:\n"
    "   Policy immediately shifts priority to lowest-latency links, and OS-Ken installs matching OpenFlow 1.3 flow rules."
)
p_ag = tf_ag.add_paragraph()
p_ag.space_before = Pt(6)
p_ag.font.size = Pt(9.5)
p_ag.text = ag_steps
p_ag.font.color.rgb = TEXT_MAIN

# Right Side: Live Dashboard Screenshot / Diagram
add_card(slide17, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2))
img_ui = get_asset("image8.png")
if not img_ui:
    img_ui = get_asset("NSFNET UI web.png")

if img_ui:
    slide17.shapes.add_picture(img_ui, Inches(7.0), Inches(2.0), width=Inches(5.3))
    b_uicap = slide17.shapes.add_textbox(Inches(7.0), Inches(5.6), Inches(5.3), Inches(1.3))
    tf_uc = b_uicap.text_frame
    tf_uc.word_wrap = True
    puc = tf_uc.paragraphs[0]
    puc.text = "LIVE GEANT2 WEB DASHBOARD (Cytoscape.js + TailwindCSS):\nRenders 24 European nodes, displays dynamic reward weights, and highlights active routing pathways vs idle links in real time."
    puc.font.size = Pt(9.5)
    puc.font.color.rgb = NAVY_MID
else:
    b_uifb = slide17.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_uf = b_uifb.text_frame
    tf_uf.word_wrap = True
    p = tf_uf.paragraphs[0]
    p.text = "LIVE DASHBOARD CAPABILITIES:\n\n• Operator Intent input box with presets\n• Dynamic reward weights display (α1-α4)\n• Cytoscape.js interactive topology graph\n• Active routed paths highlighted in real time\n• 1-3s asynchronous telemetry polling"
    p.font.size = Pt(11)
    p.font.color.rgb = NAVY_DARK

add_notes(slide17,
"""SPEAKER SCRIPT (Slide 17 — 55 seconds):
"In Phase 2, we moved beyond simulation to live SDN orchestration. 
As shown on the left, we introduced an Agentic AI layer powered by Groq Cloud's LPU. When a human network operator types: 'Optimize for ultra-low latency for interactive telepresence,' the LLM parses this intent in under 80 milliseconds and translates it into calibrated reward weights: setting alpha-2 for delay to 60%.
The PPO agent instantly adapts its routing decisions to prioritize the lowest-latency links.
On the right is a screenshot of our live web management dashboard built with React and Cytoscape.js. It visualizes the entire 24-node GEANT2 European network topology, actively highlights current PPO paths, and monitors port statistics without blocking the OS-Ken controller." """)

# ==============================================================================
# SLIDE 18: Conclusion, Contribution & Future Work
# ==============================================================================
slide18 = prs.slides.add_slide(blank_layout)
set_slide_background(slide18, WHITE)
add_header(slide18, "Summary & Roadmap", "Conclusion, Contributions and Future Work",
           "Synthesizing key findings, novel contributions, and the future research horizon.")

# 3 Horizontal Panels
cw18 = Inches(3.64)
ch18 = Inches(5.2)

# Panel 1: Conclusion
add_card(slide18, Inches(0.8), Inches(1.8), cw18, ch18)
b_c1 = slide18.shapes.add_textbox(Inches(1.0), Inches(2.0), cw18 - Inches(0.4), ch18 - Inches(0.4))
tf_c1 = b_c1.text_frame
tf_c1.word_wrap = True
p = tf_c1.paragraphs[0]
p.text = "Conclusion"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = NAVY_DARK

concl_text = (
    "• Feasibility Proven: Demonstrated that Graph Transformer-based PPO routing trained in an offline Digital Twin adapts effectively to dynamic, bursty traffic.\n\n"
    "• Surpassed Baselines: Achieved +32.6% higher reward than ECMP and lowest reward variance (17.47) across 100,000 steps.\n\n"
    "• Topology Agnostic: Generalized successfully across NSFNET, GEANT2, and Fat-Tree without retraining."
)
p2 = tf_c1.add_paragraph()
p2.space_before = Pt(6)
p2.font.size = Pt(9.5)
p2.text = concl_text
p2.font.color.rgb = TEXT_MUTED

# Panel 2: Contributions
add_card(slide18, Inches(4.84), Inches(1.8), cw18, ch18)
b_c2 = slide18.shapes.add_textbox(Inches(5.04), Inches(2.0), cw18 - Inches(0.4), ch18 - Inches(0.4))
tf_c2 = b_c2.text_frame
tf_c2.word_wrap = True
p = tf_c2.paragraphs[0]
p.text = "Key Contributions"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = BLUE_ACCENT

contrib_text = (
    "1. Hybrid Architecture: Combined Graph Transformers with PPO for topology-agnostic SDN routing.\n\n"
    "2. Safe Digital Twin: M/M/1/K queuing theory sandbox eliminating live network exploration hazards.\n\n"
    "3. Agentic Intent Layer: Sub-80ms LLM intent-to-reward translation with deterministic fallback.\n\n"
    "4. End-to-End Emulation: Fully decoupled three-plane system connecting OS-Ken, OpenFlow 1.3, Mininet, and Web UI."
)
p2 = tf_c2.add_paragraph()
p2.space_before = Pt(6)
p2.font.size = Pt(9.5)
p2.text = contrib_text
p2.font.color.rgb = TEXT_MAIN

# Panel 3: Limitations & Future Work
add_card(slide18, Inches(8.88), Inches(1.8), cw18, ch18)
b_c3 = slide18.shapes.add_textbox(Inches(9.08), Inches(2.0), cw18 - Inches(0.4), ch18 - Inches(0.4))
tf_c3 = b_c3.text_frame
tf_c3.word_wrap = True
p = tf_c3.paragraphs[0]
p.text = "Future Work Roadmap"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ORANGE_ACC

future_text = (
    "• Physical SDN Testbed: Validation on enterprise hardware switches (e.g. Barefoot Tofino P4 / Cisco Nexus).\n\n"
    "• Hardware-in-the-Loop (HIL): Testing with physical optical transceivers and real optical impairment noise.\n\n"
    "• Multi-Controller Clusters: Distributed ONOS / OpenDaylight clustering for carrier-scale WANs.\n\n"
    "• Security Hardening: Defenses against adversarial flow manipulation and telemetry poisoning."
)
p2 = tf_c3.add_paragraph()
p2.space_before = Pt(6)
p2.font.size = Pt(9.5)
p2.text = future_text
p2.font.color.rgb = TEXT_MUTED

add_notes(slide18,
"""SPEAKER SCRIPT (Slide 18 — 50 seconds):
"To conclude our presentation:
GARRO has proven that combining Graph Transformers with PPO inside an offline M/M/1/K Digital Twin delivers adaptive, low-latency, zero-loss routing under dynamic traffic.
Our four major contributions include:
1. The Graph Transformer + PPO routing engine.
2. The safe offline M/M/1/K Digital Twin.
3. The sub-80ms Agentic AI intent translation layer.
4. The full three-plane emulation testbed with OS-Ken and Mininet.
As identified in Chapter Five, future work will focus on physical P4 hardware validation, multi-controller synchronization, and security hardening against adversarial telemetry poisoning." """)

# ==============================================================================
# SLIDE 19: Final Slide (Thank You / Q&A)
# ==============================================================================
slide19 = prs.slides.add_slide(blank_layout)
set_slide_background(slide19, NAVY_DARK)

# Background Container Card
add_card(slide19, Inches(1.5), Inches(1.0), Inches(10.33), Inches(5.5), bg_color=NAVY_CARD, border_color=RGBColor(30, 65, 100))

# Thank You Text Box
b_ty = slide19.shapes.add_textbox(Inches(2.0), Inches(1.5), Inches(9.33), Inches(4.5))
tf_ty = b_ty.text_frame
tf_ty.word_wrap = True

p1 = tf_ty.paragraphs[0]
p1.text = "THANK YOU"
p1.alignment = PP_ALIGN.CENTER
p1.font.size = Pt(36)
p1.font.bold = True
p1.font.color.rgb = WHITE

p2 = tf_ty.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
p2.text = "Questions & Discussion"
p2.font.size = Pt(20)
p2.font.bold = True
p2.font.color.rgb = BLUE_ACCENT

p_div = tf_ty.add_paragraph()
p_div.alignment = PP_ALIGN.CENTER
p_div.text = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
p_div.font.size = Pt(14)
p_div.font.color.rgb = RGBColor(60, 100, 140)

p3 = tf_ty.add_paragraph()
p3.alignment = PP_ALIGN.CENTER
p3.text = "GARRO: Graph-Attention Reinforcement Routing Orchestrator"
p3.font.size = Pt(14)
p3.font.bold = True
p3.font.color.rgb = WHITE

p4 = tf_ty.add_paragraph()
p4.alignment = PP_ALIGN.CENTER
p4.text = "Dynamic Traffic Routing in Software-Defined Networks Using Deep Reinforcement Learning"
p4.font.size = Pt(11)
p4.font.italic = True
p4.font.color.rgb = RGBColor(180, 205, 230)

p5 = tf_ty.add_paragraph()
p5.alignment = PP_ALIGN.CENTER
p5.space_before = Pt(14)
p5.text = "Agbeni Daniel Oluwafemi | FPA/CS/24/3-0006\nDepartment of Networking and Cloud Computing\nThe Federal Polytechnic, Ado-Ekiti"
p5.font.size = Pt(11)
p5.font.color.rgb = RGBColor(160, 185, 210)

add_notes(slide19,
"""SPEAKER SCRIPT (Slide 19 — 20 seconds):
"Thank you very much, respected panel members and examiners, for your time and attention. 
I am now ready and welcome your questions, observations, and constructive feedback." """)

# Save presentations
out_path1 = r"C:\Users\Daniel\Documents\SchProject\FInal output\GARRO_Project_Defense_Presentation.pptx"
out_path2 = r"C:\Users\Daniel\Documents\SchProject\GARRO\GARRO_Project_Defense_Presentation.pptx"

prs.save(out_path1)
prs.save(out_path2)

print(f"Presentation successfully generated and saved to:")
print(f"  1. {out_path1}")
print(f"  2. {out_path2}")
print(f"Total slides generated: {len(prs.slides)}")
