# app.py -- SkillMap AI MVP with Gamification + Gemini helpers (Streamlit)
import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
# for resume parsing
import io
import pdfplumber
import docx
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
# For Cover Letter Feature
from fpdf import FPDF
import base64
import unicodedata

# ---------------- AI imports ----------------
try:
    import google.generativeai as genai
    from google import genai as genai_client
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False

# ---------- CONFIG ----------
load_dotenv()  # reads .env for GOOGLE_API_KEY
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "skillmap.db")
SKILLS_CSV = os.path.join(DATA_DIR, "skills.csv")
COURSES_CSV = os.path.join(DATA_DIR, "courses.csv")
QUIZ_JSON = os.path.join(DATA_DIR, "quiz_bank.json")
# cache directories for AI outputs
AI_QUIZ_CACHE = os.path.join(DATA_DIR, "ai_quizzes.json")
AI_PLAN_DIR = os.path.join(DATA_DIR, "ai_plans")
os.makedirs(AI_PLAN_DIR, exist_ok=True)

# configure Gemini if available
if AI_AVAILABLE:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if API_KEY:
        genai.configure(api_key=API_KEY)
        client = genai_client.Client(api_key=API_KEY) 
    else:
        AI_AVAILABLE = False

# FIX: Changed model name to a stable, supported identifier
MODEL_NAME = "gemini-2.5-flash"

# FIX: Define USER_ID globally for demo user access
USER_ID = 1

# ---------- UTILITIES ----------
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
ensure_data_dir()

def normalize_text(t: str) -> str:
    return re.sub(r'[^a-z0-9 ]', ' ', (t or "").lower())

def load_skills():
    if not os.path.exists(SKILLS_CSV): return []
    return [line.strip().lower() for line in open(SKILLS_CSV, encoding='utf-8') if line.strip()]

def load_courses():
    if not os.path.exists(COURSES_CSV):
        return pd.DataFrame(columns=["skill","provider","title","url"])
    return pd.read_csv(COURSES_CSV)

def load_quizbank():
    if not os.path.exists(QUIZ_JSON): return {}
    return json.load(open(QUIZ_JSON, encoding='utf-8'))

def extract_skills_from_text(text, skills):
    text = normalize_text(text)
    found = []
    for s in skills:
        if s in text:
            found.append(s)
    return sorted(set(found))

# Resume Parsing functions
def parse_resume(uploaded_file):
    """ Accepts a Streamlit UploadedFile and returns extracted plain text. """
    if uploaded_file is None: return ""
    fname = uploaded_file.name.lower()
    data = uploaded_file.read()
    if fname.endswith(".txt"):
        try: return data.decode("utf-8", errors="ignore")
        except: return data.decode("latin-1", errors="ignore")
    if fname.endswith(".docx"):
        try:
            doc = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print("DOCX parse error:", e)
            return ""
    if fname.endswith(".pdf"):
        try:
            text_pages = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt: text_pages.append(txt)
            all_text = "\n".join(text_pages).strip()
            if all_text: return all_text
        except Exception: pass
        try: 
            reader = PdfReader(io.BytesIO(data))
            pages_text = [p.extract_text() or "" for p in reader.pages]
            all_text = "\n".join(pages_text).strip()
            if all_text: return all_text
        except Exception: pass
        try: 
            ocr_texts = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    img = page.to_image(resolution=150).original
                    ocr_texts.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_texts)
        except Exception as e:
            print("OCR failed or pytesseract not available:", e)
            return ""
    try: return data.decode("utf-8", errors="ignore")
    except: return data.decode("latin-1", errors="ignore")

# FIX: Moved PII/Sanitization functions here so they are defined before ATS calls.
def strip_pii(text):
    text = re.sub(r'\S+@\S+\.\S+', '[email]', text)
    text = re.sub(r'\+?\d[\d\s\-\(\)]{6,}\d', '[phone]', text)
    return text

def sanitize_for_ats(text):
    t = text or ""
    t = re.sub(r'\S+@\S+\.\S+', '[email]', t)
    t = re.sub(r'\+?\d[\d\s\-\(\)]{6,}\d', '[phone]', t)
    return t[:4000]
# END OF FIX MOVEMENT

# ---------- AI Helper functions ----------
# FIX: All AI functions now accept the 'client' object as the first parameter.

def genai_generate(client, prompt, max_output_tokens=512, temperature=0.0):
    """Call Gemini and return text, or None on error."""
    if not AI_AVAILABLE:
        return None
    try:
        # FINAL FIX: Use keyword arguments exclusively to avoid positional argument conflict
        resp = client.models.generate_content(
            contents=prompt,
            model=MODEL_NAME, # Explicitly passed as keyword
            config={
                "candidate_count": 1,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens
            }
        )
        
        text = getattr(resp, "text", None)
        if text: return text
        return str(resp)
    except Exception as e:
        print("Gemini error:", e)
        return None

def genai_generate_json(client, prompt, schema_example, max_tokens=512, temperature=0.0):
    """Ask model to return JSON matching example. Returns parsed JSON or None."""
    if not AI_AVAILABLE: return None
    try:
        example = json.dumps(schema_example, indent=2)
        wrapper = ("Return ONLY valid JSON that matches the example schema exactly. "
                   "Do NOT include any explanation.\n\n"
                   f"EXAMPLE:\n{example}\n\nPROMPT:\n{prompt}")
        # FIX: Pass client object
        raw = genai_generate(client, wrapper, max_output_tokens=max_tokens, temperature=temperature)
        
        if not raw: return None
        
        # FIX: Robustly find JSON structure even if wrapped in markdown/text
        raw_text = raw.strip()
        
        if raw_text.startswith('```json'):
            # Strip markdown code block wrappers
            json_text = raw_text.strip('`').strip('json').strip()
        else:
            # Fallback to finding surrounding brackets
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start == -1: # Try array structure if object fails
                start = raw_text.find('[')
                end = raw_text.rfind(']')

            if start != -1 and end != -1:
                json_text = raw_text[start:end+1]
            else:
                return None # Failed to find valid JSON structure
        
        return json.loads(json_text)
    except Exception as e:
        print("GenAI JSON parse error:", e)
        return None

def generate_quiz_for_skill(client, skill_name, num_questions=10): # FIX: Default changed to 10
    """Returns list of questions (question, options, correct)."""
    cache = {}
    if os.path.exists(AI_QUIZ_CACHE): cache = json.load(open(AI_QUIZ_CACHE, encoding='utf-8'))
    if skill_name in cache: return cache[skill_name]
    
    prompt = f"""Generate {num_questions} multiple-choice questions to test a learner on the skill: "{skill_name}".
For each question return: - question: short question text (max 120 chars) - options: list of 4 answer options - correct: the index (0..3) of the correct option
Return a JSON array of questions."""
    schema_example = [{"question": "Example: What does SQL stand for?",
                       "options": ["Structured Query Language", "Simple Query Language", "Sequential Query Language", "Server Query Language"],
                       "correct": 0}]
    
    # FIX: Pass client object
    j = genai_generate_json(client, prompt, schema_example, max_tokens=600, temperature=0.2)
    
    if not j or not isinstance(j, list): return None
    
    cleaned = []
    for item in j[:num_questions]:
        if all(k in item for k in ("question","options","correct")):
            cleaned.append({"question": item["question"], "options": item["options"],
                            "correct": int(item["correct"]) if isinstance(item["correct"], int) else 0})
    
    if cleaned:
        cache[skill_name] = cleaned
        with open(AI_QUIZ_CACHE, "w", encoding="utf-8") as f: json.dump(cache, f, indent=2)
        qb = load_quizbank()
        qb.setdefault(skill_name, cleaned)
        with open(QUIZ_JSON, "w", encoding='utf-8') as f: json.dump(qb, f, indent=2)
        return cleaned
    return None

def generate_learning_path(client, user_profile_text, skill_name, target_role=None, weeks=4):
    """Returns structured learning plan dict or None."""
    keyname = f"plan_{skill_name.replace(' ','_')}.json"
    path = os.path.join(AI_PLAN_DIR, keyname)
    if os.path.exists(path): return json.load(open(path, encoding='utf-8'))
    
    prompt = f"""You are an expert learning coach. Create a {weeks}-week practical learning plan for the skill: "{skill_name}".
User profile: {user_profile_text}
If target role is provided, tailor the plan to that role: {target_role}
For each week provide 2-4 actionable goals and 1-3 recommended resources (title + short url if possible).
Return JSON with keys: skill, summary, estimated_hours, weekly_plan (array), assessment."""
    schema_example = {"skill": skill_name, "summary": "One-line summary", "estimated_hours": 20,
        "weekly_plan": [{"week": 1, "goals": ["..."], "resources": ["..."]}],
        "assessment": "One sentence"}
    
    # FIX: Pass client object
    j = genai_generate_json(client, prompt, schema_example, max_tokens=600, temperature=0.2)
    
    if j:
        with open(path, "w", encoding='utf-8') as f: json.dump(j, f, indent=2)
        return j
    return None

def ats_match_and_suggestions(client, resume_text, job_description_text, top_n=3):
    """Returns dict: {score:int, explanation:str, suggestions:[str]} or None on error."""
    resume_clean = sanitize_for_ats(resume_text) 
    job_clean = sanitize_for_ats(job_description_text)
    
    prompt = f"""You are an ATS expert. Compare resume and job description.
Return JSON:{{ "score": <0-100 integer>, "explanation": "one paragraph", "suggestions": ["s1","s2","s3"] }}
Resume:{resume_clean}
Job Description:{job_clean}
Return only JSON."""
    example = {"score": 70, "explanation": "short", "suggestions": ["s1","s2","s3"]}
    
    # FIX: Pass client object
    j = genai_generate_json(client, prompt, example, max_tokens=400, temperature=0.0)
    return j

# ---------- DATABASE & GAMIFICATION (Unchanged Logic) ----------
def ensure_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, points INTEGER DEFAULT 0, level TEXT DEFAULT 'Novice')""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_course (id INTEGER PRIMARY KEY, user_id INTEGER, skill TEXT, provider TEXT, title TEXT, url TEXT, status TEXT, progress INTEGER, enrolled_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_quiz (id INTEGER PRIMARY KEY, user_id INTEGER, skill TEXT, score INTEGER, taken_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS skill_ver (id INTEGER PRIMARY KEY, user_id INTEGER, skill TEXT, final_score INTEGER, status TEXT, verified_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS points_log (id INTEGER PRIMARY KEY, user_id INTEGER, points INTEGER, reason TEXT, timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS badges (id INTEGER PRIMARY KEY, code TEXT, title TEXT, description TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_badges (id INTEGER PRIMARY KEY, user_id INTEGER, badge_id INTEGER, awarded_at TEXT)""")
    conn.commit()
    return conn

conn = ensure_db()
c = conn.cursor()

def add_demo_user():
    c.execute("SELECT id FROM users WHERE id=1")
    if not c.fetchone():
        c.execute("INSERT INTO users(id,name,email,points,level) VALUES (1,?,?,0,'Novice')", ("Demo User","demo@example.com"))
        conn.commit()
add_demo_user()

def seed_badges():
    rows = c.execute("SELECT COUNT(*) FROM badges").fetchone()[0]
    if rows == 0:
        badges = [
            ("FIRST_VERIFIED","First Skill Verified","Awarded when you verify your first skill."),
            ("QUIZ_MASTER","Quiz Master","Pass 5 quizzes (>=60%)."),
            ("COURSE_FINISHER","Course Finisher","Complete 5 courses.")
        ]
        for code, title, desc in badges: c.execute("INSERT INTO badges(code,title,description) VALUES (?,?,?)",(code,title,desc))
        conn.commit()
seed_badges()

def store_recommended_courses(user_id, missing_skills, courses_df):
    c.execute("DELETE FROM user_course WHERE user_id=? AND status IN ('pending', 'in_progress')", (user_id,))
    conn.commit() 
    for skill in missing_skills:
        matches = courses_df[courses_df['skill'].str.lower() == skill]
        if matches.empty: continue
        for _,row in matches.head(2).iterrows():
            c.execute("SELECT id FROM user_course WHERE user_id=? AND skill=? AND title=?", (user_id, skill, row['title']))
            if not c.fetchone():
                c.execute("""INSERT INTO user_course(user_id,skill,provider,title,url,status,progress,enrolled_at)
                              VALUES (?,?,?,?,?,?,?,?)""",
                          (user_id, skill, row['provider'], row['title'], row['url'], "pending", 0, None))
    conn.commit()

def get_user_courses(user_id): return pd.read_sql_query("SELECT * FROM user_course WHERE user_id=?", conn, params=(user_id,))

def update_course_status(course_id, status, progress):
    now = datetime.utcnow().isoformat()
    c.execute("UPDATE user_course SET status=?, progress=?, enrolled_at=? WHERE id=?", (status, progress, now, course_id))
    conn.commit()

def store_quiz_result(user_id, skill, score):
    now = datetime.utcnow().isoformat()
    c.execute("INSERT INTO user_quiz(user_id,skill,score,taken_at) VALUES (?,?,?,?)", (user_id, skill, score, now))
    conn.commit()

def latest_quiz_score(user_id, skill):
    row = c.execute("SELECT score FROM user_quiz WHERE user_id=? AND skill=? ORDER BY taken_at DESC LIMIT 1", (user_id, skill)).fetchone()
    return int(row[0]) if row else 0

def set_skill_verification(user_id, skill, final_score, status):
    now = datetime.utcnow().isoformat() if status == "VERIFIED" else None
    row = c.execute("SELECT id FROM skill_ver WHERE user_id=? AND skill=?", (user_id, skill)).fetchone()
    if row:
        c.execute("UPDATE skill_ver SET final_score=?, status=?, verified_at=? WHERE id=?", (final_score, status, now, row[0]))
    else:
        c.execute("INSERT INTO skill_ver (user_id,skill,final_score,status,verified_at) VALUES (?,?,?,?,?)", (user_id, skill, final_score, status, now))
    conn.commit()
    if status == "VERIFIED":
        award_points(user_id, 100, f"Skill Verified: {skill}")
        verified_count = c.execute("SELECT COUNT(*) FROM skill_ver WHERE user_id=? AND status='VERIFIED'", (user_id,)).fetchone()[0]
        if verified_count == 1: award_badge(user_id, "FIRST_VERIFIED")

def latest_skill_ver(user_id, skill):
    row = c.execute("SELECT final_score,status,verified_at FROM skill_ver WHERE user_id=? AND skill=? ORDER BY id DESC LIMIT 1", (user_id, skill)).fetchone()
    return {"final_score": row[0], "status": row[1], "verified_at": row[2]} if row else {"final_score": 0, "status": "NOT_VERIFIED", "verified_at": None}

def award_points(user_id, points, reason):
    now = datetime.utcnow().isoformat()
    c.execute("INSERT INTO points_log(user_id,points,reason,timestamp) VALUES (?,?,?,?)", (user_id, points, reason, now))
    c.execute("UPDATE users SET points = points + ? WHERE id=?", (points, user_id))
    user_points = c.execute("SELECT points FROM users WHERE id=?", (user_id,)).fetchone()[0]
    level = 'Novice'
    if user_points >= 500: level = 'Pro'
    elif user_points >= 200: level = 'Intermediate'
    c.execute("UPDATE users SET level=? WHERE id=?", (level, user_id))
    conn.commit()

def award_badge(user_id, badge_code):
    row = c.execute("SELECT id FROM badges WHERE code=?", (badge_code,)).fetchone()
    if not row: return
    badge_id = row[0]
    already = c.execute("SELECT id FROM user_badges WHERE user_id=? AND badge_id=?", (user_id, badge_id)).fetchone()
    if already: return
    now = datetime.utcnow().isoformat()
    c.execute("INSERT INTO user_badges(user_id,badge_id,awarded_at) VALUES (?,?,?)", (user_id, badge_id, now))
    conn.commit()

def get_user_profile(user_id):
    row = c.execute("SELECT id,name,email,points,level FROM users WHERE id=?", (user_id,)).fetchone()
    return {"id": row[0], "name": row[1], "email": row[2], "points": row[3], "level": row[4]} if row else None

def get_user_badges(user_id):
    rows = c.execute("""SELECT b.code,b.title,b.description,ub.awarded_at
                        FROM user_badges ub JOIN badges b ON ub.badge_id = b.id
                        WHERE ub.user_id=?""", (user_id,)).fetchall()
    return [{"code": r[0], "title": r[1], "description": r[2], "awarded_at": r[3]} for r in rows]

def compute_final_score(P, Q, R): return int(round(0.5*P + 0.4*Q + 0.1*R))
def determine_status(final_score):
    if final_score >= 75: return "VERIFIED"
    if final_score >= 50: return "IN_PROGRESS"
    return "NOT_VERIFIED"
# ---------- END DB & GAMIFICATION ----------

# --- COVER LETTER PDF UTILITIES ---
def generate_pdf(text):
    cleaned_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in cleaned_text.split('\n'): pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin-1')

def download_pdf_button(text):
    pdf_data = generate_pdf(text)
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="cover_letter.pdf">📥 Download as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
# --- END COVER LETTER UTILITIES ---


# ---------- UI STRUCTURE ----------
st.set_page_config(page_title="SkillMap AI Master App", layout="wide")

# FIX: Load data globally so it's available to all rendering functions
skills = load_skills()
courses_df = load_courses()
quiz_bank = load_quizbank()

# Sidebar setup
with st.sidebar:
    selected_page = st.selectbox("Select Application", ["SkillMap Dashboard", "Cover Letter Generator ✉️"])
    st.markdown("---")
    
    st.header("Profile")
    # FIX: USER_ID is defined globally
    profile = get_user_profile(USER_ID)
    if profile:
        st.write(f"**{profile['name']}**")
        st.write(profile['email'])
        st.write("Points:", profile['points'], " | Level:", profile['level'])
    else: st.write("No profile")
    
    st.subheader("Badges")
    badges = get_user_badges(USER_ID)
    if badges:
        for b in badges: st.write(f"🏅 {b['title']} — {b['awarded_at']}")
    else: st.write("No badges yet.")
    
    st.markdown("---")
    st.subheader("Demo Controls")
    if st.button("Reset demo DB"):
        try: conn.close()
        except: pass
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        conn = ensure_db()
        c = conn.cursor()
        add_demo_user()
        seed_badges()
        st.rerun() 
    
    st.markdown("---")
    st.subheader("AI Status")
    st.write("Gemini available:", AI_AVAILABLE)
    if AI_AVAILABLE: st.write("Model:", MODEL_NAME)

# --- MAIN PAGE RENDERING ---

# FIX: Pass all necessary global data variables as arguments to the function
def render_skillmap_dashboard(skills, courses_df, quiz_bank):
    st.title("SkillMap AI — Dashboard (with Gamification & Gemini)")

    st.header("1) Upload / Paste Resume (or use sample)")
    uploaded = st.file_uploader("Upload resume file (.txt, .pdf, .docx) or paste below", type=["txt","pdf","docx"])
    resume_text_from_file = ""
    if uploaded:
        with st.spinner("Parsing uploaded file..."):
            resume_text_from_file = parse_resume(uploaded)
            if not resume_text_from_file:
                st.warning("Could not extract text from the uploaded file.")
    resume_area = st.text_area("Or paste resume text here", value=resume_text_from_file or "", height=150)

    st.header("2) Paste Target Job Description")
    job_area = st.text_area("Job description text", height=150)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze (Extract skills & Recommend)"):
            # FIX: Use passed arguments
            user_skills = extract_skills_from_text(resume_area, skills)
            job_skills  = extract_skills_from_text(job_area, skills)
            missing = [s for s in job_skills if s not in user_skills]
            st.session_state['user_skills'] = user_skills
            st.session_state['job_skills']  = job_skills
            st.session_state['missing']     = missing
            store_recommended_courses(USER_ID, missing, courses_df)
            st.success("Analysis done. Recommendations stored.")
    with col2:
        if st.button("Upload as 'after-learning' resume (evidence)"):
            # FIX: Use passed arguments
            user_skills_after = extract_skills_from_text(resume_area, skills)
            st.session_state['user_skills_after'] = user_skills_after
            st.success("After-learning resume stored (used as R evidence).")
        
        if AI_AVAILABLE and st.button("AI: Score Resume vs Job (ATS)"):
            with st.spinner("Running ATS match (Gemini)..."):
                ats = ats_match_and_suggestions(client, resume_area, job_area) 
                if ats:
                    st.subheader("📊 ATS Score")
                    st.write("Score:", ats.get("score"))
                    st.write("Explanation:", ats.get("explanation"))
                    st.write("Suggestions:")
                    for s in ats.get("suggestions", []): st.write("•", s)
                else:
                    st.error("ATS analysis failed or AI not available. Check API key and logs.")

    st.markdown("---")
    st.header("Dashboard")

    u_sk = st.session_state.get('user_skills', [])
    j_sk = st.session_state.get('job_skills', [])
    missing = st.session_state.get('missing', [])

    if 'user_skills' not in st.session_state:
        st.session_state['user_skills'] = []; st.session_state['job_skills'] = []; st.session_state['missing'] = []
        u_sk = []; j_sk = []; missing = []

    st.subheader("Extracted Skills")
    st.write("User skills:", u_sk)
    st.write("Job skills:", j_sk)
    st.write("Missing skills:", missing)

    st.subheader("Recommended Courses")
    uc_df = get_user_courses(USER_ID)

    if missing: uc_df = uc_df[uc_df['skill'].isin(missing)]

    if uc_df.empty:
        st.info("No recommendations yet. Click Analyze to find your skill gaps.")
    else:
        for _,row in uc_df.iterrows():
            current_status = row['status']
            with st.container(border=True):
                st.markdown(f"**Skill:** {row['skill'].title()}  |  **Course:** {row['title']} ({row['provider']}) | **Status:** {current_status.upper()}")
                
                cols = st.columns([1.5, 2, 2, 1, 3]) 

                if current_status == 'pending':
                    if cols[0].button("Start Course", key=f"start_{row['id']}"):
                        update_course_status(row['id'], 'in_progress', 0) 
                        award_points(USER_ID, 10, f"Started course {row['title']}")
                        st.rerun()
                else: cols[0].write(f"Progress: {row['progress']}%")

                cols[1].link_button("Go to Course ➡️", row['url'], help="Opens the course link in a new tab.")

                if current_status != 'completed':
                    if cols[2].button("Complete (Take Quiz)", key=f"quiz_trigger_{row['id']}"):
                        st.session_state['active_quiz'] = row['skill']
                        st.session_state['course_id_to_complete'] = row['id']
                        st.rerun()
                    
                    if cols[3].button("Sync", key=f"sync_{row['id']}"):
                        award_points(USER_ID, 10, f"Simulated course progress sync for {row['title']}") 
                        newp = min(100, int(row['progress'] or 0) + 25)
                        status = "in_progress" if newp < 100 else "completed"
                        update_course_status(row['id'], status, newp)
                        st.rerun()
                else:
                    cols[2].write("COMPLETED")
                    cols[3].write("") 

                if AI_AVAILABLE and cols[4].button("AI Plan", key=f"aiplan_{row['id']}"):
                    with st.spinner("Generating AI learning plan..."):
                        profile_text = resume_area or "Learner with basic skills"
                        plan = generate_learning_path(client, profile_text, row['skill'], target_role=None, weeks=4)
                        if plan:
                            plan_path = os.path.join(AI_PLAN_DIR, f"plan_{USER_ID}_{row['skill'].replace(' ','_')}.json")
                            with open(plan_path, "w", encoding="utf-8") as f: json.dump(plan, f, indent=2)
                            st.success("AI plan generated and saved.")
                            st.json(plan)
                        else:
                            st.error("AI plan generation failed.")
                
                st.progress(int(row['progress'] or 0))


    st.markdown("---")
    st.header("Quizzes & Verification")

    for skill in missing:
        st.subheader(skill.title())
        last_score = latest_quiz_score(USER_ID, skill)
        st.write("Last quiz score:", last_score)
        
        # Check if quiz exists in the small bank or needs AI generation
        if skill in quiz_bank:
            if st.button(f"Start Quiz: {skill}", key=f"quiz_{skill}"):
                st.session_state['active_quiz'] = skill
        else:
            if AI_AVAILABLE:
                if st.button(f"AI Generate Quiz for: {skill}", key=f"genquiz_{skill}"):
                    with st.spinner("Generating quiz via Gemini..."):
                        q = generate_quiz_for_skill(client, skill, num_questions=10) # 10 questions
                        if q:
                            # FIX: Reload quiz_bank *globally* after creation
                            st.session_state['quiz_bank_reloaded'] = True
                            st.success("AI quiz generated and stored. Rerunning to load questions.")
                            st.rerun()
                        else:
                            st.error("AI quiz generation failed.")
            else:
                st.info("No quiz available for this skill. AI is disabled.")

    if st.session_state.get('active_quiz'):
        qskill = st.session_state['active_quiz']
        st.subheader(f"Quiz: {qskill}")
        # FIX: Access quiz_bank using the passed argument
        questions = quiz_bank.get(qskill, [])
        if not questions:
             st.warning("No questions available. Please use 'AI Generate Quiz' first.")
        else:
            with st.form(key=f"form_{qskill}"):
                for i,q in enumerate(questions):
                    st.write(f"Q{i+1}: {q['question']}")
                    st.radio("Choose:", q['options'], key=f"{qskill}_q_{i}")
                submitted = st.form_submit_button("Submit Quiz")
                
                if submitted:
                    correct = 0
                    for i,q in enumerate(questions):
                        chosen = st.session_state.get(f"{qskill}_q_{i}")
                        try: chosen_idx = q['options'].index(chosen)
                        except ValueError: chosen_idx = -1 
                        if chosen_idx == q['correct']: correct += 1
                            
                    score_pct = int(round(100 * correct / len(questions))) if questions else 0
                    store_quiz_result(USER_ID, qskill, score_pct)
                    
                    if score_pct >= 60:
                        award_points(USER_ID, 30, f"Quiz passed {qskill} ({score_pct}%)")
                        course_id = st.session_state.get('course_id_to_complete')
                        if course_id:
                            update_course_status(course_id, "completed", 100) 
                            award_points(USER_ID, 50, f"Completed course for skill {qskill}")
                            completed_count = c.execute("""SELECT COUNT(*) FROM user_course WHERE user_id=? AND status='completed'""", (USER_ID,)).fetchone()[0]
                            if completed_count >= 5: award_badge(USER_ID, "COURSE_FINISHER")
                            del st.session_state['course_id_to_complete'] 
                        passed_quizzes = c.execute("SELECT COUNT(*) FROM user_quiz WHERE user_id=? AND score>=60", (USER_ID,)).fetchone()[0]
                        if passed_quizzes >= 5: award_badge(USER_ID, "QUIZ_MASTER")
                        st.success(f"Quiz submitted. Score: {score_pct}%. Associated course marked completed.")
                    else:
                        st.error(f"Quiz submitted. Score: {score_pct}%. You need 60% to pass.")

                    st.session_state['active_quiz'] = None
                    st.rerun()

    st.markdown("### Compute Verification (P=progress, Q=quiz score, R=resume evidence)")
    if st.button("Recompute Verification for all missing skills"):
        uc_df = get_user_courses(USER_ID)
        for skill in missing:
            rows = uc_df[uc_df['skill'].str.lower() == skill]
            P = int(rows['progress'].max()) if not rows.empty else 0
            Q = latest_quiz_score(USER_ID, skill)
            R = 100 if st.session_state.get('user_skills_after') and skill in st.session_state['user_skills_after'] else 0
            final = compute_final_score(P, Q, R)
            status = determine_status(final)
            set_skill_verification(USER_ID, skill, final, status)
        st.success("Verification computed.")
        st.rerun()

    st.markdown("### Skill Verification Records")
    for skill in missing:
        v = latest_skill_ver(USER_ID, skill)
        st.write(skill.title(), "| Final Score:", v['final_score'], "| Status:", v['status'], "| Verified at:", v['verified_at'])

    st.markdown("---")
    st.caption("Demo steps: Analyze → (AI: ATS / AI Plan / Generate Quiz) → Start/Complete course → Take quiz → Recompute Verification.")


def render_cover_letter_generator():
    # Custom CSS styling
    st.markdown("""
        <style>
            .title { text-align: center; font-size: 38px; font-weight: 700; color: #4B8BBE; margin-bottom: 10px;}
            .stButton>button { background-color: #4B8BBE; color: white; font-weight: bold; border-radius: 10px; padding: 10px 24px; font-size: 16px; }
            .footer { text-align: center; font-size: 13px; color: #666; margin-top: 30px; }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 class='title'>✉️ PitchPerfectAI – AI Cover Letter Generator</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Form for user input
    with st.form("cover_letter_form"):
        job_title = st.text_input("Job Title", placeholder="e.g. Software Engineer")
        company = st.text_input("Company Name", placeholder="e.g. Google")
        experience = st.text_input("Years of Experience", placeholder="e.g. 3")
        key_skills = st.text_area("Key Skills", placeholder="e.g. Python, Data Analysis, APIs")
        achievements = st.text_area("Achievements (optional)", placeholder="e.g. Increased user retention by 20%")
        tone = st.selectbox("Tone of the Letter", ["Professional", "Persuasive", "Formal", "Friendly"])
        job_description = st.text_area("Paste the Job Description", placeholder="Copy the full job posting here...")
        submitted = st.form_submit_button("Generate Cover Letter")

    # Submission handling
    if submitted:
        if not AI_AVAILABLE:
            st.error("AI is not available. Please check your GOOGLE_API_KEY setting.")
            return

        with st.spinner("✍️ Crafting your personalized cover letter..."):
            prompt = f"""
            Write a compelling cover letter for the following job application:

            Position: {job_title}
            Company: {company}
            Experience: {experience} years
            Key Skills: {key_skills}
            Achievements: {achievements if achievements else "Not specified"}
            Tone: {tone}

            Requirements:
            1. Professional business letter format
            2. Highlight experience, skills, and quantifiable achievements
            3. Use action verbs and be concise (300–400 words)
            4. Include a placeholder for the date and recipient name.

            Format the output strictly as the letter content, starting with the date line.
            """
            try:
                # FINAL FIX: Use keyword arguments to prevent positional conflict
                response = client.models.generate_content(
                    contents=prompt, # Prompt is contents=
                    model=MODEL_NAME # Model name is model=
                )
                cover_letter = response.text

                st.success("✅ Cover letter generated successfully!")
                st.text_area("📄 Generated Cover Letter", value=cover_letter, height=400)

                # PDF Download
                download_pdf_button(cover_letter)

                # ATS Match Scoring
                if job_description:
                    with st.spinner("🔍 Analyzing ATS match..."):
                        ats_prompt = f"""
                        Evaluate the following cover letter against this job description.
                        Provide: 1. An ATS match score out of 100 2. A short explanation for the score 3. Three suggestions to improve the cover letter.
                        --- COVER LETTER --- {cover_letter}
                        --- JOB DESCRIPTION --- {job_description}
                        """
                        # FINAL FIX: Use keyword arguments
                        ats_response = client.models.generate_content(
                            contents=ats_prompt, 
                            model=MODEL_NAME
                        )
                        st.markdown("---")
                        st.subheader("📊 ATS Match Analysis")
                        st.markdown(ats_response.text)
            except Exception as e:
                 st.error(f"Cover Letter generation failed. API Error: {e}")

    st.markdown("<div class='footer'>Made with ❤️ </div>", unsafe_allow_html=True)


# --- RENDER MAIN PAGE ---
if selected_page == "SkillMap Dashboard":
    # FIX: Pass global data variables into the rendering function
    render_skillmap_dashboard(skills, courses_df, quiz_bank)
elif selected_page == "Cover Letter Generator ✉️":
    render_cover_letter_generator()
