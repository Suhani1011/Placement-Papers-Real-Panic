import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
import io
import wave
import re
import json
import tempfile
import os
import base64
import fitz
import pytesseract
import speech_recognition as sr
from PIL import Image
from docx import Document
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Placement Papers, Real Panic",
    page_icon="🎤",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "started": False,
    "completed": False,
    "question_number": 0,
    "name": "",
    "role": "",
    "language": "",
    "resume_text": "",
    "generated_questions": [],
    "detected_skills": [],
    "answers": [],
    "audio_answers": [],
    "transcripts": [],
    "english_versions": [],
    "content_scores": [],
    "delivery_scores": [],
    "delivery_metrics": [],
    "feedback": [],
    "evaluation_done": False,
    "performance_history": [],
    "recording_attempt": 0,
    "dark_mode": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# THEME
# ============================================================

def apply_theme():
    if st.session_state.dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0e1117;
                color: #ffffff;
            }
            [data-testid="stHeader"] {
                background-color: #0e1117;
            }
            .stMarkdown, .stText, label, p, h1, h2, h3, h4 {
                color: #ffffff !important;
            }
            [data-testid="stWidgetLabel"] p {
                color: #ffffff !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
                color: #111111;
            }
            [data-testid="stHeader"] {
                background-color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


apply_theme()


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=api_key)
    except Exception:
        return None


# ============================================================
# AUDIO DURATION
# ============================================================

def speak_question_in_browser(question, language, question_id):
    """
    Browser-native interviewer voice.

    Use ONE SpeechSynthesisUtterance for the whole question.
    This prevents the large gaps caused by switching between
    separate Hindi and English utterances.
    """
    question_json = json.dumps(
        question,
        ensure_ascii=False
    )

    language_json = json.dumps(
        language,
        ensure_ascii=False
    )

    html = r"""
    <style>
        html, body {
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
        }

        .speaker-wrap {
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }

        .speaker-button {
            border: none;
            background: transparent;
            font-size: 28px;
            line-height: 1;
            cursor: pointer;
            padding: 2px 4px;
            margin: 0;
            border-radius: 10px;
        }

        .speaker-button:hover {
            background: rgba(128, 128, 128, 0.15);
        }
    </style>

    <div class="speaker-wrap">
        <button
            class="speaker-button"
            title="Repeat question"
            aria-label="Repeat question"
            onclick="speakQuestion()"
        >🔊</button>
    </div>

    <script>
        const questionText = __QUESTION__;
        const selectedLanguage = __LANGUAGE__;

        function getBestVoice(lang) {
            if (!window.speechSynthesis) {
                return null;
            }

            const voices =
                window.speechSynthesis.getVoices() || [];

            const exact = voices.find(
                voice =>
                    (voice.lang || "").toLowerCase() ===
                    lang.toLowerCase()
            );

            if (exact) {
                return exact;
            }

            const prefix =
                lang.split("-")[0].toLowerCase();

            return voices.find(
                voice =>
                    (voice.lang || "")
                    .toLowerCase()
                    .startsWith(prefix)
            ) || null;
        }

        function speakQuestion() {
            if (!("speechSynthesis" in window)) {
                return;
            }

            window.speechSynthesis.cancel();

            const utterance =
                new SpeechSynthesisUtterance(
                    questionText
                );

            if (selectedLanguage === "Hindi") {
                // One Hindi utterance = smooth continuous speech.
                utterance.lang = "hi-IN";
                utterance.rate = 0.96;
                utterance.pitch = 1.0;

                const voice = getBestVoice("hi-IN");

                if (voice) {
                    utterance.voice = voice;
                }
            } else {
                // English + Hinglish.
                utterance.lang = "en-IN";
                utterance.rate = 0.96;
                utterance.pitch = 1.0;

                const voice = getBestVoice("en-IN");

                if (voice) {
                    utterance.voice = voice;
                }
            }

            window.speechSynthesis.speak(
                utterance
            );
        }

        setTimeout(
            () => speakQuestion(),
            350
        );
    </script>
    """

    html = html.replace(
        "__QUESTION__",
        question_json
    )

    html = html.replace(
        "__LANGUAGE__",
        language_json
    )

    components.html(
        html,
        height=48,
        scrolling=False
    )



def get_audio_duration(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        with wave.open(audio_file, "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate > 0:
                return frames / float(rate)
        return 0
    except Exception:
        return 0


# ============================================================
# GEMINI SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_bytes, language):
    """
    Fast speech-to-text for the interview answer.

    Normal path:
    - Google Speech Recognition through the SpeechRecognition package.
    - Hindi: hi-IN
    - English: en-IN, then en-US
    - Hinglish: hi-IN, then en-IN

    Gemini remains a fallback, so the AI pipeline is still available
    without making every answer wait for a Gemini file upload.
    """
    if not audio_bytes:
        return "", "No audio recording was received."

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(
            io.BytesIO(audio_bytes)
        ) as source:
            audio_data = recognizer.record(source)
    except Exception as e:
        return "", f"Could not read the recording: {str(e)}"

    if language == "Hindi":
        recognition_languages = ["hi-IN"]

    elif language == "English":
        recognition_languages = ["en-IN", "en-US"]

    else:
        recognition_languages = ["hi-IN", "en-IN"]

    for recognition_language in recognition_languages:
        try:
            transcript = recognizer.recognize_google(
                audio_data,
                language=recognition_language,
                show_all=False
            )

            if transcript and transcript.strip():
                return transcript.strip(), ""

        except sr.UnknownValueError:
            continue

        except sr.RequestError:
            continue

        except Exception:
            continue

    # --------------------------------------------------------
    # Gemini fallback
    # --------------------------------------------------------
    client = get_gemini_client()

    if client is not None:
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            audio_file = client.files.upload(
                file=temp_path,
                config={"mime_type": "audio/wav"}
            )

            if language == "Hindi":
                prompt = """
Transcribe this interview answer in natural Hindi using Devanagari.

Keep technical terms such as Python, SQL, Excel, Power BI, Tableau,
C++, Java, JavaScript, React, API and GitHub in English.

Preserve exactly what the candidate said.
Do not summarize.
Return ONLY the transcript.
"""

            elif language == "Hinglish":
                prompt = """
Transcribe this interview answer as natural Roman Hinglish.

Use Roman script only.
Keep technical terms in English.
Preserve exactly what the candidate said.
Do not summarize.
Return ONLY the transcript.
"""

            else:
                prompt = """
Transcribe this interview answer in clear Indian English.
Preserve exactly what the candidate said.
Do not summarize.
Return ONLY the transcript.
"""

            response = client.models.generate_content(
                model="gemini-3.5-transcribe",
                contents=[
                    prompt,
                    audio_file
                ]
            )

            transcript = (
                getattr(response, "text", None)
                or ""
            ).strip()

            if transcript:
                return transcript, ""

        except Exception:
            pass

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return "", (
        "Speech-to-text could not recognize this recording. "
        "Please speak clearly and try again."
    )




# ============================================================
# ENGLISH VERSION FOR HINDI / HINGLISH
# ============================================================

def create_english_version(transcript, language):
    if language == "English":
        return transcript, ""

    client = get_gemini_client()

    if client is None:
        return "", "Gemini API key could not be loaded."

    prompt = """
Convert the following interview answer into natural,
professional English.

Preserve the EXACT meaning.
Do not add information that the candidate did not say.
Return ONLY the English version.
"""

    full_prompt = (
        prompt
        + "\n\nCandidate answer:\n"
        + transcript
    )

    import time

    # This is a short transformation task, so use the low-latency
    # Flash-Lite model first.
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=full_prompt
            )

            english_text = (
                getattr(response, "text", None)
                or ""
            ).strip()

            if english_text:
                return english_text, ""

        except Exception as e:
            message = str(e).upper()

            if (
                "503" in message
                or "UNAVAILABLE" in message
                or "429" in message
                or "RESOURCE_EXHAUSTED" in message
            ):
                time.sleep(2 ** attempt)
                continue

    return "", (
        "English improvement is temporarily unavailable. "
        "Your original answer is still available."
    )




# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def _ocr_image(image):
    """
    OCR one resume page/image.
    Use English + Hindi + Marathi when those Tesseract models are available.
    """
    try:
        # On Windows, pytesseract needs the Tesseract executable.
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        if not getattr(pytesseract.pytesseract, "tesseract_cmd", ""):
            for candidate in possible_paths:
                if Path(candidate).exists():
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    break

        available = set(
            pytesseract.get_languages(config="")
        )

        languages = [
            lang
            for lang in ("eng", "hin", "mar")
            if lang in available
        ]

        lang_string = "+".join(languages) or "eng"

        text = pytesseract.image_to_string(
            image,
            lang=lang_string,
            config="--psm 6"
        )

        return (text or "").strip()

    except Exception:
        return ""


def _ocr_pdf(pdf_bytes):
    """
    Render PDF pages and OCR them only when normal text extraction fails.
    """
    if not pdf_bytes:
        return ""

    doc = None

    try:
        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        page_texts = []

        # Resume PDFs are usually short. Limit to 6 pages for predictable
        # processing time while still covering normal resumes.
        for page_index in range(min(len(doc), 6)):
            page = doc[page_index]

            # 180 DPI is enough for most resume text and is faster than
            # rendering extremely high-resolution images.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(180 / 72, 180 / 72),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples
            )

            text = _ocr_image(image)

            if text:
                page_texts.append(text)

        return "\n\n".join(page_texts).strip()

    except Exception:
        return ""

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _extract_docx_text(uploaded_file):
    """
    Read text-based DOCX resumes.
    """
    try:
        document = Document(
            io.BytesIO(uploaded_file.getvalue())
        )

        text_parts = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())

        # Also read tables because many resumes use table-based layouts.
        for table in document.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if cells:
                    text_parts.append(" | ".join(cells))

        return "\n".join(text_parts).strip()

    except Exception:
        return ""


def _extract_image_resume_text(uploaded_file):
    """
    OCR a JPG/PNG/WebP resume image.
    """
    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        return _ocr_image(image)

    except Exception:
        return ""


def _extract_pdf_text_local(pdf_bytes):
    """
    Try fast text extraction using PyMuPDF first, then PyPDF2.
    """
    try:
        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text_parts = []

        for page in doc:
            try:
                page_text = page.get_text("text")
            except Exception:
                page_text = ""

            if page_text:
                text_parts.append(page_text)

        doc.close()

        fitz_text = "\n".join(text_parts).strip()

        if len(re.sub(r"\s+", "", fitz_text)) >= 100:
            return fitz_text

    except Exception:
        pass

    try:
        reader = PyPDF2.PdfReader(
            io.BytesIO(pdf_bytes)
        )

        text_parts = []

        for page in reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = ""

            if page_text:
                text_parts.append(page_text)

        pypdf_text = "\n".join(text_parts).strip()

        if len(re.sub(r"\s+", "", pypdf_text)) >= 100:
            return pypdf_text

    except Exception:
        pass

    return ""


def extract_resume_text(uploaded_file):
    """
    Read the resume locally in the appropriate format.

    Supported:
    - PDF (normal text + scanned/image PDF)
    - DOCX
    - PNG/JPG/JPEG/WebP

    Gemini is NOT required for resume reading, so a resume does not
    sit on a Gemini PDF-processing spinner.
    """
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()
    suffix = Path(filename).suffix

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------
    if suffix == ".docx":
        return _extract_docx_text(uploaded_file)

    # --------------------------------------------------------
    # Image resumes
    # --------------------------------------------------------
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return _extract_image_resume_text(uploaded_file)

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------
    if suffix == ".pdf":
        pdf_bytes = uploaded_file.getvalue()

        # First: fast normal PDF text extraction.
        local_text = _extract_pdf_text_local(pdf_bytes)

        if local_text:
            return local_text

        # Second: OCR only if the PDF has no usable text layer.
        with st.spinner(
            "🔎 Scanned/image PDF detected — reading the resume..."
        ):
            ocr_text = _ocr_pdf(pdf_bytes)

        return ocr_text

    return ""



def _text_looks_usable(text):
    """
    Decide whether locally extracted PDF text is good enough to use.
    """
    if not text:
        return False

    cleaned = re.sub(r"\s+", " ", text).strip()

    # Very tiny extraction is usually not a usable resume.
    if len(cleaned) < 80:
        return False

    alphanumeric_count = len(re.findall(r"[A-Za-z0-9]", cleaned))

    # Avoid treating a badly garbled extraction as a valid resume.
    if len(cleaned) > 0:
        ratio = alphanumeric_count / len(cleaned)

        if ratio < 0.20:
            return False

    return True


def extract_resume_text(uploaded_file):
    """
    Extract resume content locally.

    Priority:
    1. PyMuPDF for normal text PDFs and complex layouts.
    2. PyPDF2 as a secondary local extractor.
    3. Tesseract OCR for scanned/image-based PDFs.
    """
    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        return ""

    # --------------------------------------------------------
    # STEP 1: PyMuPDF -- fast and robust for normal PDFs
    # --------------------------------------------------------
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []

        for page in doc:
            page_text = page.get_text("text")

            if page_text:
                text_parts.append(page_text)

        doc.close()

        fitz_text = "\n".join(text_parts).strip()

        # A normal text resume should be accepted immediately.
        if len(re.sub(r"\s+", "", fitz_text)) >= 100:
            return fitz_text

    except Exception:
        fitz_text = ""

    # --------------------------------------------------------
    # STEP 2: PyPDF2 fallback
    # --------------------------------------------------------
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        local_text_parts = []

        for page in reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = ""

            if page_text:
                local_text_parts.append(page_text)

        pypdf_text = "\n".join(local_text_parts).strip()

        if len(re.sub(r"\s+", "", pypdf_text)) >= 100:
            return pypdf_text

    except Exception:
        pypdf_text = ""

    # --------------------------------------------------------
    # STEP 3: OCR fallback for scanned/image PDFs
    # --------------------------------------------------------
    with st.spinner(
        "🔎 Scanned/image PDF detected — reading the resume..."
    ):
        ocr_text = _ocr_pdf(pdf_bytes)

    return (ocr_text or "").strip()


# ============================================================
# NORMALIZE RESUME SKILLS
# ============================================================

def detect_skills(resume_text):

    skill_aliases = {
        "Python": ["python", "python3", "py"],
        "SQL": ["sql", "structured query language"],
        "Excel": ["excel", "ms excel", "microsoft excel"],
        "Tableau": ["tableau"],
        "Power BI": ["power bi", "powerbi"],
        "Machine Learning": ["machine learning", "ml"],
        "Artificial Intelligence": ["artificial intelligence", "ai"],
        "Data Analytics": ["data analytics", "data analysis"],
        "Business Analytics": ["business analytics", "business analysis"],
        "Java": ["java"],
        "C++": ["c++", "cpp", "c plus plus"],
        "C": ["c programming", "c language"],
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3", "cascading style sheets"],
        "JavaScript": ["javascript", "java script", "js"],
        "React": ["react", "reactjs", "react.js"],
        "Node.js": ["node.js", "nodejs", "node js", "node"],
        "Django": ["django"],
        "Flask": ["flask"],
        "MongoDB": ["mongodb", "mongo db"],
        "MySQL": ["mysql", "my sql"],
        "PostgreSQL": ["postgresql", "postgres sql", "postgres"],
        "AWS": ["aws", "amazon web services"],
        "Cloud": ["cloud computing", "cloud"],
        "Git": ["git"],
        "GitHub": ["github", "git hub"],
        "MATLAB": ["matlab", "mat lab"],
        "Arduino": ["arduino"],
        "Embedded Systems": ["embedded systems", "embedded system"],
        "VLSI": ["vlsi", "very large scale integration"],
        "Communication Systems": [
            "communication systems",
            "communication system"
        ],
    }

    resume_lower = resume_text.lower()
    detected = []

    for canonical_skill, aliases in skill_aliases.items():
        for alias in aliases:
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(alias)
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, resume_lower):
                detected.append(canonical_skill)
                break

    return detected


# ============================================================
# AI-GENERATED REALISTIC INTERVIEW QUESTIONS
# ============================================================

def generate_resume_questions(role, resume_text, skills, language):

    client = get_gemini_client()

    fallback_questions = {
        "English": [
            f"Tell me about yourself and why you are interested in the {role} role.",
            "Looking at your resume, tell me about one project or experience that best demonstrates your skills.",
            f"What is one important concept that a candidate applying for a {role} position should understand?",
            "Suppose you are given a problem related to your role that you have not solved before. How would you approach it?",
            "Tell me about a challenge you faced during a project or your studies and how you handled it.",
        ],

        "Hindi": [
            f"अपने बारे में बताइए और बताइए कि आप {role} की role में क्यों interested हैं।",
            "अपने resume में दिए गए किसी एक project या experience के बारे में बताइए, और उसमें आपका contribution क्या था?",
            f"{role} role के लिए एक important technical concept समझाइए।",
            "अगर आपको अपने field में कोई ऐसी problem मिले जो आपने पहले solve नहीं की है, तो आप उसे कैसे approach करेंगे?",
            "अपने किसी project या studies के दौरान आई किसी challenge के बारे में बताइए और बताइए कि आपने उसे कैसे handle किया।",
        ],

        "Hinglish": [
            f"Apne baare mein bataiye aur explain kijiye ki aap {role} role mein interested kyun hain.",
            "Apne resume mein diye gaye kisi ek project ya experience ke baare mein bataiye jo aapki skills ko best demonstrate karta hai.",
            f"{role} role ke liye important kisi ek technical concept ko explain kijiye.",
            "Agar aapko apne field se related koi aisi problem di jaaye jo aapne pehle solve nahi ki hai, to aap usse kaise approach karenge?",
            "Apne kisi project ya studies ke dauran aayi kisi challenge ke baare mein bataiye aur explain kijiye ki aapne usse kaise handle kiya.",
        ],
    }

    fallback = fallback_questions.get(
        language,
        fallback_questions["English"]
    )

    if client is None:
        return fallback

    resume_context = resume_text[:5000]
    skills_text = (
        ", ".join(skills)
        if skills
        else "No specific predefined skills detected."
    )

    if language == "English":
        language_instruction = """
Write all questions in clear professional English.
Keep technical terms exactly as technical terms.
"""

    elif language == "Hindi":
        language_instruction = """
Write the questions in NATURAL SPOKEN HINDI suitable for an
Indian college placement interview.

Use a technical-Hindi style, NOT overly formal or Sanskritized Hindi.

IMPORTANT:
- Write the Hindi sentence structure in natural Devanagari.
- Keep common technical/interview terms in English, exactly as they
  are normally spoken in Indian placement interviews.
- Examples: resume, technical skills, project, internship, role,
  Python, SQL, C++, Java, JavaScript, React, Node.js, API,
  GitHub, Power BI, Tableau, Excel, Data Analyst, Software Engineer.
- Do NOT translate technical terms into awkward Hindi words.
- Avoid overly formal words such as "तकनीकी कौशल के आधार पर" when
  "technical skills" sounds more natural in an Indian interview.
- The result should sound like a real placement interviewer
  speaking to a college student.
- QUESTION 1 MUST be an actual question. Never make it a greeting,
  welcome message, or statement.
"""

    else:
        language_instruction = """
Write the questions in natural Roman Hinglish.

IMPORTANT:
- Keep technical terms such as C++, Python, JavaScript, CSS,
  HTML, SQL, React, Node.js, Power BI, Tableau, API, GitHub,
  Java, Excel, etc. exactly as technical terms.
- Do not replace technical terms with strange Hindi translations.
- Make the questions sound like an actual Indian placement interview.
"""

    prompt = f"""
You are an experienced campus placement interviewer.

Create EXACTLY 5 interview questions for a student.

ROLE:
{role}

DETECTED RESUME SKILLS:
{skills_text}

RESUME:
{resume_context}

The interview must feel like a REAL placement interview.

Use exactly this structure:

QUESTION 1:
A genuine Personal / HR interview QUESTION.
Do NOT write a greeting, welcome message, or statement.
For example: "अपने बारे में बताइए और बताइए कि आप इस role में क्यों interested हैं?"

QUESTION 2:
Resume-based question.
It MUST use something actually present in the resume, such as a
project, skill, internship, certification, or experience.

QUESTION 3:
Easy-to-moderate technical question appropriate for the role.

QUESTION 4:
Moderate technical OR practical/situational question appropriate
for the role.

QUESTION 5:
Behavioral / problem-solving / workplace question.

IMPORTANT:
- Do not make all questions technical.
- Do not make all questions generic.
- Do not ask questions unrelated to the candidate's resume.
- Do not ask extremely advanced questions unreasonable for a fresher.
- Make the interview balanced: personal + resume + technical +
  practical + behavioral.
- Do not repeat questions.
- Do not invent skills that are not supported by the resume.
- Preserve exact technical terminology.
- Return ONLY the 5 questions, one question per line.
- Do not number them.
- Do not add explanations.

{language_instruction}
"""

    # Flash-Lite is optimized for low-latency, high-throughput tasks.
    # Use it first for the simple "generate 5 questions" step.
    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
    ]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            raw = (
                getattr(response, "text", None)
                or ""
            ).strip()

            raw = re.sub(r"```.*?\n", "", raw)
            raw = raw.replace("```", "").strip()

            lines = []

            for line in raw.splitlines():
                line = line.strip()

                line = re.sub(
                    r"^\s*(?:question\s*)?\d+[\.\):\-]\s*",
                    "",
                    line,
                    flags=re.IGNORECASE
                )

                if line:
                    lines.append(line)

            questions = lines[:5]

            if len(questions) == 5:
                return questions

        except Exception:
            continue

    return fallback


# ============================================================
# DELIVERY ANALYSIS
# ============================================================

def analyze_delivery(transcript, duration):

    if not transcript:
        return {
            "score": 0,
            "wpm": 0,
            "fillers": 0,
            "filler_words": []
        }

    words = transcript.split()
    word_count = len(words)

    if duration > 0:
        minutes = duration / 60
        wpm = round(word_count / minutes)
    else:
        wpm = 0

    filler_patterns = [
        "um",
        "uh",
        "like",
        "you know",
        "basically",
        "actually",
        "hmm"
    ]

    transcript_lower = transcript.lower()
    filler_found = []

    for filler in filler_patterns:
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, transcript_lower)

        if matches:
            filler_found.extend([filler] * len(matches))

    filler_count = len(filler_found)

    score = 100

    if wpm < 80:
        score -= 10
    elif wpm > 180:
        score -= 15
    elif wpm > 160:
        score -= 8

    score -= min(filler_count * 4, 30)
    score = max(0, min(100, score))

    return {
        "score": score,
        "wpm": wpm,
        "fillers": filler_count,
        "filler_words": filler_found
    }


# ============================================================
# GEMINI CONTENT EVALUATION
# ============================================================

def _parse_evaluation_json(text):
    """
    Parse Gemini JSON even if it accidentally wraps the JSON in
    markdown fences or adds a small amount of surrounding text.
    """
    if not text:
        return None

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    ).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Last-resort extraction of the outermost JSON object.
    first = cleaned.find("{")
    last = cleaned.rfind("}")

    if first != -1 and last > first:
        try:
            return json.loads(
                cleaned[first:last + 1]
            )
        except Exception:
            pass

    return None


def _safe_evaluation_result(message, score=0):
    """
    Return a report-safe result without pretending an unavailable
    AI evaluation means that the candidate gave a wrong answer.
    """
    return {
        "content_score": score,
        "technical_correctness": "AI evaluation temporarily unavailable.",
        "relevance": "AI evaluation temporarily unavailable.",
        "completeness": "AI evaluation temporarily unavailable.",
        "clarity": "AI evaluation temporarily unavailable.",
        "strengths": [],
        "improvements": [],
        "overall_feedback": message,
        "evaluation_available": False,
    }


def gemini_evaluate_answer(question, answer, role, resume_text):

    client = get_gemini_client()

    if client is None:
        return _safe_evaluation_result(
            "Gemini could not be connected. Your original answer is preserved."
        )

    resume_context = resume_text[:4000]

    prompt = f"""
You are an expert campus placement interview evaluator.

Evaluate the candidate's ORIGINAL interview answer.

ROLE:
{role}

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

RESUME CONTEXT:
{resume_context}

Evaluate:
1. Technical correctness
2. Relevance
3. Completeness
4. Clarity
5. Strengths
6. Specific improvements

Give a content score from 0 to 100.

Important:
- Evaluate the original answer, not English proficiency.
- Do not judge accent.
- Do not judge speaking speed.
- Do not judge personality.
- Do not heavily penalize grammar mistakes.
- The candidate may answer in Hindi or Hinglish.
- Evaluate whether the candidate actually answered the question.
- Consider that the candidate is a college student/fresher.
- For behavioral questions, do not require technical details.
- If the answer is correct but short, explain how to make it stronger.
- If the answer contains incorrect technical information, identify it.
- Never give a score of 0 merely because AI service is unavailable.
- Return ONLY valid JSON.

Use exactly this structure:
{{
    "content_score": 0,
    "technical_correctness": "short explanation",
    "relevance": "short explanation",
    "completeness": "short explanation",
    "clarity": "short explanation",
    "strengths": ["strength 1", "strength 2"],
    "improvements": ["improvement 1", "improvement 2"],
    "overall_feedback": "short personalized feedback"
}}
"""

    import time

    # Use lightweight models for short evaluation tasks first.
    # Fall back to the main Flash model if needed.
    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]

    last_error = ""

    for model_name in models_to_try:

        for attempt in range(2):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                result = _parse_evaluation_json(
                    getattr(response, "text", None)
                )

                if result is None:
                    last_error = (
                        f"{model_name} returned invalid JSON."
                    )
                    continue

                # Normalise fields so the final report always has the
                # same structure.
                score_raw = result.get("content_score", 0)

                try:
                    score = int(score_raw)
                except Exception:
                    score = 0

                result["content_score"] = max(
                    0,
                    min(100, score)
                )

                result.setdefault(
                    "technical_correctness",
                    "Not provided."
                )
                result.setdefault(
                    "relevance",
                    "Not provided."
                )
                result.setdefault(
                    "completeness",
                    "Not provided."
                )
                result.setdefault(
                    "clarity",
                    "Not provided."
                )
                result.setdefault(
                    "strengths",
                    []
                )
                result.setdefault(
                    "improvements",
                    []
                )
                result.setdefault(
                    "overall_feedback",
                    "Answer evaluated successfully."
                )

                result["evaluation_available"] = True

                return result

            except Exception as e:
                last_error = str(e)
                error_text = last_error.upper()

                # Retry temporary capacity/rate-limit failures.
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                ):
                    time.sleep(2 ** attempt)
                    continue

                # Non-transient failure: try the next model.
                break

    return _safe_evaluation_result(
        "AI evaluation is temporarily unavailable. "
        "Your original answer and delivery analysis are still preserved. "
        "Please try the interview again later.",
        score=0
    )


# ============================================================
# BATCH CONTENT EVALUATION
# ============================================================

def evaluate_all_answers_once():
    """
    Evaluate all five answers in ONE Gemini request after the interview.
    This keeps the Next Question button fast and prevents five separate
    AI calls during the interview.
    """
    client = get_gemini_client()

    if client is None:
        return [
            _safe_evaluation_result(
                "Gemini could not be connected."
            )
            for _ in st.session_state.transcripts
        ]

    items = []

    for index, transcript in enumerate(
        st.session_state.transcripts
    ):
        question = st.session_state.generated_questions[index]

        items.append(
            f"""
ANSWER {index + 1}

QUESTION:
{question}

CANDIDATE ANSWER:
{transcript}
"""
        )

    resume_context = st.session_state.resume_text[:3500]

    prompt = f"""
You are an expert campus placement interview evaluator.

Evaluate ALL candidate answers below for the role:
{st.session_state.role}

RESUME CONTEXT:
{resume_context}

{''.join(items)}

For each answer, evaluate:
- technical correctness
- relevance
- completeness
- clarity
- strengths
- improvements
- overall feedback
- content score from 0 to 100

Important:
- Evaluate the candidate's original answer.
- Hindi and Hinglish answers are valid.
- Do not judge accent, speaking speed, personality, or English ability.
- Do not give 0 simply because an answer is short.
- For behavioral questions, judge the behavior/problem-solving response.
- Return EXACTLY one result for each answer, in the same order.

Return ONLY valid JSON as an array of objects.

Each object must have:
{{
  "content_score": 0,
  "technical_correctness": "short explanation",
  "relevance": "short explanation",
  "completeness": "short explanation",
  "clarity": "short explanation",
  "strengths": ["strength 1"],
  "improvements": ["improvement 1"],
  "overall_feedback": "short personalized feedback"
}}
"""

    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]

    import time

    for model_name in models_to_try:

        for attempt in range(2):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                raw = (
                    getattr(response, "text", None)
                    or ""
                ).strip()

                raw = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    raw,
                    flags=re.IGNORECASE
                )

                raw = re.sub(
                    r"\s*```$",
                    "",
                    raw
                ).strip()

                first = raw.find("[")
                last = raw.rfind("]")

                if first == -1 or last <= first:
                    raise ValueError("Invalid evaluation JSON.")

                results = json.loads(
                    raw[first:last + 1]
                )

                if not isinstance(results, list):
                    raise ValueError("Evaluation result is not a list.")

                if len(results) != len(
                    st.session_state.transcripts
                ):
                    raise ValueError(
                        "Evaluation result count does not match answers."
                    )

                cleaned_results = []

                for result in results:
                    try:
                        score = int(
                            result.get(
                                "content_score",
                                0
                            )
                        )
                    except Exception:
                        score = 0

                    result["content_score"] = max(
                        0,
                        min(100, score)
                    )

                    result.setdefault(
                        "technical_correctness",
                        "Not provided."
                    )
                    result.setdefault(
                        "relevance",
                        "Not provided."
                    )
                    result.setdefault(
                        "completeness",
                        "Not provided."
                    )
                    result.setdefault(
                        "clarity",
                        "Not provided."
                    )
                    result.setdefault(
                        "strengths",
                        []
                    )
                    result.setdefault(
                        "improvements",
                        []
                    )
                    result.setdefault(
                        "overall_feedback",
                        "Answer evaluated successfully."
                    )

                    result["evaluation_available"] = True
                    cleaned_results.append(result)

                return cleaned_results

            except Exception as e:
                message = str(e).upper()

                if (
                    "503" in message
                    or "UNAVAILABLE" in message
                    or "429" in message
                    or "RESOURCE_EXHAUSTED" in message
                ):
                    time.sleep(2 ** attempt)
                    continue

                break

    return [
        _safe_evaluation_result(
            "AI evaluation is temporarily unavailable. "
            "Your answer is preserved and is not treated as incorrect."
        )
        for _ in st.session_state.transcripts
    ]


# ============================================================
# FRONT PAGE
# ============================================================

if not st.session_state.started:

    top_left, top_right = st.columns([7, 1])

    with top_right:
        new_dark_mode = st.toggle(
            "🌙 Dark Mode",
            value=st.session_state.dark_mode
        )

        if new_dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark_mode
            st.rerun()

    st.title("🎤 Placement Papers, Real Panic")

    st.subheader("AI-powered voice mock interview simulator")

    st.write(
        "Practice placement interviews using your resume, "
        "answer through your voice, and receive separate "
        "content and delivery feedback."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("👤 Your Name")

        role = st.selectbox(
            "💼 Select Interview Role",
            [
                "Data Analyst",
                "Business Analyst",
                "Software Engineer",
                "Electronics Engineer"
            ]
        )

    with col2:
        language = st.selectbox(
            "🌐 Interview Language",
            ["English", "Hindi", "Hinglish"]
        )

        uploaded_resume = st.file_uploader(
            "📄 Upload your Resume",
            type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
            help="Supported: normal/scanned PDF, DOCX, PNG, JPG, JPEG and WebP resumes."
        )

    st.info(
        "Your resume is used to generate "
        "role-relevant interview questions."
    )

    if st.button(
        "🚀 Start Interview",
        type="primary",
        use_container_width=True
    ):

        if not name.strip():
            st.error("Please enter your name.")

        elif uploaded_resume is None:
            st.error("Please upload your resume.")

        else:
            resume_text = extract_resume_text(uploaded_resume)

            if not resume_text:
                st.error(
                    "I couldn't extract text from this resume. "
                    "If this is a scanned/image PDF, install Tesseract OCR "
                    "on Windows and try again."
                )
                st.caption(
                    "Tesseract download: https://github.com/tesseract-ocr/tesseract"
                )

            else:
                skills = detect_skills(resume_text)

                with st.spinner(
                    "🤖 Creating your 5 personalized interview questions..."
                ):
                    questions = generate_resume_questions(
                        role,
                        resume_text,
                        skills,
                        language
                    )

                st.session_state.name = name
                st.session_state.role = role
                st.session_state.language = language
                st.session_state.resume_text = resume_text
                st.session_state.detected_skills = skills
                st.session_state.generated_questions = questions
                st.session_state.question_number = 0
                st.session_state.answers = []
                st.session_state.audio_answers = []
                st.session_state.transcripts = []
                st.session_state.english_versions = []
                st.session_state.content_scores = []
                st.session_state.delivery_scores = []
                st.session_state.delivery_metrics = []
                st.session_state.feedback = []
                st.session_state.evaluation_done = False
                st.session_state.recording_attempt = 0
                st.session_state.completed = False
                st.session_state.started = True

                st.rerun()


# ============================================================
# INTERVIEW SCREEN
# ============================================================

elif st.session_state.started and not st.session_state.completed:

    total_questions = len(
        st.session_state.generated_questions
    )

    current_question_number = st.session_state.question_number

    if current_question_number < total_questions:

        question = st.session_state.generated_questions[
            current_question_number
        ]

        st.title("🎤 Mock Interview")

        st.progress(
            current_question_number / total_questions
        )

        st.caption(
            f"Question {current_question_number + 1} "
            f"of {total_questions}"
        )

        # Keep the speaker icon and the visible heading on the same row.
        speaker_col, title_col = st.columns(
            [0.055, 0.945],
            gap="small"
        )

        with speaker_col:
            speak_question_in_browser(
                question,
                st.session_state.language,
                current_question_number
            )

        with title_col:
            st.markdown(
                "### AI Interviewer",
                unsafe_allow_html=False
            )

        st.subheader(question)

        st.write(
            f"**Role:** {st.session_state.role}"
        )

        st.write(
            f"**Language:** {st.session_state.language}"
        )

        st.divider()

        st.markdown("### 🎙️ Record Your Answer")

        # Keep the answer instruction locked to the language selected
        # when the interview started, so it never switches on later questions.
        selected_language = (
            st.session_state.get("language") or "English"
        )

        answer_instructions = {
            "Hindi": (
                "हिंदी में अपना उत्तर बोलें। "
                "आपकी आवाज़ हिंदी टेक्स्ट में दिखाई जाएगी।"
            ),
            "Hinglish": (
                "Hinglish mein apna answer boliye. "
                "Transcript Roman Hinglish mein dikhega."
            ),
            "English": (
                "Answer the question in English. "
                "Your speech will be converted into text."
            ),
        }

        st.info(
            answer_instructions.get(
                selected_language,
                answer_instructions["English"]
            )
        )

        audio = st.audio_input(
            "🎙️ Record your answer",
            sample_rate=16000,
            key=(
                f"audio_{current_question_number}_"
                f"{st.session_state.recording_attempt}"
            )
        )

        # Once a recording exists, show a clear cut/discard icon.
        # Clicking it throws away the current recording and opens a
        # fresh recorder for the same question.
        if audio is not None:
            retake_col, _ = st.columns([0.7, 7.3])

            with retake_col:
                if st.button(
                    "✕",
                    key=(
                        f"retake_{current_question_number}_"
                        f"{st.session_state.recording_attempt}"
                    ),
                    help="Discard this answer and record again",
                    use_container_width=True
                ):
                    st.session_state.recording_attempt += 1
                    st.rerun()

        transcript = ""
        english_version = ""
        duration = 0

        if audio is not None:

            audio_bytes = audio.getvalue()

            duration = get_audio_duration(audio_bytes)

            st.audio(
                audio_bytes,
                format="audio/wav"
            )

            with st.spinner(
                "🤖 Converting your speech to text..."
            ):
                transcript, speech_error = transcribe_audio(
                    audio_bytes,
                    st.session_state.language
                )

            if speech_error:
                st.error(speech_error)

            if transcript:

                st.success(
                    "Speech converted successfully!"
                )

                if st.session_state.language == "Hindi":
                    st.markdown("### 📝 What You Said — Hindi")

                elif st.session_state.language == "Hinglish":
                    st.markdown("### 📝 What You Said — Hinglish")

                else:
                    st.markdown("### 📝 What You Said — English")

                st.text_area(
                    "Original transcript",
                    value=transcript,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )

                if st.session_state.language in [
                    "Hindi",
                    "Hinglish"
                ]:

                    with st.spinner(
                        "🌐 Creating an English version for practice..."
                    ):
                        english_version, translation_error = (
                            create_english_version(
                                transcript,
                                st.session_state.language
                            )
                        )

                    if translation_error:
                        st.warning(translation_error)

                    if english_version:

                        st.markdown(
                            "### 🇬🇧 English Version — Improve Your English"
                        )

                        st.text_area(
                            "English version",
                            value=english_version,
                            height=150,
                            disabled=True,
                            label_visibility="collapsed"
                        )

                        st.caption(
                            "This English version is for learning "
                            "and interview practice."
                        )

        st.divider()

        if transcript:

            if st.button(
                "➡️ Next Question",
                type="primary",
                use_container_width=True
            ):

                # Keep Next Question instant.
                # Gemini content evaluation is performed once after all
                # five answers are collected.
                delivery_result = analyze_delivery(
                    transcript,
                    duration
                )

                ai_result = _safe_evaluation_result(
                    "Content evaluation will be completed after the interview.",
                    score=0
                )

                st.session_state.answers.append(transcript)
                st.session_state.audio_answers.append(audio_bytes)
                st.session_state.transcripts.append(transcript)
                st.session_state.english_versions.append(
                    english_version
                )
                st.session_state.content_scores.append(
                    ai_result["content_score"]
                )
                st.session_state.delivery_scores.append(
                    delivery_result["score"]
                )
                st.session_state.delivery_metrics.append(
                    delivery_result
                )
                st.session_state.feedback.append(ai_result)

                st.session_state.question_number += 1
                st.session_state.recording_attempt = 0

                if (
                    st.session_state.question_number
                    >= total_questions
                ):
                    st.session_state.completed = True

                st.rerun()

        else:
            st.warning(
                "Please record your answer before "
                "moving to the next question."
            )


# ============================================================
# FINAL REPORT
# ============================================================

else:

    # Evaluate all answers in one AI call only when the final report opens.
    if not st.session_state.get("evaluation_done", False):
        with st.spinner(
            "🤖 Preparing your final AI evaluation..."
        ):
            final_feedback = evaluate_all_answers_once()

        st.session_state.feedback = final_feedback
        st.session_state.content_scores = [
            result.get("content_score", 0)
            for result in final_feedback
        ]
        st.session_state.evaluation_done = True

    st.title("🎉 Interview Complete!")

    st.subheader(
        f"Great job, {st.session_state.name}!"
    )

    content_scores = st.session_state.content_scores
    delivery_scores = st.session_state.delivery_scores

    available_content_scores = [
        score
        for score, feedback in zip(
            content_scores,
            st.session_state.feedback
        )
        if feedback.get("evaluation_available", True)
    ]

    average_content = (
        round(
            sum(available_content_scores)
            / len(available_content_scores)
        )
        if available_content_scores
        else 0
    )

    unavailable_content_count = (
        len(content_scores) - len(available_content_scores)
    )

    average_delivery = (
        round(sum(delivery_scores) / len(delivery_scores))
        if delivery_scores else 0
    )

    overall_score = round(
        (average_content + average_delivery) / 2
    )

    st.markdown("## 📊 Your Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🧠 Content Score",
            f"{average_content}/100"
        )

    with col2:
        st.metric(
            "🎙️ Delivery Score",
            f"{average_delivery}/100"
        )

    with col3:
        st.metric(
            "🏆 Overall Score",
            f"{overall_score}/100"
        )

    if unavailable_content_count > 0:
        st.info(
            f"{unavailable_content_count} answer(s) could not be "
            "AI-evaluated because the AI service was temporarily unavailable. "
            "Those answers are not treated as incorrect."
        )

    st.divider()

    st.markdown("## 🧠 AI Content Evaluation")

    st.caption(
        "Gemini evaluates answer quality separately "
        "from speaking delivery."
    )

    for i, feedback in enumerate(
        st.session_state.feedback
    ):

        st.markdown(f"### Question {i + 1}")

        evaluation_available = feedback.get(
            "evaluation_available",
            True
        )

        if evaluation_available:
            st.write(
                f"**Content Score:** "
                f"{feedback.get('content_score', 0)}/100"
            )
        else:
            st.warning(
                "Content evaluation temporarily unavailable "
                "for this answer — this is NOT a score of 0."
            )

        st.write(
            f"**Technical Correctness:** "
            f"{feedback.get('technical_correctness', 'N/A')}"
        )

        st.write(
            f"**Relevance:** "
            f"{feedback.get('relevance', 'N/A')}"
        )

        st.write(
            f"**Completeness:** "
            f"{feedback.get('completeness', 'N/A')}"
        )

        st.write(
            f"**Clarity:** "
            f"{feedback.get('clarity', 'N/A')}"
        )

        strengths = feedback.get("strengths", [])

        if strengths:
            st.write("**✅ Strengths:**")
            for strength in strengths:
                st.write(f"- {strength}")

        improvements = feedback.get("improvements", [])

        if improvements:
            st.write("**🔧 Areas to Improve:**")
            for improvement in improvements:
                st.write(f"- {improvement}")

        st.info(
            feedback.get("overall_feedback", "")
        )

        st.divider()

    st.markdown("## 🎙️ Delivery Analysis")

    for i, metric in enumerate(
        st.session_state.delivery_metrics
    ):

        st.markdown(f"### Question {i + 1}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Speaking Speed",
                f"{metric['wpm']} WPM"
            )

        with col2:
            st.metric(
                "Filler Words",
                metric["fillers"]
            )

        with col3:
            st.metric(
                "Delivery Score",
                f"{metric['score']}/100"
            )

        if metric["wpm"] < 80:
            st.warning(
                "You were speaking quite slowly. "
                "Try maintaining a natural conversational pace."
            )

        elif metric["wpm"] > 180:
            st.warning(
                "You were speaking quite quickly. "
                "Try slowing down slightly."
            )

        else:
            st.success(
                "Your speaking speed was within a comfortable range."
            )

        if metric["fillers"] > 0:
            st.write(
                "Filler words detected: "
                + ", ".join(metric["filler_words"])
            )
        else:
            st.success(
                "No significant filler words detected."
            )

        st.divider()

    st.markdown("## 📄 Resume Analysis")

    if st.session_state.detected_skills:
        st.write(
            "Normalized skills detected from your resume:"
        )
        st.write(
            ", ".join(st.session_state.detected_skills)
        )
    else:
        st.write(
            "No predefined skills were detected."
        )

    st.markdown("## 💡 Personalized Improvement Plan")

    if average_content < 60:
        st.write(
            "🧠 Focus on improving the technical accuracy "
            "and completeness of your answers."
        )
    elif average_content < 80:
        st.write(
            "🧠 Your content is developing well. "
            "Try adding specific examples and explanations."
        )
    else:
        st.write(
            "🧠 Strong answer quality. "
            "Continue supporting answers with concrete examples."
        )

    if average_delivery < 60:
        st.write(
            "🎙️ Practice speaking clearly, controlling "
            "your pace and reducing filler words."
        )
    elif average_delivery < 80:
        st.write(
            "🎙️ Your delivery is decent. "
            "Work on confidence and smoother transitions."
        )
    else:
        st.write(
            "🎙️ Your delivery was strong. "
            "Continue practicing natural conversational speaking."
        )

    st.markdown("## 📈 Performance Tracking")

    current_result = {
        "content": average_content,
        "delivery": average_delivery,
        "overall": overall_score
    }

    # Prevent duplicate entries caused by Streamlit reruns.
    if (
        not st.session_state.performance_history
        or st.session_state.performance_history[-1] != current_result
    ):
        st.session_state.performance_history.append(
            current_result
        )

    for i, result in enumerate(
        st.session_state.performance_history
    ):
        st.write(
            f"Attempt {i + 1}: "
            f"Content {result['content']} | "
            f"Delivery {result['delivery']} | "
            f"Overall {result['overall']}"
        )

    st.markdown("## 📝 Your Interview Answers")

    for i, transcript in enumerate(
        st.session_state.transcripts
    ):

        with st.expander(f"Question {i + 1}"):

            st.write("**Original Answer:**")
            st.write(transcript)

            english_version = (
                st.session_state.english_versions[i]
            )

            if english_version:
                st.write("**🇬🇧 English Version:**")
                st.write(english_version)

    st.divider()

    st.success(
        "🤖 Gemini AI Evaluation + "
        "🎙️ Voice Recording + "
        "🌐 Hindi/Hinglish Support + "
        "🇬🇧 English Improvement + "
        "☀️/🌙 Theme Selection + "
        "📊 Separate Content & Delivery Analysis + "
        "🎯 5-question AI-generated interview "
        "are included in this MVP."
    )

    if st.button(
        "🔄 Try Another Interview",
        type="primary",
        use_container_width=True
    ):

        dark_mode = st.session_state.dark_mode

        for key in defaults:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.dark_mode = dark_mode
        st.rerun()
