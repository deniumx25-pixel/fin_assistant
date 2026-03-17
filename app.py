import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import re
import cv2
import numpy as np
from collections import defaultdict
import os
import tempfile
import subprocess
import json
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ====================== ЛОКАЛЬНЫЕ ПУТИ К OCR ======================
TESSERACT_PATH = r'C:\Users\deniu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
POPPLER_PATH   = r'C:\Users\deniu\AppData\Local\Programs\poppler-25.12.0\Library\bin'

# --- Проверка доступности OCR-компонентов: сначала локальные пути, потом PATH ---
def check_tesseract():
    if os.path.isfile(TESSERACT_PATH):
        return True
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_poppler():
    if os.path.isfile(os.path.join(POPPLER_PATH, 'pdftoppm.exe')):
        return True
    try:
        subprocess.run(['pdftoppm', '-v'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

OCR_SYSTEM_AVAILABLE = check_tesseract() and check_poppler()

# --- Импорт OCR-библиотек ---
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_LIBS_AVAILABLE = True
    if os.path.isfile(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        pytesseract.pytesseract.tesseract_cmd = 'tesseract'
except ImportError:
    OCR_LIBS_AVAILABLE = False

OCR_AVAILABLE = OCR_LIBS_AVAILABLE and OCR_SYSTEM_AVAILABLE

# --- Импорт GigaChat ---
try:
    from gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
if not GIGACHAT_CREDENTIALS and hasattr(st, "secrets"):
    GIGACHAT_CREDENTIALS = st.secrets.get("GIGACHAT_CREDENTIALS", "")

st.set_page_config(
    page_title="ИИ-ассистент финансового директора",
    page_icon="🐍",
    layout="wide"
)

# --- Инициализация session_state ---
if 'manual_years' not in st.session_state:
    st.session_state.manual_years = []
if 'gigachat_done' not in st.session_state:
    st.session_state.gigachat_done = False
if 'processed_years' not in st.session_state:
    st.session_state.processed_years = set()
if 'last_gigachat_response' not in st.session_state:
    st.session_state.last_gigachat_response = None
if 'last_fill_changes' not in st.session_state:
    st.session_state.last_fill_changes = []

with st.sidebar:
    st.header("⚙️ Настройки")
    years_to_analyze = st.slider("Количество лет для анализа", 1, 10, 3)

    with st.expander("📌 Примечание об ОФР"):
        st.info("Отчёт о финансовых результатах (ОФР) содержит данные только за два года. Для анализа за период более 2 лет загрузите ОФР за соответствующие предыдущие годы.")

    industry_options = [
        "Промышленность (добывающая, обрабатывающая, энергетика)",
        "Сельское хозяйство",
        "Строительство",
        "Транспорт (все виды)",
        "Связь и телекоммуникации",
        "Торговля (оптовая и розничная)",
        "Финансы и банковское дело",
        "Химическая и нефтехимическая промышленность",
        "Металлургия",
        "Информационные технологии и IT-услуги"
    ]
    industry = st.selectbox("Отрасль компании", options=industry_options, index=0)

    use_ocr = st.checkbox("Использовать OCR для сканов", value=False)

    if use_ocr:
        st.subheader("🔧 Качество OCR")
        ocr_quality = st.select_slider("Качество распознавания",
                                       options=["Быстрое", "Среднее", "Высокое"],
                                       value="Среднее")
        remove_lines = st.checkbox("Удалять линии таблиц", value=True)
        psm_mode = st.selectbox("Режим распознавания (PSM)",
                                options=[3, 6, 11, 12],
                                index=1,
                                format_func=lambda x: f"PSM {x} (авто:3, таблица:6, разреженный:11, переменный:12)")
    else:
        ocr_quality = "Среднее"

    if use_ocr and not OCR_AVAILABLE:
        if not OCR_LIBS_AVAILABLE:
            st.error("❌ Не установлены Python-библиотеки для OCR. Выполните: pip install pytesseract pdf2image pillow opencv-python")
        elif not OCR_SYSTEM_AVAILABLE:
            st.error(f"❌ Системные компоненты OCR отсутствуют по указанным путям:\nTesseract: {TESSERACT_PATH}\nPoppler: {POPPLER_PATH}\nПроверьте, что файлы существуют.")
        else:
            st.success("✅ OCR готов к использованию.")

    st.markdown("---")
    st.subheader("🤖 Нейросетевой анализ")
    use_gigachat = st.checkbox("Использовать GigaChat", value=False)
    fill_missing_with_ai = st.checkbox("Заполнять пропуски через GigaChat", value=False)

    if use_gigachat and not GIGACHAT_AVAILABLE:
        st.error("❌ Библиотека gigachat не установлена. Выполните: pip install gigachat")
    if use_gigachat and not GIGACHAT_CREDENTIALS:
        st.warning("⚠️ Не указан ключ GigaChat. Добавьте его в .env или Streamlit secrets.")

    debug_mode = st.checkbox("Режим отладки", value=False)

    manual_years_input = st.text_input(
        "Годы для ручного сопоставления (через запятую, например 2024,2023,2022)",
        value="",
        help="Если таблица не содержит заголовков с годами, укажите годы в порядке от нового к старому (слева направо)."
    )
    if manual_years_input.strip():
        try:
            manual_years_list = [int(y.strip()) for y in manual_years_input.split(',') if y.strip().isdigit()]
            st.session_state.manual_years = manual_years_list if manual_years_list else []
        except:
            st.warning("Неверный формат годов.")
            st.session_state.manual_years = []
    else:
        st.session_state.manual_years = []

st.title("🐍 ИИ-ассистент финансового директора")
st.markdown("Загрузите PDF-файлы с финансовой отчётностью (РСБУ). Программа извлекает данные из таблиц по бухгалтерским кодам строк, при необходимости использует OCR.")

# ------------------- Константы -------------------
CODE_TO_METRIC = {
    "2110": "Выручка",
    "2400": "Чистая прибыль",
    "2300": "Прибыль (убыток) до налогообложения",
    "1600": "Активы",
    "1400": "Долгосрочные обязательства",
    "1500": "Краткосрочные обязательства",
    "1300": "Собственный капитал",
    "1200": "Оборотные активы",
    "1250": "Денежные средства",
    "2330": "Проценты к уплате",
    "1210": "Запасы",
    "1230": "Дебиторская задолженность"
}
KEY_METRICS = list(CODE_TO_METRIC.values())
BALANCE_METRICS = ["Активы", "Долгосрочные обязательства", "Краткосрочные обязательства",
                   "Собственный капитал", "Оборотные активы", "Денежные средства",
                   "Запасы", "Дебиторская задолженность"]
INCOME_METRICS = ["Выручка", "Чистая прибыль", "Прибыль (убыток) до налогообложения", "Проценты к уплате"]

# ------------------- Отраслевые пороговые значения -------------------
INDUSTRY_THRESHOLDS = {
    "Промышленность (добывающая, обрабатывающая, энергетика)": {
        "Коэффициент текущей ликвидности": (1.5, 2.5),
        "Коэффициент быстрой ликвидности": (0.8, 1.5),
        "Коэффициент абсолютной ликвидности": (0.2, 0.5),
        "ROE": (0.12, 0.20),
        "ROA": (0.05, 0.10),
        "Рентабельность продаж": (0.10, 0.18),
        "Рентабельность до налогообложения": (0.10, 0.18),
        "Коэффициент автономии": (0.5, 0.7),
        "Финансовый рычаг": (0.5, 1.0),
        "Покрытие процентов": (3.0, 5.0),
        "Оборачиваемость активов": (0.8, 1.5),
        "Оборачиваемость дебиторской задолженности": (6.0, 12.0)
    },
    "Сельское хозяйство": {
        "Коэффициент текущей ликвидности": (1.3, 2.2),
        "Коэффициент быстрой ликвидности": (0.6, 1.2),
        "Коэффициент абсолютной ликвидности": (0.1, 0.3),
        "ROE": (0.08, 0.15),
        "ROA": (0.03, 0.07),
        "Рентабельность продаж": (0.12, 0.22),
        "Рентабельность до налогообложения": (0.10, 0.18),
        "Коэффициент автономии": (0.4, 0.6),
        "Финансовый рычаг": (0.7, 1.3),
        "Покрытие процентов": (2.0, 4.0),
        "Оборачиваемость активов": (0.4, 0.8),
        "Оборачиваемость дебиторской задолженности": (4.0, 8.0)
    },
    "Строительство": {
        "Коэффициент текущей ликвидности": (1.2, 2.0),
        "Коэффициент быстрой ликвидности": (0.5, 1.0),
        "Коэффициент абсолютной ликвидности": (0.1, 0.3),
        "ROE": (0.10, 0.18),
        "ROA": (0.04, 0.08),
        "Рентабельность продаж": (0.05, 0.12),
        "Рентабельность до налогообложения": (0.05, 0.12),
        "Коэффициент автономии": (0.3, 0.5),
        "Финансовый рычаг": (1.0, 2.0),
        "Покрытие процентов": (1.8, 3.5),
        "Оборачиваемость активов": (0.6, 1.2),
        "Оборачиваемость дебиторской задолженности": (3.0, 6.0)
    },
    "Транспорт (все виды)": {
        "Коэффициент текущей ликвидности": (1.2, 2.0),
        "Коэффициент быстрой ликвидности": (0.7, 1.3),
        "Коэффициент абсолютной ликвидности": (0.15, 0.4),
        "ROE": (0.10, 0.18),
        "ROA": (0.04, 0.09),
        "Рентабельность продаж": (0.08, 0.15),
        "Рентабельность до налогообложения": (0.07, 0.14),
        "Коэффициент автономии": (0.4, 0.6),
        "Финансовый рычаг": (0.8, 1.5),
        "Покрытие процентов": (2.5, 4.5),
        "Оборачиваемость активов": (0.5, 1.0),
        "Оборачиваемость дебиторской задолженности": (5.0, 10.0)
    },
    "Связь и телекоммуникации": {
        "Коэффициент текущей ликвидности": (1.3, 2.2),
        "Коэффициент быстрой ликвидности": (0.8, 1.5),
        "Коэффициент абсолютной ликвидности": (0.2, 0.5),
        "ROE": (0.15, 0.25),
        "ROA": (0.07, 0.14),
        "Рентабельность продаж": (0.15, 0.25),
        "Рентабельность до налогообложения": (0.14, 0.24),
        "Коэффициент автономии": (0.5, 0.7),
        "Финансовый рычаг": (0.5, 1.0),
        "Покрытие процентов": (3.5, 6.0),
        "Оборачиваемость активов": (0.6, 1.2),
        "Оборачиваемость дебиторской задолженности": (6.0, 12.0)
    },
    "Торговля (оптовая и розничная)": {
        "Коэффициент текущей ликвидности": (1.0, 2.0),
        "Коэффициент быстрой ликвидности": (0.5, 1.0),
        "Коэффициент абсолютной ликвидности": (0.1, 0.3),
        "ROE": (0.15, 0.25),
        "ROA": (0.08, 0.15),
        "Рентабельность продаж": (0.03, 0.08),
        "Рентабельность до налогообложения": (0.03, 0.08),
        "Коэффициент автономии": (0.4, 0.6),
        "Финансовый рычаг": (0.8, 1.5),
        "Покрытие процентов": (2.0, 4.0),
        "Оборачиваемость активов": (1.5, 3.0),
        "Оборачиваемость дебиторской задолженности": (6.0, 12.0)
    },
    "Финансы и банковское дело": {
        "Коэффициент текущей ликвидности": (1.0, 2.0),
        "Коэффициент быстрой ликвидности": (0.5, 1.0),
        "Коэффициент абсолютной ликвидности": (0.1, 0.3),
        "ROE": (0.08, 0.15),
        "ROA": (0.01, 0.03),
        "Рентабельность продаж": (0.20, 0.40),
        "Рентабельность до налогообложения": (0.20, 0.40),
        "Коэффициент автономии": (0.08, 0.15),
        "Финансовый рычаг": (5.0, 10.0),
        "Покрытие процентов": (1.5, 3.0),
        "Оборачиваемость активов": (0.05, 0.20),
        "Оборачиваемость дебиторской задолженности": (1.0, 3.0)
    },
    "Химическая и нефтехимическая промышленность": {
        "Коэффициент текущей ликвидности": (1.4, 2.3),
        "Коэффициент быстрой ликвидности": (0.7, 1.4),
        "Коэффициент абсолютной ликвидности": (0.2, 0.4),
        "ROE": (0.12, 0.22),
        "ROA": (0.06, 0.12),
        "Рентабельность продаж": (0.10, 0.20),
        "Рентабельность до налогообложения": (0.10, 0.20),
        "Коэффициент автономии": (0.5, 0.7),
        "Финансовый рычаг": (0.5, 1.0),
        "Покрытие процентов": (3.0, 5.0),
        "Оборачиваемость активов": (0.7, 1.3),
        "Оборачиваемость дебиторской задолженности": (5.0, 10.0)
    },
    "Металлургия": {
        "Коэффициент текущей ликвидности": (1.3, 2.2),
        "Коэффициент быстрой ликвидности": (0.6, 1.2),
        "Коэффициент абсолютной ликвидности": (0.15, 0.35),
        "ROE": (0.12, 0.22),
        "ROA": (0.06, 0.12),
        "Рентабельность продаж": (0.12, 0.22),
        "Рентабельность до налогообложения": (0.12, 0.22),
        "Коэффициент автономии": (0.45, 0.65),
        "Финансовый рычаг": (0.6, 1.2),
        "Покрытие процентов": (2.8, 5.0),
        "Оборачиваемость активов": (0.6, 1.2),
        "Оборачиваемость дебиторской задолженности": (5.0, 10.0)
    },
    "Информационные технологии и IT-услуги": {
        "Коэффициент текущей ликвидности": (1.5, 2.8),
        "Коэффициент быстрой ликвидности": (1.0, 2.0),
        "Коэффициент абсолютной ликвидности": (0.3, 0.7),
        "ROE": (0.18, 0.30),
        "ROA": (0.10, 0.20),
        "Рентабельность продаж": (0.15, 0.30),
        "Рентабельность до налогообложения": (0.15, 0.30),
        "Коэффициент автономии": (0.6, 0.8),
        "Финансовый рычаг": (0.3, 0.7),
        "Покрытие процентов": (4.0, 8.0),
        "Оборачиваемость активов": (1.0, 2.0),
        "Оборачиваемость дебиторской задолженности": (4.0, 8.0)
    }
}

current_thresholds = INDUSTRY_THRESHOLDS.get(industry, INDUSTRY_THRESHOLDS["Промышленность (добывающая, обрабатывающая, энергетика)"])

GROUPS = {
    "Ликвидность": ["Коэффициент текущей ликвидности", "Коэффициент быстрой ликвидности", "Коэффициент абсолютной ликвидности"],
    "Рентабельность": ["ROE", "ROA", "Рентабельность продаж", "Рентабельность до налогообложения"],
    "Устойчивость": ["Коэффициент автономии", "Финансовый рычаг", "Покрытие процентов"],
    "Активность": ["Оборачиваемость активов", "Оборачиваемость дебиторской задолженности"]
}
WEIGHTS = {"Ликвидность": 0.2, "Рентабельность": 0.3, "Устойчивость": 0.3, "Активность": 0.2}

# ------------------- Функции предобработки изображений -------------------
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1000:
        scale_factor = 300 / (w / 8.27)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if remove_lines:
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horiz_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horiz_kernel)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vert_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vert_kernel)
        gray = cv2.subtract(gray, horiz_lines)
        gray = cv2.subtract(gray, vert_lines)
    return gray

def scale_image(image, factor):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_CUBIC)

def extract_text_with_ocr(pdf_path, page_num):
    if not use_ocr or not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, poppler_path=POPPLER_PATH)
        if images:
            img = np.array(images[0])
            if ocr_quality == "Высокое":
                img = scale_image(img, 3)
            elif ocr_quality == "Среднее":
                img = scale_image(img, 2)
            img = preprocess_for_ocr(img)
            psms = [psm_mode, 3, 6, 11, 12]
            for psm in psms:
                try:
                    cfg = f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-'
                    text = pytesseract.image_to_string(img, config=cfg, lang='rus+eng')
                    if len(text.strip()) > 50:
                        return text
                except:
                    continue
            cfg = f'--psm {psm_mode} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-'
            return pytesseract.image_to_string(img, config=cfg, lang='rus+eng')
    except Exception as e:
        if debug_mode:
            st.warning(f"Ошибка OCR на странице {page_num}: {e}")
    return ""

def extract_full_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        if debug_mode:
            st.warning(f"Ошибка извлечения текста: {e}")
    return text

def extract_data_from_pdf(pdf_path):
    years_data = defaultdict(dict)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                page_text = page.extract_text() or ""
                if debug_mode:
                    st.info(f"Страница {page_num}: найдено таблиц: {len(tables)}")
                if (not tables or len(page_text) < 100) and use_ocr and OCR_AVAILABLE:
                    ocr_text = extract_text_with_ocr(pdf_path, page_num)
                    if ocr_text:
                        lines = [line for line in ocr_text.split('\n') if line.strip()]
                        if lines:
                            tables = [[line] for line in lines]
                            if debug_mode:
                                st.info(f"Страница {page_num}: использован OCR (строк: {len(lines)})")
                for table in tables:
                    if not table:
                        continue
                    if isinstance(table[0], str):
                        table = [[row] for row in table]
                    table = [row for row in table if any(cell and str(cell).strip() for cell in row)]
                    if not table:
                        continue
                    max_cols = max(len(row) for row in table)
                    table = [row + [None] * (max_cols - len(row)) for row in table]

                    year_indices = {}
                    header_row = None
                    for i, row in enumerate(table[:30]):
                        for j, cell in enumerate(row):
                            if cell:
                                m = re.search(r'(20\d{2})', str(cell))
                                if m:
                                    year_indices[j] = int(m.group(1))
                        if year_indices:
                            header_row = i
                            break

                    code_col = None
                    search_rows = []
                    if header_row is not None:
                        if header_row > 0:
                            search_rows.append(header_row-1)
                        search_rows.append(header_row)
                        if header_row+1 < len(table):
                            search_rows.append(header_row+1)
                        if header_row+2 < len(table):
                            search_rows.append(header_row+2)
                    else:
                        search_rows = list(range(min(5, len(table))))
                    for r in search_rows:
                        if r >= len(table):
                            continue
                        for j, cell in enumerate(table[r]):
                            if cell and re.search(r'код|code|стр[ао]ки', str(cell).lower()):
                                code_col = j
                                break
                        if code_col is not None:
                            break
                    if code_col is None:
                        best = 0
                        best_matches = -1
                        start = header_row+1 if header_row is not None else 1
                        end = min(start+15, len(table))
                        for cand in range(4):
                            matches = sum(1 for row in table[start:end] if len(row)>cand and str(row[cand]).strip() in CODE_TO_METRIC)
                            if matches > best_matches:
                                best_matches = matches
                                best = cand
                        code_col = best
                    if code_col is None:
                        code_col = 0
                    if debug_mode:
                        st.info(f"  Столбец кодов: {code_col}")

                    if year_indices:
                        for row in table[header_row+1:]:
                            if not row or len(row) <= code_col:
                                continue
                            code = str(row[code_col]).strip()
                            if code in CODE_TO_METRIC:
                                metric = CODE_TO_METRIC[code]
                                for j, year in year_indices.items():
                                    if j < len(row):
                                        val = parse_number(row[j])
                                        if val is not None and metric not in years_data[year]:
                                            years_data[year][metric] = val
                                            if debug_mode:
                                                st.info(f"    Найден {code} -> {metric} за {year}: {val}")
                    else:
                        manual = st.session_state.manual_years
                        if manual:
                            for row in table:
                                if not row or len(row) <= code_col:
                                    continue
                                code = str(row[code_col]).strip()
                                if code in CODE_TO_METRIC:
                                    metric = CODE_TO_METRIC[code]
                                    if len(row) == 1 and isinstance(row[0], str):
                                        parts = row[0].split()
                                        if len(parts) > 1:
                                            vals = parts[1:] if parts[0] == code else parts
                                            for i, year in enumerate(manual):
                                                if i < len(vals):
                                                    num = parse_number(vals[i])
                                                    if num is not None and metric not in years_data[year]:
                                                        years_data[year][metric] = num
                                    else:
                                        for i, year in enumerate(manual):
                                            col = code_col + 1 + i
                                            if col < len(row):
                                                num = parse_number(row[col])
                                                if num is not None and metric not in years_data[year]:
                                                    years_data[year][metric] = num
    except Exception as e:
        if debug_mode:
            st.error(f"Ошибка при извлечении данных: {e}")
        return {}
    return dict(years_data)

def parse_number(s):
    s = str(s).strip()
    if not s:
        return None
    if re.fullmatch(r'[-–—]+', s):
        return 0.0
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1].strip()
        num = parse_number(inner)
        return -num if num is not None else None
    s = re.sub(r'[^\d\s.,\-]', '', s)
    s = s.replace(' ', '')
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '')
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')
    s = re.sub(r'[^\d\-\.]', '', s)
    parts = s.split('.')
    if len(parts) > 2:
        s = parts[0] + '.' + ''.join(parts[1:])
    try:
        return float(s)
    except:
        return None

# ------------------- Функции для GigaChat -------------------
def analyze_with_gigachat(prompt: str) -> Optional[str]:
    if not use_gigachat or not GIGACHAT_AVAILABLE or not GIGACHAT_CREDENTIALS:
        return None
    try:
        with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False, scope='GIGACHAT_API_PERS') as giga:
            full_prompt = "Ты профессиональный финансовый аналитик.\n\n" + prompt
            response = giga.chat(full_prompt)
            return response.choices[0].message.content
    except Exception as e:
        if debug_mode:
            st.error(f"Ошибка GigaChat: {e}")
        return None

def format_metrics_summary(df_coeff, df):
    summary = ["ФИНАНСОВЫЕ МЕТРИКИ:"]
    for year in df_coeff.index:
        summary.append(f"\n{year} год:")
        for c in df_coeff.columns:
            v = df_coeff.loc[year, c]
            if pd.notna(v):
                summary.append(f"- {c}: {v:.2f}")
    summary.append("\n\nКЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:")
    for year in df.index:
        summary.append(f"\n{year} год:")
        for m in ["Выручка", "Чистая прибыль", "Активы"]:
            if m in df.columns:
                v = df.loc[year, m]
                if pd.notna(v):
                    summary.append(f"- {m}: {v:,.0f} тыс. руб.")
    return "\n".join(summary)

def fill_missing_with_gigachat(df, full_text, missing_candidates):
    if not missing_candidates:
        return df, []
    by_year = defaultdict(list)
    for code, year in missing_candidates:
        by_year[year].append(code)
    prompt = "В тексте пояснительной записки найди значения следующих показателей по кодам строк за указанные годы.\n"
    prompt += "Ответ в формате JSON без лишнего текста:\n{\n"
    for year in sorted(by_year):
        codes = by_year[year]
        if codes:
            prompt += f'  "{year}": {{\n'
            for code in codes:
                prompt += f'    "{code}": null,\n'
            prompt = prompt.rstrip(',\n') + '\n  },\n'
    prompt = prompt.rstrip(',\n') + '\n}\n\nТекст:\n' + full_text[:15000]
    resp = analyze_with_gigachat(prompt)
    if not resp:
        return df, []
    if debug_mode:
        st.info(f"Ответ GigaChat: {resp}")
    try:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        data = json.loads(m.group()) if m else json.loads(resp)
    except Exception as e:
        if debug_mode:
            st.warning(f"Не удалось распарсить JSON: {e}")
        return df, []
    changes = []
    for year_str, vals in data.items():
        try:
            year = int(year_str)
        except:
            continue
        if year not in [y for _, y in missing_candidates]:
            continue
        for code, val in vals.items():
            if code in CODE_TO_METRIC and val is not None:
                metric = CODE_TO_METRIC[code]
                if pd.isna(df.loc[year, metric]):
                    try:
                        num = float(val)
                        df.loc[year, metric] = num
                        changes.append(f"{metric} за {year} год: {num}")
                    except:
                        pass
    return df, changes

# ------------------- Загрузка файлов -------------------
uploaded_files = st.file_uploader("📎 Загрузите PDF-файлы с отчётностью", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_years_data = defaultdict(dict)
    full_texts = []
    current_years = set()
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        data = extract_data_from_pdf(tmp_path)
        if data:
            for year, metrics in data.items():
                current_years.add(year)
                for k, v in metrics.items():
                    if k not in all_years_data[year]:
                        all_years_data[year][k] = v
        text = extract_full_text_from_pdf(tmp_path)
        if text:
            full_texts.append(text)
        os.unlink(tmp_path)

    if current_years - st.session_state.processed_years:
        st.session_state.gigachat_done = False
        st.session_state.last_gigachat_response = None
        st.session_state.last_fill_changes = []

    if not all_years_data:
        st.error("❌ Не удалось извлечь данные. Убедитесь, что в файлах есть таблицы с кодами строк и годами.")
        st.stop()

    df_raw = pd.DataFrame.from_dict(all_years_data, orient='index').sort_index()
    for m in KEY_METRICS:
        if m not in df_raw.columns:
            df_raw[m] = None

    years_list = ', '.join(map(str, df_raw.index.unique()))
    st.success(f"✅ Извлечены данные: {years_list}")

    # Проверка баланса
    balance_errors = []
    balance_ok = []
    for year in df_raw.index:
        if all(col in df_raw.columns for col in ["Активы", "Собственный капитал", "Долгосрочные обязательства", "Краткосрочные обязательства"]):
            актив = df_raw.loc[year, "Активы"]
            кап = df_raw.loc[year, "Собственный капитал"]
            долг = df_raw.loc[year, "Долгосрочные обязательства"]
            крат = df_raw.loc[year, "Краткосрочные обязательства"]
            if pd.notna(актив) and pd.notna(кап) and pd.notna(долг) and pd.notna(крат):
                пассив = кап + долг + крат
                if abs(актив - пассив) > max(1, abs(актив) * 0.01):
                    balance_errors.append(f"{year} (актив {актив:,.0f} ≠ пассив {пассив:,.0f})")
                else:
                    balance_ok.append(year)
    if balance_errors:
        st.warning(f"⚠️ Баланс не сходится по годам: {', '.join(balance_errors)}")
    elif balance_ok:
        st.success(f"✅ Актив и пассив сходятся по годам: {', '.join(map(str, balance_ok))}")
    else:
        st.info("ℹ️ Недостаточно данных для проверки баланса.")

    years_available = sorted(df_raw.index.unique())
    years_to_use = years_available[-years_to_analyze:] if len(years_available) >= years_to_analyze else years_available
    df = df_raw.loc[years_to_use].copy()

    required_for_analysis = ["Выручка", "Активы", "Долгосрочные обязательства", "Краткосрочные обязательства",
                             "Собственный капитал", "Оборотные активы", "Денежные средства",
                             "Чистая прибыль", "Прибыль (убыток) до налогообложения", "Проценты к уплате",
                             "Запасы", "Дебиторская задолженность"]

    # Заполнение пропусков через GigaChat
    fill_changes = []
    if use_gigachat and fill_missing_with_ai and full_texts and GIGACHAT_AVAILABLE and GIGACHAT_CREDENTIALS and not st.session_state.gigachat_done:
        candidates = []
        for year in years_to_use:
            bal_ok = any(pd.notna(df.loc[year].get(m)) for m in BALANCE_METRICS if m in df.columns)
            inc_ok = any(pd.notna(df.loc[year].get(m)) for m in INCOME_METRICS if m in df.columns)
            for code, metric in CODE_TO_METRIC.items():
                if metric not in df.columns or pd.isna(df.loc[year, metric]):
                    if metric in BALANCE_METRICS and bal_ok:
                        candidates.append((code, year))
                    elif metric in INCOME_METRICS and inc_ok:
                        candidates.append((code, year))
        if candidates:
            with st.spinner("GigaChat ищет недостающие данные..."):
                combined = "\n".join(full_texts)
                df, fill_changes = fill_missing_with_gigachat(df, combined, candidates)
                st.session_state.last_fill_changes = fill_changes
        st.session_state.gigachat_done = True
        st.session_state.processed_years = set(df_raw.index)
    else:
        fill_changes = st.session_state.last_fill_changes

    if fill_changes:
        with st.expander("🤖 Данные, добавленные GigaChat"):
            for c in fill_changes:
                st.write(f"- {c}")

    missing_passive = []
    for year in years_to_use:
        for m in ["Долгосрочные обязательства", "Краткосрочные обязательства", "Собственный капитал"]:
            if pd.isna(df.loc[year].get(m)):
                missing_passive.append(f"{year}: {m}")
    if missing_passive:
        st.warning("⚠️ Не удалось извлечь некоторые показатели пассива (1400,1500,1300).")
        st.info("💡 Рекомендуем объединить страницы актива и пассива в один PDF, либо ввести значения вручную в редакторе ниже.")

    missing = []
    for year in years_to_use:
        for m in required_for_analysis:
            if pd.isna(df.loc[year, m]):
                missing.append(f"{year}: отсутствует **{m}**")
    if missing:
        st.warning("⚠️ Обнаружены пропуски в данных. Результаты могут быть неточными.")
        for msg in missing[:5]:
            st.write(f"- {msg}")

    st.subheader("✏️ Редактирование всех извлечённых данных")
    editable = df[KEY_METRICS].T.reset_index()
    editable.columns = ['Показатель'] + [str(c) for c in df.index]
    edited = st.data_editor(editable, use_container_width=True, num_rows="dynamic", key="editor")
    edited_years = [col for col in edited.columns if col != 'Показатель']
    for y in edited_years:
        for m in KEY_METRICS:
            row = edited[edited['Показатель'] == m].index
            if not row.empty:
                val = edited.loc[row[0], y]
                try:
                    if val == "" or val is None:
                        df.loc[int(y), m] = None
                    else:
                        df.loc[int(y), m] = float(val)
                except:
                    df.loc[int(y), m] = None
    st.success("✅ Данные обновлены.")

    st.subheader("📋 Источники данных")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Показатели бухгалтерского баланса**")
        st.dataframe(df[BALANCE_METRICS].T.style.format("{:.2f}", na_rep="данные не найдены"))
    with col2:
        st.markdown("**Показатели отчёта о финансовых результатах**")
        st.dataframe(df[INCOME_METRICS].T.style.format("{:.2f}", na_rep="данные не найдены"))

    # Расчёт коэффициентов
    def safe_div(a, b):
        return a / b if a is not None and b is not None and b != 0 else None
    def safe_sub(a, b):
        return a - b if a is not None and b is not None else None

    coeff_dict = defaultdict(dict)
    for year in years_to_use:
        r = df.loc[year]
        долгоср = r.get("Долгосрочные обязательства")
        краткоср = r.get("Краткосрочные обязательства")
        total_liab = долгоср + краткоср if долгоср is not None and краткоср is not None else None
        coeff_dict[year] = {
            "Коэффициент текущей ликвидности": safe_div(r.get("Оборотные активы"), r.get("Краткосрочные обязательства")),
            "Коэффициент быстрой ликвидности": safe_div(safe_sub(r.get("Оборотные активы"), r.get("Запасы")), r.get("Краткосрочные обязательства")),
            "Коэффициент абсолютной ликвидности": safe_div(r.get("Денежные средства"), r.get("Краткосрочные обязательства")),
            "ROE": safe_div(r.get("Чистая прибыль"), r.get("Собственный капитал")),
            "ROA": safe_div(r.get("Чистая прибыль"), r.get("Активы")),
            "Рентабельность продаж": safe_div(r.get("Чистая прибыль"), r.get("Выручка")),
            "Рентабельность до налогообложения": safe_div(r.get("Прибыль (убыток) до налогообложения"), r.get("Выручка")),
            "Коэффициент автономии": safe_div(r.get("Собственный капитал"), r.get("Активы")),
            "Финансовый рычаг": safe_div(total_liab, r.get("Собственный капитал")),
            "Покрытие процентов": safe_div(r.get("Прибыль (убыток) до налогообложения"), r.get("Проценты к уплате")),
            "Оборачиваемость активов": safe_div(r.get("Выручка"), r.get("Активы")),
            "Оборачиваемость дебиторской задолженности": safe_div(r.get("Выручка"), r.get("Дебиторская задолженность")),
        }

    df_coeff = pd.DataFrame(coeff_dict).T
    st.subheader("📈 Финансовые коэффициенты")
    st.dataframe(df_coeff.T.style.format("{:.2f}", na_rep="—"))

    if len(years_to_use) >= 2:
        st.subheader("📉 Динамика коэффициентов")
        avail = list(df_coeff.columns)
        for c in avail:
            if f"show_{c}" not in st.session_state:
                st.session_state[f"show_{c}"] = True
        b1, b2, _ = st.columns([1,1,5])
        with b1:
            if st.button("✓ Выбрать все", key="select_all"):
                for c in avail:
                    st.session_state[f"show_{c}"] = True
                st.rerun()
        with b2:
            if st.button("✗ Сбросить все", key="deselect_all"):
                for c in avail:
                    st.session_state[f"show_{c}"] = False
                st.rerun()
        st.write("**Выберите коэффициенты:**")
        cols = st.columns(3)
        for i, c in enumerate(avail):
            with cols[i % 3]:
                st.checkbox(c, key=f"show_{c}")
        selected = [c for c in avail if st.session_state[f"show_{c}"]]
        split = st.checkbox("Разделить на отдельные графики", False)
        if not selected:
            st.warning("Не выбрано ни одного коэффициента.")
        else:
            if split:
                n_cols = 2
                rows = (len(selected) + n_cols - 1) // n_cols
                for r in range(rows):
                    cs = st.columns(n_cols)
                    for ci in range(n_cols):
                        idx = r * n_cols + ci
                        if idx < len(selected):
                            c = selected[idx]
                            with cs[ci]:
                                fig, ax = plt.subplots(figsize=(8,4))
                                ax.plot(df_coeff.index, df_coeff[c], marker='o')
                                ax.set_title(c)
                                ax.set_xlabel("Год")
                                ax.set_ylabel("Значение")
                                ax.grid()
                                st.pyplot(fig)
                                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(14,6))
                for c in selected:
                    ax.plot(df_coeff.index, df_coeff[c], marker='o', label=c)
                ax.legend(loc='upper left', bbox_to_anchor=(1,1))
                ax.set_xlabel("Год")
                ax.grid()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Для графика нужно минимум два года.")

    # Интегральная оценка
    st.subheader("🏆 Интегральная оценка")
    def norm(val, low, high):
        if pd.isna(val):
            return None
        if val >= high:
            return 100
        if val <= low:
            return 0
        return (val - low) / (high - low) * 100

    scores = {}
    for year in years_to_use:
        total = 0
        groups = {}
        for group, coeffs in GROUPS.items():
            gsum = 0
            cnt = 0
            for c in coeffs:
                v = df_coeff.loc[year, c] if c in df_coeff.columns else None
                if c in current_thresholds:
                    lo, hi = current_thresholds[c]
                    n = norm(v, lo, hi)
                    if n is not None:
                        gsum += n
                        cnt += 1
            if cnt > 0:
                groups[group] = gsum / cnt
                total += groups[group] * WEIGHTS[group]
        scores[year] = (total, groups)
    avg_total = sum(t for t,_ in scores.values()) / len(scores)
    st.metric("Средний интегральный балл", f"{avg_total:.1f} / 100")
    num = len(years_to_use)
    cpr = 4
    for i in range(0, num, cpr):
        cols = st.columns(min(cpr, num - i))
        for j, year in enumerate(years_to_use[i:i+cpr]):
            with cols[j]:
                total, groups = scores[year]
                st.metric(f"{year} год", f"{total:.1f} / 100")
                for g, s in groups.items():
                    st.write(f"{g}: {s:.1f}")

    # Анализ рисков
    st.subheader("⚠️ Анализ потенциальных рисков")
    risks = defaultdict(list)
    for year in years_to_use:
        miss = sum(pd.isna(df.loc[year].get(m)) for m in required_for_analysis)
        if miss > 3:
            risks[year].append(f"⚠️ Недостаточно данных: пропущено {miss} показателей.")
        for c, v in df_coeff.loc[year].items():
            if pd.isna(v):
                continue
            if c in current_thresholds:
                lo, hi = current_thresholds[c]
                if v < lo:
                    risks[year].append(f"{c} = {v:.2f} (ниже нормы {lo:.2f})")
                elif v > hi * 1.5 and c in ["Финансовый рычаг", "Оборачиваемость активов"]:
                    risks[year].append(f"{c} = {v:.2f} (чрезмерно высок)")
    def prob_word(cnt):
        if cnt % 10 == 1 and cnt % 100 != 11:
            return "проблема"
        if 2 <= cnt % 10 <= 4 and (cnt % 100 < 10 or cnt % 100 >= 20):
            return "проблемы"
        return "проблем"
    if len(years_to_use) >= 2:
        for year in years_to_use:
            lst = risks.get(year, [])
            cnt = len(lst)
            w = prob_word(cnt)
            if lst:
                with st.expander(f"**{year} год: {cnt} {w}**"):
                    for r in lst:
                        st.write(f"- {r}")
            else:
                with st.expander(f"**{year} год: 0 проблем**"):
                    st.success("Проблем не обнаружено.")
    else:
        if risks:
            for year, lst in risks.items():
                cnt = len(lst)
                w = prob_word(cnt)
                st.warning(f"**{year} год: {cnt} {w}**")
                for r in lst:
                    st.write(f"- {r}")
        else:
            st.success("По коэффициентам отклонений не обнаружено.")

    # ---------- Анализ повторяемости рисков (исправленный блок) ----------
    if len(years_to_use) >= 2:
        with st.expander("🔁 Анализ повторяемости рисков"):
            patterns = defaultdict(list)
            for year in years_to_use:
                for c, v in df_coeff.loc[year].items():
                    if pd.isna(v):
                        continue
                    if c in current_thresholds:
                        lo, hi = current_thresholds[c]
                        if v < lo:
                            patterns[c].append((year, "ниже нормы"))
                        elif v > hi * 1.5 and c in ["Финансовый рычаг", "Оборачиваемость активов"]:
                            patterns[c].append((year, "чрезмерно высок"))
            found = False
            for c, occ in patterns.items():
                if len(occ) >= 2:
                    found = True
                    years = [str(y) for y,_ in occ]
                    yrs = sorted([y for y,_ in occ])
                    intervals = [yrs[i]-yrs[i-1] for i in range(1,len(yrs))]
                    
                    # Формирование сообщения о повторяемости
                    if intervals:
                        if all(i == 1 for i in intervals):
                            freq_msg = "постоянная проблема (каждый год)"
                        elif len(set(intervals)) == 1:
                            interval = intervals[0]
                            # склонение для "раз в X лет"
                            if interval % 10 == 1 and interval % 100 != 11:
                                year_word = "год"
                            elif 2 <= interval % 10 <= 4 and (interval % 100 < 10 or interval % 100 >= 20):
                                year_word = "года"
                            else:
                                year_word = "лет"
                            freq_msg = f"раз в {interval} {year_word}"
                        else:
                            # разные интервалы
                            int_str = ", ".join(map(str, intervals))
                            freq_msg = f"интервалы: {int_str} лет"
                    else:
                        freq_msg = ""
                    
                    st.markdown(f"**{c}** повторяется в годах: {', '.join(years)}. {freq_msg}")
            if not found:
                st.info("Нет повторяющихся рисков.")

    # Рекомендации
    st.subheader("💡 Рекомендации для менеджмента")
    avg = df_coeff.mean()
    recs = []
    for c, v in avg.items():
        if pd.isna(v):
            continue
        if c in current_thresholds:
            lo, hi = current_thresholds[c]
            if v < lo:
                action = ("увеличить ликвидность" if "ликвидности" in c else
                          "повысить рентабельность" if "RO" in c or "Рентабельность" in c else
                          "снизить долговую нагрузку" if "рычаг" in c else
                          "оптимизировать управление")
                recs.append(f"🔻 **{c}** (средний {v:.2f}) ниже нормы. Рекомендуется: {action}.")
            elif v > hi and c in ["Финансовый рычаг", "Оборачиваемость активов"]:
                recs.append(f"🔺 **{c}** (средний {v:.2f}) выше нормы. Возможно избыточное использование заёмных средств.")
    if not recs:
        recs.append("Все ключевые показатели в норме. Рекомендуется поддерживать текущую политику.")
    for r in recs:
        st.write(r)

       # Нейросетевой анализ (полнотекстовый) – один раз
    if use_gigachat and full_texts and GIGACHAT_AVAILABLE and GIGACHAT_CREDENTIALS:
        st.subheader("🤖 Анализ пояснительной записки (GigaChat)")
        if st.session_state.last_gigachat_response is not None:
            st.markdown(st.session_state.last_gigachat_response)
            st.download_button(
                label="📥 Скачать анализ",
                data=st.session_state.last_gigachat_response,
                file_name="ai_financial_analysis.txt",
                mime="text/plain"
            )
        else:
            with st.spinner("GigaChat анализирует текст..."):
                combined_text = "\n".join(full_texts)
                metrics_summary = format_metrics_summary(df_coeff, df)
                ai_prompt = f"""
                Ты - опытный финансовый аналитик. Проанализируй следующий текст пояснительной записки 
                к бухгалтерской отчетности и выяви ключевые качественные факторы, риски и возможности для компании.
                
                Также учти рассчитанные финансовые метрики:
                {metrics_summary}
                
                Текст пояснительной записки:
                {combined_text[:15000]}
                
                Пожалуйста, предоставь структурированный анализ в следующем формате:
                
                1. КЛЮЧЕВЫЕ РИСКИ (перечисли 3-5 основных рисков, упомянутых в тексте)
                2. ВОЗМОЖНОСТИ (перечисли 2-3 потенциальные возможности для компании)
                3. КАЧЕСТВЕННЫЕ ФАКТОРЫ (важные нематериальные факторы, влияющие на финансовое состояние)
                4. ОБЩИЙ ВЫВОД (краткое резюме о перспективах компании)
                """
                ai_analysis = analyze_with_gigachat(ai_prompt)
                if ai_analysis:
                    st.session_state.last_gigachat_response = ai_analysis
                    st.markdown(ai_analysis)
                    st.download_button(
                        label="📥 Скачать анализ",
                        data=ai_analysis,
                        file_name="ai_financial_analysis.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Не удалось получить анализ от GigaChat.")

    # Качество данных
    total = len(KEY_METRICS) * len(years_to_use)
    filled = df[KEY_METRICS].notna().sum().sum()
    missing = total - filled
    qual = filled / total if total else 0
    reliab = max(10, 100 - 20 * missing)
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("📊 Качество распознавания (доля заполненных)", f"{qual:.0%}")
    with col_b:
        st.metric("🛡️ Надёжность данных (штраф за пропуски)", f"{reliab:.0f}%")
    if missing:
        st.caption(f"Пропущено {missing} значений. Каждый пропуск снижает надёжность на 20% (минимум 10%).")
    if qual < 0.7:
        st.error("📌 Рекомендуется использовать более чёткие отчёты или проверить наличие кодов строк.")
    elif qual < 0.9:
        st.warning("📌 Часть данных не извлечена, возможны неточности.")
    else:
        st.success("📌 Данные пригодны для анализа.")
    st.caption("🔍 Извлечение данных выполнено по кодам строк из таблиц PDF (с улучшенным поиском).")