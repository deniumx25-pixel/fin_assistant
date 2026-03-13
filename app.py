import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os
import tempfile
import subprocess
import json
from typing import Optional, List, Tuple, Set
from dotenv import load_dotenv

# Загрузка переменных окружения из .env (только для локальной разработки)
load_dotenv()

# --- Проверка доступности OCR-компонентов в системе (через PATH) ---
def check_tesseract():
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_poppler():
    try:
        subprocess.run(['pdftoppm', '-v'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

OCR_SYSTEM_AVAILABLE = check_tesseract() and check_poppler()

# --- Импорт OCR-библиотек (опционально) ---
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_LIBS_AVAILABLE = True
except ImportError:
    OCR_LIBS_AVAILABLE = False

OCR_AVAILABLE = OCR_LIBS_AVAILABLE and OCR_SYSTEM_AVAILABLE

# --- Импорт GigaChat (только GigaChat) ---
try:
    from gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

# --- Получение ключа GigaChat ---
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
if not GIGACHAT_CREDENTIALS and hasattr(st, "secrets"):
    GIGACHAT_CREDENTIALS = st.secrets.get("GIGACHAT_CREDENTIALS", "")

st.set_page_config(
    page_title="ИИ-ассистент финансового директора",
    page_icon="🐍",
    layout="wide"
)

with st.sidebar:
    st.header("⚙️ Настройки")
    years_to_analyze = st.slider(
        "Количество лет для анализа",
        min_value=1, max_value=10, value=3,
        help="Сколько последних лет отчётности использовать для расчётов."
    )
    
    with st.expander("📌 Примечание об ОФР"):
        st.info(
            "Отчёт о финансовых результатах (ОФР) содержит данные только за два года "
            "(текущий и предыдущий). Для анализа за период более 2 лет загрузите ОФР "
            "за соответствующие предыдущие годы."
        )
    
    # --- Выбор отрасли ---
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
    industry = st.selectbox(
        "Отрасль компании",
        options=industry_options,
        index=0,
        help="Пороговые значения коэффициентов будут подобраны под выбранную отрасль."
    )
    
    use_ocr = st.checkbox(
        "Использовать OCR для сканов",
        value=False,
        help="Требуется Tesseract и Poppler (будут автоматически доступны в облаке)."
    )
    
    if use_ocr:
        if not OCR_LIBS_AVAILABLE:
            st.error("❌ Не установлены Python-библиотеки для OCR. Выполните:\n"
                     "```bash\npip install pytesseract pdf2image pillow\n```")
        elif not OCR_SYSTEM_AVAILABLE:
            st.error("❌ Системные компоненты OCR отсутствуют. "
                     "В облаке они будут установлены через packages.txt. "
                     "Локально установите Tesseract и Poppler.")
        else:
            st.success("✅ OCR готов к использованию.")
    
    st.markdown("---")
    st.subheader("🤖 Нейросетевой анализ")
    use_gigachat = st.checkbox(
        "Использовать GigaChat для анализа пояснительных записок",
        value=False,
        help="Требуется ключ GigaChat (укажите в .env или secrets)."
    )
    fill_missing_with_ai = st.checkbox(
        "Заполнять пропуски через GigaChat",
        value=False,
        help="Если данные не извлечены, GigaChat попытается найти их в тексте."
    )
    
    if use_gigachat and not GIGACHAT_AVAILABLE:
        st.error("❌ Библиотека gigachat не установлена. Выполните:\n"
                 "```bash\npip install gigachat\n```")
    if use_gigachat and not GIGACHAT_CREDENTIALS:
        st.warning("⚠️ Не указан ключ GigaChat. Добавьте его в .env или Streamlit secrets.")
    
    debug_mode = st.checkbox("Режим отладки", value=False)

st.title("🐍 ИИ-ассистент финансового директора")
st.markdown("""
Загрузите **PDF-файлы** с финансовой отчётностью (РСБУ).  
Программа извлекает данные из таблиц по бухгалтерским кодам строк, при необходимости использует OCR.
GigaChat может анализировать пояснительные записки и заполнять пропуски.
""")

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

# ------------------- Функции для извлечения данных -------------------
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

def extract_text_with_ocr(pdf_path, page_num):
    if not use_ocr or not OCR_AVAILABLE:
        return ""
    try:
        # В облаке Poppler доступен через PATH, поэтому poppler_path не указываем
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        if images:
            text = pytesseract.image_to_string(images[0], lang='rus+eng')
            return text
    except Exception as e:
        if debug_mode:
            st.warning(f"Ошибка OCR на странице {page_num}: {e}")
    return ""

def extract_full_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
    except Exception as e:
        if debug_mode:
            st.warning(f"Ошибка извлечения текста: {e}")
    return full_text

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
                        lines = ocr_text.split('\n')
                        ocr_table = []
                        for line in lines:
                            cells = re.split(r'\s{2,}', line.strip())
                            if len(cells) > 1:
                                ocr_table.append(cells)
                        if ocr_table:
                            tables = [ocr_table]
                            if debug_mode:
                                st.info(f"Страница {page_num}: использован OCR для извлечения таблицы.")
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        if debug_mode:
                            st.info(f"  Таблица {table_idx}: слишком мало строк, пропущена")
                        continue
                    year_indices = {}
                    header_row_idx = None
                    for row_idx, row in enumerate(table[:20]):
                        for col_idx, cell in enumerate(row):
                            if cell:
                                year_match = re.search(r'(20\d{2})', str(cell))
                                if year_match:
                                    year = int(year_match.group(1))
                                    year_indices[col_idx] = year
                        if year_indices:
                            header_row_idx = row_idx
                            break
                    if not year_indices:
                        if debug_mode:
                            st.info(f"  Таблица {table_idx}: не найдены годы, пропущена")
                        continue
                    if debug_mode:
                        st.info(f"  Таблица {table_idx}: найдены годы {dict(year_indices)} (строка {header_row_idx})")
                    best_col = 0
                    max_matches = -1
                    for candidate_col in range(4):
                        matches = 0
                        for row in table[header_row_idx+1: header_row_idx+16]:
                            if row and len(row) > candidate_col:
                                code_cell = str(row[candidate_col]).strip()
                                if code_cell in CODE_TO_METRIC:
                                    matches += 1
                        if matches > max_matches:
                            max_matches = matches
                            best_col = candidate_col
                    code_col_idx = best_col
                    if debug_mode:
                        st.info(f"  Таблица {table_idx}: выбран столбец кодов {code_col_idx} (совпадений: {max_matches})")
                    for row in table[header_row_idx + 1:]:
                        if not row or len(row) <= code_col_idx:
                            continue
                        code_cell = str(row[code_col_idx]).strip()
                        if code_cell in CODE_TO_METRIC:
                            metric = CODE_TO_METRIC[code_cell]
                            for col_idx, year in year_indices.items():
                                if col_idx < len(row):
                                    cell_val = row[col_idx]
                                    if cell_val is not None:
                                        num = parse_number(cell_val)
                                        if num is not None:
                                            if metric not in years_data[year]:
                                                years_data[year][metric] = num
                                                if debug_mode:
                                                    st.info(f"    Найден код {code_cell} ({metric}) для года {year}: {num}")
    except Exception as e:
        if debug_mode:
            st.error(f"Ошибка при извлечении данных из PDF: {e}")
        return {}
    return dict(years_data)

# ------------------- Функции для GigaChat (упрощённые) -------------------
def analyze_with_gigachat(prompt: str) -> Optional[str]:
    if not use_gigachat or not GIGACHAT_AVAILABLE or not GIGACHAT_CREDENTIALS:
        return None
    try:
        if debug_mode:
            st.info("Попытка подключения к GigaChat...")
        with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False, scope='GIGACHAT_API_PERS') as giga:
            full_prompt = "Ты профессиональный финансовый аналитик.\n\n" + prompt
            response = giga.chat(full_prompt)
            return response.choices[0].message.content
    except Exception as e:
        if debug_mode:
            st.error(f"Ошибка при обращении к GigaChat: {e}")
            import traceback
            st.error(traceback.format_exc())
        return None

def format_metrics_summary(df_coeff: pd.DataFrame, df: pd.DataFrame) -> str:
    summary = []
    summary.append("ФИНАНСОВЫЕ МЕТРИКИ:")
    for year in df_coeff.index:
        summary.append(f"\n{year} год:")
        for coeff in df_coeff.columns:
            val = df_coeff.loc[year, coeff]
            if pd.notna(val):
                summary.append(f"- {coeff}: {val:.2f}")
    summary.append("\n\nКЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:")
    for year in df.index:
        summary.append(f"\n{year} год:")
        for metric in ["Выручка", "Чистая прибыль", "Активы"]:
            if metric in df.columns:
                val = df.loc[year, metric]
                if pd.notna(val):
                    summary.append(f"- {metric}: {val:,.0f} тыс. руб.")
    return "\n".join(summary)

def fill_missing_with_gigachat(df: pd.DataFrame, full_text: str, missing_candidates: List[Tuple[str, int]]) -> Tuple[pd.DataFrame, List[str]]:
    if not missing_candidates:
        return df, []
    
    missing_by_year = defaultdict(list)
    for code, year in missing_candidates:
        missing_by_year[year].append(code)
    
    prompt = "В тексте пояснительной записки к бухгалтерской отчетности найди значения следующих показателей по кодам строк за указанные годы.\n"
    prompt += "Если данные отсутствуют, укажи null. Ответ должен быть строго в формате JSON без дополнительного текста:\n"
    prompt += "{\n"
    for year in sorted(missing_by_year.keys()):
        codes = missing_by_year[year]
        if codes:
            prompt += f'  "{year}": {{\n'
            for code in codes:
                prompt += f'    "{code}": null,\n'
            prompt = prompt.rstrip(',\n') + '\n  },\n'
    prompt = prompt.rstrip(',\n') + '\n}\n\n'
    prompt += "Текст:\n" + full_text[:15000]
    
    response = analyze_with_gigachat(prompt)
    if not response:
        return df, []
    
    if debug_mode:
        st.info(f"Ответ GigaChat: {response}")
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)
    except Exception as e:
        if debug_mode:
            st.warning(f"Не удалось распарсить ответ GigaChat: {e}")
        return df, []
    
    changes = []
    for year_str, values in data.items():
        try:
            year = int(year_str)
        except:
            continue
        if year not in [y for _, y in missing_candidates]:
            continue
        for code, val in values.items():
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

# ------------------- Инициализация состояния сессии -------------------
if 'gigachat_done' not in st.session_state:
    st.session_state.gigachat_done = False
if 'processed_years' not in st.session_state:
    st.session_state.processed_years = set()
if 'last_gigachat_response' not in st.session_state:
    st.session_state.last_gigachat_response = None
if 'last_fill_changes' not in st.session_state:
    st.session_state.last_fill_changes = []

# ------------------- Загрузка файлов и основной анализ -------------------
uploaded_files = st.file_uploader("📎 Загрузите PDF-файлы с отчётностью", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_years_data = defaultdict(dict)
    full_texts = []
    current_years = set()
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name
        file_data = extract_data_from_pdf(tmp_path)
        if file_data is None:
            file_data = {}
        full_text = extract_full_text_from_pdf(tmp_path)
        if full_text:
            full_texts.append(full_text)
        for year, metrics in file_data.items():
            current_years.add(year)
            for metric, value in metrics.items():
                if metric not in all_years_data[year]:
                    all_years_data[year][metric] = value
        os.unlink(tmp_path)

    # Если появились новые года (которых не было в processed_years), сбрасываем флаг
    if current_years - st.session_state.processed_years:
        st.session_state.gigachat_done = False
        st.session_state.last_gigachat_response = None
        st.session_state.last_fill_changes = []

    if not all_years_data:
        st.error("❌ Не удалось извлечь данные. Убедитесь, что в файлах есть таблицы с кодами строк и годами. Если страницы отсканированы, включите OCR.")
        st.stop()

    df_raw = pd.DataFrame.from_dict(all_years_data, orient='index').sort_index()
    for metric in KEY_METRICS:
        if metric not in df_raw.columns:
            df_raw[metric] = None

    years_list = ', '.join(map(str, df_raw.index.unique()))
    st.success(f"✅ Извлечены данные: {years_list}")

    # ------------------- Проверка сходимости баланса (Актив = Пассив) -------------------
    balance_errors = []
    balance_ok_years = []
    for year in df_raw.index:
        if ("Активы" in df_raw.columns and 
            "Собственный капитал" in df_raw.columns and
            "Долгосрочные обязательства" in df_raw.columns and
            "Краткосрочные обязательства" in df_raw.columns):
            
            актив = df_raw.loc[year, "Активы"]
            кап = df_raw.loc[year, "Собственный капитал"]
            долг = df_raw.loc[year, "Долгосрочные обязательства"]
            крат = df_raw.loc[year, "Краткосрочные обязательства"]
            
            if pd.notna(актив) and pd.notna(кап) and pd.notna(долг) and pd.notna(крат):
                пассив_расчетный = кап + долг + крат
                # Допустимая погрешность: 1% или 1 тыс. руб.
                if abs(актив - пассив_расчетный) > max(1, abs(актив) * 0.01):
                    balance_errors.append(f"{year} (актив {актив:,.0f} ≠ пассив {пассив_расчетный:,.0f})")
                else:
                    balance_ok_years.append(year)
    if balance_errors:
        st.warning(f"⚠️ Баланс не сходится по следующим годам: {', '.join(balance_errors)}")
    else:
        if balance_ok_years:
            st.success(f"✅ Актив и пассив сходятся по всем проверенным годам: {', '.join(map(str, balance_ok_years))}")
        else:
            st.info("ℹ️ Недостаточно данных для проверки сходимости баланса.")

    years_available = sorted(df_raw.index.unique())
    years_to_use = years_available[-years_to_analyze:] if len(years_available) >= years_to_analyze else years_available
    df = df_raw.loc[years_to_use].copy()

    required_for_analysis = ["Выручка", "Активы", "Долгосрочные обязательства", "Краткосрочные обязательства",
                             "Собственный капитал", "Оборотные активы", "Денежные средства",
                             "Чистая прибыль", "Прибыль (убыток) до налогообложения", "Проценты к уплате",
                             "Запасы", "Дебиторская задолженность"]

    # ---------- Заполнение пропусков через GigaChat (только один раз) ----------
    fill_changes = []
    if use_gigachat and fill_missing_with_ai and full_texts and GIGACHAT_AVAILABLE and GIGACHAT_CREDENTIALS and not st.session_state.gigachat_done:
        missing_candidates = []
        for year in years_to_use:
            balance_filled = any(pd.notna(df.loc[year].get(m)) for m in BALANCE_METRICS if m in df.columns)
            income_filled = any(pd.notna(df.loc[year].get(m)) for m in INCOME_METRICS if m in df.columns)
            for code, metric in CODE_TO_METRIC.items():
                if metric not in df.columns or pd.isna(df.loc[year, metric]):
                    if metric in BALANCE_METRICS and balance_filled:
                        missing_candidates.append((code, year))
                    elif metric in INCOME_METRICS and income_filled:
                        missing_candidates.append((code, year))
        if missing_candidates:
            with st.spinner("GigaChat пытается найти недостающие данные в тексте..."):
                combined_text = "\n".join(full_texts)
                df, fill_changes = fill_missing_with_gigachat(df, combined_text, missing_candidates)
                st.session_state.last_fill_changes = fill_changes
        st.session_state.gigachat_done = True
        st.session_state.processed_years = set(df_raw.index)
    else:
        fill_changes = st.session_state.last_fill_changes

    if fill_changes:
        with st.expander("🤖 Данные, добавленные GigaChat"):
            for change in fill_changes:
                st.write(f"- {change}")

    # Проверка наличия пассивных показателей и рекомендация
    missing_passive = []
    for year in years_to_use:
        for m in ["Долгосрочные обязательства", "Краткосрочные обязательства", "Собственный капитал"]:
            if pd.isna(df.loc[year].get(m)):
                missing_passive.append(f"{year}: {m}")
    if missing_passive:
        st.warning("⚠️ Не удалось автоматически извлечь некоторые показатели пассива (1400,1500,1300).")
        st.info("💡 Возможно, таблица пассива находится на отдельной странице без указания годов. Рекомендуем объединить страницы актива и пассива в один PDF и загрузить заново, либо введите значения вручную в редакторе ниже, либо включите опцию 'Заполнять пропуски через GigaChat'.")

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
    editable_df = df[KEY_METRICS].T.reset_index()
    editable_df.columns = ['Показатель'] + [str(col) for col in df.index]
    edited = st.data_editor(editable_df, use_container_width=True, num_rows="dynamic", key="data_editor")
    edited_years = [col for col in edited.columns if col != 'Показатель']
    for year in edited_years:
        for metric in KEY_METRICS:
            row_idx = edited[edited['Показатель'] == metric].index
            if not row_idx.empty:
                val = edited.loc[row_idx[0], year]
                try:
                    if val == "" or val is None:
                        df.loc[int(year), metric] = None
                    else:
                        df.loc[int(year), metric] = float(val)
                except:
                    df.loc[int(year), metric] = None
    st.success("✅ Данные обновлены.")

    st.subheader("📋 Источники данных")
    col_bal, col_inc = st.columns(2)
    with col_bal:
        st.markdown("**Показатели бухгалтерского баланса**")
        st.dataframe(df[BALANCE_METRICS].T.style.format("{:.2f}", na_rep="данные не найдены"))
    with col_inc:
        st.markdown("**Показатели отчёта о финансовых результатах**")
        st.dataframe(df[INCOME_METRICS].T.style.format("{:.2f}", na_rep="данные не найдены"))

    # Расчёт коэффициентов
    def safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b
    def safe_sub(a, b):
        return a - b if a is not None and b is not None else None

    coeff_dict = defaultdict(dict)
    for year in years_to_use:
        r = df.loc[year]
        долгоср = r.get("Долгосрочные обязательства")
        краткоср = r.get("Краткосрочные обязательства")
        total_liabilities = долгоср + краткоср if долгоср is not None and краткоср is not None else None
        coeff_dict[year] = {
            "Коэффициент текущей ликвидности": safe_div(r.get("Оборотные активы"), r.get("Краткосрочные обязательства")),
            "Коэффициент быстрой ликвидности": safe_div(safe_sub(r.get("Оборотные активы"), r.get("Запасы")), r.get("Краткосрочные обязательства")),
            "Коэффициент абсолютной ликвидности": safe_div(r.get("Денежные средства"), r.get("Краткосрочные обязательства")),
            "ROE": safe_div(r.get("Чистая прибыль"), r.get("Собственный капитал")),
            "ROA": safe_div(r.get("Чистая прибыль"), r.get("Активы")),
            "Рентабельность продаж": safe_div(r.get("Чистая прибыль"), r.get("Выручка")),
            "Рентабельность до налогообложения": safe_div(r.get("Прибыль (убыток) до налогообложения"), r.get("Выручка")),
            "Коэффициент автономии": safe_div(r.get("Собственный капитал"), r.get("Активы")),
            "Финансовый рычаг": safe_div(total_liabilities, r.get("Собственный капитал")),
            "Покрытие процентов": safe_div(r.get("Прибыль (убыток) до налогообложения"), r.get("Проценты к уплате")),
            "Оборачиваемость активов": safe_div(r.get("Выручка"), r.get("Активы")),
            "Оборачиваемость дебиторской задолженности": safe_div(r.get("Выручка"), r.get("Дебиторская задолженность")),
        }

    df_coeff = pd.DataFrame(coeff_dict).T
    st.subheader("📈 Финансовые коэффициенты")
    st.dataframe(df_coeff.T.style.format("{:.2f}", na_rep="—"))

    if len(years_to_use) >= 2:
        st.subheader("📉 Динамика коэффициентов")
        available_coeffs = list(df_coeff.columns)
        for coeff in available_coeffs:
            if f"show_{coeff}" not in st.session_state:
                st.session_state[f"show_{coeff}"] = True

        col_btn1, col_btn2, _ = st.columns([1, 1, 5])
        with col_btn1:
            if st.button("✓ Выбрать все", key="select_all_btn"):
                for coeff in available_coeffs:
                    st.session_state[f"show_{coeff}"] = True
                st.rerun()
        with col_btn2:
            if st.button("✗ Сбросить все", key="deselect_all_btn"):
                for coeff in available_coeffs:
                    st.session_state[f"show_{coeff}"] = False
                st.rerun()

        st.write("**Выберите коэффициенты для отображения:**")
        num_cols = 3
        cols = st.columns(num_cols)
        for idx, coeff in enumerate(available_coeffs):
            with cols[idx % num_cols]:
                st.checkbox(coeff, key=f"show_{coeff}")

        selected_coeffs = [coeff for coeff in available_coeffs if st.session_state[f"show_{coeff}"]]
        split_option = st.checkbox("Разделить на отдельные графики", value=False)

        if not selected_coeffs:
            st.warning("Не выбрано ни одного коэффициента.")
        else:
            if split_option:
                n_cols = 2
                rows = (len(selected_coeffs) + n_cols - 1) // n_cols
                for row in range(rows):
                    cols = st.columns(n_cols)
                    for col_idx in range(n_cols):
                        idx = row * n_cols + col_idx
                        if idx < len(selected_coeffs):
                            coeff = selected_coeffs[idx]
                            with cols[col_idx]:
                                fig, ax = plt.subplots(figsize=(8,4))
                                ax.plot(df_coeff.index, df_coeff[coeff], marker='o', linewidth=2)
                                ax.set_title(coeff)
                                ax.set_xlabel("Год")
                                ax.set_ylabel("Значение")
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(14,6))
                for coeff in selected_coeffs:
                    ax.plot(df_coeff.index, df_coeff[coeff], marker='o', label=coeff)
                ax.legend(loc='upper left', bbox_to_anchor=(1,1))
                ax.set_xlabel("Год")
                ax.set_ylabel("Значение")
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Для построения графика необходимо как минимум два года данных.")

    # Интегральная оценка (с использованием отраслевых порогов)
    st.subheader("🏆 Интегральная оценка финансового состояния")
    def normalize(value, low, high):
        if pd.isna(value) or value is None:
            return None
        if value >= high:
            return 100
        elif value <= low:
            return 0
        else:
            return (value - low) / (high - low) * 100

    scores_by_year = {}
    for year in years_to_use:
        total = 0
        group_scores = {}
        for group, coeff_list in GROUPS.items():
            group_sum = 0
            count = 0
            for c in coeff_list:
                val = df_coeff.loc[year, c] if c in df_coeff.columns else None
                if c in current_thresholds:
                    low, high = current_thresholds[c]
                    norm = normalize(val, low, high)
                    if norm is not None:
                        group_sum += norm
                        count += 1
            if count > 0:
                group_avg = group_sum / count
                group_scores[group] = group_avg
                total += group_avg * WEIGHTS[group]
        scores_by_year[year] = {"total": total, "groups": group_scores}
    avg_total = sum(scores_by_year[year]["total"] for year in years_to_use) / len(years_to_use)
    st.metric("Средний интегральный балл за период", f"{avg_total:.1f} / 100")
    num_years = len(years_to_use)
    cols_per_row = 4
    for i in range(0, num_years, cols_per_row):
        cols = st.columns(min(cols_per_row, num_years - i))
        for j, year in enumerate(years_to_use[i:i+cols_per_row]):
            with cols[j]:
                st.metric(f"{year} год", f"{scores_by_year[year]['total']:.1f} / 100")
                for g, s in scores_by_year[year]['groups'].items():
                    st.write(f"{g}: {s:.1f}")

    # Анализ рисков (с отраслевыми порогами)
    st.subheader("⚠️ Анализ потенциальных рисков")
    risks_by_year = defaultdict(list)
    for year in years_to_use:
        year_data = df.loc[year]
        missing_count = sum(pd.isna(year_data.get(m)) for m in required_for_analysis)
        if missing_count > 3:
            risks_by_year[year].append(f"⚠️ Недостаточно данных: пропущено {missing_count} показателей.")
        for coeff_name, value in df_coeff.loc[year].items():
            if pd.isna(value):
                continue
            if coeff_name in current_thresholds:
                low, high = current_thresholds[coeff_name]
                if value < low:
                    risks_by_year[year].append(f"{coeff_name} = {value:.2f} (ниже нормы {low:.2f})")
                elif value > high * 1.5 and coeff_name in ["Финансовый рычаг", "Оборачиваемость активов"]:
                    risks_by_year[year].append(f"{coeff_name} = {value:.2f} (чрезмерно высок)")
    def problem_word(count):
        if count % 10 == 1 and count % 100 != 11:
            return "проблема"
        elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
            return "проблемы"
        else:
            return "проблем"
    if len(years_to_use) >= 2:
        for year in years_to_use:
            risk_list = risks_by_year.get(year, [])
            count = len(risk_list)
            word = problem_word(count)
            if risk_list:
                with st.expander(f"**{year} год: {count} {word}**"):
                    for r in risk_list:
                        st.write(f"- {r}")
            else:
                with st.expander(f"**{year} год: 0 проблем**"):
                    st.success("Проблем не обнаружено.")
    else:
        if risks_by_year:
            for year, risk_list in risks_by_year.items():
                count = len(risk_list)
                word = problem_word(count)
                st.warning(f"**{year} год: {count} {word}**")
                for r in risk_list:
                    st.write(f"- {r}")
        else:
            st.success("По коэффициентам значительных отклонений не обнаружено.")

    # Анализ повторяемости (с отраслевыми порогами)
    if len(years_to_use) >= 2:
        with st.expander("🔁 Анализ повторяемости рисков"):
            risk_patterns = defaultdict(list)
            for year in years_to_use:
                for coeff_name, value in df_coeff.loc[year].items():
                    if pd.isna(value):
                        continue
                    if coeff_name in current_thresholds:
                        low, high = current_thresholds[coeff_name]
                        if value < low:
                            risk_patterns[coeff_name].append((year, "ниже нормы"))
                        elif value > high * 1.5 and coeff_name in ["Финансовый рычаг", "Оборачиваемость активов"]:
                            risk_patterns[coeff_name].append((year, "чрезмерно высок"))
            def year_word(num):
                if num % 10 == 1 and num % 100 != 11:
                    return "год"
                elif 2 <= num % 10 <= 4 and (num % 100 < 10 or num % 100 >= 20):
                    return "года"
                else:
                    return "лет"
            found_pattern = False
            for coeff_name, occurrences in risk_patterns.items():
                if len(occurrences) >= 2:
                    found_pattern = True
                    years = [str(y) for y,_ in occurrences]
                    years_sorted = sorted([y for y,_ in occurrences])
                    intervals = [years_sorted[i]-years_sorted[i-1] for i in range(1,len(years_sorted))]
                    if intervals:
                        interval_str = ", ".join(map(str, intervals))
                        st.warning(f"**{coeff_name}** повторяется в годах: {', '.join(years)}. Интервалы: {interval_str} {year_word(intervals[0]) if len(intervals)==1 else 'лет'}.")
                    else:
                        st.warning(f"**{coeff_name}** повторяется в годах: {', '.join(years)}.")
            if not found_pattern:
                st.info("Нет повторяющихся рисков.")

    # Рекомендации (с отраслевыми порогами)
    st.subheader("💡 Рекомендации для менеджмента")
    avg_coeff = df_coeff.mean()
    recommendations = []
    for coeff_name, avg_val in avg_coeff.items():
        if pd.isna(avg_val):
            continue
        if coeff_name in current_thresholds:
            low, high = current_thresholds[coeff_name]
            if avg_val < low:
                action = ("увеличить ликвидность" if "ликвидности" in coeff_name else
                          "повысить рентабельность" if "RO" in coeff_name or "Рентабельность" in coeff_name else
                          "снизить долговую нагрузку" if "рычаг" in coeff_name else
                          "оптимизировать управление")
                recommendations.append(f"🔻 **{coeff_name}** (средний {avg_val:.2f}) ниже нормы. Рекомендуется: {action}.")
            elif avg_val > high and coeff_name in ["Финансовый рычаг", "Оборачиваемость активов"]:
                recommendations.append(f"🔺 **{coeff_name}** (средний {avg_val:.2f}) выше нормы. Возможно избыточное использование заёмных средств.")
    if not recommendations:
        recommendations.append("Все ключевые показатели в норме. Рекомендуется поддерживать текущую политику.")
    for rec in recommendations:
        st.write(rec)

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
    total_cells = len(KEY_METRICS) * len(years_to_use)
    filled = df[KEY_METRICS].notna().sum().sum()
    missing_count = total_cells - filled
    quality = filled / total_cells if total_cells else 0
    reliability = max(10, 100 - 20 * missing_count)
    st.markdown("---")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.metric("📊 Качество распознавания (доля заполненных)", f"{quality:.0%}")
    with col_q2:
        st.metric("🛡️ Надёжность данных (штраф за пропуски)", f"{reliability:.0f}%")
    if missing_count > 0:
        st.caption(f"Пропущено {missing_count} значений. Каждый пропуск снижает надёжность на 20% (минимум 10%).")
    if quality < 0.7:
        st.error("📌 Рекомендуется использовать более чёткие отчёты или проверить наличие кодов строк.")
    elif quality < 0.9:
        st.warning("📌 Часть данных не извлечена, возможны неточности.")
    else:
        st.success("📌 Данные пригодны для анализа.")
    st.caption("🔍 Извлечение данных выполнено по кодам строк из таблиц PDF (с улучшенным поиском).")