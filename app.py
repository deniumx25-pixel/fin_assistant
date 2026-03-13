import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os
import tempfile
import subprocess

# --- Проверка доступности OCR-компонентов в системе ---
def check_tesseract():
    """Проверяет, доступен ли Tesseract в системе."""
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_poppler():
    """Проверяет, доступен ли pdftoppm (часть Poppler) в системе."""
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

# OCR доступен только если и библиотеки, и системные компоненты присутствуют
OCR_AVAILABLE = OCR_LIBS_AVAILABLE and OCR_SYSTEM_AVAILABLE

st.set_page_config(
    page_title="ИИ-ассистент финансового директора",
    page_icon="🐍",
    layout="wide"
)

# --- Боковая панель с настройками ---
with st.sidebar:
    st.header("⚙️ Настройки")
    years_to_analyze = st.slider(
        "Количество лет для анализа",
        min_value=1, max_value=10, value=3,
        help="Сколько последних лет отчётности использовать для расчётов."
    )
    use_ocr = st.checkbox(
        "Использовать OCR для сканов",
        value=False,
        help="Если страницы отсканированы, включите эту опцию (требуется Tesseract и Poppler)."
    )
    debug_mode = st.checkbox("Режим отладки", value=False, help="Показывать промежуточные данные извлечения.")
    
    if use_ocr and not OCR_AVAILABLE:
        if not OCR_LIBS_AVAILABLE:
            st.error("Библиотеки OCR не установлены (pytesseract, pdf2image, pillow).")
        elif not OCR_SYSTEM_AVAILABLE:
            st.error("Системные компоненты OCR отсутствуют. Для работы OCR установите Tesseract и Poppler.")
        else:
            st.error("OCR недоступен по неизвестной причине.")
        st.info("Отключите OCR для работы с текстовыми PDF.")
    
    # Информационное сообщение об ОФР появляется только при выборе 3 и более лет
    if years_to_analyze >= 3:
        st.info(
            "📌 **Примечание:** Отчёт о финансовых результатах (ОФР) содержит данные только за два года "
            "(текущий и предыдущий). Для анализа за период более 2 лет необходимо загрузить ОФР за соответствующие "
            "предыдущие годы (например, из отдельных файлов)."
        )

st.title("🐍 ИИ-ассистент финансового директора")
st.markdown("""
Загрузите **PDF-файлы** с финансовой отчётностью.  
Программа извлекает данные из таблиц по бухгалтерским кодам строк, при необходимости используя OCR.
""")

# ------------------- Настройки и константы -------------------
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

THRESHOLDS = {
    "Коэффициент текущей ликвидности": (1.5, 2.5),
    "Коэффициент быстрой ликвидности": (0.8, 1.5),
    "Коэффициент абсолютной ликвидности": (0.2, 0.5),
    "ROE": (0.1, 0.2),
    "ROA": (0.05, 0.1),
    "Рентабельность продаж": (0.1, 0.2),
    "Рентабельность до налогообложения": (0.1, 0.2),
    "Коэффициент автономии": (0.5, 0.7),
    "Финансовый рычаг": (0.5, 1.0),
    "Покрытие процентов": (3, 5),
    "Оборачиваемость активов": (0.8, 1.5),
    "Оборачиваемость дебиторской задолженности": (5, 10)
}

GROUPS = {
    "Ликвидность": ["Коэффициент текущей ликвидности", "Коэффициент быстрой ликвидности", "Коэффициент абсолютной ликвидности"],
    "Рентабельность": ["ROE", "ROA", "Рентабельность продаж", "Рентабельность до налогообложения"],
    "Устойчивость": ["Коэффициент автономии", "Финансовый рычаг", "Покрытие процентов"],
    "Активность": ["Оборачиваемость активов", "Оборачиваемость дебиторской задолженности"]
}
WEIGHTS = {"Ликвидность": 0.2, "Рентабельность": 0.3, "Устойчивость": 0.3, "Активность": 0.2}

# ------------------- Функции для извлечения данных -------------------
def parse_number(s):
    """
    Преобразует строку в число.
    - Если строка содержит только символы тире (-, –, —), возвращает 0.0.
    - Иначе пытается извлечь число.
    """
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
    """Извлекает текст со страницы PDF с помощью OCR (если включено и доступно)."""
    if not use_ocr or not OCR_AVAILABLE:
        return ""
    try:
        # Полагаемся на PATH, не передаём poppler_path
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        if images:
            # pytesseract должен быть доступен через PATH
            text = pytesseract.image_to_string(images[0], lang='rus+eng')
            return text
    except Exception as e:
        if debug_mode:
            st.warning(f"Ошибка OCR на странице {page_num}: {e}")
    return ""

def extract_data_from_pdf(pdf_path):
    """
    Основная функция извлечения данных.
    """
    years_data = defaultdict(dict)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Извлекаем таблицы
            tables = page.extract_tables()
            page_text = page.extract_text() or ""

            if debug_mode:
                st.info(f"Страница {page_num}: найдено таблиц: {len(tables)}")

            # Если таблиц нет или текст слишком короткий, и включён OCR, пробуем распознать
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

            # Обрабатываем каждую таблицу
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    if debug_mode:
                        st.info(f"  Таблица {table_idx}: слишком мало строк, пропущена")
                    continue

                # Ищем строку с годами в первых 20 строках
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

                # Определяем столбец с кодами: выбираем из первых 4 столбцов тот, где больше всего совпадений
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

                # Извлекаем данные
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
    return years_data

# ------------------- Загрузка файлов и основной анализ -------------------
uploaded_files = st.file_uploader("📎 Загрузите PDF-файлы с отчётностью", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_years_data = defaultdict(dict)

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name

        file_data = extract_data_from_pdf(tmp_path)

        for year, metrics in file_data.items():
            for metric, value in metrics.items():
                if metric not in all_years_data[year]:
                    all_years_data[year][metric] = value

        os.unlink(tmp_path)

    if not all_years_data:
        st.error("❌ Не удалось извлечь данные ни за один год. Убедитесь, что в файлах есть таблицы с кодами строк и годами. Если страницы отсканированы, включите OCR в настройках.")
        st.stop()

    df_raw = pd.DataFrame.from_dict(all_years_data, orient='index').sort_index()
    for metric in KEY_METRICS:
        if metric not in df_raw.columns:
            df_raw[metric] = None

    years_list = ', '.join(map(str, df_raw.index.unique()))
    st.success(f"✅ Извлечены данные: {years_list}")

    years_available = sorted(df_raw.index.unique())
    years_to_use = years_available[-years_to_analyze:] if len(years_available) >= years_to_analyze else years_available
    df = df_raw.loc[years_to_use].copy()

    required_for_analysis = ["Выручка", "Активы", "Долгосрочные обязательства", "Краткосрочные обязательства",
                             "Собственный капитал", "Оборотные активы", "Денежные средства",
                             "Чистая прибыль", "Прибыль (убыток) до налогообложения", "Проценты к уплате",
                             "Запасы", "Дебиторская задолженность"]

    # Проверка наличия пассивных показателей и рекомендация
    missing_passive = []
    for year in years_to_use:
        for m in ["Долгосрочные обязательства", "Краткосрочные обязательства", "Собственный капитал"]:
            if pd.isna(df.loc[year].get(m)):
                missing_passive.append(f"{year}: {m}")
    if missing_passive:
        st.warning("⚠️ Не удалось автоматически извлечь некоторые показатели пассива (1400,1500,1300).")
        st.info("💡 Возможно, таблица пассива находится на отдельной странице без указания годов. Рекомендуем объединить страницы актива и пассива в один PDF (например, через ilovepdf.com) и загрузить заново. Либо введите значения вручную в редакторе ниже.")

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
    st.markdown("При необходимости вы можете вручную исправить значения в таблице ниже. Изменения автоматически сохраняются.")

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

    # ------------------- Расчёт коэффициентов -------------------
    def safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        try:
            return a / b
        except:
            return None

    def safe_sub(a, b):
        if a is None or b is None:
            return None
        return a - b

    coeff_dict = defaultdict(dict)
    for year in years_to_use:
        r = df.loc[year]
        долгоср = r.get("Долгосрочные обязательства")
        краткоср = r.get("Краткосрочные обязательства")
        if долгоср is not None and краткоср is not None:
            total_liabilities = долгоср + краткоср
        else:
            total_liabilities = None

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

        # Доступные коэффициенты (без пропусков)
        available_coeffs = list(df_coeff.columns)

        # Инициализация session_state для каждого коэффициента (по умолчанию все выбраны)
        for coeff in available_coeffs:
            if f"show_{coeff}" not in st.session_state:
                st.session_state[f"show_{coeff}"] = True

        # Кнопки выбора всех / сброса
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

        # Отображение чекбоксов в несколько колонок
        st.write("**Выберите коэффициенты для отображения:**")
        num_cols = 3
        cols = st.columns(num_cols)
        for idx, coeff in enumerate(available_coeffs):
            with cols[idx % num_cols]:
                st.checkbox(
                    coeff,
                    key=f"show_{coeff}",
                    value=st.session_state[f"show_{coeff}"]
                )

        # Собираем выбранные коэффициенты
        selected_coeffs = [coeff for coeff in available_coeffs if st.session_state[f"show_{coeff}"]]

        # Опция разделения графиков
        split_option = st.checkbox("Разделить на отдельные графики", value=False)

        if not selected_coeffs:
            st.warning("Не выбрано ни одного коэффициента. График не построен.")
        else:
            if split_option:
                # Разделение на отдельные графики (по 2 в ряд)
                n_cols = 2
                rows = (len(selected_coeffs) + n_cols - 1) // n_cols
                for row in range(rows):
                    cols = st.columns(n_cols)
                    for col_idx in range(n_cols):
                        idx = row * n_cols + col_idx
                        if idx < len(selected_coeffs):
                            coeff = selected_coeffs[idx]
                            with cols[col_idx]:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(df_coeff.index, df_coeff[coeff], marker='o', linewidth=2)
                                ax.set_title(coeff)
                                ax.set_xlabel("Год")
                                ax.set_ylabel("Значение")
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
            else:
                # Единый график со всеми выбранными коэффициентами
                fig, ax = plt.subplots(figsize=(14, 6))
                for coeff in selected_coeffs:
                    ax.plot(df_coeff.index, df_coeff[coeff], marker='o', label=coeff)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.set_xlabel("Год")
                ax.set_ylabel("Значение")
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Для построения графика необходимо как минимум два года данных.")

    # Интегральная оценка
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
                if c in THRESHOLDS:
                    low, high = THRESHOLDS[c]
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

    col1, col2, col3 = st.columns(3)
    for i, year in enumerate(years_to_use):
        with [col1, col2, col3][i]:
            st.metric(f"{year} год", f"{scores_by_year[year]['total']:.1f} / 100")
            for g, s in scores_by_year[year]['groups'].items():
                st.write(f"{g}: {s:.1f}")

    # ------------------- Анализ рисков -------------------
    st.subheader("⚠️ Анализ потенциальных рисков")

    risks_by_year = defaultdict(list)
    for year in years_to_use:
        year_data = df.loc[year]
        missing_count = sum(pd.isna(year_data.get(m)) for m in required_for_analysis)
        if missing_count > 3:
            risks_by_year[year].append(f"⚠️ Недостаточно данных: пропущено {missing_count} показателей.")

        for coeff_name, value in df_coeff.loc[year].items():
            if pd.isna(value) or value is None:
                continue
            if coeff_name in THRESHOLDS:
                low, high = THRESHOLDS[coeff_name]
                if value < low:
                    risk_text = f"{coeff_name} = {value:.2f} (ниже нормы {low:.2f})"
                    risks_by_year[year].append(risk_text)
                elif value > high * 1.5:
                    if coeff_name in ["Финансовый рычаг", "Оборачиваемость активов"]:
                        risk_text = f"{coeff_name} = {value:.2f} (чрезмерно высок)"
                        risks_by_year[year].append(risk_text)

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

    # Анализ повторяемости
    if len(years_to_use) >= 2:
        with st.expander("🔁 Анализ повторяемости рисков"):
            risk_patterns = defaultdict(list)
            for year in years_to_use:
                for coeff_name, value in df_coeff.loc[year].items():
                    if pd.isna(value) or value is None:
                        continue
                    if coeff_name in THRESHOLDS:
                        low, high = THRESHOLDS[coeff_name]
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
                    years = [str(y) for y, _ in occurrences]
                    years_sorted = sorted([y for y, _ in occurrences])
                    intervals = []
                    for i in range(1, len(years_sorted)):
                        intervals.append(years_sorted[i] - years_sorted[i-1])
                    if intervals:
                        interval_str = ", ".join(map(str, intervals))
                        st.warning(f"**{coeff_name}** повторяется в годах: {', '.join(years)}. Интервалы: {interval_str} {year_word(intervals[0]) if len(intervals)==1 else 'лет'}.")
                    else:
                        st.warning(f"**{coeff_name}** повторяется в годах: {', '.join(years)}.")
            if not found_pattern:
                st.info("Нет повторяющихся рисков.")

    # Рекомендации
    st.subheader("💡 Рекомендации для менеджмента")

    avg_coeff = df_coeff.mean()
    recommendations = []
    for coeff_name, avg_val in avg_coeff.items():
        if pd.isna(avg_val) or avg_val is None:
            continue
        if coeff_name in THRESHOLDS:
            low, high = THRESHOLDS[coeff_name]
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

    # Качество и надёжность данных
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