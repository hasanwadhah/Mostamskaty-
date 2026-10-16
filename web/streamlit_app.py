"""مستمسكاتي على المتصفح — Copyright (c) 2026 حسن وضاح. All rights reserved."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
from iraqi_id_printer import Card, COPYRIGHT, PRESETS, layout_pages, render_page
from web.workflow import WebEditor, prepare_export

st.set_page_config(page_title="مستمسكاتي | حسن وضاح", page_icon="📄", layout="wide")
crop_component = components.declare_component("mostamskaty_crop", path=str(Path(__file__).parent / "crop_component"))

CSS = """
<style>
.stApp {direction:rtl} [data-testid="stSidebar"] {direction:rtl}
h1,h2,h3,p,label,[data-testid="stCaptionContainer"] {text-align:right}
h1 {color:#14395b} .block-container {max-width:1400px;padding-top:2rem}
[data-testid="stSidebar"] {min-width:330px;max-width:400px}
[data-testid="stFileUploader"] {direction:rtl}
[data-testid="stImage"] img {max-width:100%;height:auto;box-shadow:0 5px 22px #23375018}
.stButton button,.stDownloadButton button {border-radius:8px}
.hero {background:#14395b;color:#fff;padding:20px 25px;border-radius:14px;margin-bottom:18px}
.hero h1 {color:white;margin:0}.hero p {margin:4px 0;color:#d5e6f6}
</style>
"""


def initialize():
    if "cards" not in st.session_state:
        st.session_state.cards = [Card() for _ in range(4)]
        st.session_state.editor = None
        st.session_state.document_revision = 0
        st.session_state.upload_versions = [0] * 4
        st.session_state.exports = {}
        for index in range(4):
            st.session_state[f"width_{index}"] = 8.6
            st.session_state[f"height_{index}"] = 5.6


def apply_preset():
    width, height = PRESETS[st.session_state.preset]
    for index, card in enumerate(st.session_state.cards):
        card.width_cm, card.height_cm = width, height
        st.session_state[f"width_{index}"] = width
        st.session_state[f"height_{index}"] = height


def changed():
    st.session_state.document_revision += 1
    st.session_state.exports = {}


def queue_upload(key: str, index: int, side: str):
    uploaded = st.session_state.get(key)
    if uploaded is not None and st.session_state.editor is None:
        st.session_state.pending_upload = (uploaded.getvalue(), index, side)


def process_upload():
    pending = st.session_state.pop("pending_upload", None)
    if pending is None:
        return
    raw, index, side = pending
    card = st.session_state.cards[index]
    try:
        with st.spinner("جارٍ فتح الصورة وكشف حدودها…"):
            st.session_state.editor = WebEditor.from_upload(raw, card.width_cm / card.height_cm, index, side)
    except Exception as exc:
        st.error(f"تعذر فتح الصورة: {exc}")


def sidebar():
    with st.sidebar:
        st.header("إعداد المستمسكات")
        if st.button("مسح الجلسة بالكامل", width="stretch"):
            st.session_state.clear()
            st.rerun()
        st.caption("تطوير: حسن وضاح • بدون اشتراك داخل البرنامج")
        editing = st.session_state.editor is not None
        st.selectbox("النمط — مقاسات البطاقات الأربع", list(PRESETS), key="preset",
                     on_change=apply_preset, disabled=editing)
        horizontal = st.slider("التباعد الأفقي (mm)", 0.0, 30.0, 4.0, .5, key="horizontal")
        vertical = st.slider("التباعد العمودي (mm)", 0.0, 30.0, 4.0, .5, key="vertical")
        for index, card in enumerate(st.session_state.cards):
            with st.expander(f"البطاقة {index + 1}", expanded=index == 0):
                editing = st.session_state.editor is not None
                dims = st.columns(2)
                card.width_cm = dims[0].number_input("العرض (cm)", min_value=1., max_value=30.,
                                                     step=.1, key=f"width_{index}", disabled=editing)
                card.height_cm = dims[1].number_input("الارتفاع (cm)", min_value=1., max_value=30.,
                                                       step=.1, key=f"height_{index}", disabled=editing)
                for side, caption in (("front", "الوجه"), ("back", "الظهر")):
                    key = f"upload_{index}_{side}_{st.session_state.upload_versions[index]}"
                    st.file_uploader(f"رفع {caption}", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                     key=key, disabled=st.session_state.editor is not None,
                                     on_change=queue_upload, args=(key, index, side),
                                     help="حتى 12 MB للصورة؛ تُعالج نسخة حتى 2200 بكسل لتناسب الاستضافة المجانية.")
                    image = getattr(card, side)
                    if image is not None:
                        st.image(image, caption=f"✓ {caption} المعتمد", width="stretch")
                        if st.button(f"تعديل {caption} المعتمد", key=f"edit_{index}_{side}",
                                     disabled=st.session_state.editor is not None):
                            st.session_state.editor = WebEditor.from_card(
                                image, card.width_cm / card.height_cm, index, side)
                            st.rerun()
                if st.button("حذف البطاقة", key=f"clear_{index}", disabled=st.session_state.editor is not None,
                             width="stretch"):
                    card.front = card.back = None
                    st.session_state.upload_versions[index] += 1
                    changed()
                    st.rerun()
        st.caption(COPYRIGHT)
    return horizontal, vertical


def show_editor(editor: WebEditor):
    side_label = "الوجه" if editor.side == "front" else "الظهر"
    st.subheader(f"تحرير البطاقة {editor.card_index + 1} — {side_label}")
    st.caption("اسحب زوايا المستمسك الأربع، ثم اعتمد الصورة لإضافتها إلى A4.")
    value = crop_component(image=editor.image_url(), points=editor.points.tolist(),
                           width=editor.current.shape[1], height=editor.current.shape[0],
                           revision=editor.revision, key=f"crop_{editor.identity}",
                           default={"revision": editor.revision, "points": editor.points.tolist()})
    try:
        editor.update_points(value)
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
    if editor.message:
        st.caption(editor.message)
    actions = [
        ("تعرف على الحواف", "detect"), ("قص يدوي", "manual"), ("تطبيق القص", "crop"),
        ("تحسين الألوان", "enhance"), ("ضبط الميلان", "deskew"), ("تدوير 90°", "rotate"),
        ("ضبط الحواف", "border"), ("استعادة الأصل", "reset"),
    ]
    for start in range(0, len(actions), 3):
        columns = st.columns(3)
        for column, (label, action) in zip(columns, actions[start:start + 3]):
            if column.button(label, key=f"tool_{action}", width="stretch"):
                try:
                    with st.spinner("جارٍ معالجة الصورة…"):
                        editor.apply(action)
                    st.rerun()
                except Exception as exc:
                    st.error(f"تعذر تنفيذ العملية: {exc}")
    accept, cancel = st.columns(2)
    if accept.button("اعتماد وتثبيت", type="primary", width="stretch"):
        try:
            with st.spinner("جارٍ اعتماد المستمسك…"):
                result = editor.accept()
            setattr(st.session_state.cards[editor.card_index], editor.side, result)
            st.session_state.upload_versions[editor.card_index] += 1
            st.session_state.editor = None
            changed()
            st.rerun()
        except Exception as exc:
            st.error(f"راجع الزوايا قبل الاعتماد: {exc}")
    if cancel.button("إلغاء التحرير", width="stretch"):
        st.session_state.upload_versions[editor.card_index] += 1
        st.session_state.editor = None
        st.rerun()
    st.divider()


def show_preview(pages):
    if st.session_state.get("selected_page", 1) > len(pages):
        del st.session_state["selected_page"]
    page = st.selectbox("صفحات A4", list(range(1, len(pages) + 1)), key="selected_page",
                        format_func=lambda n: f"الصفحة {n} / {len(pages)} — A4")
    columns = st.columns([1, 2, 1])
    with columns[1]:
        st.image(render_page(pages[page - 1], 96), width="stretch")
    st.caption("A4 • 210 × 297 mm • الوجه يمينًا والظهر يسارًا • هامش 10 mm • المقاس الفعلي 100%")
    if len(pages) > 1:
        st.info(f"النتيجة {len(pages)} صفحات. الوثائق الكبيرة تُرتّب عموديًا وتُرحّل دون تصغير.")


def export_controls(horizontal, vertical, active):
    st.subheader("تنزيل النتيجة والطباعة")
    columns = st.columns(2)
    dpi = columns[0].selectbox("جودة التصدير", [300, 600], format_func=lambda v: f"{v} dpi", key="export_dpi")
    format_name = columns[1].selectbox("صيغة الملف", ["PDF", "PNG", "JPG"], key="export_format")
    cards = st.session_state.cards
    signature = (st.session_state.document_revision, horizontal, vertical, dpi,
                 tuple((c.width_cm, c.height_cm) for c in cards))
    if st.session_state.get("export_signature") != signature:
        st.session_state.exports = {}
        st.session_state.export_signature = signature
    disabled = not active or st.session_state.editor is not None
    if st.button("تجهيز ملف التنزيل", type="primary", disabled=disabled, width="stretch"):
        try:
            with st.spinner("جارٍ تجهيز جميع الصفحات بالمقاسات الفعلية…"):
                st.session_state.exports[format_name] = prepare_export(cards, horizontal, vertical, dpi, format_name)
        except Exception as exc:
            st.error(f"تعذر التصدير: {exc}")
    prepared = st.session_state.exports.get(format_name)
    if prepared and not disabled:
        data, mime, filename = prepared
        st.download_button("تنزيل الملف الجاهز", data=data, file_name=filename, mime=mime,
                           type="primary", width="stretch")
    st.caption("للطباعة: نزّل PDF وافتحه، ثم اختر الحجم الفعلي / Actual Size / 100%. لا توجد طباعة مباشرة إلى طابعتك من خادم الويب.")


def main():
    initialize()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>مستمسكاتي</h1><p>إعداد المستمسكات العراقية للطباعة — تطوير حسن وضاح</p></div>',
                unsafe_allow_html=True)
    st.info("نسخة المتصفح: تُرفع الصور إلى خادم الاستضافة لمعالجتها في جلستك. استخدم نسخة ويندوز إذا أردت المعالجة على جهازك دون رفع الصور.")
    process_upload()
    horizontal, vertical = sidebar()
    editor = st.session_state.editor
    if editor is not None:
        show_editor(editor)
    try:
        pages = layout_pages(st.session_state.cards, horizontal, vertical)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    active = any(card.front is not None or card.back is not None for card in st.session_state.cards)
    if not active:
        st.info("ابدأ برفع الوجه أو الظهر من البطاقة 1 في الشريط الجانبي، ثم اعتمد الصورة.")
    if editor is not None:
        with st.expander("معاينة A4 الحالية"):
            show_preview(pages)
    else:
        show_preview(pages)
    export_controls(horizontal, vertical, active)
    st.caption("كشف الحواف تقريبي ويحتاج مراجعة الزوايا. نسخة الويب الخفيفة تستخدم OpenCV دون مفاتيح API أو اشتراك ذكاء اصطناعي.")
    st.caption(COPYRIGHT)


if __name__ == "__main__":
    main()
