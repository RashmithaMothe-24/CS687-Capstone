"""
Page 7 — Admin & Export
- Inspect project artifacts (data/models/reports).
- Download individual files or a bundled ZIP.
- Optional cleanup utilities (clear feedback or all generated artifacts).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import io
import os
import zipfile

import streamlit as st

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0:
            return f"{size:,.2f} {u}"
        size /= 1024.0
    return f"{size:,.2f} PB"


def _iter_files(root_dirs: List[Path]) -> List[Tuple[str, Path, int]]:
    """
    Return list of (relative_display_path, absolute_path, size_bytes)
    for files under the given roots.
    """
    out: List[Tuple[str, Path, int]] = []
    for root in root_dirs:
        if not root.exists():
            continue
        for r, _, fs in os.walk(root):
            for f in fs:
                ap = Path(r) / f
                rp = ap.relative_to(Path(".")).as_posix()
                try:
                    size = ap.stat().st_size
                except Exception:
                    size = 0
                out.append((rp, ap, size))
    out.sort(key=lambda x: x[0])
    return out


def _zip_selected(files: List[Path]) -> bytes:
    """
    Create an in-memory ZIP containing the given absolute file paths.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ap in files:
            if ap.exists() and ap.is_file():
                # Write with a project-relative path
                arcname = ap.relative_to(Path(".")).as_posix()
                try:
                    zf.write(ap, arcname=arcname)
                except Exception:
                    # Skip files that cannot be read
                    continue
    buf.seek(0)
    return buf.read()


def _delete_files(files: List[Path]) -> int:
    deleted = 0
    for ap in files:
        try:
            if ap.exists() and ap.is_file():
                ap.unlink()
                deleted += 1
        except Exception:
            pass
    return deleted


def render():
    st.header("7) Admin & Export")

    st.markdown(
        """
Use this page to inspect generated artifacts, download individual files or a bundled ZIP,
and (optionally) clear feedback data or all generated artifacts.
        """
    )

    # --------------------------------------------------------------------------
    # Artifact listing
    # --------------------------------------------------------------------------
    st.subheader("Artifacts")

    roots = [DATA_DIR, MODELS_DIR, REPORTS_DIR]
    files = _iter_files(roots)

    if not files:
        st.info("No artifacts found yet. After running earlier pages, artifacts will appear here.")
        return

    # Tabular view with per-file download
    cols = st.columns([6, 2, 2])
    cols[0].markdown("**Path**")
    cols[1].markdown("**Size**")
    cols[2].markdown("**Download**")

    selectable_files: List[Path] = []
    for i, (rel, ap, size) in enumerate(files):
        with cols[0]:
            st.code(rel, language="text")
        with cols[1]:
            st.text(_human_size(size))
        with cols[2]:
            try:
                data = ap.read_bytes()
                st.download_button(
                    label="Download",
                    data=data,
                    file_name=ap.name,
                    mime="application/octet-stream",
                    key=f"dl_{i}",
                )
            except Exception as e:
                st.caption(f"Unavailable: {e}")
        selectable_files.append(ap)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Bundle download (ZIP)
    # --------------------------------------------------------------------------
    st.subheader("Bundle Export (ZIP)")

    with st.expander("Select files to include in the ZIP bundle"):
        indices = list(range(len(selectable_files)))
        default_selected = [i for i, (_, ap, _) in enumerate(files) if ap.parent in [MODELS_DIR, DATA_DIR]]
        sel = st.multiselect(
            "Choose files to include",
            options=indices,
            default=default_selected,
            format_func=lambda i: files[i][0],
        )

    if st.button("Create ZIP bundle"):
        chosen = [selectable_files[i] for i in sel] if sel else []
        if not chosen:
            st.warning("No files selected.")
        else:
            try:
                zip_bytes = _zip_selected(chosen)
                st.success(f"Created ZIP with {len(chosen)} files.")
                st.download_button(
                    label="Download bundle.zip",
                    data=zip_bytes,
                    file_name="bundle.zip",
                    mime="application/zip",
                )
            except Exception as e:
                st.error(f"Failed to create ZIP: {e}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Cleanup utilities
    # --------------------------------------------------------------------------
    st.subheader("Cleanup Utilities")

    # Clear feedback only
    fb_path = DATA_DIR / "feedback_labels.csv"
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Clear only feedback labels file (data/feedback_labels.csv)")
        confirm_fb = st.checkbox("Confirm delete feedback file")
        if st.button("Delete feedback file"):
            if confirm_fb:
                if fb_path.exists():
                    try:
                        fb_path.unlink()
                        st.success("Deleted data/feedback_labels.csv")
                    except Exception as e:
                        st.error(f"Failed to delete feedback file: {e}")
                else:
                    st.info("Feedback file not found.")
            else:
                st.warning("Please tick the confirmation checkbox first.")

    # Clear all generated artifacts
    with c2:
        st.caption("Clear generated artifacts (processed datasets, model bundles, reports)")
        confirm_all = st.checkbox("Confirm delete all artifacts (irreversible)")
        if st.button("Delete all artifacts"):
            if confirm_all:
                targets = []
                # data
                for name in ["processed_train.csv", "processed_valid.csv", "feedback_labels.csv", "datadict.json", "feature_manifest.json"]:
                    p = DATA_DIR / name
                    targets.append(p)
                # models
                for name in ["anomaly_model.joblib", "pipeline.joblib"]:
                    p = MODELS_DIR / name
                    targets.append(p)
                # reports (delete all files)
                for r, _, fs in os.walk(REPORTS_DIR):
                    for f in fs:
                        targets.append(Path(r) / f)

                deleted = _delete_files(targets)
                st.success(f"Deleted {deleted} files.")
            else:
                st.warning("Please tick the confirmation checkbox first.")

    st.caption("Note: Source CSVs you uploaded are not deleted by the 'all artifacts' option unless they are named exactly as above.")
