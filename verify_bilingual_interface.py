"""Headless regression checks for the version-40 bilingual interface."""
from pathlib import Path

from i18n import get_language, set_language, tr
from gui import RunViewer

BASE = Path(__file__).resolve().parent

set_language("en")
assert get_language() == "en"
assert tr("config_window_title") == "SAR simulation configuration"
assert RunViewer._legend_groups()[0][0] == "REAL ENVIRONMENT"

set_language("it")
assert get_language() == "it"
assert tr("config_window_title") == "Configurazione simulazione SAR"
assert RunViewer._legend_groups()[0][0] == "AMBIENTE REALE"

for path in [
    BASE / "sar_logo.png",
    BASE / "manuals" / "SAR_User_Manual_EN.docx",
    BASE / "manuals" / "SAR_User_Manual_EN.pdf",
    BASE / "manuals" / "SAR_User_Manual_EN.txt",
    BASE / "manuals" / "Manuale_Utente_SAR_IT.docx",
    BASE / "manuals" / "Manuale_Utente_SAR_IT.pdf",
    BASE / "manuals" / "Manuale_Utente_SAR_IT.txt",
]:
    assert path.exists() and path.stat().st_size > 0, path

print("Bilingual interface and manuals: OK")
