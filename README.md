# SAR simulation environment 

**Authors:** Zoe Betta and Antonio Sgorbissa

This distribution extends version 39 with a bilingual welcome screen and a fully localised user interface.

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py
```

At startup, select **English** or **Italiano**. The selected language is used by the configuration window, the Pygame viewer, the Panda3D window title, final plots/tables, and console messages.

## Manuals

The `manuals` folder contains complete user manuals in both languages:

- `SAR_User_Manual_EN.docx`
- `SAR_User_Manual_EN.pdf`
- `SAR_User_Manual_EN.txt`
- `Manuale_Utente_SAR_IT.docx`
- `Manuale_Utente_SAR_IT.pdf`
- `Manuale_Utente_SAR_IT.txt`

The welcome screen includes an **Open manual / Apri manuale** button.

## Main controls

- `SPACE`: pause/resume
- `1`, `2`, `3`, `4`: 1x, 2x, 4x, 8x
- **Max Turbo**: fastest simulation without rendering
- **Enable/Disable 3D view**: Panda3D first-person view
- `ESC`: end the current episode

See the manuals for configuration options, experimental-design guidance, metrics, output files, and statistical interpretation.
