# ERNA GUI

A simple Streamlit app for analyzing evoked resonant neural activity (ERNA) signals.
It loads .smr and .smrx files, shows stimulation data, finds peaks and troughs, and lets you export results.

<img width="1032" height="1153" alt="interface_overview" src="https://github.com/user-attachments/assets/000d0ca6-1390-4979-955b-965a3575d8f1" />

## Setup

Make sure Python is installed, then install the required packages:

**Command:**

```
pip install streamlit plotly numpy pandas scipy neo
```

For .smrx files also install sonpy and a recent neo:

```
pip install sonpy "neo>=0.14.6"
```

Older neo versions cannot read the current sonpy (1.9.12), and neo 0.14.6 needs Python 3.10 or newer. Platform support for sonpy:

* Windows: Python 3.10 to 3.14.
* macOS: Python 3.14 from python.org, running in Intel mode (Rosetta on Apple Silicon). Homebrew and conda Pythons cannot load sonpy because its macOS build is linked to the python.org install location.
* Linux: Python 3.14 on x86_64 only.

## How to Run

Open a terminal (Mac/Linux) or Command Prompt (Windows).

**macOS / Linux:**

```
streamlit run /path/to/erna_gui.py
```

**Windows:**

```
streamlit run C:\path\to\erna_gui.py
```

The app will open automatically in your web browser at a local address.

## Example File

Try the included example file:

```
data/example_file.smr
```

## What It Does

* Loads .smr and .smrx files recorded with Spike2
* Displays raw signals and evoked field averages
* Lets you select peaks, troughs, and analysis windows interactively
* Exports results to CSV and interactive HTML files
