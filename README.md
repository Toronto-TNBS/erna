# ERNA GUI

A simple Streamlit app for analyzing evoked resonant neural activity (ERNA) signals.
It loads .smr files, shows stimulation data, finds peaks and troughs, and lets you export results.

<img width="1032" height="1153" alt="interface_overview" src="https://github.com/user-attachments/assets/000d0ca6-1390-4979-955b-965a3575d8f1" />

## Setup

Make sure Python is installed, then install the required packages:

**Command:**

```
pip install streamlit plotly numpy pandas scipy neo
```

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

* Loads .smr files recorded with Spike2
* Displays raw signals and evoked field averages
* Lets you select peaks, troughs, and analysis windows interactively
* Exports results to CSV and interactive HTML files
