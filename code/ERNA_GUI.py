import os
import warnings
warnings.filterwarnings("ignore", message="Units .* can not be converted")
import numpy as np
import scipy as sp
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from math import floor
import neo
import io
from scipy.signal import find_peaks
import subprocess
import sys

@st.cache_data
def filename_list(sidebar_path):
    smr_files = []

    for file in os.listdir(sidebar_path):
        if file.endswith('.smr'):
            smr_files.append(file)

    smr_files.sort()

    file_numbers = list(map(str,list(range(0,len(smr_files)))))

    numbered_smr_files =[]

    for x in range(len(smr_files)):
        numbered_smr_files.append(file_numbers[x] + ': ' + smr_files[x])

    return numbered_smr_files, smr_files

@st.cache_data
def import_smr(filename, path, WaveChan):
    FilePath = os.path.join(path, filename)

    reader = neo.io.Spike2IO(filename=FilePath)
    block = reader.read_block(lazy=False)
    seg = block.segments[0]

    signal = seg.analogsignals[WaveChan]

    raw_data = np.array(signal, dtype=float).flatten()
    fs = float(signal.sampling_rate)

    t = np.arange(len(raw_data)) / fs
    t_start = float(signal.t_start)
    t_stop = float(signal.t_stop)

    return raw_data, fs, t_start, t_stop, t

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def main():

    numbered_smr_files, smr_files = filename_list(sidebar_path)

    sidebar_filename_numbered = st.sidebar.selectbox('Select a file',numbered_smr_files)
    sidebar_filename = smr_files[numbered_smr_files.index(sidebar_filename_numbered)]

    file_indices = [i for i, s in enumerate(smr_files) if sidebar_filename in s]

    FilePath = os.path.join(sidebar_path, sidebar_filename)

    reader = neo.io.Spike2IO(filename=FilePath)
    block = reader.read_block(lazy=False)
    seg = block.segments[0]

    channels_all = []
    for i, signal in enumerate(seg.analogsignals):
        chan_names = signal.array_annotations.get("channel_names", [])
        if len(chan_names) > 0:
            channels_all.extend(chan_names)
        else:
            channels_all.append(f"Channel {i+1}")


    selected_name = st.sidebar.selectbox('Enter channel to import: ', channels_all)
    select_channel = channels_all.index(selected_name)

    raw_data, fs, t_start, t_stop, t = import_smr(sidebar_filename, sidebar_path, select_channel)
    n_samples = len(raw_data)

    # --- Auto-detect stimulation frequency ---
    # Threshold: 3 standard deviations above the mean
    signal_mean = np.mean(raw_data)
    signal_std = np.std(raw_data)
    pos_thresh = signal_mean + 3 * signal_std
    neg_thresh = -signal_mean + 3 * signal_std  # for inverted signal

    # Minimum distance between stim artifacts: 4 ms
    min_dist_4ms = max(1, round(0.004 * fs))

    # Detect stim artifacts in both polarities with 3SD threshold and 4ms min distance
    pos_detect = find_peaks(raw_data, height=pos_thresh, distance=min_dist_4ms)[0]
    neg_detect = find_peaks(-raw_data, height=neg_thresh, distance=min_dist_4ms)[0]

    # Pick the polarity with more detections — that's the stim artifact direction
    if len(neg_detect) > len(pos_detect):
        stim_peaks = neg_detect
    else:
        stim_peaks = pos_detect

    # Compute stim frequency from median inter-peak interval
    if len(stim_peaks) > 2:
        intervals = np.diff(stim_peaks)
        # Filter out outlier intervals (pauses, double-detections)
        med_interval = np.median(intervals)
        regular = intervals[(intervals > med_interval * 0.5) & (intervals < med_interval * 2.0)]
        if len(regular) > 0:
            detected_freq = round(fs / np.median(regular))
        else:
            detected_freq = round(fs / med_interval)
    elif len(stim_peaks) == 2:
        detected_freq = round(fs / np.diff(stim_peaks)[0])
    else:
        detected_freq = 100

    select_stim_frequency = st.sidebar.number_input(
        "Stimulation frequency (auto-detected)",
        min_value=1, max_value=300, value=int(detected_freq))

    st.sidebar.caption(f"Detected: {detected_freq} Hz")

    # --- Final stim artifact detection with detected frequency ---
    # Use half the stim period as min distance for final detection
    min_distance = max(1, round((1 / select_stim_frequency) / 2 * fs))
    pos = find_peaks(raw_data, height=pos_thresh, distance=min_distance)[0]
    neg = find_peaks(-raw_data, height=neg_thresh, distance=min_distance)[0]

    peak_locs = neg if (
        len(neg) > 0 and (
            np.max(np.abs(raw_data[neg])) >
            np.max(np.abs(raw_data[pos])) if len(pos) > 0 else True
        )
    ) else pos

    st.sidebar.caption(f"Stim artifacts detected: {len(peak_locs)}")

    if len(peak_locs) == 0:
        st.error("No stim artifacts detected. Try adjusting the channel or frequency.")
        return

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Evoked Fields", "Database"])

    with tab2:
        win = int(np.floor((1/select_stim_frequency)*fs))

        x = st.slider('Window slider',min_value=-50,max_value=50, value = -10)

        # Collect valid evoked fields using a list (faster than np.append in loop)
        evoked_list = []
        for i in peak_locs:
            start = i + x
            stop = i + win + x
            if start < 0 or stop > n_samples:
                continue
            evoked_list.append(raw_data[start:stop])

        if len(evoked_list) == 0:
            st.warning("No valid evoked fields found. Try adjusting the window slider.")
            average_EV = np.zeros(win)
        else:
            evoked_fields = np.array(evoked_list)
            average_EV = np.mean(evoked_fields, axis=0)

        t_EV = np.array(range(0,len(average_EV)))

        fig2 = go.Figure(data=go.Scatter(
                x = t_EV,
                y = average_EV,
                mode = 'lines',
                name='Stim_EV',
                line = dict(color='black'),
                showlegend=False))

        left, right = st.columns(2)

        POI_peaks = left.multiselect('Select peaks of interest', range(0,len(average_EV)))
        POI_troughs = right.multiselect('Select troughs of interest', range(0,len(average_EV)))

        POI_baseline = 0

        auto_win = st.slider('Auto-calulation window:', min_value=1, max_value=20, value = 10)

        fig2.add_trace(go.Scatter(
                x=POI_peaks,
                y=average_EV[POI_peaks],
                mode='markers',
                marker = dict(color = 'red'),
                showlegend=False))

        fig2.add_trace(go.Scatter(
                x=POI_troughs,
                y=average_EV[POI_troughs],
                mode='markers',
                marker = dict(color = 'blue'),
                showlegend=False))

        fig2.add_trace(go.Scatter(
                x=[POI_baseline],
                y=[average_EV[POI_baseline]],
                mode='markers',
                marker = dict(color = 'orange'),
                showlegend=False))

        for h in POI_peaks:

            fig2.add_shape(type='line',
                x0=h-auto_win,x1=h-auto_win,
                y0=min(average_EV),y1=max(average_EV),
                line = dict(color='red'),
                name='Auto Calc Start')

            fig2.add_shape(type='line',
                x0=h+auto_win,x1=h+auto_win,
                y0=min(average_EV),y1=max(average_EV),
                line = dict(color='red'),
                name='Auto Calc Stop')

        for h2 in POI_troughs:

            fig2.add_shape(type='line',
                x0=h2-auto_win,x1=h2-auto_win,
                y0=min(average_EV),y1=max(average_EV),
                line = dict(color='blue'),
                name='Auto Calc Start')

            fig2.add_shape(type='line',
                x0=h2+auto_win,x1=h2+auto_win,
                y0=min(average_EV),y1=max(average_EV),
                line = dict(color='blue'),
                name='Auto Calc Stop')


        fig2.update_layout(
                width=500,
                )

        st.plotly_chart(fig2)

    with tab1:

        fig = go.Figure(data=go.Scatter(
        x = t,
        y = raw_data,
        mode = 'lines',
        name='Raw Data',
        line = dict(color='darkgreen'),
        showlegend=False))

        database = []

        for idx, r in enumerate(peak_locs):

            baseline_idx = POI_baseline + x + r
            # Skip if baseline index is out of bounds
            if baseline_idx < 0 or baseline_idx >= n_samples:
                continue

            database.append(['Baseline',idx+1, baseline_idx/fs, raw_data[baseline_idx], 0])

            fig.add_trace(go.Scatter(
                    x=[baseline_idx/fs],
                    y=[raw_data[baseline_idx]],
                    mode='markers',
                    marker = dict(color = 'orange'),
                    showlegend=False))

            for idx1, d in enumerate(POI_peaks):

                search_start = d + x + r - auto_win
                search_stop = d + x + r + auto_win
                # Skip if search window is out of bounds
                if search_start < 0 or search_stop > n_samples:
                    continue

                max_peaks = np.argmax(raw_data[search_start:search_stop])

                loc = search_start + max_peaks

                database.append(['Peak '+str(idx1+1),int(idx+1),loc/fs, raw_data[loc],(loc/fs - baseline_idx/fs)*1000])

                fig.add_trace(go.Scatter(
                        x=[loc/fs],
                        y=[raw_data[loc]],
                        mode='markers',
                        marker = dict(color = 'red'),
                        showlegend=False))


            for idx2, d2 in enumerate(POI_troughs):

                search_start2 = d2 + x + r - auto_win
                search_stop2 = d2 + x + r + auto_win
                # Skip if search window is out of bounds
                if search_start2 < 0 or search_stop2 > n_samples:
                    continue

                min_peaks = np.argmin(raw_data[search_start2:search_stop2])

                loc2 = search_start2 + min_peaks

                database.append(['Trough '+str(idx2+1),idx+1,loc2/fs,raw_data[loc2],(loc2/fs - baseline_idx/fs)*1000])

                fig.add_trace(go.Scatter(
                        x=[loc2/fs],
                        y=[raw_data[loc2]],
                        mode='markers',
                        marker = dict(color = 'blue'),
                        showlegend=False))

        st.plotly_chart(fig)

        buffer = io.StringIO()
        fig.write_html(buffer, include_plotlyjs='cdn')
        html_bytes = buffer.getvalue().encode()

        st.download_button(
            label='Download HTML',
            data=html_bytes,
            file_name=sidebar_filename[0:-4]+".html",
            mime='text/html'
        )

    with tab3:
        if len(database) == 0:
            st.warning("No data to display.")
            return

        # Ensure consistent structure for database
        if isinstance(database[0], (int, float, str)):
            database = [database]

        df = pd.DataFrame(database, columns=['Type', 'Stim', 'Timestamp (s)', 'Voltage (V)', 'Latency (ms)'])
        df[["Stim", "Timestamp (s)", "Voltage (V)", "Latency (ms)"]] = df[["Stim", "Timestamp (s)", "Voltage (V)", "Latency (ms)"]].apply(pd.to_numeric)
        df.sort_values(by=['Timestamp (s)'], inplace=True)

        st.dataframe(df)
        st.download_button(label="Download data as CSV", data=convert_df(df), file_name=sidebar_filename[0:-4]+'.csv', mime='text/csv')


st.title('ERNA Analysis')
st.subheader('By Srdjan Sumarac')

sidebar_path = st.sidebar.text_input('Path to .smr files')

if len(sidebar_path) > 0:
    main()

if __name__ == "__main__" and "STREAMLIT_RUNNING" not in os.environ:
    os.environ["STREAMLIT_RUNNING"] = "1"
    script_path = os.path.abspath(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
