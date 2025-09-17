import random
import numpy as np
import simpy
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import openpyxl 
import streamlit as st
from io import BytesIO

# ---------------- PARAMETERS ----------------
class g:
    sim_duration_days = 365 * 25   # 25 years
    annual_growth_rate = 0.025    # 2.5% per year

    # ---------------- Baseline Data ----------------
    Unit = 'EXETER'
    prevalent_ICHD = 511
    prevalent_PD   = 79
    prevalent_HHD  = 16
    prevalent_Tx   = 586
    number_of_stations = 20

    # ---------------- Other parameters ----------------
    mean_consult_time = 4  # hours
    age_increment_per_day = 1 / 365.0

    # Age distributions
    min_age, max_age = 18, 90
    ichd_age_mean,pd_age_mean,hhd_age_mean=66,66,66 
    median_age_dialysis = 63.2
    mean_age_dialysis_transplant = 60
    tx_age_mean=57
    
    # ----------- Growth-driven arrivals ------------
    @staticmethod
    def expected_new_patients(years, modality):
        if modality == "ICHD":
            base = g.prevalent_ICHD
        elif modality == "PD":
            base = g.prevalent_PD
        elif modality == "HHD":
            base = g.prevalent_HHD
        elif modality == "Transplant":
            base = g.prevalent_Tx
        else:
            return 0
        prev = base * ((1 + g.annual_growth_rate) ** (years - 1))
        curr = base * ((1 + g.annual_growth_rate) ** years)
        new_patients = curr - prev
        if modality == "HHD" and new_patients < 1:
            return 1
        return int(new_patients)

    @staticmethod
    def interarrival_days(modality, year):
        new_patients = g.expected_new_patients(year, modality)
        if new_patients <= 0:
            return float("inf")
        return 365.0 / new_patients
          

# ---------------- PATIENT CLASS ----------------
class Patient:
    def __init__(self, p_id, patient_type):
        self.id = p_id
        self.type = patient_type 
        self.entry_age = np.random.triangular(g.min_age,g.median_age_dialysis,g.max_age) 
        self.age = self.entry_age 
        self.q_time_station = 0 
        self.next_eligible_day = 0 


# ---------------- MODEL CLASS ----------------
class Model:
    def __init__(self, run_number, seed=None):
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.station = simpy.Resource(self.env, capacity=g.number_of_stations)
        self.run_number = run_number
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.results_df = pd.DataFrame(columns=[
            'Patient Id', 'Patient Type', 'Entry Age',
            'Q time station', 'Time in dialysis station', 'Exit Age', 'Year'
        ])
        self.mean_q_time_station = 0
        self.queue_monitor = []

        # Add prevalent patients
        for _ in range(g.prevalent_ICHD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'ICHD')
            self.env.process(self.activity_generator_ICHD(p))
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': 0
            }
        for _ in range(g.prevalent_PD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'PD')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': 0
            }
        for _ in range(g.prevalent_HHD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'HHD')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': 0
            }
        for _ in range(g.prevalent_Tx):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': 0
            }

    # ---------------- Arrival Generators ----------------
    def generator_ICHD_arrivals(self):  
        year = 1 
        while self.env.now < g.sim_duration_days: 
            self.patient_counter += 1 
            p = Patient(self.patient_counter, 'ICHD') 
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': year
            }
            self.env.process(self.activity_generator_ICHD(p))
            interarrival = g.interarrival_days("ICHD", year)
            yield self.env.timeout(random.expovariate(1.0 / interarrival))
            year = int(self.env.now / 365) + 1 

    def generator_PD_arrivals(self):
        year = 1
        while self.env.now < g.sim_duration_days: 
            interarrival = g.interarrival_days("PD", year) 
            if not np.isfinite(interarrival) or interarrival <= 0: 
                break 
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'PD')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': year
            }
            yield self.env.timeout(random.expovariate(1.0 / interarrival)) 
            year = int(self.env.now / 365) + 1 

    def generator_HHD_arrivals(self):
        year = 1
        while self.env.now < g.sim_duration_days:
            interarrival = g.interarrival_days("HHD", year)
            if not np.isfinite(interarrival) or interarrival <= 0:
                break
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'HHD')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': year
            }
            yield self.env.timeout(random.expovariate(1.0 / interarrival))
            year = int(self.env.now / 365) + 1

    def generator_Tx_arrivals(self):
        year = 1
        while self.env.now < g.sim_duration_days:
            interarrival = g.interarrival_days("Transplant", year)
            if not np.isfinite(interarrival) or interarrival <= 0:
                break
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'Exit Age': p.age,'Year': year
            }
            yield self.env.timeout(random.expovariate(1.0 / interarrival))
            year = int(self.env.now / 365) + 1

    # ---------------- Activities ----------------
    def activity_generator_ICHD(self, patient):
        total_dialysis_time = 0
        session_count = 0  
        while patient.age < g.max_age:
            if self.env.now < patient.next_eligible_day:
                yield self.env.timeout(1)
                patient.age += g.age_increment_per_day
                continue

            time_entered_queue = self.env.now
            with self.station.request() as req:
                yield req  
                time_left_queue = self.env.now
                patient.q_time_station += (time_left_queue - time_entered_queue)
                mean_duration_days = g.mean_consult_time / 24.0
                sampled_time = random.expovariate(1.0 / mean_duration_days)
                total_dialysis_time += sampled_time
                session_count += 1
                patient.age += sampled_time * g.age_increment_per_day  
                yield self.env.timeout(sampled_time)  
            
            rest_days = 1
            patient.age += rest_days * g.age_increment_per_day
            yield self.env.timeout(rest_days)

            if session_count > 0 and session_count % 3 == 0:
                patient.next_eligible_day = self.env.now + 2

        self.results_df.loc[len(self.results_df)] = {
        'Patient Id': patient.id,'Patient Type': patient.type,'Entry Age': patient.entry_age,
        'Q time station': patient.q_time_station,'Total dialysis time': total_dialysis_time,
        'Sessions': session_count,'Exit Age': patient.age,'Year Exit': int(self.env.now / 365)
        }     

    # ---------------- Queue Monitoring ----------------
    def monitor_queue(self, interval=30):
        while True:
            q_len = len(self.station.queue)
            self.queue_monitor.append((self.env.now, q_len))
            yield self.env.timeout(interval)

    # ---------------- Run & Results ----------------
    def calculate_run_results(self):
        self.mean_q_time_station = self.results_df['Q time station'].mean()

    def run(self):
        self.env.process(self.generator_ICHD_arrivals())
        self.env.process(self.generator_PD_arrivals())
        self.env.process(self.generator_HHD_arrivals())
        self.env.process(self.generator_Tx_arrivals())
        self.env.process(self.monitor_queue())

        self.env.run(until=g.sim_duration_days)
        self.calculate_run_results()
        return self.results_df, self.mean_q_time_station, self.queue_monitor


# ---------------- STREAMLIT APP ----------------
st.title("Renal Unit Simulation Model")

# -------------------------------
# Sidebar with grouped parameters
# -------------------------------
def get_g_defaults():
    return {
        "sim_duration_days": g.sim_duration_days,
        "annual_growth_rate": g.annual_growth_rate,
        "prevalent_ICHD": g.prevalent_ICHD,
        "prevalent_PD": g.prevalent_PD,
        "prevalent_HHD": g.prevalent_HHD,
        "prevalent_Tx": g.prevalent_Tx,
        "number_of_stations": g.number_of_stations,
        "mean_consult_time": g.mean_consult_time,
        "min_age": g.min_age,
        "max_age": g.max_age,
        "num_runs": 5
    }

# Initialize session state defaults if not set
if "params" not in st.session_state:
    st.session_state.params = get_g_defaults()

# Reset button
if st.sidebar.button("Reset to defaults"):
    st.session_state.params = get_g_defaults()

st.sidebar.header("Simulation Parameters")

# Growth & Duration
with st.sidebar.expander("Growth & Duration", expanded=True):
    sim_years = st.slider(
        "Simulation years",
        min_value=1, max_value=50,
        value=st.session_state.params["sim_duration_days"] // 365
    )
    g.sim_duration_days = 365 * sim_years
    st.session_state.params["sim_duration_days"] = g.sim_duration_days

    g.annual_growth_rate = st.number_input(
        "Annual growth rate (%)",
        min_value=0.0, max_value=0.1,
        value=st.session_state.params["annual_growth_rate"],
        step=0.005
    )
    st.session_state.params["annual_growth_rate"] = g.annual_growth_rate

# Patient Prevalence
with st.sidebar.expander("Patient Prevalence", expanded=False):
    g.prevalent_ICHD = st.number_input(
        "Prevalent ICHD", 0, 2000,
        value=st.session_state.params["prevalent_ICHD"]
    )
    g.prevalent_PD = st.number_input(
        "Prevalent PD", 0, 2000,
        value=st.session_state.params["prevalent_PD"]
    )
    g.prevalent_HHD = st.number_input(
        "Prevalent HHD", 0, 2000,
        value=st.session_state.params["prevalent_HHD"]
    )
    g.prevalent_Tx = st.number_input(
        "Prevalent Tx", 0, 2000,
        value=st.session_state.params["prevalent_Tx"]
    )
    st.session_state.params.update({
        "prevalent_ICHD": g.prevalent_ICHD,
        "prevalent_PD": g.prevalent_PD,
        "prevalent_HHD": g.prevalent_HHD,
        "prevalent_Tx": g.prevalent_Tx,
    })

# Dialysis Unit Setup
with st.sidebar.expander("Dialysis Unit Setup", expanded=False):
    g.number_of_stations = st.number_input(
        "Number of dialysis stations", 1, 200,
        value=st.session_state.params["number_of_stations"]
    )
    st.session_state.params["number_of_stations"] = g.number_of_stations

    g.mean_consult_time = st.slider(
        "Mean dialysis session length (hours)", 1, 8,
        value=st.session_state.params["mean_consult_time"]
    )
    st.session_state.params["mean_consult_time"] = g.mean_consult_time

# Simulation Settings
with st.sidebar.expander("Simulation Settings", expanded=False):
    g.min_age = st.number_input(
        "Minimum patient age", 0, 30,
        value=st.session_state.params["min_age"]
    )
    g.max_age = st.number_input(
        "Maximum patient age", 70, 120,
        value=st.session_state.params["max_age"]
    )
    st.session_state.params.update({
        "min_age": g.min_age,
        "max_age": g.max_age
    })

    num_runs = st.slider(
        "Number of simulation runs", 1, 50,
        value=st.session_state.params["num_runs"]
    )
    st.session_state.params["num_runs"] = num_runs

# -------------------------------
# Main Run
# -------------------------------
if st.button("Run Simulation"):
    start_time = time.time()
    mean_queue_times = []
    all_queue_monitors = []
    all_results_list = []

    for run_number in range(1, num_runs + 1):
        model = Model(run_number=run_number, seed=run_number)
        results, mean_q, q_monitor = model.run()
        mean_queue_times.append(mean_q)
        all_queue_monitors.append(q_monitor)
        all_results_list.append(results)

    average_mean_q = sum(mean_queue_times) / num_runs
    st.write(f"### Average mean queue time over {num_runs} runs: {average_mean_q:.2f} days")

    q_monitor_array = np.array([np.array(q)[:, 1] for q in all_queue_monitors])
    avg_queue_length = np.mean(q_monitor_array, axis=0)

    q_df = pd.DataFrame(all_queue_monitors[0], columns=["Day", "QueueLength"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(q_df["Day"] / 365, avg_queue_length, label="Average Queue length")
    for yr in [5, 10, 15, 20, 25]:
        ax.axvline(x=yr, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Years")
    ax.set_ylabel("Queue length")
    ax.set_title(f"Average Dialysis Queue over {num_runs} Runs")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    volume_tables = []
    for results in all_results_list:
        vt = results.groupby(['Year', 'Patient Type']).size().unstack(fill_value=0)
        volume_tables.append(vt)

    avg_volume_table = sum(volume_tables) / len(volume_tables)
    avg_volume_table = avg_volume_table.round(2)
    st.write("### Average patient volumes by year and modality")
    st.dataframe(avg_volume_table)

    # Download Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        avg_volume_table.to_excel(writer, sheet_name="Avg Patients by Year")
    st.download_button(
        label="Download results as Excel",
        data=output.getvalue(),
        file_name="Renal_patient_volumes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Simulation finished in {time.time() - start_time:.2f} seconds")
