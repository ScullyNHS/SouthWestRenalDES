import random
import numpy as np
import simpy
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import openpyxl 
import plotly.express as px
import streamlit as st
from io import BytesIO

# ---------------- PARAMETERS ----------------
class g:
    sim_duration_days = 365 * 25   # 25 years
    annual_growth_rate = 0.025    # 2.5% per year
    sim_duration_years = int(sim_duration_days/365)
    year_duration = 365

    # ---------------- Baseline Data ----------------
    Unit = 'EXETER'
    prevalent_ICHD = 511
    prevalent_PD   = 79
    prevalent_HHD  = 16
    prevalent_LTx   = 195
    prevalent_CTx   = 391

    proportion_ICHD = 0.428691275
    proportion_PD = 0.066275168
    proportion_HHD = 0.013422819
    proportion_LTx = 0.163590604
    proportion_CTx = 0.328020134

    number_of_stations = 113

    # ---------------- Other parameters ----------------
    mean_consult_time = 4  # hours
    age_increment_per_day = 1 / 365.0

    # Age distributions
    min_age, max_age = 18, 90
    ichd_age_mean,pd_age_mean,hhd_age_mean=66,66,66 
    median_age_dialysis = 63.2
    mean_age_dialysis_transplant = 60
    tx_age_mean=57

    # Dialysis duration
    max_ichd_sessions = 1560
    max_24yr_sessions = 52*3*22
    max_29yr_sessions = 52*3*15
    max_34yr_sessions = 52*3*14
    max_39yr_sessions = 52*3*12
    max_44yr_sessions = 52*3*10
    max_49yr_sessions = 52*3*9
    max_54yr_sessions = 52*3*8
    max_59yr_sessions = 52*3*7
    max_64yr_sessions = 52*3*5
    max_69yr_sessions = 52*3*4
    max_74yr_sessions = 52*3*3
    max_79yr_sessions = 52*3*2
    max_80yr_sessions = 52*3*1
    max_CTx_sessions = 78
    
    # New patients per year

    new_KRT_patients = 210

    # ----------- Growth-driven arrivals ------------
    @staticmethod
    def expected_new_patients(years, modality):
        if modality == "ICHD":
            base = g.new_KRT_patients*g.proportion_ICHD
        elif modality == "PD":
            base = g.new_KRT_patients*g.proportion_PD
        elif modality == "HHD":
            base = g.new_KRT_patients*g.proportion_HHD
        elif modality == "Pre-emptive Transplant":
            base = g.new_KRT_patients*g.proportion_LTx
        elif modality == "Non-pre-emptive Transplant":
            base = g.new_KRT_patients*g.proportion_CTx
        else:
            return 0
        # prev = base * ((1 + g.annual_growth_rate) ** (years - 1))
        # curr = base * ((1 + g.annual_growth_rate) ** years)
        new_patients = base * ((1 + g.annual_growth_rate) ** years)
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
        self.max_sessions = 0 # Patient defined number of sessions


# ---------------- MODEL CLASS ----------------
class Model:
    def __init__(self, run_number, seed=None):
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.station = simpy.Resource(self.env, capacity=g.number_of_stations)
        self.station_usage_count = 0 # Counter for dialysis station usage
        self.run_number = run_number
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.results_df = pd.DataFrame(columns=[
            'Run Number','Patient Id', 'Patient Type', 'Entry Age',
            'Q time station', 'Time in dialysis station', 'No of Sessions', 'Exit Age', 'Year'
        ])

        # Create a dataframe to store snapshots of session counts
        self.sessions_per_year = pd.DataFrame(columns=[
            'Year',
            'Session Count'
        ])

        self.mean_q_time_station = 0
        self.queue_monitor = []

        # Add prevalent patients
        for _ in range(g.prevalent_ICHD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'ICHD')
            if p.entry_age >= 18 and p.entry_age < 25:
                p.max_sessions = g.max_24yr_sessions
            elif p.entry_age >= 25 and p.entry_age < 30:
                p.max_sessions = g.max_29yr_sessions
            elif p.entry_age >= 30 and p.entry_age < 35:
                p.max_sessions = g.max_34yr_sessions
            elif p.entry_age >= 35 and p.entry_age < 40:
                p.max_sessions = g.max_39yr_sessions
            elif p.entry_age >= 40 and p.entry_age < 45:
                p.max_sessions = g.max_44yr_sessions
            elif p.entry_age >= 45 and p.entry_age < 50:
                p.max_sessions = g.max_49yr_sessions
            elif p.entry_age >= 50 and p.entry_age < 55:
                p.max_sessions = g.max_54yr_sessions
            elif p.entry_age >= 55 and p.entry_age < 60:
                p.max_sessions = g.max_59yr_sessions
            elif p.entry_age >= 60 and p.entry_age < 65:
                p.max_sessions = g.max_64yr_sessions
            elif p.entry_age >= 65 and p.entry_age < 70:
                p.max_sessions = g.max_69yr_sessions
            elif p.entry_age >= 70 and p.entry_age < 75:
                p.max_sessions = g.max_74yr_sessions
            elif p.entry_age >= 75 and p.entry_age < 80:
                p.max_sessions = g.max_79yr_sessions
            elif p.entry_age >= 80:
                p.max_sessions = g.max_80yr_sessions
            else:
                return 0
            self.env.process(self.activity_generator_ICHD(p))
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'No of Sessions':0,'Exit Age': p.age,'Year': 0
            }
        for _ in range(g.prevalent_PD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'PD')
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'No of Sessions':0,'Exit Age': p.age,'Year': 0
            }
        for _ in range(g.prevalent_HHD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'HHD')
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'No of Sessions':0,'Exit Age': p.age,'Year': 0
            }
        for _ in range(g.prevalent_LTx):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Pre-emptive Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'No of Sessions': 0,
                'Exit Age': p.age,
                'Year': 0
            }

        for _ in range(g.prevalent_CTx):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Non-pre-emptive Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'No of Sessions': 0,
                'Exit Age': p.age,
                'Year': 0
            }

    # ---------------- Arrival Generators ----------------
    def generator_ICHD_arrivals(self):  
        year = 1 
        while self.env.now < g.sim_duration_days: 
            self.patient_counter += 1 
            p = Patient(self.patient_counter, 'ICHD')
            if p.entry_age >= 18 and p.entry_age < 25:
                p.max_sessions = g.max_24yr_sessions
            elif p.entry_age >= 25 and p.entry_age < 30:
                p.max_sessions = g.max_29yr_sessions
            elif p.entry_age >= 30 and p.entry_age < 35:
                p.max_sessions = g.max_34yr_sessions
            elif p.entry_age >= 35 and p.entry_age < 40:
                p.max_sessions = g.max_39yr_sessions
            elif p.entry_age >= 40 and p.entry_age < 45:
                p.max_sessions = g.max_44yr_sessions
            elif p.entry_age >= 45 and p.entry_age < 50:
                p.max_sessions = g.max_49yr_sessions
            elif p.entry_age >= 50 and p.entry_age < 55:
                p.max_sessions = g.max_54yr_sessions
            elif p.entry_age >= 55 and p.entry_age < 60:
                p.max_sessions = g.max_59yr_sessions
            elif p.entry_age >= 60 and p.entry_age < 65:
                p.max_sessions = g.max_64yr_sessions
            elif p.entry_age >= 65 and p.entry_age < 70:
                p.max_sessions = g.max_69yr_sessions
            elif p.entry_age >= 70 and p.entry_age < 75:
                p.max_sessions = g.max_74yr_sessions
            elif p.entry_age >= 75 and p.entry_age < 80:
                p.max_sessions = g.max_79yr_sessions
            elif p.entry_age >= 80:
                p.max_sessions = g.max_80yr_sessions
            else:
                return 0 
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'No of Sessions': 0,'Exit Age': p.age,'Year': year
            }
            self.env.process(self.activity_generator_ICHD(p))
            interarrival = g.interarrival_days("ICHD", year)
            yield self.env.timeout(random.expovariate(1.0 / interarrival))
            year = int(self.env.now / 365) + 1 
    
    def generator_CTx_arrivals(self):
        year = 1
        while self.env.now < g.sim_duration_days:
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Non-pre-emptive Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0, 'No of Sessions': 0, 
                'Exit Age': p.age,'Year': year
            }
            ## Cadaver patients require ICHD before their Tx
            ## This generator and activity account for this            
            self.env.process(self.activity_generator_CTx(p)) # Start patient activity process
            interarrival = g.interarrival_days("Non-pre-emptive Transplant", year)
            if not np.isfinite(interarrival) or interarrival <= 0:
                break
                      
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
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'No of Sessions': 0,'Exit Age': p.age,'Year': year
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
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0,'No of Sessions': 0,'Exit Age': p.age,'Year': year
            }
            yield self.env.timeout(random.expovariate(1.0 / interarrival))
            year = int(self.env.now / 365) + 1

    def generator_LTx_arrivals(self):
        year = 1
        while self.env.now < g.sim_duration_days:
            interarrival = g.interarrival_days("Pre-emptive Transplant", year)
            if not np.isfinite(interarrival) or interarrival <= 0:
                break
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Pre-emptive Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Run Number': run_number,'Patient Id': p.id,'Patient Type': p.type,'Entry Age': p.entry_age,
                'Q time station': 0,'Time in dialysis station': 0, 'No of Sessions': 0,
                 'Exit Age': p.age,'Year': year
            }
            yield self.env.timeout(random.expovariate(1.0 / interarrival))
            year = int(self.env.now / 365) + 1
        
    # Usage count of stations
    def station_increment(self):
        self.station_usage_count += 1
    
    def reset_station_increment(self):
        self.station_usage_count = 0
    

    # ---------------- Activities ----------------
    def activity_generator_ICHD(self, patient):
        total_dialysis_time = 0
        session_count = 0  # Count the number of dialysis sessions
        while (patient.age < g.max_age) and (session_count < patient.max_sessions): # Patient continues until max age/sessions
            if self.env.now < patient.next_eligible_day: #check if patient is eligible for next session
                yield self.env.timeout(1) # Wait for 1 day
                patient.age += g.age_increment_per_day # Increment age
                continue

            time_entered_queue = self.env.now # Record time when patient enters the queue
            with self.station.request() as req: # Request a dialysis station
                yield req  # Wait until the request is granted
                time_left_queue = self.env.now # Record time when patient leaves the queue
                patient.q_time_station += (time_left_queue - time_entered_queue) # Update total queue time
                mean_duration_days = g.mean_consult_time / 24.0 # Convert hours to days
                sampled_time = random.expovariate(1.0 / mean_duration_days) # Dialysis session duration
                total_dialysis_time += sampled_time # Update total dialysis time
                session_count += 1 # Increment session count
                model.station_increment() # increment station usage count
                patient.age += sampled_time * g.age_increment_per_day  # Increment age based on session duration
                yield self.env.timeout(sampled_time)  # Yield to allow other processes to run
            
            rest_days = 1 # 1 day rest between sessions
            patient.age += rest_days * g.age_increment_per_day # Increment age during rest
            yield self.env.timeout(rest_days) # Wait for rest period

            if session_count > 0 and session_count % 3 == 0: # After every 3 sessions
                patient.next_eligible_day = self.env.now + 2 # 2 days break after 3 sessions
    
        # Once the patient reaches the maximum age, record their details in the results table
        self.results_df.loc[len(self.results_df)] = {
        'Run Number': run_number,
        'Patient Id': patient.id,
        'Patient Type': patient.type,
        'Entry Age': patient.entry_age,
        'Q time station': patient.q_time_station,
        'Total dialysis time': total_dialysis_time,
        'No of Sessions': session_count,
        'Exit Age': patient.age,
        'Year': int(self.env.now / 365)
        }

    def activity_generator_CTx(self, patient):
        total_dialysis_time = 0
        session_count = 0  # Count the number of dialysis sessions
        while (session_count < g.max_CTx_sessions): # Patient continues until max age/sessions
            if self.env.now < patient.next_eligible_day: #check if patient is eligible for next session
                yield self.env.timeout(1) # Wait for 1 day
                patient.age += g.age_increment_per_day # Increment age
                continue

            time_entered_queue = self.env.now # Record time when patient enters the queue
            with self.station.request() as req: # Request a dialysis station
                yield req  # Wait until the request is granted
                time_left_queue = self.env.now # Record time when patient leaves the queue
                patient.q_time_station += (time_left_queue - time_entered_queue) # Update total queue time
                mean_duration_days = g.mean_consult_time / 24.0 # Convert hours to days
                sampled_time = random.expovariate(1.0 / mean_duration_days) # Dialysis session duration
                total_dialysis_time += sampled_time # Update total dialysis time
                session_count += 1 # Increment session count
                model.station_increment() # increment station usage count
                patient.age += sampled_time * g.age_increment_per_day  # Increment age based on session duration
                yield self.env.timeout(sampled_time)  # Yield to allow other processes to run
            
            rest_days = 1 # 1 day rest between sessions
            patient.age += rest_days * g.age_increment_per_day # Increment age during rest
            yield self.env.timeout(rest_days) # Wait for rest period

            if session_count > 0 and session_count % 3 == 0: # After every 3 sessions
                patient.next_eligible_day = self.env.now + 2 # 2 days break after 3 sessions

        # Once the patient reaches the maximum age, record their details in the results table
        self.results_df.loc[len(self.results_df)] = {
        'Run Number': run_number,
        'Patient Id': patient.id,
        'Patient Type': patient.type,
        'Entry Age': patient.entry_age,
        'Q time station': patient.q_time_station,
        'Total dialysis time': total_dialysis_time,
        'No of Sessions': session_count,
        'Exit Age': patient.age,
        'Year': int(self.env.now / 365)
        }   

    # ---------------- Queue Monitoring ----------------
    def monitor_queue(self, interval=30):
        while True:
            q_len = len(self.station.queue)
            self.queue_monitor.append((self.env.now, q_len))
            yield self.env.timeout(interval)

    #---------------- Station Monitoring ------------
    def yearly_station_snapshot(self):
        for year in range(1, g.sim_duration_years +1):
            yield self.env.timeout(g.year_duration)
            self.sessions_per_year.loc[len(self.sessions_per_year)] = {
            'Year':year, 
            'Session Count':self.station_usage_count}
            self.reset_station_increment()

    # ---------------- Run & Results ----------------
    def calculate_run_results(self):
        # Calculate mean queue time for this run
        self.mean_q_time_station = self.results_df['Q time station'].mean()

    def run(self):
        self.env.process(self.generator_ICHD_arrivals()) # Start the process that adds new ICHD patients to the system
        self.env.process(self.generator_PD_arrivals()) # Start the process that adds new PD patients to the system
        self.env.process(self.generator_HHD_arrivals()) # Start the process that adds new HHD patients to the system
        self.env.process(self.generator_LTx_arrivals()) # Start the process that adds new Pre-emptive Transplant patients to the system
        self.env.process(self.generator_CTx_arrivals()) # Start the process that adds new Non-pre-emptive Transplant patients to the system
        self.env.process(self.monitor_queue()) # Start queue monitoring
        self.env.process(self.yearly_station_snapshot()) # Start station snapshot monitoring

        self.env.run(until=g.sim_duration_days) # Run the simulation
        self.calculate_run_results() # Calculate results
        return self.results_df, self.sessions_per_year ,self.mean_q_time_station, self.queue_monitor #  Return results


# ---------------- STREAMLIT APP ----------------
st.title("Renal Unit Simulation Model")

st.write("This app allows you to test scenarios that might impact the demand and capacity of renal units and centres. This can be done by changing the parameters to the left and clicking Run Simulation below.")
st.image("flow 2.drawio.png")

# -------------------------------
# Sidebar with grouped parameters
# -------------------------------
def get_g_defaults():
    return {
        "sim_duration_days": g.sim_duration_days,
        "annual_growth_rate": g.annual_growth_rate*100,
        "prevalent_ICHD": g.prevalent_ICHD,
        "prevalent_PD": g.prevalent_PD,
        "prevalent_HHD": g.prevalent_HHD,
        "prevalent_LTx": g.prevalent_LTx,
        "prevalent_CTx": g.prevalent_CTx,
        "new_KRT_patients": g.new_KRT_patients,
        "proportion_ICHD": g.proportion_ICHD,
        "proportion_PD": g.proportion_PD,
        "proportion_HHD": g.proportion_HHD,
        "proportion_LTx": g.proportion_LTx,
        "proportion_CTx": g.proportion_CTx,
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
        "Annual Percentage Growth in Incidence",
        min_value=0.0, max_value=10.0,
        value=st.session_state.params["annual_growth_rate"],
        step=0.005
    )
    st.session_state.params["annual_growth_rate"]//100 = g.annual_growth_rate

# Patient Prevalence
with st.sidebar.expander("Patient Prevalence", expanded=False):
    g.prevalent_ICHD = st.number_input(
        "Prevalent ICHD", 0, 5000,
        value=st.session_state.params["prevalent_ICHD"]
    )
    g.prevalent_PD = st.number_input(
        "Prevalent PD", 0, 5000,
        value=st.session_state.params["prevalent_PD"]
    )
    g.prevalent_HHD = st.number_input(
        "Prevalent HHD", 0, 5000,
        value=st.session_state.params["prevalent_HHD"]
    )
    g.prevalent_LTx = st.number_input(
        "Prevalent Pre-Emptive Tx", 0, 5000,
        value=st.session_state.params["prevalent_LTx"]
    )
    g.prevalent_CTx = st.number_input(
        "Prevalent Non-Pre-Emptive Tx", 0, 5000,
        value=st.session_state.params["prevalent_CTx"]
    )
    st.session_state.params.update({
        "prevalent_ICHD": g.prevalent_ICHD,
        "prevalent_PD": g.prevalent_PD,
        "prevalent_HHD": g.prevalent_HHD,
        "prevalent_LTx": g.prevalent_LTx,
        "prevalent_CTx": g.prevalent_CTx,
    })

# Patient Incidence
with st.sidebar.expander("Patient Incidence", expanded=False):

    g.new_KRT_patients = st.number_input(
        "KRT patient incidence per year", 0, 10000,
        value=st.session_state.params["new_KRT_patients"]
    )
    g.proportion_ICHD = st.number_input(
        "Proportion ICHD", 
        min_value=0.0, max_value=1.0,
        value=st.session_state.params["proportion_ICHD"],
        step=0.005
    )
    g.proportion_ICHD = st.number_input(
        "Proportion PD", 
        min_value=0.0, max_value=1.0,
        value=st.session_state.params["proportion_PD"],
        step=0.005
    )
    g.proportion_ICHD = st.number_input(
        "Proportion HHD", 
        min_value=0.0, max_value=1.0,
        value=st.session_state.params["proportion_HHD"],
        step=0.005
    )
    g.proportion_ICHD = st.number_input(
        "Proportion PTx", 
        min_value=0.0, max_value=1.0,
        value=st.session_state.params["proportion_LTx"],
        step=0.005
    )
    g.proportion_ICHD = st.number_input(
        "Proportion NPTx", 
        min_value=0.0, max_value=1.0,
        value=st.session_state.params["proportion_CTx"],
        step=0.005
    )
    st.session_state.params.update({
        "new_KRT_patients": g.new_KRT_patients,
        "proportion_ICHD": g.proportion_ICHD,
        "proportion_PD": g.proportion_PD,
        "proportion_HHD": g.proportion_HHD,
        "proportion_LTx": g.proportion_LTx,
        "proportion_CTx": g.proportion_CTx,
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
    #all_results_list = []
    all_results = []
    all_sessions_list = []

    for run_number in range(1, num_runs + 1):
        model = Model(run_number=run_number, seed=run_number)
        results, sessions, mean_q, q_monitor = model.run()
        mean_queue_times.append(mean_q)
        all_queue_monitors.append(q_monitor)
        #all_results_list.append(results)
        all_results.append(results)
        all_sessions_list.append(sessions)

    all_results_list = pd.concat(all_results, ignore_index=True)
    
    # average_mean_q = sum(mean_queue_times) / num_runs
    # st.write(f"### Average mean queue time over {num_runs} runs: {average_mean_q:.2f} days")

    # q_monitor_array = np.array([np.array(q)[:, 1] for q in all_queue_monitors])
    # avg_queue_length = np.mean(q_monitor_array, axis=0)

    # q_df = pd.DataFrame(all_queue_monitors[0], columns=["Day", "QueueLength"])
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(q_df["Day"] / 365, avg_queue_length, label="Average Queue length")
    # for yr in [5, 10, 15, 20, 25]:
    #     ax.axvline(x=yr, color="red", linestyle="--", alpha=0.6)
    # ax.set_xlabel("Years")
    # ax.set_ylabel("Queue length")
    # ax.set_title(f"Average Dialysis Queue over {num_runs} Runs")
    # ax.legend()
    # ax.grid(True)
    # st.pyplot(fig)

    # volume_tables = []
    # for results in all_results_list:
    #     vt = results.groupby(['Year', 'Patient Type']).size().unstack(fill_value=0)
    #     volume_tables.append(vt)

    # avg_volume_table = sum(volume_tables) / len(volume_tables)
    # avg_volume_table = avg_volume_table.round(2)
    new_patients_filter = all_results_list[all_results_list['No of Sessions']==0]
    new_patients_df = new_patients_filter.groupby(['Run Number','Year', 'Patient Type']).size().reset_index(name='count')

    # Sum and average
    avg_volume_table = new_patients_df.groupby(['Year', 'Patient Type'])['count'].mean().unstack(fill_value=0)
    avg_volume_table_2 = new_patients_df.groupby(['Year', 'Patient Type'])['count'].mean().reset_index(name='count')
    avg_volume_table_2 = avg_volume_table_2[avg_volume_table_2['Year']!=0]

    st.write("### Average Patient Incidence by Year and Modality")
    #st.dataframe(avg_volume_table_2)

    fig1 = px.line(
        avg_volume_table_2,
        x="Year",
        y="count",
        facet_col = "Patient Type",
        facet_col_wrap = 2,
        labels={"count": "Incidence", "index": "Year", "variable": "Patient Type"},
        title="Average Incidence By Year"
    )

    fig1.update_yaxes(matches=None)

    st.plotly_chart(fig1, use_container_width=True)

     # Download Excel
    # output = BytesIO()
    # with pd.ExcelWriter(output, engine="openpyxl") as writer:
    #     avg_volume_table.to_excel(writer, sheet_name="Avg Patients by Year")
    # st.download_button(
    #     label="Download results as Excel",
    #     data=output.getvalue(),
    #     file_name="Renal_patient_incidence.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )

    # ## Sessions

    avg_sessions_table = sum(all_sessions_list) / len(all_sessions_list)
    avg_sessions_table = avg_sessions_table.round(2)  
    st.write("### Average Number of Sessions by Year")
    #st.dataframe(avg_sessions_table)

    # Line chart
    fig2 = px.line(
        avg_sessions_table,
        x="Year",
        y="Session Count",
        labels={"value": "Average No. of Sessions", "index": "Year"},
        title="Average No. of Sessions By Year"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Download Excel
    # output = BytesIO()
    # with pd.ExcelWriter(output, engine="openpyxl") as writer:
    #     avg_sessions_table.to_excel(writer, sheet_name="Avg Sessions by Year")
    # st.download_button(
    #     label="Download results as Excel",
    #     data=output.getvalue(),
    #     file_name="Renal_session_volumes.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )

    # ## Exit patients output

    exit_patients_filter = all_results_list[all_results_list['No of Sessions']>0]
    exit_patients_df = exit_patients_filter.groupby(['Run Number','Year', 'Patient Type']).size().reset_index(name='count')

    # Sum and average
    avg_exit_table = exit_patients_df.groupby(['Year', 'Patient Type'])['count'].mean().unstack(fill_value=0)
    avg_exit_table_2 = exit_patients_df.groupby(['Year', 'Patient Type'])['count'].mean().reset_index(name='count')

    st.write("### Average Number of Patients Completing Dialysis by Year")
    st.dataframe(avg_exit_table)

    fig3 = px.line(
        avg_exit_table_2,
        x="Year",
        y="count",
        facet_col = "Patient Type",
        facet_col_wrap = 2,
        labels={"count": "Average No. of Pts Completing Dialysis", "index": "Year", "variable": "Patient Type"},
        title="Average Dialysis Completions By Year"
    )

    fig3.update_yaxes(matches=None)

    st.plotly_chart(fig3, use_container_width=True)

    # Download Excel
    # output = BytesIO()
    # with pd.ExcelWriter(output, engine="openpyxl") as writer:
    #     avg_exit_table.to_excel(writer, sheet_name="Avg Patients by Year")
    # st.download_button(
    #     label="Download results as Excel",
    #     data=output.getvalue(),
    #     file_name="Renal_exit_volumes.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )


    # ## Pt Prevalence output

    prevalence_df = new_patients_df.merge(
        exit_patients_df,
        on=["Run Number", "Year", "Patient Type"],
        how="left",
        suffixes=("_new", "_exit")  # renames the Count columns
    )

    prevalence_df_ichd = prevalence_df[ prevalence_df["Patient Type"] == "ICHD" ]

    # Calculate the difference between new and exit patients and prevalence
    prevalence_df_ichd["count_exit"] = prevalence_df_ichd["count_exit"].fillna(0)
    prevalence_df_ichd["Count_diff"] = prevalence_df_ichd["count_new"] - prevalence_df_ichd["count_exit"]
    prevalence_df_ichd["Count_Prev"] = (
    prevalence_df_ichd.groupby("Run Number")["Count_diff"].cumsum()
    )
    
    
    # Sum and average
    avg_prev_table = prevalence_df_ichd.groupby(['Year', 'Patient Type'])['Count_Prev'].mean().unstack(fill_value=0)
    st.write("### Average Prevalence by Year")
    #st.dataframe(avg_prev_table)

    # Line chart
    fig = px.line(
        avg_prev_table,
        labels={"value": "Average Prevalence", "index": "Year", "variable": "Patient Type"},
        title="Average Prevalence By Year"
    )

    st.plotly_chart(fig, use_container_width=True)

     # Download Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        avg_prev_table.to_excel(writer, sheet_name="Avg Patients by Year")
        avg_volume_table.to_excel(writer, sheet_name="Avg Incidence by Year")
        avg_sessions_table.to_excel(writer, sheet_name="Avg Sessions by Year")
        avg_exit_table.to_excel(writer, sheet_name="Avg HD Comp by Year")
    st.download_button(
        label="Download results as Excel",
        data=output.getvalue(),
        file_name="Renal_Unit_Simulation_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Simulation finished in {time.time() - start_time:.2f} seconds")

