import random
import numpy as np
import simpy
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import openpyxl 

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
    ichd_age_mean,pd_age_mean,hhd_age_mean=66,66,66 #mean from Mickie's data
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
        if modality == "HHD" and new_patients < 1: # Ensure at least one new HHD patient if growth is very low
            return 1
        return int(new_patients)

    @staticmethod
    def interarrival_days(modality, year): # Calculate interarrival time in days
        new_patients = g.expected_new_patients(year, modality) # New patients expected this year
        if new_patients <= 0:
            return float("inf") # No new patients, return infinite interarrival time
        return 365.0 / new_patients # Interarrival time in days
          

# ---------------- PATIENT CLASS ----------------
class Patient:
    def __init__(self, p_id, patient_type):
        self.id = p_id # Unique patient ID
        self.type = patient_type # 'ICHD', 'PD', 'HHD', 'Transplant'
        self.entry_age = np.random.triangular(g.min_age,g.median_age_dialysis,g.max_age) # Triangular distribution for entry age
        self.age = self.entry_age # Current age
        self.q_time_station = 0 # Time spent waiting for a station
        self.next_eligible_day = 0 # Next day the patient is eligible for a station


# ---------------- MODEL CLASS ----------------
class Model:
    def __init__(self, run_number, seed=None): 
        self.env = simpy.Environment() # Simulation environment
        self.patient_counter = 0 # Counter for unique patient IDs
        self.station = simpy.Resource(self.env, capacity=g.number_of_stations) # Dialysis stations
        self.run_number = run_number # Current run number
        if seed is not None: 
            random.seed(seed)
            np.random.seed(seed)

        # Create a dataframe to store results for each patient (who they are, their wait times, etc.)
        self.results_df = pd.DataFrame(columns=[
            'Patient Id', 
            'Patient Type', 
            'Entry Age',
            'Q time station', 
            'Time in dialysis station', 
            'Exit Age', 
            'Year'
        ])
        self.mean_q_time_station = 0 # This will hold the average queue time for all patients in this run
        self.queue_monitor = [] # List to store queue lengths over time

        # Add existing (prevalent) ICHD patients
        # Only ICHD patients are simulated as processes because they use the shared dialysis station resource;
        for _ in range(g.prevalent_ICHD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'ICHD')
            self.env.process(self.activity_generator_ICHD(p))
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'Exit Age': p.age,
                'Year': 0
            }

        # Add existing (prevalent) PD patients
        # For PD, record their details directly (they don't use the shared station resource)
        for _ in range(g.prevalent_PD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'PD')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'Exit Age': p.age,
                'Year': 0
            }
        for _ in range(g.prevalent_HHD):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'HHD')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'Exit Age': p.age,
                'Year': 0
            }
        for _ in range(g.prevalent_Tx):
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'Transplant')
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'Exit Age': p.age,
                'Year': 0
            }

    # ---------------- Arrival Generators ----------------
    def generator_ICHD_arrivals(self):  
        year = 1 # Start from year 1
        while self.env.now < g.sim_duration_days: # Continue until simulation duration
            self.patient_counter += 1 # Increment patient counter
            p = Patient(self.patient_counter, 'ICHD') # Create new ICHD patient
            # Record initial patient details in results table
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'Exit Age': p.age,
                'Year': year
            }
            self.env.process(self.activity_generator_ICHD(p)) # Start patient activity process
            interarrival = g.interarrival_days("ICHD", year) # Calculate interarrival time
            yield self.env.timeout(random.expovariate(1.0 / interarrival)) # Wait for next arrival
            year = int(self.env.now / 365) + 1 # Update year

    def generator_PD_arrivals(self):
        year = 1
        while self.env.now < g.sim_duration_days: 
            interarrival = g.interarrival_days("PD", year) 
            if not np.isfinite(interarrival) or interarrival <= 0: 
                break 
            self.patient_counter += 1
            p = Patient(self.patient_counter, 'PD')
            # Record this patient's details in the results table
            # For PD patients, queue time and dialysis station time are set to zero (they don't use the shared station resource)
            # Exit age is set to their starting age, since their pathway isn't simulated further here
            self.results_df.loc[len(self.results_df)] = {
                'Patient Id': p.id,
                'Patient Type': p.type,
                'Entry Age': p.entry_age,
                'Q time station': 0,
                'Time in dialysis station': 0,
                'Exit Age': p.age,
                'Year': year
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
        session_count = 0  # Count the number of dialysis sessions
        while patient.age < g.max_age: # Patient continues until max age
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
                patient.age += sampled_time * g.age_increment_per_day  # Increment age based on session duration
                yield self.env.timeout(sampled_time)  # Yield to allow other processes to run
            
            rest_days = 1 # 1 day rest between sessions
            patient.age += rest_days * g.age_increment_per_day # Increment age during rest
            yield self.env.timeout(rest_days) # Wait for rest period

            if session_count > 0 and session_count % 3 == 0: # After every 3 sessions
                patient.next_eligible_day = self.env.now + 2 # 2 days break after 3 sessions

        # Once the patient reaches the maximum age, record their details in the results table
        self.results_df.loc[len(self.results_df)] = {
        'Patient Id': patient.id,
        'Patient Type': patient.type,
        'Entry Age': patient.entry_age,
        'Q time station': patient.q_time_station,
        'Total dialysis time': total_dialysis_time,
        'Sessions': session_count,
        'Exit Age': patient.age,
        'Year Exit': int(self.env.now / 365)
        }

    # ---------------- Queue Monitoring ----------------
    def monitor_queue(self, interval=30): # Monitor every 30 days
        while True:
            q_len = len(self.station.queue) # Count the number of patients currently waiting in the queue
            self.queue_monitor.append((self.env.now, q_len)) # Record the current simulation time and queue length in a list
            yield self.env.timeout(interval) # Wait for next interval

    # ---------------- Run & Results ----------------
    def calculate_run_results(self):
        # Calculate mean queue time for this run
        self.mean_q_time_station = self.results_df['Q time station'].mean()

    def run(self):
        self.env.process(self.generator_ICHD_arrivals()) # Start the process that adds new ICHD patients to the system
        self.env.process(self.generator_PD_arrivals()) # Start the process that adds new PD patients to the system
        self.env.process(self.generator_HHD_arrivals()) # Start the process that adds new HHD patients to the system
        self.env.process(self.generator_Tx_arrivals()) # Start the process that adds new Transplant patients to the system
        self.env.process(self.monitor_queue()) # Start queue monitoring

        self.env.run(until=g.sim_duration_days) # Run the simulation
        self.calculate_run_results() # Calculate results
        return self.results_df, self.mean_q_time_station, self.queue_monitor #  Return results


# ---------------- MAIN RUN ----------------
if __name__ == "__main__":
    start_time = time.time() #to identify how long the simulation takes
    g.sim_duration_days = 365 * 25  # change here for changing simulation years
    num_runs = 10  # Number of simulation runs
    mean_queue_times = []
    all_queue_monitors = []
    all_results_list = []

    for run_number in range(1, num_runs + 1):
        model = Model(run_number=run_number, seed=run_number)  # Use different seed for each run
        results, mean_q, q_monitor = model.run()
        mean_queue_times.append(mean_q)
        all_queue_monitors.append(q_monitor)
        all_results_list.append(results)

    # Calculate average mean queue time
    average_mean_q = sum(mean_queue_times) / num_runs
    print(f"Average mean queue time over {num_runs} runs: {average_mean_q:.2f}")

    # Aggregate queue monitor data for plotting
    import numpy as np
    q_monitor_array = np.array([np.array(q)[:,1] for q in all_queue_monitors])
    avg_queue_length = np.mean(q_monitor_array, axis=0)

    # Use the time points from the first run for plotting
    q_df = pd.DataFrame(all_queue_monitors[0], columns=["Day", "QueueLength"])
    plt.figure(figsize=(12, 6))
    plt.plot(q_df["Day"] / 365, avg_queue_length, label="Average Queue length")
    for i, yr in enumerate([5, 10, 15, 20, 25]):  # change here when you change the years
        label = f"{yr} years" if i == 0 else None
        plt.axvline(x=yr, color="red", linestyle="--", alpha=0.6, label=label)

    plt.xlabel("Years")
    plt.ylabel("Queue length (waiting for dialysis station)")
    plt.title(f"Average Dialysis Queue over {num_runs} Runs")
    plt.legend()
    plt.grid(True)
    plt.show()
    

# Calculate average patient volumes by year and modality across all runs
volume_tables = []
for results in all_results_list:
    vt = results.groupby(['Year', 'Patient Type']).size().unstack(fill_value=0)
    volume_tables.append(vt)

# Sum and average
avg_volume_table = sum(volume_tables) / len(volume_tables)
avg_volume_table = avg_volume_table.round(2)  

# Save to Excel
avg_volume_table.to_excel(r"\\tsclient\T\Renal Network\Renal HSMA Project\Renal_patient_volumes_avg_test1.xlsx", sheet_name='Avg Patients by Year')

print("Averaged patient volumes by year and modality saved to Renal_patient_volumes_avg.xlsx")
print(avg_volume_table)

print("Process finished --- %s seconds ---" % (time.time() - start_time)) #to identify how long the simulation takes
