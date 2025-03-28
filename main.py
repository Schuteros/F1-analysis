import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastf1.plotting
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

ORDER = 10  # Sensitivity of peak detection
SMOOTHING_WINDOW = 25  # Adjust to control smoothing

def detect_sections(distance, speed):
    """Find local maxima and minima, refine them, and compute midpoints."""
    speed = savgol_filter(speed, window_length=SMOOTHING_WINDOW, polyorder=2)
    max_idxs, _ = find_peaks(speed, distance=ORDER)  # Find local maxima (acceleration points)
    speed = -speed
    min_idxs, _ = find_peaks(speed, distance=ORDER  )  # Find local minima (braking points)
    extremes = np.sort(np.concatenate([max_idxs, min_idxs]))  # Combine maxima and minima
    extremes = np.insert(extremes, 0, 0)  # Add start of the lap
    extremes = np.append(extremes, len(speed) - 1)  # Add end of the lap
    section_midpoints = (extremes[1:] + extremes[:-1]) / 2  # Compute midpoints
    section_midpoints = distance[section_midpoints.astype(int)]  # Convert midpoints to distance values
    
    return section_midpoints, max_idxs, min_idxs

def plot_section_graph(distance, speed, max_idxs, min_idxs, section_midpoints):
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(distance, speed, label="Speed", color="blue")

    # Plot refined local maxima
    ax.scatter(distance[max_idxs], speed[max_idxs], color='green', label="Refined Maxima (Acceleration)", zorder=3)

    # Plot refined local minima (braking points)
    ax.scatter(distance[min_idxs], speed[min_idxs], color='red', label="Refined Minima (Braking)", zorder=3)

    # Plot section midpoints
    ax.scatter(section_midpoints, np.interp(section_midpoints, distance, speed), color='black', label="Section Midpoints", zorder=3)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.legend()
    
    plt.show()
    

def calculate_section_times(laps, section_midpoints):
    section_times = {i: [] for i in range(len(section_midpoints) + 1)}

    for _, lap in laps.iterrows():
        data = lap.get_car_data().add_distance()
        data.loc[:, "Time (s)"] = data["Time"].dt.total_seconds()

        first_midpoint_time = data.iloc[(data['Distance'] - section_midpoints[0]).abs().argmin()]['Time (s)']
        section_times[0].append(first_midpoint_time)

        for i in range(1, len(section_midpoints)):
            prev_time = data.iloc[(data['Distance'] - section_midpoints[i - 1]).abs().argmin()]['Time (s)']
            current_time = data.iloc[(data['Distance'] - section_midpoints[i]).abs().argmin()]['Time (s)']
            section_times[i].append(current_time - prev_time)

        last_time = data.iloc[(data['Distance'] - section_midpoints[-1]).abs().argmin()]['Time (s)']
        end_time = data['Time (s)'].iloc[-1]
        section_times[len(section_midpoints)].append(end_time - last_time)

    average_section_times = {section: np.mean(times) if times else 0 for section, times in section_times.items()}
    return average_section_times

def get_section_times(laps, section_midpoints):
    section_times = {i: [] for i in range(len(section_midpoints) + 1)}

    for _, lap in laps.iterrows():
        data = lap.get_car_data().add_distance()
        data.loc[:, "Time (s)"] = data["Time"].dt.total_seconds()

        first_midpoint_time = data.iloc[(data['Distance'] - section_midpoints[0]).abs().argmin()]['Time (s)']
        section_times[0].append(first_midpoint_time)

        for i in range(1, len(section_midpoints)):
            prev_time = data.iloc[(data['Distance'] - section_midpoints[i - 1]).abs().argmin()]['Time (s)']
            current_time = data.iloc[(data['Distance'] - section_midpoints[i]).abs().argmin()]['Time (s)']
            section_times[i].append(current_time - prev_time)

        last_time = data.iloc[(data['Distance'] - section_midpoints[-1]).abs().argmin()]['Time (s)']
        end_time = data['Time (s)'].iloc[-1]
        section_times[len(section_midpoints)].append(end_time - last_time)
    return section_times

def find_and_normalize_section_times(average_section_times, car_average_section_times):
    section_time_differences = {sec: car_average_section_times[sec] - avg for sec, avg in average_section_times.items()}
    
    std_dev = np.std(list(section_time_differences.values())) or 1  # Avoid division by zero
    mean_diff = np.mean(list(section_time_differences.values()))
    
    normalized_section_time_differences = {sec: (diff - mean_diff) / std_dev for sec, diff in section_time_differences.items()}
    
    return normalized_section_time_differences

def calculate_driver_consistency(gp, best_driver, section_midpoints, car_laps):
    fastest_laps = gp.laps.pick_drivers(best_driver).pick_quicklaps().pick_accurate()
    fastest_car_section_times = get_section_times(fastest_laps, section_midpoints)
    car_section_times = get_section_times(car_laps, section_midpoints)
    fastest_car_section_sd = {}
    for section, times in fastest_car_section_times.items():
        fastest_car_section_sd[section] = np.std(times)

    car_section_sd = {}
    for section, times in car_section_times.items():
        car_section_sd[section] = np.std(times)

    driver_section_consistency = {sec: (std - fastest_car_section_sd[sec]) / (fastest_car_section_sd[sec] or 1)
                                  for sec, std in car_section_sd.items()}

    return driver_section_consistency

def analyze_performance(gp, section_midpoints, best_driver, driver_section_times):
    # Step 1: Compute section times for selected driver
    laps = gp.laps.pick_quicklaps().pick_accurate()
    avg_section_times = calculate_section_times(laps, section_midpoints)

    # Step 3: Compute Z-scores for time difference
    time_diff_z_scores = find_and_normalize_section_times(avg_section_times, driver_section_times)

    # Step 4: Compute driver consistency
    consistency_z_scores = calculate_driver_consistency(gp, best_driver, section_midpoints, laps)

    # Step 5: Create DataFrame with results
    df = pd.DataFrame({
        "Time Diff Z-Score": time_diff_z_scores,
        "Consistency": consistency_z_scores
    })

    return df

def problem_analysis(first_car_analysis, second_car_analysis):
    full_car_analysis =  pd.DataFrame(columns=["Time Diff Z-Score", "Consistency", "Problem"])
    for i, row in first_car_analysis.iterrows():
        full_car_analysis.loc[i, "Time Diff Z-Score"] = row["Time Diff Z-Score"]
        full_car_analysis.loc[i, "Consistency"] = row["Consistency"]
        if row["Time Diff Z-Score"] > 0.75:
            if row["Consistency"] > 0.2:
                if second_car_analysis.iloc[i]["Consistency"] > 0.2:
                    full_car_analysis.loc[i, "Problem"] = "Car"
                else: 
                    full_car_analysis.loc[i, "Problem"] = "Driver"
            else:
                if second_car_analysis.iloc[i]["Time Diff Z-Score"] > 0.75:
                    full_car_analysis.loc[i, "Problem"] = "Car"
                else: 
                    full_car_analysis.loc[i, "Problem"] = "Driver"
        elif row["Consistency"] > 0.2: 
            if second_car_analysis.iloc[i]["Consistency"] > 0.2:
                full_car_analysis.loc[i, "Problem"] = "Car"
            else: 
                full_car_analysis.loc[i, "Problem"] = "Driver"
        else:
            full_car_analysis.loc[i, "Problem"] = "None"
        
    return full_car_analysis


def get_average_telemetry(car_laps):
    """
    Compute the average telemetry data for all laps and return it as a DataFrame.
    """
    all_laps_data = []  # Store telemetry data per lap

    for _, lap in car_laps.iterrows():  # Iterate through each lap
        telemetry = lap.get_car_data().add_distance()  # Get telemetry for this lap

        if telemetry.empty:
            continue  # Skip empty laps

        # Convert boolean columns to integers
        if "Brake" in telemetry.columns:
            telemetry["Brake"] = telemetry["Brake"].astype(int)

        # Round each distance to the nearest 0.5m bin
        telemetry["DistanceBin"] = (telemetry["Distance"] / 0.5).round() * 0.5

        # Select only numeric columns
        numeric_cols = telemetry.select_dtypes(include=['number']).columns

        # Group by DistanceBin and compute average
        lap_avg = telemetry.groupby("DistanceBin", as_index=False)[numeric_cols].mean()

        # Store the result
        all_laps_data.append(lap_avg)

    if not all_laps_data:
        print("No valid lap data found.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid data is found

    # Combine all lap data and compute the final average per distance bin
    combined_telemetry = pd.concat(all_laps_data).groupby("DistanceBin", as_index=False).mean()

    # Drop bins that have all NaN values
    combined_telemetry = combined_telemetry.dropna(how="all")

    # Drop SessionTime and Time columns if they exist
    combined_telemetry = combined_telemetry.drop(columns=["SessionTime", "Time"], errors='ignore')

    # Return the cleaned and averaged telemetry data as a DataFrame
    return combined_telemetry

def synchronize_speed_data(distance, speed, avg_speed_distance, avg_speed):
    """
    Resamples both speed datasets onto a common distance grid.
    
    Parameters:
    - distance: Car's distance array (telemetry).
    - speed: Car's speed array.
    - avg_speed_distance: Distance array for the avg_speed data.
    - avg_speed: Average speed array at those distances.
    
    Returns:
    - new_distance: Synchronized distance array.
    - speed_interp: Car speed resampled to new distance points.
    - avg_speed_interp: Avg speed resampled to new distance points.
    """

    # Calculate the number of resampling points dynamically
    num_points = (len(distance) + len(avg_speed_distance)) // 2

    # Create a common distance range
    min_dist = max(min(distance), min(avg_speed_distance))
    max_dist = min(max(distance), max(avg_speed_distance))
    new_distance = np.linspace(min_dist, max_dist, num_points)

    # Interpolate both datasets to the new distance points
    speed_interp = np.interp(new_distance, distance, speed)
    avg_speed_interp = np.interp(new_distance, avg_speed_distance, avg_speed)

    return new_distance, speed_interp, avg_speed_interp

def get_average_speeds(car_laps):
    all_laps_data = []  # Store distance and speed data per lap

    for _, lap in car_laps.iterrows():  # Iterate through each lap
        telemetry = lap.get_car_data().add_distance()  # Get telemetry for this lap

        if telemetry.empty:
            continue  # Skip empty laps

        # Round each distance to the nearest 0.5m bin
        telemetry["DistanceBin"] = (telemetry["Distance"] / 0.5).round() * 0.5

        # Select only DistanceBin and Speed columns
        lap_avg = telemetry.groupby("DistanceBin", as_index=False)["Speed"].mean()

        all_laps_data.append(lap_avg)

    if not all_laps_data:
        print("No valid lap data found.")
        return None

    # Combine all lap data and compute the final average per distance bin
    combined_data = pd.concat(all_laps_data).groupby("DistanceBin").mean()

    # Drop bins that have all NaN values
    combined_data = combined_data.dropna(how="all")

    return combined_data  # Return averaged distance and speed data


def detect_slow_subsections(car_laps, problematic_sections, midpoints, all_laps):
    """
    Detect slow subsections within problematic sections based on speed drops.
    Synchronizes avg_speed with car speed data before comparison.
    Returns a new DataFrame with merged subsections added.
    """
    car_data = get_average_speeds(car_laps)
    averaged_data = get_average_speeds(all_laps)

    # Create a new empty DataFrame to store problematic sections and their subsections
    new_problematic_sections = pd.DataFrame(columns=["Problem", "Time Diff Z-Score", "Consistency", "Subsections"])

    # Synchronize distance, speed, and avg_speed
    new_distance, speed_interp, avg_speed_interp = synchronize_speed_data(
        car_data.index, car_data["Speed"], averaged_data.index, averaged_data["Speed"]
    )

    for i, section in problematic_sections.iterrows():
        slow_subsections = []  # List to hold subsections for each problem section
        if section["Problem"] not in ["Car", "Driver"]:
            new_problematic_sections = pd.concat([
                new_problematic_sections,
                pd.DataFrame([{
                    "Problem": section["Problem"],
                    "Time Diff Z-Score": section["Time Diff Z-Score"],
                    "Consistency": section["Consistency"],
                    "Subsections": []
                }])
            ], ignore_index=True)
            continue  # Skip non-car/driver issues

        # Define start and end of the problematic section
        start_idx = np.searchsorted(new_distance, midpoints[i - 1]) if i > 0 else 0
        end_idx = np.searchsorted(new_distance, midpoints[i]) if i < len(midpoints) else len(new_distance)

        # Find subsections where speed is below the interpolated average speed
        slow_start = None
        for idx in range(start_idx, end_idx):
            if speed_interp[idx] < avg_speed_interp[idx]:  # Car is slower than expected
                if slow_start is None:
                    slow_start = float(new_distance[idx])  # Start a new subsection
            else:
                if slow_start is not None:
                    # End the current subsection and add it to the list
                    slow_subsections.append((slow_start, float(new_distance[idx - 1])))
                    slow_start = None

        # Add any remaining open subsection
        if slow_start is not None:
            slow_subsections.append((slow_start, float(new_distance[end_idx - 1])))

        # Merge consecutive subsections into a single one if the gap is less than 6 meters
        merged_subsections = []
        for subsection in slow_subsections:
            if not merged_subsections or subsection[0] - merged_subsections[-1][1] > 6:
                merged_subsections.append(subsection)
            else:
                # Merge the current subsection with the previous one
                merged_subsections[-1] = (merged_subsections[-1][0], subsection[1])

        # Remove subsections where the distance between points is less than 50
        filtered_subsections = [
            (start, end) for start, end in merged_subsections if end - start >= 50
        ]

        # Add the section and its filtered subsections to the new DataFrame
        new_problematic_sections = pd.concat([
            new_problematic_sections,
            pd.DataFrame([{
                "Problem": section["Problem"],
                "Time Diff Z-Score": section["Time Diff Z-Score"],
                "Consistency": section["Consistency"],
                "Subsections": filtered_subsections
            }])
        ], ignore_index=True)

    return new_problematic_sections


HIGH_BRAKE_THRESHOLD = 0.8
HIGH_THROTTLE_THRESHOLD = 0.8
HIGH_SPEED_THRESHOLD = 290
LOW_THROTTLE_THRESHOLD = 0.2
LOW_BRAKE_THRESHOLD = 0.2


def find_general_problems(problematic_sections, car_laps):
    telemetry = get_average_telemetry(car_laps)
    problem_dict = {
        "Braking": [],
        "Acceleration": [],
        "Cornering": [],
        "DRS Inefficiency": [],
        "Straight-Line Speed Issue": [],
        "Other": []
    }
    
    for _, section in problematic_sections.iterrows():
        if section["Problem"] == "Car":
            for subsection in section["Subsections"]:
                start, end = subsection
                problem_data = telemetry[(telemetry["DistanceBin"] >= start) & (telemetry["DistanceBin"] <= end)]
                
                # Compute mean values for analysis
                mean_brake = problem_data["Brake"].mean()
                mean_throttle = problem_data["Throttle"].mean()
                mean_drs = problem_data["DRS"].mean()
                mean_speed = problem_data["Speed"].mean()
                
                # Determine category and store the subsection
                if ((mean_brake > HIGH_BRAKE_THRESHOLD) or (mean_throttle > HIGH_THROTTLE_THRESHOLD) or (mean_throttle < LOW_THROTTLE_THRESHOLD and mean_brake < LOW_BRAKE_THRESHOLD) or (mean_drs > 0) or (mean_speed > HIGH_SPEED_THRESHOLD and mean_throttle > HIGH_THROTTLE_THRESHOLD)):
                    if mean_brake > HIGH_BRAKE_THRESHOLD:
                        problem_dict["Braking"].append((start, end))
                    if mean_throttle > HIGH_THROTTLE_THRESHOLD:
                        problem_dict["Acceleration"].append((start, end))
                    if mean_throttle < LOW_THROTTLE_THRESHOLD and mean_brake < LOW_BRAKE_THRESHOLD:
                        problem_dict["Cornering"].append((start, end))
                    if mean_drs > 0:
                        problem_dict["DRS Inefficiency"].append((start, end))
                    if mean_speed > HIGH_SPEED_THRESHOLD and mean_throttle > HIGH_THROTTLE_THRESHOLD:
                        problem_dict["Straight-Line Speed Issue"].append((start, end))
                else:
                    problem_dict["Other"].append((start, end))

    # Convert to DataFrame
    df = pd.DataFrame({
        "Problem": problem_dict.keys(),
        "Subsections": [subsections if subsections else None for subsections in problem_dict.values()]
    })
    
    return df
                
                
                

def main():
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme='fastf1')

    gp = fastf1.get_session(2025, "AUSTRALIA", "Race")
    gp.load()
    print(gp.results)
    best_driver = gp.results[gp.results['ClassifiedPosition'] == "1"]['Abbreviation'].iloc[0]
    print(best_driver)
    
    fastest_lap = gp.laps.pick_fastest()
    fastest_lap_data = fastest_lap.get_car_data().add_distance()

    distance = fastest_lap_data['Distance'].to_numpy()
    speed = fastest_lap_data['Speed'].to_numpy()

    # Detect sections based on refined peak detection
    section_midpoints, max_idxs, min_idxs = detect_sections(distance, speed)
    plot_section_graph(distance, speed, max_idxs, min_idxs, section_midpoints)

    first_car_laps = gp.laps.pick_drivers("LAW").pick_accurate().pick_quicklaps()
    print(first_car_laps)
    second_car_laps = gp.laps.pick_drivers("VER").pick_accurate().pick_quicklaps()
    all_laps = gp.laps.pick_accurate().pick_quicklaps()

    first_car_average_section_times = calculate_section_times(first_car_laps, section_midpoints)
    second_car_average_section_times = calculate_section_times(second_car_laps, section_midpoints)
    first_car_analysis = analyze_performance(gp, section_midpoints, best_driver, first_car_average_section_times)
    second_car_analysis = analyze_performance(gp,section_midpoints, best_driver, second_car_average_section_times)

    problem_car_analysis = problem_analysis(first_car_analysis, second_car_analysis)
    problem_car_analysis = detect_slow_subsections(first_car_laps, problem_car_analysis, section_midpoints, all_laps)
            
    general_problems = find_general_problems(problem_car_analysis, first_car_laps)
    
    for _, problem in general_problems.iterrows():
        print(f"Problem: {problem['Problem']}")
        if problem['Subsections']:
            for subsection in problem['Subsections']:
                print(f"Subsection: {subsection}")
    

# Using the special variable 
# __name__
if __name__=="__main__":
    main()