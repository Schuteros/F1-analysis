import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastf1.plotting
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

ORDER = 25  # Sensitivity of peak detection
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
    Also integrates the position data from the fastest lap with the telemetry data.
    """
    all_laps_data = []  # Store telemetry data per lap

    # Iterate through each lap and get telemetry data
    for _, lap in car_laps.iterrows():
        telemetry = lap.get_car_data().add_distance()  # Get telemetry for this lap

        if telemetry.empty:
            continue  # Skip empty laps

        # Convert boolean columns to integers
        if "Brake" in telemetry.columns:
            telemetry["Brake"] = telemetry["Brake"].astype(int)

        if "DRS" in telemetry.columns:
            telemetry["DRS"] = telemetry["DRS"].apply(lambda drs: 1 if drs in [10, 12, 14] else 0)

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

    # Drop SessionTime and Time columns if they exist
    combined_telemetry = combined_telemetry.drop(columns=["SessionTime", "Time"], errors='ignore')

    # Get the position data from the fastest lap
    pos_data = car_laps.pick_fastest().get_pos_data()
    if not pd.api.types.is_timedelta64_dtype(pos_data['SessionTime']):
        pos_data['SessionTime'] = pd.to_timedelta(pos_data['SessionTime'])
    # Ensure the position data is indexed by 'SessionTime' for merging
    pos_data = pos_data.set_index('SessionTime')

    # Now retrieve the telemetry from the fastest lap
    fastest_lap_telemetry = car_laps.pick_fastest().get_car_data().add_distance()
    if not pd.api.types.is_timedelta64_dtype(fastest_lap_telemetry['SessionTime']):
        fastest_lap_telemetry['SessionTime'] = pd.to_timedelta(fastest_lap_telemetry['SessionTime'])
    fastest_lap_telemetry = fastest_lap_telemetry.set_index('SessionTime')

    # Merge the position data with the telemetry data using SessionTime (method='nearest' for alignment)
    merged_data = pd.merge_asof(fastest_lap_telemetry, pos_data[['X', 'Y']], left_index=True, right_index=True, direction='nearest')

    # Now compute DistanceBin for the merged data based on telemetry Distance
    merged_data["DistanceBin"] = (merged_data["Distance"] / 0.5).round() * 0.5

    # Merge the combined telemetry with the merged data from the fastest lap
    final_telemetry = pd.merge(combined_telemetry, merged_data[['DistanceBin', 'X', 'Y']], on='DistanceBin', how='left')

    # Drop rows where X or Y is NaN (i.e., no valid position data)
    final_telemetry = final_telemetry.dropna(subset=['X', 'Y'])

    # Calculate curvature and add it to the telemetry
    final_telemetry["Curvature"] = calculate_curvature(final_telemetry["X"].values, final_telemetry["Y"].values)

    # Debugging: Print the final telemetry columns
    print("Final Telemetry Columns:", final_telemetry.columns)

    # Return the cleaned and averaged telemetry data as a DataFrame
    return final_telemetry





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


def visualize_subsection(x, y):
    """
    Visualizes the entire path from the given x and y coordinates in the x-y plane.
    
    Arguments:
    x -- List of x coordinates
    y -- List of y coordinates
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Path Subsection", color='blue', linestyle='-', marker='o')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Visualization of Path Subsection")
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


def calculate_curvature(x, y):
    """
    Function to calculate curvature using 3 consecutive points.
    Pads the result to match the input length.
    """
    curvatures = []
    for i in range(1, len(x) - 1):
        # Get the three consecutive points
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i + 1], y[i + 1]

        # Calculate the lengths of the sides of the triangle
        L1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        L2 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        L3 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Calculate the area of the triangle
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Calculate the curvature
        if L1 * L2 * L3 != 0:  # Avoid division by zero
            curvature = (2 * area) / (L1 * L2 * L3)
        else:
            curvature = 0

        curvatures.append(curvature)

    # Pad the curvature list to match the length of x and y
    return [0] + curvatures + [0]


HIGH_BRAKE_THRESHOLD = 0.6
HIGH_THROTTLE_THRESHOLD = 90
HIGH_SPEED_THRESHOLD = 290
LOW_THROTTLE_THRESHOLD = 20
LOW_BRAKE_THRESHOLD = 0.2
CURVATURE_THRESHOLD = 0.002
DRS_THRESHOLD = 0.4


def find_general_problems(problematic_sections, car_average_telemetry):
    telemetry = car_average_telemetry
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
                # Extract the position data (X, Y)
                x_positions = problem_data["X"].values / 10
                y_positions = problem_data["Y"].values / 10
                
                # Calculate curvature for each lap
                curvatures = calculate_curvature(x_positions, y_positions)
                
                # Compute mean values for analysis
                mean_brake = problem_data["Brake"].mean()
                mean_throttle = problem_data["Throttle"].mean()
                mean_drs = problem_data["DRS"].mean()
                mean_speed = problem_data["Speed"].mean()

                # Identify whether the current section is part of a corner (based on curvature threshold)
                cornering_section = any(c > CURVATURE_THRESHOLD for c in curvatures)

                # If the car is in a corner and has issues with braking or acceleration, classify it as cornering
                if mean_drs > DRS_THRESHOLD:  # Check DRS inefficiency first
                    problem_dict["DRS Inefficiency"].append((start, end))
                elif cornering_section:  # Then check if it's a cornering issue
                    problem_dict["Cornering"].append((start, end))
                elif mean_speed > HIGH_SPEED_THRESHOLD and mean_throttle > HIGH_THROTTLE_THRESHOLD:  # Then check for straight-line speed
                    problem_dict["Straight-Line Speed Issue"].append((start, end))
                elif mean_throttle > HIGH_THROTTLE_THRESHOLD and mean_brake < LOW_BRAKE_THRESHOLD:  # Acceleration issue comes next
                    problem_dict["Acceleration"].append((start, end))
                else:  # If no other issue is detected, categorize as "Other"
                    problem_dict["Other"].append((start, end))
                
    # Convert to DataFrame
    df = pd.DataFrame({
        "Problem": problem_dict.keys(),
        "Subsections": [subsections if subsections else None for subsections in problem_dict.values()]
    })
    
    return df
                
                
def plot_track_with_problems(telemetry, general_problems, problem_car_analysis):
    """
    Plots the entire race track and overlays problem sections with different colors.

    Parameters:
    - telemetry: DataFrame containing the telemetry data with 'X' and 'Y' columns.
    - general_problems: DataFrame containing problem sections and their subsections.
    - problem_car_analysis: DataFrame containing car and driver problems.
    """
    plt.figure(figsize=(10, 8))

    # Plot the full track
    plt.plot(telemetry["X"], telemetry["Y"], color="gray", label="Track", linewidth=1)

    # Define colors for each problem type
    problem_colors = {
        "Braking": "red",
        "Acceleration": "blue",
        "Cornering": "green",
        "DRS Inefficiency": "purple",
        "Straight-Line Speed Issue": "orange",
        "Other": "black",
        "Driver": "cyan"  # Color for driver-related problems
    }

    # Overlay car-related problem sections
    for _, problem in general_problems.iterrows():
        problem_type = problem["Problem"]
        if problem["Subsections"]:
            for subsection in problem["Subsections"]:
                start, end = subsection
                # Extract the subsection from telemetry
                subsection_data = telemetry[(telemetry["DistanceBin"] >= start) & (telemetry["DistanceBin"] <= end)]
                plt.plot(
                    subsection_data["X"],
                    subsection_data["Y"],
                    color=problem_colors.get(problem_type, "black"),
                    linewidth=2,
                    label=problem_type if problem_type not in plt.gca().get_legend_handles_labels()[1] else ""
                )

    # Overlay driver-related problems
    for _, row in problem_car_analysis.iterrows():
        if row["Problem"] == "Driver" and row["Subsections"]:
            for subsection in row["Subsections"]:
                start, end = subsection
                # Extract the subsection from telemetry
                subsection_data = telemetry[(telemetry["DistanceBin"] >= start) & (telemetry["DistanceBin"] <= end)]
                if not subsection_data.empty:
                    plt.plot(
                        subsection_data["X"],
                        subsection_data["Y"],
                        color=problem_colors["Driver"],
                        linewidth=2,
                        label="Driver Problem" if "Driver Problem" not in plt.gca().get_legend_handles_labels()[1] else ""
                    )

    # Add labels and legend
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Race Track with Problem Sections")
    plt.legend()
    plt.grid(True)
    plt.show()
                

def find_straights(average_telemetry, curvature_threshold=0.0002, gap_threshold=50, min_straight_length=20, braking_zones=None):
    """
    Finds straights based on curvature thresholds using average telemetry data and adjusts the end of straights
    to align with the start of braking zones if applicable.

    Parameters:
    - average_telemetry: DataFrame containing the average telemetry data with 'Distance' and 'Curvature'.
    - curvature_threshold: Maximum curvature value to consider a section as a straight.
    - gap_threshold: Maximum gap (in meters) between consecutive straights to merge them.
    - min_straight_length: Minimum length (in meters) for a section to be considered a straight.
    - braking_zones: List of tuples with start and end distances of braking zones (optional).

    Returns:
    - List of tuples with start and end distances of each straight.
    """
    # Ensure the required columns are present
    if not {"Distance", "Curvature"}.issubset(average_telemetry.columns):
        raise ValueError("Average telemetry data must contain 'Distance' and 'Curvature' columns.")

    # Identify straights based on curvature thresholds
    is_straight = average_telemetry["Curvature"] < curvature_threshold

    # Group consecutive points that satisfy the straight condition
    straights = []
    straight_start = None
    for i, straight in enumerate(is_straight):
        if straight and straight_start is None:
            straight_start = average_telemetry.iloc[i]["Distance"]
        elif not straight and straight_start is not None:
            straight_end = average_telemetry.iloc[i - 1]["Distance"]
            # Only add the straight if it meets the minimum length requirement
            if straight_end - straight_start >= min_straight_length:
                straights.append((straight_start, straight_end))
            straight_start = None

    # Handle the case where the last section is a straight
    if straight_start is not None:
        straight_end = average_telemetry.iloc[-1]["Distance"]
        if straight_end - straight_start >= min_straight_length:
            straights.append((straight_start, straight_end))

    # Debugging: Print detected straights before merging
    print("Detected Straights Before Merging:", straights)

    # Iterative merging of consecutive straights
    while True:
        i = 0
        while i < len(straights) - 1:
            if straights[i + 1][0] - straights[i][1] <= gap_threshold:
                # Merge the current straight with the next one
                straights[i] = (straights[i][0], straights[i + 1][1])  # Create a new tuple
                straights.pop(i + 1)  # Remove the next straight since it has been merged
                continue  # Restart the loop to check again from the current index
            else:
                i += 1

        break  # Exit the loop if no merging occurred

    # Adjust the end of straights to align with the start of braking zones
    if braking_zones:
        for i, (straight_start, straight_end) in enumerate(straights):
            for braking_start, _ in braking_zones:
                if braking_start > straight_start and braking_start <= straight_end:
                    # Adjust the end of the straight to the start of the braking zone
                    straights[i] = (straight_start, braking_start)
                    break  # Move to the next straight after adjusting

    # Debugging: Print detected straights after merging and adjustment
    print("Detected Straights After Merging and Adjustment:", straights)

    return straights


def plot_sections(telemetry, sections, title="Track Sections", section_label="Section"):
    """
    Plots specific sections of the track based on the given telemetry and sections.

    Parameters:
    - telemetry: DataFrame containing the telemetry data with 'X', 'Y', and 'DistanceBin' columns.
    - sections: List of tuples, where each tuple contains the start and end distances of a section.
    - title: Title of the plot (default: "Track Sections").
    - section_label: Label for the sections being plotted (default: "Section").
    """
    plt.figure(figsize=(10, 8))

    # Plot the full track
    plt.plot(telemetry["X"], telemetry["Y"], color="gray", label="Full Track", linewidth=1)

    # Plot each section
    for i, (start, end) in enumerate(sections):
        # Extract the subsection from telemetry
        subsection_data = telemetry[(telemetry["DistanceBin"] >= start) & (telemetry["DistanceBin"] <= end)]
        if not subsection_data.empty:
            plt.plot(
                subsection_data["X"],
                subsection_data["Y"],
                label=f"{section_label} {i + 1}",
                linewidth=2
            )

    # Add labels and legend
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def find_braking_zones(average_telemetry, brake_threshold=0.2, gap_threshold=00, min_braking_length=20):
    """
    Finds braking zones based on brake application thresholds using average telemetry data.

    Parameters:
    - average_telemetry: DataFrame containing the average telemetry data with 'Distance' and 'Brake'.
    - brake_threshold: Minimum brake value to consider a section as a braking zone.
    - gap_threshold: Maximum gap (in meters) between consecutive braking zones to merge them.
    - min_braking_length: Minimum length (in meters) for a section to be considered a braking zone.

    Returns:
    - List of tuples with start and end distances of each braking zone.
    """
    # Ensure the required columns are present
    if not {"Distance", "Brake"}.issubset(average_telemetry.columns):
        raise ValueError("Average telemetry data must contain 'Distance' and 'Brake' columns.")

    # Identify braking zones based on brake application thresholds
    braking_zones = []
    is_braking = average_telemetry["Brake"] > brake_threshold

    # Group consecutive points that satisfy the braking condition
    braking_start = None
    for i, braking in enumerate(is_braking):
        if braking and braking_start is None:
            braking_start = average_telemetry.iloc[i]["Distance"]
        elif not braking and braking_start is not None:
            braking_end = average_telemetry.iloc[i - 1]["Distance"]
            # Only add the braking zone if it meets the minimum length requirement
            if braking_end - braking_start >= min_braking_length:
                braking_zones.append((braking_start, braking_end))
            braking_start = None

    # Handle the case where the last section is a braking zone
    if braking_start is not None:
        braking_end = average_telemetry.iloc[-1]["Distance"]
        if braking_end - braking_start >= min_braking_length:
            braking_zones.append((braking_start, braking_end))

    # Merge consecutive braking zones if the gap between them is less than the gap_threshold
    merged_braking_zones = []
    for zone in braking_zones:
        if not merged_braking_zones or zone[0] - merged_braking_zones[-1][1] > gap_threshold:
            merged_braking_zones.append(zone)
        else:
            # Merge the current zone with the previous one
            merged_braking_zones[-1] = (merged_braking_zones[-1][0], zone[1])

    return merged_braking_zones


def find_corners(average_telemetry, curvature_threshold=0.0002, gap_threshold=50, min_corner_length=10):
    """
    Finds corners based on curvature thresholds using average telemetry data.

    Parameters:
    - average_telemetry: DataFrame containing the average telemetry data with 'Distance', 'Curvature', and optionally 'Speed'.
    - curvature_threshold: Minimum curvature value to consider a section as a corner.
    - gap_threshold: Maximum gap (in meters) between consecutive corners to merge them.
    - min_corner_length: Minimum length (in meters) for a section to be considered a corner.

    Returns:
    - List of tuples with start and end distances of each corner.
    """
    # Ensure the required columns are present
    if not {"Distance", "Curvature"}.issubset(average_telemetry.columns):
        raise ValueError("Average telemetry data must contain 'Distance' and 'Curvature' columns.")

    # Debugging: Print curvature statistics
    print("Curvature Statistics:")
    print(average_telemetry["Curvature"].describe())

    # Smooth the curvature values to reduce noise
    from scipy.signal import savgol_filter
    average_telemetry["Curvature"] = savgol_filter(average_telemetry["Curvature"], window_length=7, polyorder=2)

    # Identify corners based on curvature thresholds
    corners = []
    is_corner = average_telemetry["Curvature"] > curvature_threshold

    # Group consecutive points that satisfy the corner condition
    corner_start = None
    for i, corner in enumerate(is_corner):
        if corner and corner_start is None:
            corner_start = average_telemetry.iloc[i]["Distance"]
        elif not corner and corner_start is not None:
            corner_end = average_telemetry.iloc[i - 1]["Distance"]
            # Only add the corner if it meets the minimum length requirement
            if corner_end - corner_start >= min_corner_length:
                corners.append((corner_start, corner_end))
            corner_start = None

    # Handle the case where the last section is a corner
    if corner_start is not None:
        corner_end = average_telemetry.iloc[-1]["Distance"]
        if corner_end - corner_start >= min_corner_length:
            corners.append((corner_start, corner_end))

    # Merge consecutive corners if the gap between them is less than the gap_threshold
    merged_corners = []
    for corner in corners:
        if not merged_corners or corner[0] - merged_corners[-1][1] > gap_threshold:
            merged_corners.append(corner)
        else:
            # Merge the current corner with the previous one
            merged_corners[-1] = (merged_corners[-1][0], corner[1])

    return merged_corners


def find_exits(average_telemetry, acceleration_threshold=3, min_exit_length=20):
    """
    Finds exit phases based on acceleration thresholds using average telemetry data.

    Parameters:
    - average_telemetry: DataFrame containing the average telemetry data with 'Distance' and 'Speed'.
    - acceleration_threshold: Minimum acceleration (in m/s^2) to consider a section as an exit.
    - min_exit_length: Minimum length (in meters) for a section to be considered an exit.

    Returns:
    - List of tuples with start and end distances of each exit phase.
    """
    # Ensure the required columns are present
    if not {"Distance", "Speed"}.issubset(average_telemetry.columns):
        raise ValueError("Average telemetry data must contain 'Distance' and 'Speed' columns.")

    # Calculate acceleration as the derivative of speed with respect to time
    # Ensure Speed is in m/s (convert from km/h if necessary)
    if average_telemetry["Speed"].max() > 100:  # likely in km/h
        speed_mps = average_telemetry["Speed"] / 3.6
    else:
        speed_mps = average_telemetry["Speed"]

    # Estimate time difference between points (assuming uniform sampling, or use Distance and Speed)
    # dt = ds / v
    ds = average_telemetry["Distance"].diff()
    dt = ds / speed_mps.replace(0, np.nan)
    average_telemetry["Acceleration"] = speed_mps.diff() / dt
    average_telemetry["Acceleration"] = average_telemetry["Acceleration"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print("Acceleration Statistics:" + str(average_telemetry["Acceleration"].describe()))

    # Identify exit phases based on acceleration thresholds
    is_exit = average_telemetry["Acceleration"] > acceleration_threshold

    # Group consecutive points that satisfy the exit condition
    exits = []
    exit_start = None
    for i, exit_phase in enumerate(is_exit):
        if exit_phase and exit_start is None:
            exit_start = average_telemetry.iloc[i]["Distance"]
        elif not exit_phase and exit_start is not None:
            exit_end = average_telemetry.iloc[i - 1]["Distance"]
            # Only add the exit if it meets the minimum length requirement
            if exit_end - exit_start >= min_exit_length:
                exits.append((exit_start, exit_end))
            exit_start = None

    # Handle the case where the last section is an exit
    if exit_start is not None:
        exit_end = average_telemetry.iloc[-1]["Distance"]
        if exit_end - exit_start >= min_exit_length:
            exits.append((exit_start, exit_end))

    return exits


def find_and_override_drs_zones(average_telemetry, straights, braking_zones, corners, exits, throttle_threshold=80):
    """
    Finds DRS zones based on average telemetry data and removes overlapping parts from other sections.

    Parameters:
    - average_telemetry: DataFrame containing the average telemetry data with 'Distance', 'Throttle', and 'DRS' columns.
    - straights: List of tuples, where each tuple contains the start and end distances of a straight.
    - braking_zones: List of tuples, where each tuple contains the start and end distances of a braking zone.
    - corners: List of tuples, where each tuple contains the start and end distances of a corner.
    - exits: List of tuples, where each tuple contains the start and end distances of an exit.
    - throttle_threshold: Minimum throttle percentage (as a fraction) to consider a section as a DRS zone.

    Returns:
    - List of tuples with start and end distances of each DRS zone.
    - Updated lists of straights, braking zones, corners, and exits with DRS zones removed.
    """
    drs_zones = []

    # Identify DRS zones based on DRS > 0 and throttle > throttle_threshold
    is_drs_zone = (average_telemetry["DRS"] > 0) & (average_telemetry["Throttle"] >= throttle_threshold)

    # Group consecutive points that satisfy the DRS condition
    drs_start = None
    for i, drs in enumerate(is_drs_zone):
        if drs and drs_start is None:
            drs_start = average_telemetry.iloc[i]["Distance"]
        elif not drs and drs_start is not None:
            drs_end = average_telemetry.iloc[i - 1]["Distance"]
            drs_zones.append((drs_start, drs_end))
            drs_start = None

    # Handle the case where the last section is a DRS zone
    if drs_start is not None:
        drs_end = average_telemetry.iloc[-1]["Distance"]
        drs_zones.append((drs_start, drs_end))

    # Helper function to remove DRS zones from a section list
    def remove_drs_from_sections(sections, drs_zones):
        updated_sections = []
        for section in sections:
            section_start, section_end = section
            current_section = [(section_start, section_end)]

            for drs_start, drs_end in drs_zones:
                new_section = []
                for sub_section_start, sub_section_end in current_section:
                    # If the DRS zone overlaps with the section, split it
                    if drs_start <= sub_section_end and drs_end >= sub_section_start:
                        # Add the part of the section before the DRS zone
                        if sub_section_start < drs_start:
                            new_section.append((sub_section_start, drs_start))
                        # Add the part of the section after the DRS zone
                        if sub_section_end > drs_end:
                            new_section.append((drs_end, sub_section_end))
                    else:
                        # If no overlap, keep the section as is
                        new_section.append((sub_section_start, sub_section_end))
                current_section = new_section

            updated_sections.extend(current_section)
        return updated_sections

    # Remove DRS zones from each section type
    updated_straights = remove_drs_from_sections(straights, drs_zones)
    updated_braking_zones = remove_drs_from_sections(braking_zones, drs_zones)
    updated_corners = remove_drs_from_sections(corners, drs_zones)
    updated_exits = remove_drs_from_sections(exits, drs_zones)

    return drs_zones, updated_straights, updated_braking_zones, updated_corners, updated_exits


def calculate_sections(average_telemetry, curvature_threshold=0.0002, brake_threshold=0.2, acceleration_threshold=3.75
                       , min_corner_length=10):
    """
    Calculates all sections (corners, braking zones, exits, straights) in order.

    Parameters:
    - average_telemetry: DataFrame containing the average telemetry data with 'Distance', 'X', 'Y', 'Brake', 'Curvature', 'Speed'.
    - curvature_threshold: Minimum curvature value to consider a section as a corner.
    - brake_threshold: Minimum brake value to consider a section as a braking zone.
    - acceleration_threshold: Minimum acceleration (in m/s^2) to consider a section as an exit.
    - min_corner_length: Minimum length (in meters) for a corner.

    Returns:
    - Dictionary containing all sections: corners, braking zones, exits, and straights.
    """
    # Step 1: Identify corners
    corners = find_corners(average_telemetry, curvature_threshold, min_corner_length=min_corner_length)

    # Step 2: Identify braking zones
    braking_zones = find_braking_zones(average_telemetry, brake_threshold)

    # Step 3: Identify straights
    straights = find_straights(average_telemetry, curvature_threshold, braking_zones=braking_zones)

    # Step 4: Identify exits based on acceleration
    exits = find_exits(average_telemetry, acceleration_threshold=acceleration_threshold)

    # Return all sections
    return {
        "corners": corners,
        "braking_zones": braking_zones,
        "exits": exits,
        "straights": straights,
    }


def remove_overlaps(target_sections, reference_sections, min_corner_length=50):
    """
    Removes overlapping parts of target_sections that overlap with reference_sections by splitting at the midpoint,
    unless the reference section (e.g., a corner) is smaller than a specified minimum length.

    Parameters:
    - target_sections: List of tuples (start, end) representing the target sections.
    - reference_sections: List of tuples (start, end) representing the reference sections.
    - min_corner_length: Minimum length (in meters) for a reference section to allow merging through the midpoint.

    Returns:
    - List of tuples (start, end) with overlaps handled appropriately.
    """
    updated_sections = []
    for target_start, target_end in target_sections:
        current_section = [(target_start, target_end)]

        for ref_start, ref_end in reference_sections:
            new_section = []
            for sub_start, sub_end in current_section:
                # If the reference section overlaps with the target section
                if ref_start <= sub_end and ref_end >= sub_start:
                    # Check the length of the reference section
                    ref_length = ref_end - ref_start
                    if ref_length < min_corner_length:
                        # If the reference section is too small, do not split at the midpoint
                        continue
                    else:
                        # Find the midpoint of the overlap
                        overlap_start = max(sub_start, ref_start)
                        overlap_end = min(sub_end, ref_end)
                        midpoint = (overlap_start + overlap_end) / 2

                        # Split the section at the midpoint
                        if sub_start < midpoint:
                            new_section.append((sub_start, midpoint))
                        if sub_end > midpoint:
                            new_section.append((midpoint, sub_end))
                else:
                    # If no overlap, keep the section as is
                    new_section.append((sub_start, sub_end))
            current_section = new_section

        updated_sections.extend(current_section)
    return updated_sections


def main():
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme='fastf1')

    gp = fastf1.get_session(2025, "CHINESE", "Sprint")
    gp.load()
    print(gp.results)
    best_driver = gp.results[gp.results['ClassifiedPosition'] == "1"]['Abbreviation'].iloc[0]
    print(best_driver)

    first_car_laps = gp.laps.pick_drivers("HUL").pick_accurate().pick_quicklaps()
    first_car_average_telemetry = get_average_telemetry(first_car_laps)

    # Calculate all sections
    sections = calculate_sections(first_car_average_telemetry)

    # Visualize each section type
    plot_sections(first_car_average_telemetry, sections["corners"], title="Corners", section_label="Corner")
    plot_sections(first_car_average_telemetry, sections["braking_zones"], title="Braking Zones", section_label="Braking Zone")
    plot_sections(first_car_average_telemetry, sections["exits"], title="Exits", section_label="Exit")
    plot_sections(first_car_average_telemetry, sections["straights"], title="Straights", section_label="Straight")
    #plot_sections(first_car_average_telemetry, sections["drs_zones"], title="DRS Zones", section_label="DRS Zone")


# Using the special variable 
# __name__
if __name__=="__main__":
    main()