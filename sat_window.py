#!/usr/bin/env python3

import os
import requests
import json
import calendar
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, Topos, EarthSatellite
from datetime import datetime, timedelta, UTC, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def load_config(config_path="config.json"):
	with open(config_path) as f:
		config = json.load(f)

	local_tz = ZoneInfo(config["timezone"])
	sats = config["sats"]
	gs = config["ground_station"]
	gs_lat = gs["lat"]
	gs_lon = gs["lon"]
	gs_alt = gs["alt"]
	target = config["target"]
	target_az = target["az"]
	target_el = target["el"]
	radius_deg = target["radius_deg"]
	tle_url = config["tle_url"]

	day_name_to_int = {day: i for i, day in enumerate(calendar.day_name)}
	schedule = {}
	for day_name, windows in config["schedule"].items():
		day_idx = day_name_to_int[day_name]
		schedule[day_idx] = [
			(time.fromisoformat(start), time.fromisoformat(end))
			for start, end in windows
		]

	return {
		"timezone": local_tz,
		"sats": sats,
		"gs_lat": gs_lat,
		"gs_lon": gs_lon,
		"gs_alt": gs_alt,
		"target_az": target_az,
		"target_el": target_el,
		"radius_deg": radius_deg,
		"schedule": schedule,
		"tle_url": tle_url
	}



def fetch_tles(tle_url, tle_file="tle.txt", max_age_hours=24):
	max_age = timedelta(hours=max_age_hours)  # Convert hours to timedelta

	# Check if local file exists and is recent
	if os.path.exists(tle_file):
		file_time = datetime.fromtimestamp(os.path.getmtime(tle_file))
		if datetime.now() - file_time < max_age:
			with open(tle_file, 'r') as f:
				tle_raw = f.read().split("\n")[:-1]
		else:
			tle_raw = download_tle(tle_url, tle_file)
	else:
		tle_raw = download_tle(tle_url, tle_file)

	# Parse TLE data
	tles = []

	for i in range(0, len(tle_raw), 3):
		sat_name = tle_raw[i].strip()
		line1 = tle_raw[i + 1].strip()
		line2 = tle_raw[i + 2].strip()

		tle = {"name": sat_name, "line1": line1, "line2": line2}
		tles.append(tle)

	return tles

def download_tle(tle_url, tle_file="tle.txt"):
	response = requests.get(tle_url)
	if response.status_code == 200:
		print("New TLE data downloaded")
		with open(tle_file, 'w') as f:
			f.write(response.text)
		tle_raw = response.text.split("\r\n")[:-1]
		return tle_raw
	return []


def angular_distance(az1, el1, az2, el2):
	az1, el1, az2, el2 = map(np.radians, [az1, el1, az2, el2])
	cos_d = np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
	return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def get_passes(satellite, days=5, min_alt=10.0):
	t0 = ts.now()
	t1 = ts.utc(datetime.now(UTC) + timedelta(days=days))

	times, events = satellite.find_events(topos, t0, t1, altitude_degrees=min_alt)

	pass_list = []
	current_pass = {}

	for ti, event in zip(times, events):
		event_type = ("rise", "culminate", "set")[event]
		alt, az, _ = (satellite - topos).at(ti).altaz()
		current_pass[event_type] = {
			"time": ti.utc_datetime(),
			"alt": alt.degrees,
			"az": az.degrees,
		}
		if event_type == "set":
			pass_list.append(current_pass)
			current_pass = {}

	return pass_list

def pass_within_target(pass_data, satellite):
	start = pass_data['rise']['time']
	end = pass_data['set']['time']

	times = ts.utc(
		[t.year for t in (start + timedelta(seconds=10 * i) for i in range(int((end - start).total_seconds() / 10) + 1))],
		[t.month for t in (start + timedelta(seconds=10 * i) for i in range(int((end - start).total_seconds() / 10) + 1))],
		[t.day for t in (start + timedelta(seconds=10 * i) for i in range(int((end - start).total_seconds() / 10) + 1))],
		[t.hour for t in (start + timedelta(seconds=10 * i) for i in range(int((end - start).total_seconds() / 10) + 1))],
		[t.minute for t in (start + timedelta(seconds=10 * i) for i in range(int((end - start).total_seconds() / 10) + 1))],
		[t.second + t.microsecond / 1e6 for t in (start + timedelta(seconds=10 * i) for i in range(int((end - start).total_seconds() / 10) + 1))]
	)

	for t in times:
		alt, az, _ = (satellite - topos).at(t).altaz()
		dist = angular_distance(az.degrees, alt.degrees, target_az, target_el)

		if dist <= radius_deg:
			return True

	return False

def pass_within_schedule(dt_utc):
	local = dt_utc.astimezone(local_tz)
	day = local.weekday()
	time_of_day = local.time()

	for start, end in schedule.get(day, []):
		if start <= time_of_day <= end:
			return True
	return False

def plot_pass(pass_data, satellite, steps=100):

	start_utc = pass_data['rise']['time']
	end_utc = pass_data['set']['time']
	start_local = start_utc.astimezone(local_tz)
	end_local = end_utc.astimezone(local_tz)

	time_range = np.linspace(start_utc.timestamp(), end_utc.timestamp(), steps)
	times_dt = [datetime.utcfromtimestamp(t).replace(tzinfo=ZoneInfo("UTC")) for t in time_range]

	times = ts.utc([t.year for t in times_dt],
				   [t.month for t in times_dt],
				   [t.day for t in times_dt],
				   [t.hour for t in times_dt],
				   [t.minute for t in times_dt],
				   [t.second + t.microsecond / 1e6 for t in times_dt])


	az_list = []
	el_list = []

	for t in times:
		alt, az, _ = (satellite - topos).at(t).altaz()
		az_list.append(np.radians(az.degrees))     # az in radians
		el_list.append(90 - alt.degrees) 

	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111, polar=True)
	ax.plot(az_list, el_list, color='blue', linewidth=2)  # No label

	ax.set_theta_zero_location("N")
	ax.set_theta_direction(-1)
	ax.set_rmax(90)
	ax.set_rticks([30, 60, 90])
	ax.set_rlabel_position(0)
	ax.grid(True)
	ax.set_title(f"Sky Track: {satellite.name}", va='bottom')

	start_str = start_local.strftime("Start: %a %Y-%m-%d %H:%M:%S")
	end_str = end_local.strftime("End: %a %Y-%m-%d %H:%M:%S")
	fig.text(0.01, 0.01, start_str, ha='left', va='bottom', fontsize=10)
	fig.text(0.99, 0.01, end_str, ha='right', va='bottom', fontsize=10)

	plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='sat_window')
	parser.add_argument('-c', '--config', required=True)
	args = parser.parse_args()
	# Load config
	print("Loading config...")
	config = load_config(args.config)
	sats = config["sats"]
	gs_lat = config["gs_lat"]
	gs_lon = config["gs_lon"]
	gs_alt = config["gs_alt"]
	target_az = config["target_az"]
	target_el = config["target_el"]
	radius_deg = config["radius_deg"]
	schedule = config["schedule"]
	local_tz = config["timezone"]
	tle_url = config["tle_url"]

	# Setup
	ts = load.timescale()
	eph = load("de421.bsp")
	topos = Topos(latitude_degrees=gs_lat, longitude_degrees=gs_lon, elevation_m=gs_alt)

	# Get all TLEs then fitler for ones of interest
	print("Fetching TLE data...")
	all_tles = fetch_tles(tle_url)
	tles = []
	for tle in all_tles:
		if tle["name"] in sats:
			tles.append(tle)
	# meat
	total_passes = 0
	for tle in tles:
		satellite = EarthSatellite(tle["line1"], tle["line2"], tle["name"], ts)
		all_passes = get_passes(satellite)
		total_passes += len(all_passes)
		
		# filter based on target area of sky
		target_passes = []
		for sat_pass in all_passes:
			if pass_within_target(sat_pass, satellite):
				target_passes.append(sat_pass)

		# filter based on schedule
		schedule_passes = []
		for sat_pass in target_passes:
			if pass_within_schedule(sat_pass['culminate']['time']):
				schedule_passes.append(sat_pass)

		tle['passes'] = schedule_passes

	filtered_passes = []

	for sat in tles:
		sat_name = sat["name"]
		for sat_pass in sat["passes"]:
			filtered_passes.append({
				"satellite": sat_name,
				"tle": {"line1": sat["line1"], "line2": sat["line2"]},
				"pass": sat_pass
			})

	filtered_passes.sort(key=lambda x: x["pass"]["rise"]["time"].astimezone(local_tz))

	# print sorted pass data
	print(f"{len(filtered_passes)}/{total_passes} passes fit criteria")
	print(f"\n{'#':<3} {'Satellite':<10} {'Event':<11} {'Date':<10} {'Time':<8} {'Az(°)':>6} {'El(°)':>6}")
	print("-" * 60)

	for i, sat_pass in enumerate(filtered_passes, start=1):
		sat_name = sat_pass["satellite"]
		first_event = True 

		for event_key in ['rise', 'culminate', 'set']:
			event = sat_pass["pass"][event_key]
			t_local = event['time'].astimezone(local_tz)
			date = t_local.strftime("%a %m-%d")
			time_str = t_local.strftime("%H:%M:%S")
			az = event['az']
			el = event['alt']
			label = event_key.capitalize()

			if first_event:
				print(f"\n{i:<3} {sat_name:<10} {label:<11} {date:<10} {time_str:<8} {az:6.1f} {el:6.1f}")
				first_event = False
			else:
				print(f"{'':<3} {'':<10} {label:<11} {date:<10} {time_str:<8} {az:6.1f} {el:6.1f}")  # Indent

	# plot desired passes
	while True:
		choice = input("\nEnter pass number to plot (or 'q' to quit): ")
		if choice.lower() == "q":
			break

		try:
			choice = int(choice) - 1
			if 0 <= choice < len(filtered_passes):
				selected_pass = filtered_passes[choice]
				sat = EarthSatellite(selected_pass["tle"]["line1"], selected_pass["tle"]["line2"], selected_pass["satellite"], ts)
				plot_pass(selected_pass["pass"], sat)
			else:
				print("Invalid selection. Try again.")
		except ValueError:
			print("Invalid selection. Enter a number from the list or 'q' to quit.")
