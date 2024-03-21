# -*- coding: utf-8 -*-
""" 
    Using partial code from 
    Vrachimis et al. https://github.com/KIOS-Research/BattLeDIM
"""
import pandas as pd
import numpy as np
import wntr
import pickle
import os
import argparse
import time
from math import sqrt, ceil
import warnings, copy
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def create_cli_parser():
    # ----- ----- ----- ----- ----- -----
    # Command line arguments
    # ----- ----- ----- ----- ----- -----
    parser  = argparse.ArgumentParser()
    parser.add_argument('--wdn',
                        default = 'l_town',
                        type    = str,
                        choices = ['hanoi', 'fossolo', 'pescara', 'l_town', 'zhijiang'],
                        help    = "specify the WDS for which you want to simulate the scenarios; default is l_town ")
    parser.add_argument('--start_scenario',
                        default = 1,
                        type    = int,
                        help    = "specify the start scenario name, must be an integer; default is 1")
    parser.add_argument('--end_scenario',
                        default = 50,
                        type    = int,
                        help    = "specify the end scenario name, must be an integer; default is 50")
    parser.add_argument('--sigma_dem',
                        default = 0.1,
                        type    = float,
                        help    = "Specify the standard deviation of the noise to be added to the demand patterns; default is 0.1.")   
    parser.add_argument('--sigma_dia',
                        default = 1/30,
                        type    = float,
                        help    = "Specify the standard deviation of the noise to be added to the diameters; default is 1/30")   
    parser.add_argument('--_seed',
                        default = None,
                        type    = int,
                        help    = "Specify the random seed for noise; default is None, where it will be set to the scenario name for every scenario.")   
    parser.add_argument('--start_time',
                        default = '2018-01-01 00:00',
                        type    = str,
                        help    = "Specify the start time of the simulation; default is 2018-01-01 00:00, the samples will be sampled every 30 minutes starting from this time.")   
    parser.add_argument('--end_time',
                        default = '2018-01-14 23:30',
                        type    = str,
                        help    = "Specify the end time of the simulation; default is 2018-01-14 23:30.")   
    return parser


class DatasetCreator:
    def __init__(self, scenario_folder, inp_file, start_time, end_time):

        self.scenario_folder = scenario_folder
        print(f'Run input file: "{inp_file}"')
        self.results_folder = os.path.join(os.getcwd(), scenario_folder, "Results-Clean")
        # Create Results folder
        if not os.path.isdir(self.results_folder):
            os.system('mkdir ' + self.results_folder)

        # demand-driven (DD) or pressure dependent demand (PDD)
        Mode_Simulation = 'DD'  

        # Load EPANET network file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.wn.options.hydraulic.demand_model = Mode_Simulation

        self.nodes = self.wn.get_graph().nodes()
        self.links = self.wn.link_name_list

        # Get time step
        self.time_step = round(self.wn.options.time.hydraulic_timestep)
        # Create time_stamp
        self.time_stamp = pd.date_range(start_time, end_time, freq=str(self.time_step / 60) + "min")

        # Simulation duration in steps
        self.wn.options.time.duration = (len(self.time_stamp) - 1) * self.time_step  
    

    def dataset_generator(self, scenario_times=[]):
        # Path of EPANET Input File
        print(f"Dataset Generator run...")
        
        # Save the water network model to a file before using it in a simulation
        with open(os.path.join(os.getcwd(), self.scenario_folder,'self.wn.pickle'), 'wb') as f:
            pickle.dump(self.wn, f)

        # Run wntr simulator
        scenario_start_time = time.time()
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        scenario_end_time = time.time()
        scenario_time = scenario_end_time - scenario_start_time
        scenario_times.append(scenario_time)
        print('... simulation done!')
        
        if results.node["pressure"].empty:
            print("Negative pressures.")
            return -1

        if results:
            decimal_size = 6            

            # Create xlsx file with Measurements
            def export_measurements(pressure_sensors, flow_sensors, file_out="Measurements.xlsx"):
                total_pressures = {'Timestamp': self.time_stamp}
                total_demands = {'Timestamp': self.time_stamp}
                total_flows = {'Timestamp': self.time_stamp}
                total_levels = {'Timestamp': self.time_stamp}
                total_heads = {'Timestamp': self.time_stamp}
                for j in range(0, self.wn.num_nodes):
                    node_id = self.wn.node_name_list[j]
                    if node_id in pressure_sensors:
                        pres = results.node['pressure'][node_id]
                        pres = pres[:len(self.time_stamp)]
                        pres = [round(elem, decimal_size) for elem in pres]
                        total_pressures[node_id] = pres

                        head = results.node['head'][node_id]
                        head = head[:len(self.time_stamp)]
                        head = [round(elem, decimal_size) for elem in head]
                        total_heads[node_id] = head

                        dem = results.node['demand'][node_id]
                        dem = dem[:len(self.time_stamp)]
                        dem = [round(elem, decimal_size) for elem in dem] 
                        total_demands[node_id] = dem

                        level_pres = results.node['pressure'][node_id]
                        level_pres = level_pres[:len(self.time_stamp)]
                        level_pres = [round(elem, decimal_size) for elem in level_pres]
                        total_levels[node_id] = level_pres

                for j in range(0, self.wn.num_links):
                    link_id = self.wn.link_name_list[j]

                    if link_id not in flow_sensors:
                        continue
                    flows = results.link['flowrate'][link_id]
                    flows = [round(elem, decimal_size) for elem in flows]
                    flows = flows[:len(self.time_stamp)]
                    total_flows[link_id] = flows

                # Loading original demands
                dem_multiplier = self.wn.options.hydraulic.demand_multiplier
                n_timesteps = len(self.time_stamp)
                orig_demands = {'Timestamp': self.time_stamp}
                for node_id in self.wn.node_name_list:
                    node_dem = 0
                    if self.wn.nodes._data[node_id].node_type == 'Junction':
                        for patterns in self.wn.nodes._data[node_id].demand_timeseries_list._list:
                            if patterns.pattern.multipliers is not None:
                                node_dem += (patterns.base_value * dem_multiplier) * patterns.pattern.multipliers[: n_timesteps]                                    
                            else:
                                node_dem += patterns.base_value * dem_multiplier
                    try:
                        repeat_idx = ceil(n_timesteps / len(node_dem))
                        node_dem_copy = copy.deepcopy(node_dem)
                        for i in range(1, repeat_idx):
                            node_dem = np.concatenate((node_dem, node_dem_copy))
                        orig_demands[node_id] = node_dem[: n_timesteps] #* 3600 * 1000
                    except:
                        orig_demands[node_id] = node_dem              


                # Create a Pandas dataframe from the data.                
                df1 = pd.DataFrame(total_pressures)
                df2 = pd.DataFrame(total_demands)
                df3 = pd.DataFrame(total_flows)
                df4 = pd.DataFrame(total_levels)
                df5 = pd.DataFrame(total_heads)
                df6 = pd.DataFrame(orig_demands)

                print("Minimum Pressure: ", df1.min(numeric_only=True).min(), "Maximum Pressure: ", df1.max(numeric_only=True).max())
                print("Minimum Head: ", df5.min(numeric_only=True).min(), "Maximum Head: ", df5.max(numeric_only=True).max())

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                writer = pd.ExcelWriter(os.path.join(self.results_folder, file_out), engine='xlsxwriter')

                # Convert the dataframe to an XlsxWriter Excel object.
                # Pressures (m), Demands (m^3/s), Flows (m^3/s), Levels (m), Heads (m)
                df1.to_excel(writer, sheet_name='Pressures (m)', index=False)
                df2.to_excel(writer, sheet_name='Demands (m3_s)', index=False)
                df3.to_excel(writer, sheet_name='Flows (m3_s)', index=False)
                df4.to_excel(writer, sheet_name='Levels (m)', index=False)
                df5.to_excel(writer, sheet_name='Heads (m)', index=False)
                df6.to_excel(writer, sheet_name='Orig_Demands (m3_s)', index=False)

                # Close the Pandas Excel writer and output the Excel file.
                writer.save()

                # Export as .csv files -- .csv files are much faster parsed by pandas than huge .xlsx files!
                df1.to_csv(os.path.join(self.results_folder, file_out.replace(".xlsx", "_Pressures.csv")), index=False)
                df2.to_csv(os.path.join(self.results_folder, file_out.replace(".xlsx", "_Demands.csv")), index=False)
                df3.to_csv(os.path.join(self.results_folder, file_out.replace(".xlsx", "_Flows.csv")), index=False)
                df4.to_csv(os.path.join(self.results_folder, file_out.replace(".xlsx", "_Levels.csv")), index=False)
                df5.to_csv(os.path.join(self.results_folder, file_out.replace(".xlsx", "_Heads.csv")), index=False)
                df6.to_csv(os.path.join(self.results_folder, file_out.replace(".xlsx", "_Orig_Demands.csv")), index=False)

            # Export all measurements 
            export_measurements(self.nodes, self.links, "Measurements_All.xlsx")

            # Clean up
            os.remove(os.path.join(os.getcwd(), self.scenario_folder,'self.wn.pickle'))
        else:
            print('Results empty.')
            return -1

        return scenario_times
    


def run(wdn = 'l_town', start_scenario=1, end_scenario=50, scenario_times=[], sigma_dem=0.1, sigma_dia=1/30, 
        in_seed=None, start_time='2018-01-01 00:00', end_time='2018-01-14 23:30'):

    for s in range(start_scenario, end_scenario + 1):
        scenario = 's' + str(s)
        save_dir = os.path.join(os.getcwd(),"networks",  wdn)
        scenario_dir = os.path.join(save_dir, scenario)
        if not os.path.isdir(scenario_dir):
            os.system('mkdir ' + scenario_dir)

        if wdn == "hanoi":
            inp_file = os.path.join(scenario_dir, "Hanoi_CMH_Scenario-" + str(s) + ".inp")
        else:
            inp_file = os.path.join(save_dir, wdn + ".inp")

        t = time.time()
    
        wn = wntr.network.WaterNetworkModel(inp_file)
        if in_seed is None:
            _seed = s
        else:
            _seed = in_seed
        np.random.seed(_seed)
        print('Seed used for scenario ', s)

        pattern_offset = np.round(np.random.normal(0, sigma_dem, size = 1), 6) 

        if wdn != "hanoi" and wdn != "l_town":
            _len = 48*7*2
            pattern = np.round(np.random.normal(1, .1, size = _len), 6).clip(0) 
            wn.add_pattern(
                name = "random_week", 
                pattern = pattern + pattern_offset
                )
        if wdn == "l_town":
            _len = len(wn.patterns._data['P-Residential'].multipliers)
            wn.patterns._data['P-Residential'].multipliers = \
                wn.patterns._data['P-Residential'].multipliers + \
                    np.round(np.random.normal(0, sigma_dem, size = 1), 6).clip(0) 

        if wdn != "hanoi":        
            for key, value in wn.links._data.items():
                wn.links._data[key].diameter = \
                    wn.links._data[key].diameter * ( 1 + np.random.normal(0, sigma_dia, size=1)[0] ) 
            
            filename = os.path.join(scenario_dir, wdn + ".inp")
            out_inp = wntr.epanet.io.InpFile()
            out_inp.write(
                filename, 
                wn = wn,
                units = None, 
                version = 2.2, 
                force_coordinates = False
                )
            inp_file = filename

        # Call dataset creator        
        L = DatasetCreator(scenario_dir, inp_file, start_time, end_time)
        scenario_times = L.dataset_generator(scenario_times)

        print('\nScenario ' + scenario + ' generated. Total Elapsed time is ' + str(time.time() - t) + ' seconds.\n')

    return scenario_times



if __name__ == '__main__':
    parser = create_cli_parser()
    args = parser.parse_args()  

    scenario_times = []
    scenario_times = run(
                        args.wdn, 
                        args.start_scenario, 
                        args.end_scenario, 
                        scenario_times, 
                        args.sigma_dem, 
                        args.sigma_dia, 
                        args._seed, 
                        args.start_time, 
                        args.end_time
                        )    

    print("Total Simulation Time for all Scenarios: ", np.sum(scenario_times))
    print("Scenarios Simulation Times: ", scenario_times)