""" This code is used for plotting precision related images.

__author__ =
    "Aswin, Deepti, Sahu"
__copyright__ = CDS, IISc, 2025

"""
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

for run_i in tqdm(range(1, 9)):
    class_of_methods = "ab"

    main_scheme_num, ref_scheme_num = "1", "2"
    main_scheme = f"{class_of_methods}{main_scheme_num}"
    ref_scheme = f"{class_of_methods}{ref_scheme_num}"
    time_stepping = "adptv"
    precision_chosen1 = 32
    precision_chosen2 = 64
    run_num = f"00{run_i}"

    time_stepping_str = None
    if time_stepping == "const":
        time_stepping_str = "constant"
    elif time_stepping == "adptv":  
        time_stepping_str = "adaptive"

    # Read the data, set the precision to full while reading
    df_main1 = pd.read_csv(f"../../output_{precision_chosen1}_{main_scheme}_{ref_scheme}_{time_stepping}_run{run_num}_main.csv", dtype="float64")
    df_main2 = pd.read_csv(f"../../output_{precision_chosen2}_{main_scheme}_{ref_scheme}_{time_stepping}_run{run_num}_main.csv", dtype="float64")
    df_conv = pd.read_csv(f"../../conv_output_run{run_num}.csv", dtype="float64")

    # print(f"{df_main['dt'].min()}")

    #########################################################################
    ### PLOT 1
    #########################################################################
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_main1["time"], df_main1["temp"], label=f"Main - {main_scheme.upper()} - {precision_chosen1} bit", ls="-")
    ax.plot(df_main2["time"], df_main2["temp"], label=f"Main - {main_scheme.upper()} - {precision_chosen2} bit", ls="--")
    ax.plot(df_conv["time"], df_conv["temp"], label=f"CV Reactor", ls="dotted")

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Temperature vs Time with {time_stepping_str} time stepping")

    ax.grid(which="both", visible=True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig(f"../../output_{precision_chosen1}_{main_scheme}_{ref_scheme}_{time_stepping}_run{run_num}_temp_precision.png", dpi=600)
    plt.close("all")

    # plt.show(block=False)
    #########################################################################


    for species_name in ["H2", "H", "O2", "OH", "O", "H2O", "HO2", "H2O2", "N2"]:
        #########################################################################
        ### PLOT 3
        #########################################################################
        # species_name = "H2O2"
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(df_main1["time"], df_main1[species_name], label=f"Main - {main_scheme.upper()} - {precision_chosen1} bit", ls="-")
        ax.plot(df_main2["time"], df_main2[species_name], label=f"Main - {main_scheme.upper()} - {precision_chosen2} bit", ls="--")    
        ax.plot(df_conv["time"], df_conv[species_name], label=f"CV Reactor", ls="dotted")

        ax.set_xlabel("Time")
        ax.set_ylabel(species_name)
        ax.set_title(f"{species_name} vs Time with {time_stepping_str} time stepping")

        ax.grid(which="both", visible=True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout(pad=0.5)
        plt.savefig(f"../../output_{precision_chosen1}_{main_scheme}_{ref_scheme}_{time_stepping}_run{run_num}_{species_name}_precision.png", dpi=600)
        plt.close("all")

        # plt.show(block=False)
        #########################################################################




    #########################################################################
    ### PLOT 4
    #########################################################################
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(df_main1["time"], df_main1["dt"], label=f"Main - {main_scheme.upper()} - {precision_chosen1} bit", ls="-")
    ax.semilogy(df_main2["time"], df_main2["dt"], label=f"Main - {main_scheme.upper()} - {precision_chosen2} bit", ls="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("dt")
    ax.set_title(f"Time step vs Time with {time_stepping_str} time stepping")

    ax.grid(which="both", visible=True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig(f"../../output_{precision_chosen1}_{main_scheme}_{ref_scheme}_{time_stepping}_run{run_num}_dt_precision.png", dpi=600)
    plt.close("all")

    # plt.show()
    #########################################################################"


    #########################################################################
    ### PLOT 4
    #########################################################################
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(df_main1["time"], df_main1["error"], label=f"Main - {main_scheme.upper()} - {precision_chosen1} bit", ls="-")
    ax.semilogy(df_main2["time"], df_main2["error"], label=f"Main - {main_scheme.upper()} - {precision_chosen2} bit", ls="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title(f"Time step vs Error with {time_stepping_str} time stepping")

    ax.grid(which="both", visible=True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig(f"../../output_{precision_chosen1}_{main_scheme}_{ref_scheme}_{time_stepping}_run{run_num}_error_precision.png", dpi=600)
    plt.close("all")

    # plt.show()
    #########################################################################"
