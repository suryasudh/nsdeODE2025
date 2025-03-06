import matplotlib.pyplot as plt
import pandas as pd

class_of_methods = "rk"

main_scheme_num, ref_scheme_num = "2", "3"
main_scheme = f"{class_of_methods}{main_scheme_num}"
ref_scheme = f"{class_of_methods}{ref_scheme_num}"
time_stepping = "adptv"
precision_chosen = 64

# Read the data, set the precision to full while reading
df_main = pd.read_csv(f"../../output_{precision_chosen}_{main_scheme}_{ref_scheme}_{time_stepping}_run001_main.csv", dtype="float64")
df_ref = pd.read_csv(f"../../output_{precision_chosen}_{main_scheme}_{ref_scheme}_{time_stepping}_run001_ref.csv", dtype="float64")

print(f"{df_main['dt'].min()}")

"""
#########################################################################
### PLOT 1
#########################################################################
fig, ax = plt.subplots()

ax.plot(df_main["time"], df_main["temp"], label=f"Main - {main_scheme.upper()}")
ax.plot(df_ref["time"], df_ref["temp"], label=f"Ref - {ref_scheme.upper()}")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature")
ax.set_title(f"Temperature vs Time with {time_stepping} time stepping\n and {precision_chosen} bit precision")

plt.legend()
# plt.savefig(f"../../output_{precision_chosen}_{main_scheme}_{ref_scheme}_{time_stepping}_run001_temp.png", dpi=600)

plt.show()
#########################################################################



#########################################################################
### PLOT 2
#########################################################################
fig, ax = plt.subplots()

ax.plot(df_main["time"], df_main["OH"], label=f"Main - {main_scheme.upper()}")
ax.plot(df_ref["time"], df_ref["OH"], label=f"Ref - {ref_scheme.upper()}")
ax.set_xlabel("Time")
ax.set_ylabel("OH")
ax.set_title(f"Temperature vs Time with {time_stepping} time stepping\n and {precision_chosen} bit precision")

plt.legend()
# plt.savefig(f"../../output_{precision_chosen}_{main_scheme}_{ref_scheme}_{time_stepping}_run001_OH.png", dpi=600)

plt.show()
#########################################################################

#########################################################################
### PLOT 3
#########################################################################
fig, ax = plt.subplots()

ax.semilogy(df_main["time"], df_main["dt"], label=f"Main - {main_scheme.upper()}")
# ax.plot(df_ref["time"], df_ref["OH"], label=f"Ref - {ref_scheme.upper()}")
ax.set_xlabel("Time")
ax.set_ylabel("dt")
ax.set_title(f"Time step vs Time with {time_stepping} time stepping\n and {precision_chosen} bit precision")

plt.legend()
# plt.savefig(f"../../output_{precision_chosen}_{main_scheme}_{ref_scheme}_{time_stepping}_run001_OH.png", dpi=600)

plt.show()
#########################################################################"
"""