# file: fbb_app.py
import streamlit as st
import pandas as pd
from datetime import date

from fbb_simulation import (
    run_simulation,
    build_room_occupancy_slots,
    TIMES_DAY,
    plot_tagespeak,
    plot_meetingrooms,
    build_week_weighting_from_weeks,
    scale_week_weighting,
)

st.set_page_config(page_title="Office Simulation Dashboard", layout="wide")
st.title("🏢 Office Simulation Dashboard")

# -------------------------------------------------
# Defaults
# -------------------------------------------------
# default_start_date = st.sidebar.date_input("Start Date", date(2025, 1, 1))
# default_end_date = st.sidebar.date_input("End Date", date(2025, 12, 31))

default_start_date = start_date = date(2025, 1, 1)
default_end_date = date(2025, 12, 31)

default_week_factor = {
    "mon": 0.175,
    "tue": 0.245,
    "wed": 0.23,
    "thu": 0.23,
    "fri": 0.12,
}
default_week_scale = {
    1: 33,
    2: 33,
    3: 33,
    4: 33,
    5: 37,
    6: 37,
    7: 37,
    8: 37,
    9: 35,
    10: 35,
    11: 35,
    12: 35,
    13: 35,
    14: 33,
    15: 33,
    16: 33,
    17: 33,
    18: 31,
    19: 31,
    20: 31,
    21: 31,
    22: 37,
    23: 37,
    24: 37,
    25: 37,
    26: 37,
    27: 33,
    28: 33,
    29: 33,
    30: 33,
    31: 27,
    32: 27,
    33: 27,
    34: 27,
    35: 27,
    36: 37,
    37: 37,
    38: 37,
    39: 37,
    40: 37,
    41: 37,
    42: 37,
    43: 37,
    44: 39,
    45: 39,
    46: 39,
    47: 39,
    48: 39,
    49: 27,
    50: 27,
    51: 27,
    52: 27,
}
default_profiles = {
    "Abteilung_A": {
        "num_employees": 40,
        "employment_rate": 0.8,
        "office": 0.7,
        "meeting": 0.3,
        "not_office": 0.3,
        "week_factor": default_week_factor,  # Fixed week factor per profile for now
    },
    "Team_B": {
        "num_employees": 30,
        "employment_rate": 0.75,
        "office": 0.6,
        "meeting": 0.25,
        "not_office": 0.4,
        "week_factor": default_week_factor,  # Fixed week factor per profile for now
    },
    "Funktion_C": {
        "num_employees": 30,
        "employment_rate": 0.85,
        "office": 0.8,
        "meeting": 0.4,
        "not_office": 0.2,
        "week_factor": default_week_factor,  # Fixed week factor per profile for now
    },
}
default_meeting_size_dist = {
    2: 0.45,
    3: 0.25,
    4: 0.15,
    5: 0.04,
    6: 0.03,
    7: 0.02,
    8: 0.01,
    9: 0.01,
    10: 0.01,
    11: 0.01,
    12: 0.01,
    13: 0.01,
}
default_meeting_duration_dist = {
    0.5: 0.1,
    1.0: 0.6,
    1.5: 0.1,
    2.0: 0.15,
    2.5: 0.01,
    3.0: 0.01,
    3.5: 0.01,
    4.0: 0.02,
}
default_meeting_start_time_dist = {
    8: 0.07,
    8.5: 0.05,
    9: 0.09,
    9.5: 0.05,
    10: 0.11,
    10.5: 0.06,
    11: 0.07,
    11.5: 0.03,
    12: 0.02,
    12.5: 0.01,
    13: 0.08,
    13.5: 0.07,
    14: 0.08,
    14.5: 0.04,
    15: 0.06,
    15.5: 0.03,
    16: 0.05,
    16.5: 0.03,
}
default_meeting_room_max_size = {"klein": 4, "mittel": 10, "gross": 20}


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_dict(d: dict) -> dict:
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else d


# -------------------------------------------------
# Sidebar inputs
# -------------------------------------------------
st.sidebar.header("Simulation Settings")
iterations = st.sidebar.slider("Iterations", 1, 200, 20)
seed = 42  # st.sidebar.number_input("Random Seed", value=42, step=1)
min_bg = st.sidebar.slider("Min BG", 0.0, 1.0, 0.4, 0.1)
max_bg = st.sidebar.slider("Max BG", 0.0, 1.0, 1.0, 0.1)
step_bg = st.sidebar.slider("Step BG", 0.01, 0.2, 0.1, 0.01)
tolerance = st.sidebar.slider("Employment Rate Tolerance", 0.0, 0.2, 0.05, 0.01)
weeks_not_working = st.sidebar.slider("Weeks Not Working", 0, 12, 7)
min_cleardesk_hours = st.sidebar.slider("Cleardesk Hours", 0.5, 4.0, 1.5, 0.5)
sharing_factor = st.sidebar.slider("Sharing Ratio", 0.0, 1.0, 0.8, 0.05)
cut_off_quantile = st.sidebar.slider("Cut-off Quantile", 0.0, 0.5, 0.2, 0.05)


# Weekday Factor
# st.sidebar.subheader("Weekday Factor")
# week_factor_df = pd.DataFrame(
#     list(default_week_factor.items()), columns=["day", "factor"]
# )
# week_factor_df = st.sidebar.data_editor(week_factor_df, num_rows="dynamic")
# week_factor = dict(zip(week_factor_df["day"], week_factor_df["factor"]))
# week_factor = normalize_dict(week_factor)

# Meeting Size Distribution
st.sidebar.subheader("Meeting Size Distribution")
size_df = pd.DataFrame(
    list(default_meeting_size_dist.items()), columns=["size", "probability"]
)
size_df = st.sidebar.data_editor(size_df, num_rows="dynamic")
meeting_size_dist = dict(zip(size_df["size"], size_df["probability"]))
meeting_size_dist = normalize_dict(meeting_size_dist)

meeting_size_dist = default_meeting_size_dist  # Keep fixed for now

# Meeting Duration Distribution
st.sidebar.subheader("Meeting Duration Distribution")
duration_df = pd.DataFrame(
    list(default_meeting_duration_dist.items()), columns=["duration", "probability"]
)
duration_df = st.sidebar.data_editor(duration_df, num_rows="dynamic")
meeting_duration_dist = dict(zip(duration_df["duration"], duration_df["probability"]))
meeting_duration_dist = normalize_dict(meeting_duration_dist)

meeting_duration_dist = default_meeting_duration_dist  # Keep fixed for now

# Meeting Start Time Distribution
st.sidebar.subheader("Meeting Start Time Distribution")
start_df = pd.DataFrame(
    list(default_meeting_start_time_dist.items()), columns=["start_time", "probability"]
)
start_df = st.sidebar.data_editor(start_df, num_rows="dynamic")
meeting_start_time_dist = dict(zip(start_df["start_time"], start_df["probability"]))
meeting_start_time_dist = normalize_dict(meeting_start_time_dist)

# meeting_start_time_dist = default_meeting_start_time_dist  # Keep fixed for now

# Meeting Room Max Size
st.sidebar.subheader("Meeting Room Max Size")
room_size_df = pd.DataFrame(
    list(default_meeting_room_max_size.items()), columns=["room", "capacity"]
)
room_size_df = st.sidebar.data_editor(room_size_df, num_rows="dynamic")
meeting_room_max_size = dict(zip(room_size_df["room"], room_size_df["capacity"]))

meeting_room_max_size = default_meeting_room_max_size  # Keep fixed for now

# Profiles
st.sidebar.subheader("Profiles")

profiles_df = (
    pd.DataFrame.from_dict(default_profiles, orient="index")
    .reset_index()
    .rename(columns={"index": "unit"})
)

# week_factor in Spalten expandieren
week_df = profiles_df["week_factor"].apply(pd.Series)
week_df.columns = [f"wf_{c}" for c in week_df.columns]  # Prefix für Übersicht

profiles_df = pd.concat([profiles_df.drop(columns=["week_factor"]), week_df], axis=1)

edited_df = st.sidebar.data_editor(profiles_df, num_rows="dynamic")

# zurück zu dict
profiles = {}

for _, row in edited_df.iterrows():
    wf_cols = {
        c.replace("wf_", ""): row[c] for c in edited_df.columns if c.startswith("wf_")
    }
    profiles[row["unit"]] = {
        "num_employees": row["num_employees"],
        "employment_rate": row["employment_rate"],
        "office": row["office"],
        "meeting": row["meeting"],
        "not_office": row["not_office"],
        "week_factor": wf_cols,
    }
# print(profiles)


# # Week Scale
st.sidebar.subheader("Week Scale")
week_scale_df = pd.DataFrame(
    list(default_week_scale.items()), columns=["week", "scale"]
)
week_scale_df = st.sidebar.data_editor(week_scale_df, num_rows="dynamic")
week_scale = dict(zip(week_scale_df["week"], week_scale_df["scale"]))

week_scale = default_week_scale

# -------------------------------------------------
# Run Simulation
# -------------------------------------------------
if st.sidebar.button("Run Simulation"):
    weeks_in_range = (
        pd.date_range(start=default_start_date, end=default_end_date, freq="W-MON")
        .isocalendar()
        .week.unique()
    )
    week_weighting = build_week_weighting_from_weeks(weeks=weeks_in_range, weight=1.0)
    week_weighting = scale_week_weighting(week_weighting, week_scale)
    # Normalize for readability (run_simulation normalizes again for the slice)
    sw = sum(week_weighting.values())
    if sw > 0:
        week_weighting = {k: v / sw for k, v in week_weighting.items()}

    all_data, all_meetings = run_simulation(
        start_date=default_start_date,
        end_date=default_end_date,
        # week_factor=week_factor,
        profiles=profiles,
        min_bg=min_bg,
        max_bg=max_bg,
        step_bg=step_bg,
        employment_rate_variability=tolerance,
        weeks_not_working=weeks_not_working,
        iterations=iterations,
        seed=seed,
        min_cleardesk_hours=min_cleardesk_hours,
        meeting_room_max_size=meeting_room_max_size,
        week_weighting=week_weighting,
        meeting_size_dist=meeting_size_dist,
        meeting_duration_dist=meeting_duration_dist,
        meeting_start_time_dist=meeting_start_time_dist,
    )

    all_meetingrooms = build_room_occupancy_slots(
        all_meetings,
        slot_times=TIMES_DAY,
        by=("replication", "weekNumber", "date", "meeting_room_size"),
        room_col="room_id",
        include_idle=True,
    )

    st.success("Simulation complete!")

    # -------------------------------------------------
    # Visualizations
    # -------------------------------------------------
    st.subheader("📊 Einzelarbeitsplätze – Tagespeak")
    total_employees = sum(profile["num_employees"] for profile in profiles.values())
    fig1 = plot_tagespeak(all_data, total_employees, cut_off_quantile, sharing_factor)
    st.pyplot(fig1)

    st.subheader("📊 Meetingräume – Tagespeak")
    for size in ["klein", "mittel", "gross"]:
        st.markdown(f"### {size.capitalize()} Meetingräume")
        fig2 = plot_meetingrooms(all_meetingrooms, size)
        st.pyplot(fig2)

    # # -------------------------------------------------
    # # KPIs
    # # -------------------------------------------------
    # st.subheader("📈 KPIs")
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.markdown("#### BG pro Einheit")
    #     bg_rep_mean = all_data.groupby(["replication", "unit"], observed=False)[
    #         "bg"
    #     ].mean()
    #     bg_mean = bg_rep_mean.groupby("unit", observed=False).mean()
    #     bg_std = bg_rep_mean.groupby("unit", observed=False).std()
    #     st.dataframe(pd.DataFrame({"mean": bg_mean, "std": bg_std}))

    # with col2:
    #     st.markdown("#### Anteil Office pro Einheit")
    #     office_sum = all_data.groupby(["replication", "unit"], observed=False)[
    #         "present"
    #     ].sum()
    #     office_share_mean = office_sum.groupby("unit", observed=False).mean()
    #     office_share_std = office_sum.groupby("unit", observed=False).std()
    #     st.dataframe(pd.DataFrame({"mean": office_share_mean, "std": office_share_std}))

    # with col3:
    #     st.markdown("#### Meeting Rate pro Einheit")
    #     meeting_sum = all_data.groupby(["replication", "unit"], observed=False)[
    #         "meeting"
    #     ].sum()
    #     meeting_sum_mean = meeting_sum.groupby("unit", observed=False).mean()
    #     meeting_sum_std = meeting_sum.groupby("unit", observed=False).std()
    #     st.dataframe(pd.DataFrame({"mean": meeting_sum_mean, "std": meeting_sum_std}))
