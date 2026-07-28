# file: fbb_app.py
import streamlit as st
import pandas as pd
from datetime import date

# WICHTIG: aus deiner vorhandenen Datei importieren
from fbb_simulation import (
    run_simulation,
    plot_tagespeak,
    plot_meetingrooms,
    compute_tagespeak_stats,
    compute_meetingroom_stats,
    compute_reserve_am_jahrespeak,
    compute_tage_ohne_standard_mit_arbeitsmoeglichkeit,
    compute_tage_ohne_standard_und_arbeitsmoeglichkeit,
    build_week_weighting_from_weeks,
    scale_week_weighting,
)

st.set_page_config(page_title="Office Simulation Dashboard", layout="wide")
st.title("🏢 Office Simulation Dashboard")

# ---------------------------
# Defaults
# ---------------------------
default_start_date = date(2025, 1, 1)
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
        "num_fixed_employees": 0,
        "employment_rate": 0.8,
        "office": 0.7,
        "meeting": 0.3,
        "not_office": 0.3,
        "week_factor": default_week_factor,
    },
    "Team_B": {
        "num_employees": 30,
        "num_fixed_employees": 0,
        "employment_rate": 0.75,
        "office": 0.6,
        "meeting": 0.25,
        "not_office": 0.4,
        "week_factor": default_week_factor,
    },
    "Funktion_C": {
        "num_employees": 30,
        "num_fixed_employees": 0,
        "employment_rate": 0.85,
        "office": 0.8,
        "meeting": 0.4,
        "not_office": 0.2,
        "week_factor": default_week_factor,
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
default_meeting_room_max_size = {"klein": 4, "mittel": 8, "gross": 16}

# ---------------------------
# Helpers
# ---------------------------
def normalize_dict(d: dict) -> dict:
    """Skaliert die Werte eines dict so, dass sie sich zu 1 aufsummieren."""

    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else d

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Simulation Settings")
iterations = st.sidebar.slider("Iterations", 1, 200, 20)
seed = 42
min_bg = st.sidebar.slider("Min BG", 0.0, 1.0, 0.4, 0.1)
max_bg = st.sidebar.slider("Max BG", 0.0, 1.0, 1.0, 0.1)
step_bg = st.sidebar.slider("Step BG", 0.01, 0.2, 0.1, 0.01)
tolerance = (
    0.05  # = st.sidebar.slider("Employment Rate Tolerance", 0.0, 0.2, 0.05, 0.01)
)
weeks_not_working = st.sidebar.slider("Weeks Not Working", 0, 12, 7)
min_cleardesk_hours = st.sidebar.slider("Cleardesk Hours", 0.5, 4.0, 2.0, 0.5)
cut_off_quantile = st.sidebar.slider("Cut-off Quantile", 0.0, 0.5, 0.2, 0.05)

# Profiles
st.sidebar.subheader("Profiles")
profiles_df = (
    pd.DataFrame.from_dict(default_profiles, orient="index")
    .reset_index()
    .rename(columns={"index": "unit"})
)
week_df = profiles_df["week_factor"].apply(pd.Series)
week_df.columns = [f"wf_{c}" for c in week_df.columns]
profiles_df = pd.concat([profiles_df.drop(columns=["week_factor"]), week_df], axis=1)
edited_df = st.sidebar.data_editor(profiles_df, num_rows="dynamic")

profiles = {}
for _, row in edited_df.iterrows():
    wf_cols = {
        c.replace("wf_", ""): row[c] for c in edited_df.columns if c.startswith("wf_")
    }
    raw_num_fixed = row.get("num_fixed_employees", 0)
    num_fixed_employees = int(raw_num_fixed) if pd.notna(raw_num_fixed) else 0
    profiles[row["unit"]] = {
        "num_employees": int(row["num_employees"]),
        "num_fixed_employees": num_fixed_employees,
        "employment_rate": float(row["employment_rate"]),
        "office": float(row["office"]),
        "meeting": float(row["meeting"]),
        "not_office": float(row["not_office"]),
        "week_factor": normalize_dict(wf_cols),
    }

# Meeting Size Distribution
st.sidebar.subheader("Meeting Size Distribution")
size_df = pd.DataFrame(
    list(default_meeting_size_dist.items()), columns=["size", "probability"]
)
size_df = st.sidebar.data_editor(size_df, num_rows="dynamic")
meeting_size_dist = normalize_dict(dict(zip(size_df["size"], size_df["probability"])))

# Meeting Duration Distribution
st.sidebar.subheader("Meeting Duration Distribution")
duration_df = pd.DataFrame(
    list(default_meeting_duration_dist.items()), columns=["duration", "probability"]
)
duration_df = st.sidebar.data_editor(duration_df, num_rows="dynamic")
meeting_duration_dist = normalize_dict(
    dict(zip(duration_df["duration"], duration_df["probability"]))
)

# Meeting Start Time Distribution
st.sidebar.subheader("Meeting Start Time Distribution")
start_df = pd.DataFrame(
    list(default_meeting_start_time_dist.items()), columns=["start_time", "probability"]
)
start_df = st.sidebar.data_editor(start_df, num_rows="dynamic")
meeting_start_time_dist = normalize_dict(
    dict(zip(start_df["start_time"], start_df["probability"]))
)

# Meeting Room Max Size
st.sidebar.subheader("Meeting Room Max Size")
room_size_df = pd.DataFrame(
    list(default_meeting_room_max_size.items()), columns=["room", "capacity"]
)
room_size_df = st.sidebar.data_editor(room_size_df, num_rows="dynamic")
meeting_room_max_size = dict(zip(room_size_df["room"], room_size_df["capacity"]))


# Week Scale
st.sidebar.subheader("Week Scale")
week_scale_df = pd.DataFrame(
    list(default_week_scale.items()), columns=["week", "scale"]
)
week_scale_df = st.sidebar.data_editor(week_scale_df, num_rows="dynamic")
week_scale = dict(zip(week_scale_df["week"], week_scale_df["scale"]))


# ---------------------------
# Cache: Simulation + Occupancy
# ---------------------------
@st.cache_data(show_spinner=False)
def _cached_simulation(
    start_date,
    end_date,
    profiles,
    min_bg,
    max_bg,
    step_bg,
    tolerance,
    weeks_not_working,
    iterations,
    seed,
    min_cleardesk_hours,
    meeting_room_max_size,
    week_weighting,
    meeting_size_dist,
    meeting_duration_dist,
    meeting_start_time_dist,
):
    """Cache-Wrapper um run_simulation: teure Simulation, identische Inputs → Wiederverwendung."""

    return run_simulation(
        start_date=start_date,
        end_date=end_date,
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
        return_slot_totals=True,
    )


# ---------------------------
# Run Simulation Button
# ---------------------------
if st.sidebar.button("Run Simulation"):
    try:
        with st.status("Simulation läuft …", expanded=True) as status:
            st.write("📦 Baue Wochengewichtung …")
            weeks_in_range = (
                pd.date_range(
                    start=default_start_date, end=default_end_date, freq="W-MON"
                )
                .isocalendar()
                .week.unique()
            )
            week_weighting = build_week_weighting_from_weeks(
                weeks=weeks_in_range, weight=1.0
            )
            week_weighting = scale_week_weighting(week_weighting, week_scale)
            s = sum(week_weighting.values())
            if s > 0:
                week_weighting = {k: v / s for k, v in week_weighting.items()}

            st.write("🧮 Starte Simulation …")
            all_data, all_meetings, slot_totals = _cached_simulation(
                default_start_date,
                default_end_date,
                profiles,
                min_bg,
                max_bg,
                step_bg,
                tolerance,
                weeks_not_working,
                iterations,
                seed,
                min_cleardesk_hours,
                meeting_room_max_size,
                week_weighting,
                meeting_size_dist,
                meeting_duration_dist,
                meeting_start_time_dist,
            )

            status.update(label="Fertig ✅", state="complete")

        total_fixed_employees = sum(
            int(profile.get("num_fixed_employees", 0)) for profile in profiles.values()
        )

        # Ergebnisse persistieren: Folge-Eingaben (Standardarbeitsplätze,
        # Risikokennwerte) sollen ohne erneute (teure) Simulation reagieren.
        st.session_state["sim_results"] = {
            "all_data": all_data,
            "all_meetings": all_meetings,
            "slot_totals": slot_totals,
            "total_fixed_employees": total_fixed_employees,
        }

    except Exception as e:
        # Sichtbarer Fehler statt App-Abbruch
        st.error("Die Simulation ist fehlgeschlagen.")
        st.exception(e)
        st.session_state.pop("sim_results", None)

# ---------------------------
# Ergebnisse (reagieren live auf Sidebar- und Folge-Eingaben,
# ohne die Simulation neu laufen zu lassen)
# ---------------------------
if "sim_results" in st.session_state:
    res = st.session_state["sim_results"]
    all_data = res["all_data"]
    all_meetings = res["all_meetings"]
    slot_totals = res["slot_totals"]
    total_fixed_employees = res["total_fixed_employees"]

    st.success("Simulation complete!")

    # ---------------------------
    # Kennzahlen
    # ---------------------------
    tagespeak_stats = compute_tagespeak_stats(
        all_data,
        cut_off_quantile=cut_off_quantile,
        num_fixed_employees=total_fixed_employees,
    )

    room_order = ["klein", "mittel", "gross"]
    min_meeting_size = int(min(meeting_size_dist.keys())) if meeting_size_dist else 2
    room_bounds = {}
    prev_max = min_meeting_size - 1
    for r in room_order:
        cap = int(meeting_room_max_size.get(r, prev_max))
        room_bounds[r] = (prev_max + 1, cap)
        prev_max = cap
    room_stats = {
        size: compute_meetingroom_stats(all_meetings, size) for size in room_order
    }

    st.subheader("Einzelarbeitsplätze")
    col1, col2 = st.columns(2)
    col1.metric("Absoluter Jahrespeak", f"{tagespeak_stats['max_daily_peak']:.0f}")
    col2.metric("Max. Ø-Tagespeak", f"{tagespeak_stats['max_avg_peak']:.0f}")
    with st.expander("Verteilung ansehen"):
        fig1 = plot_tagespeak(
            all_data,
            cut_off_quantile,
            total_fixed_employees,
        )
        st.pyplot(fig1, clear_figure=True)

    st.subheader("Meetingräume")
    cols = st.columns(len(room_order))
    for col, size in zip(cols, room_order):
        lo, hi = room_bounds[size]
        col.metric(
            f"{size.capitalize()} ({lo}–{hi} P)",
            f"{room_stats[size]['max_avg_peak']:.0f}",
        )
    with st.expander("Verteilung ansehen"):
        for size in room_order:
            st.markdown(f"### {size.capitalize()} Meetingräume")
            fig2 = plot_meetingrooms(all_meetings, size)
            st.pyplot(fig2, clear_figure=True)

    # ---------------------------
    # Standardarbeitsplätze (editierbare Folge-Eingaben nach der Simulation)
    # ---------------------------
    st.subheader("Standardarbeitsplätze")

    def _ap_input_row(label, help_text, suggested_value, key, note=None):
        """Zeigt Label + Simulationsvorschlag links und ein editierbares Zahlenfeld rechts an."""
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{label}**", help=help_text)
            st.caption(f"Simulationsvorschlag: {suggested_value:.0f}")
            if note:
                st.caption(note)
        with c2:
            return c2.number_input(
                label,
                min_value=0,
                value=int(round(suggested_value)),
                key=key,
                label_visibility="collapsed",
            )

    shared_ap = _ap_input_row(
        "Shared",
        "Anzahl geteilter Standardarbeitsplätze (Sharing-Pool). "
        "Vorschlag aus dem Max. Ø-Tagespeak der Simulation.",
        tagespeak_stats["max_avg_peak"],
        "std_shared",
    )
    zugewiesen_ap = _ap_input_row(
        "Zugewiesen",
        "Anzahl fest zugewiesener (fixer) Arbeitsplätze, kein Sharing. "
        "Vorschlag aus den Fix-AP-Angaben der Profile.",
        total_fixed_employees,
        "std_zugewiesen",
    )
    arbeitsmoeglichkeiten = _ap_input_row(
        "Arbeitsmöglichkeiten",
        "Zusätzliche flexible Arbeitsmöglichkeiten (z.B. Touchdown, Fokusräume, "
        "Lounge) – nicht Teil der Simulation, manuell zu planen.",
        0,
        "std_arbeitsmoeglichkeiten",
    )
    sekundaer_ap = _ap_input_row(
        "Sekundärarbeitsplätze",
        "Arbeitsplätze in anderen Flächentypen (z.B. Cafeteria, Lounge) – "
        "nicht Teil der Simulation, manuell zu planen.",
        0,
        "std_sekundaer",
        note="m²-Bedarf in anderen Flächentypen enthalten",
    )

    # ---------------------------
    # Risikokennwerte
    # ---------------------------
    st.subheader("Risikokennwerte")

    reserve = compute_reserve_am_jahrespeak(
        tagespeak_stats["max_daily_peak"],
        shared_ap,
        zugewiesen_ap,
        arbeitsmoeglichkeiten,
        sekundaer_ap,
    )
    tage_mit_arbeitsmoeglichkeit = compute_tage_ohne_standard_mit_arbeitsmoeglichkeit(
        slot_totals, shared_ap, zugewiesen_ap, arbeitsmoeglichkeiten
    )
    tage_mit_sekundaer = compute_tage_ohne_standard_und_arbeitsmoeglichkeit(
        slot_totals, shared_ap, zugewiesen_ap, arbeitsmoeglichkeiten
    )

    st.metric(
        "Verbleibende freie Einzel-APs bei absolutem Jahrespeak",
        f"{reserve:+.0f}",
        help=(
            "Alle Einzelarbeitsplätze (Standard, Arbeitsmöglichkeiten, Sekundär) "
            "abzüglich des absoluten Belegungs-Jahrespeaks. "
            "Positiv = Reserve, negativ = Überlastung."
        ),
    )
    c1, c2 = st.columns(2)
    c1.markdown(
        "**Personenstunden ohne Standard-AP, aber mit Arbeitsmöglichkeit**",
        help=(
            "Anzahl Tage pro Jahr (Ø über Replikationen) mit nennenswerter "
            "Einschränkung: an mehr als 2% der Einzel-AP-Bedarfsstunden dieses "
            "Tages ist kein Standardarbeitsplatz, aber eine Arbeitsmöglichkeit "
            "verfügbar."
        ),
    )
    c1.caption("Anzahl Tage pro Jahr mit nennenswerter Einschränkung")
    c1.metric(" ", f"{tage_mit_arbeitsmoeglichkeit:.0f}", label_visibility="collapsed")

    c2.markdown(
        "**Personenstunden ohne Alternative zur Nutzung eines Sekundär-APs**",
        help=(
            "Anzahl Tage pro Jahr (Ø über Replikationen) mit nennenswerter "
            "Einschränkung: an mehr als 2% der Einzel-AP-Bedarfsstunden dieses "
            "Tages ist weder ein Standardarbeitsplatz noch eine Arbeitsmöglichkeit, "
            "aber ein Sekundärarbeitsplatz verfügbar."
        ),
    )
    c2.caption("Anzahl Tage pro Jahr mit nennenswerter Einschränkung")
    c2.metric(" ", f"{tage_mit_sekundaer:.0f}", label_visibility="collapsed")
