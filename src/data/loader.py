"""
Dataset loader for the P2P Energy Trading system.

Parses the multi-sheet Excel dataset into structured HouseholdProfile
and CommunityData objects. Handles all 19 sheets including household
loads, PV generation, BESS parameters, EV specifications, pricing,
and mobility patterns.
"""

import numpy as np
import openpyxl
from pathlib import Path
from typing import Optional

from .schemas import (
    HouseholdProfile,
    BESSProfile,
    EVProfile,
    CommunityData,
)

# Constants
NUM_HOUSEHOLDS = 250
NUM_TIMESTEPS = 96  # 15-min intervals per day


class DatasetLoader:
    """Loads and parses the energy community Excel dataset.
    
    The dataset contains 19 sheets with static parameters and
    time-series data for 250 households over 96 timesteps (1 day).
    
    Usage:
        loader = DatasetLoader("path/to/dataset.xlsx")
        community = loader.load_all()
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")
        self._wb = None

    def _open(self):
        """Open workbook (read-only mode for performance)."""
        if self._wb is None:
            self._wb = openpyxl.load_workbook(
                str(self.filepath), read_only=True, data_only=True
            )

    def _close(self):
        """Close workbook."""
        if self._wb is not None:
            self._wb.close()
            self._wb = None

    def _read_timeseries_sheet(self, sheet_name: str) -> np.ndarray:
        """Read a time-series sheet into a (96, 250) numpy array.
        
        These sheets have timestep indices in column A (rows 2-97)
        and household data in columns B-IQ (250 households).
        """
        ws = self._wb[sheet_name]
        data = np.zeros((NUM_TIMESTEPS, NUM_HOUSEHOLDS))
        for t, row in enumerate(
            ws.iter_rows(
                min_row=2, max_row=NUM_TIMESTEPS + 1,
                min_col=2, max_col=NUM_HOUSEHOLDS + 1,
                values_only=True,
            )
        ):
            for h, val in enumerate(row):
                data[t, h] = float(val) if val is not None else 0.0
        return data

    def _read_general_info(self) -> list:
        """Parse General Information sheet for PV/BESS allocation per player.
        
        Returns list of (player_id, has_pv, has_bess) tuples.
        """
        ws = self._wb["General Information"]
        players = []
        for row in ws.iter_rows(
            min_row=5, max_row=NUM_HOUSEHOLDS + 4,
            min_col=1, max_col=3,
            values_only=True,
        ):
            pid, pv, ess = row
            if pid is not None and isinstance(pid, (int, float)):
                players.append((int(pid), bool(pv), bool(ess)))
        return players

    def _read_bess_params(self) -> dict:
        """Parse BESS sheet for battery parameters per household.
        
        Returns dict mapping household index (0-based) to BESSProfile.
        """
        ws = self._wb["BESS"]
        rows = {}
        labels = []
        for row in ws.iter_rows(
            min_row=1, max_row=8,
            min_col=1, max_col=NUM_HOUSEHOLDS + 1,
            values_only=True,
        ):
            row_list = list(row)
            label = row_list[0]
            labels.append(label)
            rows[label] = row_list[1:]

        bess_dict = {}
        for h in range(NUM_HOUSEHOLDS):
            model = int(rows["Model"][h]) if rows["Model"][h] else 0
            cap = float(rows["Capacity (kW)"][h]) if rows["Capacity (kW)"][h] else 0.0
            if cap > 0:
                bess_dict[h] = BESSProfile(
                    has_bess=True,
                    model_id=model,
                    capacity_kwh=cap,
                    max_charge_kw=float(rows["Charge (kW)"][h] or 0),
                    max_discharge_kw=float(rows["Discharge (kW)"][h] or 0),
                    efficiency=float(rows["Eficiency"][h] or 0.95),
                    initial_soc_kwh=float(rows["Initial (kW)"][h] or cap * 0.5),
                    target_final_soc_kwh=float(rows["Final (kW)"][h] or cap * 0.5),
                )
            else:
                bess_dict[h] = BESSProfile(has_bess=False)
        return bess_dict

    def _read_limits(self) -> dict:
        """Parse Limits sheet for contracted power limits per household.
        
        Returns dict mapping household index to (buy_limit, sell_limit, fixed_cost).
        """
        ws = self._wb["Limits"]
        rows = {}
        for row in ws.iter_rows(
            min_row=1, max_row=4,
            min_col=1, max_col=NUM_HOUSEHOLDS + 1,
            values_only=True,
        ):
            row_list = list(row)
            label = row_list[0]
            if label is not None:
                rows[label] = row_list[1:]

        limits_dict = {}
        for h in range(NUM_HOUSEHOLDS):
            buy_lim = float(rows.get("Power Buy (kW)", [17.25] * NUM_HOUSEHOLDS)[h] or 17.25)
            sell_lim = float(rows.get("Power Sell (kW)", [17.25] * NUM_HOUSEHOLDS)[h] or 17.25)
            # Handle the fixed costs key with special character
            fixed_key = [k for k in rows.keys() if "Fixed" in str(k)]
            fixed_cost = float(rows[fixed_key[0]][h]) if fixed_key else 0.7647
            limits_dict[h] = (buy_lim, sell_lim, fixed_cost)
        return limits_dict

    def _read_ev_static(self, ev_num: int) -> dict:
        """Parse EVs sheet for static EV parameters.
        
        Args:
            ev_num: 1 for EV1 (regular), 2 for EV2 (premium)
            
        Returns dict mapping household index to EVProfile.
        """
        ws = self._wb["EVs"]
        all_rows = []
        for row in ws.iter_rows(
            min_row=1, max_row=46,
            min_col=1, max_col=NUM_HOUSEHOLDS + 1,
            values_only=True,
        ):
            all_rows.append(list(row))

        # EV1 data starts at row index 5 (row 6), EV2 at row index 26 (row 27)
        offset = 5 if ev_num == 1 else 26
        ev_dict = {}
        for h in range(NUM_HOUSEHOLDS):
            try:
                model = int(all_rows[offset + 1][h + 1] or 0)
                cap = float(all_rows[offset + 2][h + 1] or 0)
                charge = float(all_rows[offset + 3][h + 1] or 0)
                discharge = float(all_rows[offset + 4][h + 1] or 0)
                eff = float(all_rows[offset + 5][h + 1] or 0.9)
                # Initial/final may be formulas — use 20% of capacity as default
                initial = cap * 0.2
                final = initial
                consumption = float(all_rows[offset + 8][h + 1] or 150)
                departure = int(all_rows[offset + 9][h + 1] or 0)
                arrival = int(all_rows[offset + 10][h + 1] or 0)
                distance = float(all_rows[offset + 11][h + 1] or 0)
                # Trip consumption may be formula
                trip_cons = (distance * consumption) / 1000.0
                charger_type = int(all_rows[offset + 16][h + 1] or 0)
                pub_charger = int(all_rows[offset + 17][h + 1] or 0)
                pub_power = float(all_rows[offset + 18][h + 1] or 0)
                pub_price = float(all_rows[offset + 19][h + 1] or 0)

                ev_dict[h] = EVProfile(
                    model_id=model,
                    capacity_kwh=cap,
                    max_charge_kw=charge,
                    max_discharge_kw=discharge,
                    efficiency=eff,
                    initial_soc_kwh=initial,
                    target_final_soc_kwh=final,
                    consumption_wh_per_km=consumption,
                    departure_period=departure,
                    arrival_period=arrival,
                    distance_km=distance,
                    trip_consumption_kwh=trip_cons,
                    charger_type=charger_type,
                    public_charger_id=pub_charger,
                    public_charger_power_kw=pub_power,
                    public_charger_price=pub_price,
                )
            except (IndexError, TypeError, ValueError):
                ev_dict[h] = EVProfile()
        return ev_dict

    def load_all(self, max_households: Optional[int] = None) -> CommunityData:
        """Load the complete dataset and return a CommunityData object.
        
        Args:
            max_households: If set, only load this many households (for faster iteration).
            
        Returns:
            CommunityData with all household profiles populated.
        """
        self._open()
        n = min(max_households or NUM_HOUSEHOLDS, NUM_HOUSEHOLDS)
        print(f"Loading dataset for {n} households...")

        # 1. Read static allocation
        player_info = self._read_general_info()

        # 2. Read BESS parameters
        bess_params = self._read_bess_params()

        # 3. Read contracted power limits
        limits = self._read_limits()

        # 4. Read time-series sheets
        print("  Loading time-series data...")
        load_data = self._read_timeseries_sheet("Load")
        pv_data = self._read_timeseries_sheet("PV")
        buy_price = self._read_timeseries_sheet("Buy Price")
        sell_price = self._read_timeseries_sheet("Sell Price")

        # 5. Read EV time-series
        print("  Loading EV data...")
        ev1_buy_price = self._read_timeseries_sheet("EV1 Buy Price")
        ev2_buy_price = self._read_timeseries_sheet("EV2 Buy Price")
        ev1_load = self._read_timeseries_sheet("EV1 Load")
        ev2_load = self._read_timeseries_sheet("EV2 Load")
        ev1_max_discharge = self._read_timeseries_sheet("Max EV1 Dis-Charge")
        ev2_max_discharge = self._read_timeseries_sheet("Max EV2 Dis-Charge")
        ev1_at_home = self._read_timeseries_sheet("EV1 at Home (x)")
        ev2_at_home = self._read_timeseries_sheet("EV2 at Home (x)")
        ev1_at_station = self._read_timeseries_sheet("EV1 at Charging Station (x)")
        ev2_at_station = self._read_timeseries_sheet("EV2 at Charging Station (x)")

        # 6. Read EV static parameters
        ev1_static = self._read_ev_static(1)
        ev2_static = self._read_ev_static(2)

        # 7. Build HouseholdProfile objects
        print("  Building household profiles...")
        households = []
        for i in range(n):
            pid, has_pv, has_bess = player_info[i] if i < len(player_info) else (i + 1, False, False)
            buy_lim, sell_lim, fixed_cost = limits.get(i, (17.25, 17.25, 0.7647))

            # Attach EV time-series to EV profiles
            ev1 = ev1_static.get(i, EVProfile())
            ev1.buy_price = ev1_buy_price[:, i]
            ev1.load_profile = ev1_load[:, i]
            ev1.max_discharge_profile = ev1_max_discharge[:, i]
            ev1.at_home = ev1_at_home[:, i]
            ev1.at_station = ev1_at_station[:, i]

            ev2 = ev2_static.get(i, EVProfile())
            ev2.buy_price = ev2_buy_price[:, i] if i < ev2_buy_price.shape[1] else np.zeros(NUM_TIMESTEPS)
            ev2.load_profile = ev2_load[:, i]
            ev2.max_discharge_profile = ev2_max_discharge[:, i]
            ev2.at_home = ev2_at_home[:, i]
            ev2.at_station = ev2_at_station[:, i]

            household = HouseholdProfile(
                player_id=int(pid),
                has_pv=has_pv,
                has_bess=has_bess,
                power_buy_limit_kw=buy_lim,
                power_sell_limit_kw=sell_lim,
                fixed_cost_eur=fixed_cost,
                bess=bess_params.get(i, BESSProfile(has_bess=False)),
                ev1=ev1,
                ev2=ev2,
                load_profile_kw=load_data[:, i],
                pv_profile_kw=pv_data[:, i],
                buy_price_eur=buy_price[:, i],
                sell_price_eur=sell_price[:, i],
            )
            households.append(household)

        # 8. Build CommunityData
        community = CommunityData(households=households)
        community.compute_aggregates()

        self._close()
        print(f"  Loaded: {community.num_households} households, "
              f"{community.num_pv} PV, {community.num_bess} BESS, "
              f"{community.num_evs} EVs")
        return community
