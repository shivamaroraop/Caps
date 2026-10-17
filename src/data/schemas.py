"""
Data schemas for the P2P Energy Trading system.

Defines structured data classes for all entities in the energy community:
households, batteries, EVs, PV systems, and the community aggregate.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BESSProfile:
    """Battery Energy Storage System parameters for a single household."""
    has_bess: bool
    model_id: int = 0
    capacity_kwh: float = 0.0       # Total energy capacity
    max_charge_kw: float = 0.0      # Maximum charging power
    max_discharge_kw: float = 0.0   # Maximum discharging power
    efficiency: float = 0.95        # Round-trip efficiency (0-1)
    initial_soc_kwh: float = 0.0    # Starting energy level
    target_final_soc_kwh: float = 0.0  # Desired end-of-day energy level


@dataclass
class EVProfile:
    """Electric Vehicle parameters for a single EV (EV1 or EV2)."""
    model_id: int = 0
    capacity_kwh: float = 0.0
    max_charge_kw: float = 0.0
    max_discharge_kw: float = 0.0
    efficiency: float = 0.90
    initial_soc_kwh: float = 0.0
    target_final_soc_kwh: float = 0.0
    consumption_wh_per_km: float = 150.0
    departure_period: int = 0         # 15-min period index (0-95)
    arrival_period: int = 0           # 15-min period index (0-95)
    distance_km: float = 0.0
    trip_consumption_kwh: float = 0.0
    charger_type: int = 0
    public_charger_id: int = 0
    public_charger_power_kw: float = 0.0
    public_charger_price: float = 0.0

    # Time-series data (96 timesteps)
    buy_price: Optional[np.ndarray] = None        # (96,)
    load_profile: Optional[np.ndarray] = None     # (96,)
    max_discharge_profile: Optional[np.ndarray] = None  # (96,)
    at_home: Optional[np.ndarray] = None          # (96,) binary
    at_station: Optional[np.ndarray] = None       # (96,) binary


@dataclass
class HouseholdProfile:
    """Complete profile for a single household/prosumer agent.
    
    Contains all static parameters and time-series data needed
    to simulate one household for a single day (96 timesteps).
    """
    player_id: int
    has_pv: bool
    has_bess: bool

    # Contracted power limits
    power_buy_limit_kw: float = 17.25
    power_sell_limit_kw: float = 17.25
    fixed_cost_eur: float = 0.7647

    # BESS parameters
    bess: Optional[BESSProfile] = None

    # EV parameters
    ev1: Optional[EVProfile] = None
    ev2: Optional[EVProfile] = None

    # Time-series data (96 timesteps for one day)
    load_profile_kw: Optional[np.ndarray] = None      # (96,) household load
    pv_profile_kw: Optional[np.ndarray] = None         # (96,) PV generation
    buy_price_eur: Optional[np.ndarray] = None          # (96,) grid buy price
    sell_price_eur: Optional[np.ndarray] = None         # (96,) grid sell price

    @property
    def daily_load_kwh(self) -> float:
        """Total daily energy consumption in kWh (15-min intervals)."""
        if self.load_profile_kw is not None:
            return float(np.sum(self.load_profile_kw) * 0.25)
        return 0.0

    @property
    def daily_pv_kwh(self) -> float:
        """Total daily PV generation in kWh."""
        if self.pv_profile_kw is not None:
            return float(np.sum(self.pv_profile_kw) * 0.25)
        return 0.0

    @property
    def self_sufficiency_ratio(self) -> float:
        """Fraction of load that could theoretically be met by PV."""
        if self.daily_load_kwh > 0:
            return min(1.0, self.daily_pv_kwh / self.daily_load_kwh)
        return 0.0


@dataclass
class CommunityData:
    """Aggregate data for the entire energy community.
    
    Contains list of all household profiles and community-level
    computed statistics used as global observations in MARL.
    """
    households: list = field(default_factory=list)  # List[HouseholdProfile]
    num_households: int = 0
    num_pv: int = 0
    num_bess: int = 0
    num_evs: int = 0
    timesteps_per_day: int = 96
    timestep_minutes: int = 15

    # Community aggregate time-series (computed from households)
    total_load_kw: Optional[np.ndarray] = None    # (96,)
    total_pv_kw: Optional[np.ndarray] = None      # (96,)
    net_demand_kw: Optional[np.ndarray] = None    # (96,) load - pv

    def compute_aggregates(self):
        """Compute community-level aggregate profiles from individual households."""
        if not self.households:
            return
        
        self.num_households = len(self.households)
        self.num_pv = sum(1 for h in self.households if h.has_pv)
        self.num_bess = sum(1 for h in self.households if h.has_bess)
        self.num_evs = self.num_households * 2  # 2 EVs per household

        self.total_load_kw = np.zeros(self.timesteps_per_day)
        self.total_pv_kw = np.zeros(self.timesteps_per_day)

        for h in self.households:
            if h.load_profile_kw is not None:
                self.total_load_kw += h.load_profile_kw
            if h.pv_profile_kw is not None:
                self.total_pv_kw += h.pv_profile_kw

        self.net_demand_kw = self.total_load_kw - self.total_pv_kw

    def get_household(self, player_id: int) -> Optional[HouseholdProfile]:
        """Retrieve a household by its player ID."""
        for h in self.households:
            if h.player_id == player_id:
                return h
        return None

    def get_prosumers(self) -> list:
        """Return only households that have PV generation."""
        return [h for h in self.households if h.has_pv]

    def get_bess_owners(self) -> list:
        """Return only households that have battery storage."""
        return [h for h in self.households if h.has_bess]
