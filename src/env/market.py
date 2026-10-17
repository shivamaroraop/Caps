"""Peer-to-peer double auction with vectorised merit-order clearing.

Two mechanisms are provided:

``uniform``
    A standard uniform-price double auction.  Bids are sorted descending, asks
    ascending, and the traded volume is the largest quantity at which the
    marginal bid still covers the marginal ask.  Every matched unit settles at
    one common clearing price.  This is the mechanism used in the reported
    experiments: because an agent's quote determines *whether* it trades but
    not *what* it pays, the incentive to shade prices is far weaker than under
    pay-as-bid, which keeps the learning problem well posed.

``midpoint``
    Pay-as-bid pairwise matching at the midpoint of each matched buyer/seller
    quote.  Retained for ablation against the legacy implementation.

The legacy code cleared the market with a Python ``while`` loop popping from
sorted lists.  Here the clearing point is found with two ``searchsorted`` calls
over the union of quantity breakpoints, which is O(N log N) and vectorises over
the parallel-environment axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClearingResult:
    buy_kwh: np.ndarray      # (E, N) energy bought from peers
    sell_kwh: np.ndarray     # (E, N) energy sold to peers
    price: np.ndarray        # (E,)   clearing price EUR/kWh
    volume_kwh: np.ndarray   # (E,)   total matched energy
    feasible_kwh: np.ndarray  # (E,)  min(total demand, total supply): the
                             #        maximum that *could* have been matched


def _clear_one(bid_q, bid_p, ask_q, ask_p, fallback_price, mechanism):
    """Clear a single market. Arrays are 1-D over agents."""
    n = bid_q.shape[0]
    buy = np.zeros(n)
    sell = np.zeros(n)

    b_idx = np.flatnonzero(bid_q > 1e-9)
    s_idx = np.flatnonzero(ask_q > 1e-9)
    feasible = float(min(bid_q.sum(), ask_q.sum()))
    if b_idx.size == 0 or s_idx.size == 0:
        return buy, sell, float(fallback_price), 0.0, feasible

    # Merit order: most willing buyers first, cheapest sellers first.
    b_order = b_idx[np.argsort(-bid_p[b_idx], kind="stable")]
    s_order = s_idx[np.argsort(ask_p[s_idx], kind="stable")]
    bq, bp = bid_q[b_order], bid_p[b_order]
    sq, sp = ask_q[s_order], ask_p[s_order]
    cb = np.cumsum(bq)
    cs = np.cumsum(sq)

    # Candidate clearing quantities are the cumulative breakpoints of either
    # side, capped by whichever side runs out first.
    cap = min(cb[-1], cs[-1])
    breaks = np.unique(np.concatenate([cb, cs]))
    breaks = breaks[breaks <= cap + 1e-12]
    if breaks.size == 0:
        return buy, sell, float(fallback_price), 0.0, feasible

    # Marginal price on each side just *below* a breakpoint.
    probe = breaks - 1e-9
    mb = bp[np.clip(np.searchsorted(cb, probe, side="left"), 0, bp.size - 1)]
    ms = sp[np.clip(np.searchsorted(cs, probe, side="left"), 0, sp.size - 1)]
    ok = mb + 1e-12 >= ms
    if not ok.any():
        return buy, sell, float(fallback_price), 0.0, feasible

    k = int(np.flatnonzero(ok)[-1])
    volume = float(breaks[k])
    if volume <= 1e-9:
        return buy, sell, float(fallback_price), 0.0, feasible

    # Pro-rata allocation down the merit order.
    def allocate(order, qty, cum):
        alloc = np.minimum(qty, np.maximum(volume - (cum - qty), 0.0))
        return np.clip(alloc, 0.0, qty)

    buy[b_order] = allocate(b_order, bq, cb)
    sell[s_order] = allocate(s_order, sq, cs)

    if mechanism == "uniform":
        price = 0.5 * (mb[k] + ms[k])
    else:  # midpoint / pay-as-bid over matched pairs, volume-weighted
        matched_b = buy[b_order] > 1e-12
        matched_s = sell[s_order] > 1e-12
        wb = (buy[b_order][matched_b] * bp[matched_b]).sum()
        ws = (sell[s_order][matched_s] * sp[matched_s]).sum()
        price = 0.5 * (wb + ws) / max(volume, 1e-12)

    return buy, sell, float(price), volume, feasible


def clear_double_auction(bid_qty: np.ndarray, bid_price: np.ndarray,
                         ask_qty: np.ndarray, ask_price: np.ndarray,
                         fallback_price: np.ndarray,
                         mechanism: str = "uniform") -> ClearingResult:
    """Clear ``E`` independent markets of ``N`` agents each.

    Parameters
    ----------
    bid_qty, ask_qty : (E, N) energy quantities in kWh (already feasibility
        capped by the caller against real deficit/surplus and connection limits)
    bid_price, ask_price : (E, N) EUR/kWh quotes
    fallback_price : (E,) price reported when nothing trades
    """
    E, N = bid_qty.shape
    buy = np.zeros((E, N))
    sell = np.zeros((E, N))
    price = np.zeros(E)
    volume = np.zeros(E)
    feasible = np.zeros(E)
    for e in range(E):
        b, s, p, v, f = _clear_one(bid_qty[e], bid_price[e], ask_qty[e], ask_price[e],
                                   fallback_price[e], mechanism)
        buy[e], sell[e], price[e], volume[e], feasible[e] = b, s, p, v, f
    return ClearingResult(buy, sell, price, volume, feasible)
