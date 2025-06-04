"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import math
from collections import defaultdict
import logging
import json
from typing import Optional
from pathlib import Path
import bw2calc
import numpy as np
import sparse
import pandas as pd
from prettytable import PrettyTable
import bw2data
from tqdm import tqdm
from textwrap import fill
from functools import lru_cache

from .utils import (
    format_data,
    get_flow_matrix_positions,
    safe_eval_cached,
    validate_parameter_lengths,
    get_str,
    make_hashable,
)
from .matrix_builders import initialize_lcia_matrix, build_technosphere_edges_matrix
from .flow_matching import (
    preprocess_cfs,
    matches_classifications,
    normalize_classification_entries,
    build_cf_index,
    cached_match_with_index,
    preprocess_flows,
    build_index,
    compute_cf_memoized_factory,
    resolve_candidate_consumers,
    normalize_signature_data,
    group_edges_by_signature,
    compute_average_cf,
)
from .georesolver import GeoResolver
from .uncertainty import sample_cf_distribution
from .filesystem_constants import DATA_DIR

# delete the logs
with open("edgelcia.log", "w", encoding="utf-8"):
    pass

# initiate the logger
logging.basicConfig(
    filename="edgelcia.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(funcName)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def add_cf_entry(
    cfs_mapping, supplier_info, consumer_info, direction, indices, value, uncertainty
):

    if direction == "biosphere-technosphere":
        supplier_info["matrix"] = "biosphere"
    else:
        supplier_info["matrix"] = "technosphere"
    consumer_info["matrix"] = "technosphere"

    entry = {
        "supplier": supplier_info,
        "consumer": consumer_info,
        "positions": indices,
        "direction": direction,
        "value": value,
    }

    if uncertainty is not None:
        entry["uncertainty"] = uncertainty

    cfs_mapping.append(entry)


class EdgeLCIA:
    """
    Class that implements the calculation of the regionalized life cycle impact assessment (LCIA) results.
    Relies on bw2data.LCA class for inventory calculations and matrices.
    """

    def __init__(
        self,
        demand: dict,
        method: Optional[tuple] = None,
        weight: Optional[str] = "population",
        parameters: Optional[dict] = None,
        scenario: Optional[str] = None,
        filepath: Optional[str] = None,
        allowed_functions: Optional[dict] = None,
        use_distributions: Optional[bool] = False,
        random_seed: Optional[int] = None,
        iterations: Optional[int] = 100,
    ):
        """
        Initialize an EdgeLCIA object for exchange-level life cycle impact assessment.

        Parameters
        ----------
        demand : dict
            A Brightway-style demand dictionary defining the functional unit.
        method : tuple, optional
            Method name as a tuple (e.g., ("AWARE", "2.0")), used to locate the CF JSON file.
        weight : str, optional
            Weighting variable used for region aggregation/disaggregation (e.g., "population", "gdp").
        parameters : dict, optional
            Dictionary of parameter values or scenarios for symbolic CF evaluation.
        scenario : str, optional
            Name of the default scenario (must match a key in `parameters`).
        filepath : str, optional
            Explicit path to the JSON method file; overrides `method` if provided.
        allowed_functions : dict, optional
            Additional safe functions available to CF evaluation expressions.
        use_distributions : bool, optional
            Whether to interpret CF uncertainty fields and perform Monte Carlo sampling.
        random_seed : int, optional
            Seed for reproducible uncertainty sampling.
        iterations : int, optional
            Number of Monte Carlo samples to draw if uncertainty is enabled.

        Notes
        -----
        After initialization, the standard evaluation sequence is:
        1. `lci()`
        2. `map_exchanges()`
        3. Optionally: regional mapping methods
        4. `evaluate_cfs()`
        5. `lcia()`
        6. Optionally: `statistics()`, `generate_df_table()`
        """
        self.cf_index = None
        self.scenario_cfs = None
        self.method_metadata = None
        self.demand = demand
        self.weights = None
        self.consumer_lookup = None
        self.reversed_consumer_lookup = None
        self.reversed_supplier_lookup = None
        self.processed_technosphere_edges = None
        self.processed_biosphere_edges = None
        self.raw_cfs_data = None
        self.unprocessed_technosphere_edges = []
        self.unprocessed_biosphere_edges = []
        self.score = None
        self.cfs_number = None
        self.filepath = Path(filepath) if filepath else None
        self.reversed_biosphere = None
        self.reversed_activity = None
        self.characterization_matrix = None
        self.method = method  # Store the method argument in the instance
        self.position_to_technosphere_flows_lookup = None
        self.technosphere_flows_lookup = defaultdict(list)
        self.technosphere_edges = []
        self.technosphere_flow_matrix = None
        self.biosphere_edges = []
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
        self.biosphere_characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.ignored_method_exchanges = list()
        self.weight: str = weight

        # Accept both "parameters" and "scenarios" for flexibility
        self.parameters = parameters or {}

        self.scenario = scenario  # New: store default scenario
        self.scenario_length = validate_parameter_lengths(parameters=self.parameters)
        self.use_distributions = use_distributions
        self.iterations = iterations
        self.random_seed = random_seed if random_seed is not None else 42
        self.random_state = np.random.default_rng(self.random_seed)

        self.lca = bw2calc.LCA(demand=self.demand)
        self._load_raw_lcia_data()
        self.cfs_mapping = []

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fh = logging.FileHandler("edgelcia.log")
            formatter = logging.Formatter(
                "%(asctime)s,%(msecs)d %(name)s %(funcName)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.logger.propagate = False

        self.SAFE_GLOBALS = {
            "__builtins__": None,
            "abs": abs,
            "max": max,
            "min": min,
            "round": round,
            "pow": pow,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log10": math.log10,
        }

        # Allow user-defined trusted functions explicitly
        if allowed_functions:
            self.SAFE_GLOBALS.update(allowed_functions)

        self._cached_supplier_keys = self._get_candidate_supplier_keys()

    def _load_raw_lcia_data(self):
        if self.filepath is None:
            self.filepath = DATA_DIR / f"{'_'.join(self.method)}.json"
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Store full method metadata except exchanges and parameters
        self.raw_cfs_data, self.method_metadata = format_data(raw, self.weight)
        self.raw_cfs_data = normalize_classification_entries(self.raw_cfs_data)
        self.cfs_number = len(self.raw_cfs_data)

        # Extract parameters or scenarios from method file if not already provided
        if not self.parameters:
            self.parameters = raw.get("scenarios", raw.get("parameters", {}))

        # Fallback to default scenario
        if self.scenario and self.scenario not in self.parameters:
            raise ValueError(
                f"Scenario '{self.scenario}' not found in available parameters: {list(self.parameters)}"
            )

        grouping_mode = self._detect_cf_grouping_mode()
        self.cfs_lookup = preprocess_cfs(self.raw_cfs_data, by=grouping_mode)

        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight", "position"}
        }

        self.cf_index = build_cf_index(self.raw_cfs_data, self.required_supplier_fields)

    def _initialize_weights(self):
        if self.weights is not None:
            return

        if not self.raw_cfs_data:
            self.weights = {}
            return

        self.weights = {}
        for cf in self.raw_cfs_data:
            consumer = cf.get("consumer", {})
            location = consumer.get("location")
            weight = cf.get("weight")
            if location and weight is not None:
                loc_str = get_str(location)
                self.weights[loc_str] = weight

        if hasattr(self, "_geo") and self._geo is not None:
            self._geo._cached_lookup.cache_clear()

    def _get_candidate_supplier_keys(self):
        if hasattr(self, "_cached_supplier_keys"):
            return self._cached_supplier_keys

        grouping_mode = self._detect_cf_grouping_mode()
        cfs_lookup = preprocess_cfs(self.raw_cfs_data, by=grouping_mode)

        keys = set()
        for cf_list in cfs_lookup.values():
            for cf in cf_list:
                filtered = {
                    k: cf["supplier"].get(k)
                    for k in self.required_supplier_fields
                    if cf["supplier"].get(k) is not None
                }

                # Normalize classification field
                if "classifications" in filtered:
                    c = filtered["classifications"]
                    if isinstance(c, dict):
                        filtered["classifications"] = tuple(
                            (scheme, tuple(vals)) for scheme, vals in sorted(c.items())
                        )
                    elif isinstance(c, list):
                        filtered["classifications"] = tuple(c)

                keys.add(make_hashable(filtered))

        self._cached_supplier_keys = keys
        return keys

    def _detect_cf_grouping_mode(self):
        has_consumer_locations = any(
            "location" in cf.get("consumer", {}) for cf in self.raw_cfs_data
        )
        has_supplier_locations = any(
            "location" in cf.get("supplier", {}) for cf in self.raw_cfs_data
        )
        if has_consumer_locations and not has_supplier_locations:
            return "consumer"
        elif has_supplier_locations and not has_consumer_locations:
            return "supplier"
        else:
            return "both"

    def _resolve_parameters_for_scenario(
        self, scenario_idx: int, scenario_name: Optional[str] = None
    ) -> dict:
        scenario_name = scenario_name or self.scenario

        param_set = self.parameters.get(scenario_name)

        resolved = {}
        if param_set is not None:
            for k, v in param_set.items():
                if isinstance(v, dict):
                    resolved[k] = v.get(str(scenario_idx), list(v.values())[-1])
                else:
                    resolved[k] = v
        return resolved

    def _update_unprocessed_edges(self):
        self.processed_biosphere_edges = {
            pos
            for cf in self.cfs_mapping
            if cf["direction"] == "biosphere-technosphere"
            for pos in cf["positions"]
        }

        self.processed_technosphere_edges = {
            pos
            for cf in self.cfs_mapping
            if cf["direction"] == "technosphere-technosphere"
            for pos in cf["positions"]
        }

        print(
            f"Processed edges: {len(self.processed_biosphere_edges) + len(self.processed_technosphere_edges)}"
        )

        self.unprocessed_biosphere_edges = [
            edge
            for edge in self.biosphere_edges
            if edge not in self.processed_biosphere_edges
        ]

        self.unprocessed_technosphere_edges = [
            edge
            for edge in self.technosphere_edges
            if edge not in self.processed_technosphere_edges
        ]

    def _equality_supplier_signature(self, supplier_info: dict) -> tuple:
        """
        Return a hashable supplier signature based on required supplier fields.
        If no fields are required, return an empty tuple (matches any supplier).
        """
        if not self.required_supplier_fields:
            return ()

        filtered = {
            k: supplier_info.get(k)
            for k in self.required_supplier_fields
            if supplier_info.get(k) is not None
        }

        # Normalize classifications
        if "classifications" in filtered:
            classifications = filtered["classifications"]

            # Brightway format: list or tuple of (scheme, code)
            if isinstance(classifications, (list, tuple)):
                try:
                    filtered["classifications"] = tuple(
                        sorted((str(s), str(c)) for s, c in classifications)
                    )
                except Exception:
                    self.logger.warning(
                        f"Malformed classification tuples: {classifications}"
                    )
                    filtered["classifications"] = ()

            # CF format: dict of scheme -> list of codes
            elif isinstance(classifications, dict):
                filtered["classifications"] = tuple(
                    (scheme, tuple(sorted(map(str, codes))))
                    for scheme, codes in sorted(classifications.items())
                )
            else:
                self.logger.warning(
                    f"Unexpected classifications format: {classifications}"
                )
                filtered["classifications"] = ()

        return make_hashable(filtered)

    def _preprocess_lookups(self):
        """
        Preprocess supplier and consumer flows into lookup dictionaries.
        """

        # Constants for ignored fields
        IGNORED_FIELDS = {"matrix", "operator", "weight", "classifications"}

        self.required_consumer_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        }

        self.supplier_lookup = preprocess_flows(
            flows_list=(
                self.biosphere_flows
                if all(
                    cf["supplier"].get("matrix") == "biosphere"
                    for cf in self.raw_cfs_data
                )
                else self.technosphere_flows
            ),
            mandatory_fields=self.required_supplier_fields,
        )

        self.consumer_lookup = preprocess_flows(
            flows_list=self.technosphere_flows,
            mandatory_fields=self.required_consumer_fields,
        )

        self.reversed_supplier_lookup = {
            pos: key
            for key, positions in self.supplier_lookup.items()
            for pos in positions
        }

        self.reversed_consumer_lookup = {
            pos: key
            for key, positions in self.consumer_lookup.items()
            for pos in positions
        }

    def _get_consumer_info(self, consumer_idx):
        """
        Returns a dict with consumer flow info, including a fallback for missing fields
        like 'location' from the position_to_technosphere_flows_lookup.
        """
        info = dict(self.reversed_consumer_lookup.get(consumer_idx, {}))
        if "location" not in info:
            fallback = self.position_to_technosphere_flows_lookup.get(consumer_idx, {})
            if fallback and "location" in fallback:
                info["location"] = fallback["location"]
        return info

    def lci(self) -> None:
        """
        Perform the life cycle inventory (LCI) calculation and extract relevant exchanges.

        This step computes the inventory matrix using Brightway2 and stores the
        biosphere and/or technosphere exchanges relevant for impact assessment.

        It also builds lookups for flow indices, supplier and consumer locations,
        and initializes flow matrices used in downstream CF mapping.

        Must be called before `map_exchanges()` or any mapping or evaluation step.
        """

        self.lca.lci()

        if all(
            cf["supplier"].get("matrix") == "technosphere" for cf in self.raw_cfs_data
        ):
            self.technosphere_flow_matrix = build_technosphere_edges_matrix(
                self.lca.technosphere_matrix, self.lca.supply_array
            )
            self.technosphere_edges = set(
                list(zip(*self.technosphere_flow_matrix.nonzero()))
            )
        else:
            self.biosphere_edges = set(list(zip(*self.lca.inventory.nonzero())))

        unique_biosphere_flows = set(x[0] for x in self.biosphere_edges)

        if len(unique_biosphere_flows) > 0:
            self.biosphere_flows = get_flow_matrix_positions(
                {
                    k: v
                    for k, v in self.lca.biosphere_dict.items()
                    if v in unique_biosphere_flows
                }
            )

        self.technosphere_flows = get_flow_matrix_positions(
            {k: v for k, v in self.lca.activity_dict.items()}
        )

        self.reversed_activity, _, self.reversed_biosphere = self.lca.reverse_dict()

        # Build technosphere flow lookups as in the original implementation.
        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }

    def map_exchanges(self):
        """
        Match inventory exchanges to characterization factors based on supplier and consumer criteria.

        This method performs direct matching of biosphere or technosphere exchanges
        against the CF definitions in the method file. It supports matching based on:

        - Flow `name`, `reference product`, `location`
        - Matrix type (`biosphere` or `technosphere`)
        - `classifications` (e.g., CPC codes)

        It populates the internal `cfs_mapping` with the positions and values of matched CFs.

        This is typically the first mapping step after inventory calculation (`lci()`).

        Notes
        -----
        - This step only handles CFs with explicit, direct matches.
        - Exchanges not matched in this step may be handled by subsequent mapping steps:
          `map_aggregate_locations()`, `map_dynamic_locations()`, etc.
        - Matching logic uses both string and classification matching with caching for efficiency.

        Raises
        ------
        ValueError
            If required metadata is missing from the inventory (e.g., flow location or classifications).
        """

        self._initialize_weights()
        self._preprocess_lookups()

        edges = (
            self.biosphere_edges
            if all(
                cf["supplier"].get("matrix") == "biosphere" for cf in self.raw_cfs_data
            )
            else self.technosphere_edges
        )

        print(f"Mapping {len(edges)} exchanges...")

        supplier_index = build_index(
            self.supplier_lookup, self.required_supplier_fields
        )
        consumer_index = build_index(
            self.consumer_lookup, self.required_consumer_fields
        )

        seen_positions = []

        for i, cf in enumerate(tqdm(self.raw_cfs_data, desc="Mapping exchanges")):
            supplier_criteria = cf["supplier"]
            consumer_criteria = cf["consumer"]

            # Step 1: Classifications filter
            if "classifications" in supplier_criteria:
                cf_class = supplier_criteria["classifications"]
                classification_matches = [
                    idx
                    for idx in self.reversed_supplier_lookup
                    if matches_classifications(
                        cf_class,
                        dict(self.reversed_supplier_lookup[idx]).get(
                            "classifications", []
                        ),
                    )
                ]
            else:
                classification_matches = None

            # Step 2: Other filters (location, name, etc.)
            cached_match_with_index.index = supplier_index
            cached_match_with_index.lookup_mapping = self.supplier_lookup
            cached_match_with_index.reversed_lookup = self.reversed_supplier_lookup

            nonclass_criteria = {
                k: v for k, v in supplier_criteria.items() if k != "classifications"
            }

            nonclass_matches = cached_match_with_index(
                make_hashable(nonclass_criteria),
                tuple(
                    sorted(
                        {
                            k
                            for k in self.required_supplier_fields
                            if k != "classifications"
                        }
                    )
                ),
            )

            # Step 3: Combine
            if classification_matches is not None:
                supplier_candidates = list(
                    set(classification_matches) & set(nonclass_matches)
                )
            else:
                supplier_candidates = nonclass_matches

            # --- Consumer matching ---
            if not any(
                k
                for k in consumer_criteria
                if k not in {"matrix", "weight", "position"}
            ):
                consumer_candidates = list(self.consumer_lookup.values())
                consumer_candidates = [
                    pos for sublist in consumer_candidates for pos in sublist
                ]
            else:
                cached_match_with_index.index = consumer_index
                cached_match_with_index.lookup_mapping = self.consumer_lookup
                cached_match_with_index.reversed_lookup = (
                    self.position_to_technosphere_flows_lookup
                )
                consumer_candidates = cached_match_with_index(
                    make_hashable(consumer_criteria),
                    tuple(sorted(self.required_consumer_fields)),
                )

            # --- Combine supplier + consumer ---
            positions = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in edges
            ]

            positions = [pos for pos in positions if pos not in seen_positions]

            if positions:
                cf_entry = {
                    "supplier": supplier_criteria,
                    "consumer": consumer_criteria,
                    "direction": f"{supplier_criteria['matrix']}-{consumer_criteria['matrix']}",
                    "positions": positions,
                    "value": cf["value"],
                    "uncertainty": cf.get("uncertainty"),
                }
                self.cfs_mapping.append(cf_entry)
                seen_positions.extend(positions)

        self._update_unprocessed_edges()

    def map_aggregate_locations(self) -> None:
        """
        Map unmatched exchanges using CFs from broader (aggregated) regions.

        This method resolves cases where a direct match was not found by using CFs
        defined at a higher aggregation level (e.g., region = "RER" instead of "FR").

        It computes weighted averages for aggregate CFs using a user-specified
        weighting variable (e.g., population, GDP, resource use) from the method metadata.

        Typical use case: national-level exchanges matched to region-level CFs
        when no country-specific CF is available.

        Notes
        -----
        - Weight values are extracted from the `weight` field in each CF.
        - Uses a two-pass matching strategy: fast signature-based prefiltering, then fallback.

        Preconditions
        -------------
        - `lci()` must be called
        - `map_exchanges()` must be called
        - Weight metadata must be available for aggregation

        Updates
        -------
        - Extends `cfs_mapping` with newly matched aggregate CFs.
        - Updates internal lists of `processed_*` and `unprocessed_*` edges.
        """

        self._initialize_weights()
        print("Handling static regions...")

        if not hasattr(self, "_unmatched_hashes"):
            self._unmatched_hashes = set()

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }
        candidate_supplier_keys = self._cached_supplier_keys

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            processed_flows = set(processed_flows)
            edges_index = defaultdict(list)

            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_loc = dict(self.reversed_consumer_lookup[consumer_idx]).get(
                    "location"
                )
                if not consumer_loc:
                    fallback = self.position_to_technosphere_flows_lookup.get(
                        consumer_idx, {}
                    )
                    consumer_loc = fallback.get("location")
                if not consumer_loc:
                    continue

                edges_index[consumer_loc].append((supplier_idx, consumer_idx))

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for location, edges in edges_index.items():
                if location in ["RoW", "RoE"]:
                    continue

                # 🔁 Use the shared utility function to get subregions
                candidate_locations = resolve_candidate_consumers(
                    geo=self.geo,
                    location=location,
                    weights=self.weights,
                    containing=True,
                )

                if not candidate_locations:
                    continue

                for supplier_idx, consumer_idx in edges:
                    if (supplier_idx, consumer_idx) in processed_flows:
                        continue

                    supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                    consumer_info = self._get_consumer_info(consumer_idx)

                    if "location" not in consumer_info:
                        fallback = self.position_to_technosphere_flows_lookup.get(
                            consumer_idx, {}
                        )
                        if fallback and "location" in fallback:
                            consumer_info["location"] = fallback["location"]

                    sig = self._equality_supplier_signature(supplier_info)

                    if sig in candidate_supplier_keys:
                        prefiltered_groups[sig].append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                candidate_locations,
                            )
                        )
                    else:
                        if any(op in cf_operators for op in ["contains", "startswith"]):
                            remaining_edges.append(
                                (
                                    supplier_idx,
                                    consumer_idx,
                                    supplier_info,
                                    consumer_info,
                                    candidate_locations,
                                )
                            )
                        else:
                            self.logger.info(
                                f"No match or fallback for edge ({supplier_idx}, {consumer_idx})"
                            )

            # Pass 1
            for sig, group_edges in tqdm(
                prefiltered_groups.items(), desc="Processing static groups (pass 1)"
            ):
                supplier_info = group_edges[0][2]
                consumer_info = group_edges[0][3]
                candidate_locations = group_edges[0][-1]

                candidate_suppliers = (
                    candidate_locations
                    if (
                        "location" in self.required_supplier_fields
                        and self.geo.resolve(
                            supplier_info.get("location"), containing=True
                        )
                    )
                    else []
                )
                candidate_consumers = (
                    candidate_locations
                    if (
                        "location" in self.required_consumer_fields
                        and self.geo.resolve(
                            consumer_info.get("location"), containing=True
                        )
                    )
                    else []
                )

                new_cf, matched_cf_obj = compute_average_cf(
                    candidate_suppliers=candidate_suppliers,
                    candidate_consumers=candidate_consumers,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    weight=self.weights,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    cf_index=self.cf_index,
                    logger=self.logger,
                )

                if new_cf != 0:
                    for (
                        supplier_idx,
                        consumer_idx,
                        supplier_info,
                        consumer_info,
                        _,
                    ) in group_edges:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

            # Pass 2
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
                logger=self.logger,
            )

            grouped_edges = group_edges_by_signature(
                edge_list=[
                    (supplier_idx, consumer_idx, supplier_info, consumer_info)
                    for (supplier_idx, consumer_idx, supplier_info, consumer_info, _) in remaining_edges
                ],
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                geo=self.geo,
                weights=self.weights,
            )

            for (
                s_key,
                c_key,
                (candidate_suppliers, candidate_consumers),
            ), edge_group in tqdm(
                grouped_edges.items(), desc="Processing static groups (pass 2)"
            ):
                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key, c_key, candidate_suppliers, candidate_consumers
                )
                if new_cf != 0:
                    for supplier_idx, consumer_idx in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

        self._update_unprocessed_edges()

    def map_dynamic_locations(self) -> None:
        """
        Handle location-matching for dynamic or relative regions such as 'RoW' or 'RoE'.

        This method computes CFs for exchanges whose consumer location is a dynamic placeholder
        like "Rest of World" (RoW) by averaging over all regions **not** explicitly covered
        by the inventory.

        It uses the known supplier-consumer relationships in the inventory to identify
        excluded subregions, and builds CFs from the remaining regions using a weighted average.

        Typical use case: inventory exchanges with generic locations that need fallback handling
        (e.g., average CF for "RoW" that excludes countries already modeled explicitly).

        Notes
        -----
        - Technosphere exchange structure is analyzed to determine uncovered locations.
        - CFs are matched using exchange signatures and spatial exclusions.
        - Weighted averages are computed from the remaining eligible subregions.

        Preconditions
        -------------
        - `lci()` and `map_exchanges()` must be called
        - `weights` must be defined (e.g., population, GDP, etc.)
        - Suitable for methods with CFs that include relative or global coverage

        Updates
        -------
        - Adds dynamic-region CFs to `cfs_mapping`
        - Updates internal lists of processed and unprocessed exchanges
        """

        self._initialize_weights()
        print("Handling dynamic regions...")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }
        candidate_supplier_keys = self._cached_supplier_keys

        for flow in self.technosphere_flows:
            key = (flow["name"], flow.get("reference product"))
            self.technosphere_flows_lookup[key].append(flow["location"])

        raw_exclusion_locs = {
            loc
            for locs in self.technosphere_flows_lookup.values()
            for loc in locs
            if loc not in ["RoW", "RoE"]
        }
        decomposed_exclusions = self.geo.batch(
            locations=list(raw_exclusion_locs), containing=True
        )
        #for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
        for direction in ["technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            processed_flows = set(processed_flows)
            prefiltered_groups = defaultdict(list)
            remaining_edges = []
            print(f"Processing {len(unprocessed_edges)} unprocessed edges in {direction} direction...")
            counter, incounter = 0, 0
            # Iterate over unprocessed edges
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_info = self._get_consumer_info(consumer_idx)
                location = consumer_info.get("location")
                if location not in ["RoW", "RoE"]:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                sig = self._equality_supplier_signature(supplier_info)

                cons_act = self.position_to_technosphere_flows_lookup.get(
                    consumer_idx, {}
                )
                name, reference_product = cons_act.get("name"), cons_act.get(
                    "reference product"
                )
                exclusions = self.technosphere_flows_lookup.get(
                    (name, reference_product), []
                )

                excluded_subregions = []
                for loc in exclusions:
                    excluded_subregions.extend(decomposed_exclusions.get(loc, [loc]))

                candidate_consumers = resolve_candidate_consumers(
                    geo=self.geo,
                    location="GLO",
                    weights=self.weights,
                    containing=True,
                    exceptions=excluded_subregions,
                )
                
                if sig in candidate_supplier_keys:
                    prefiltered_groups[sig].append(
                        (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            candidate_consumers,
                        )
                    )
                else:
                    counter += 1
                    if any(op in cf_operators for op in ["contains", "startswith"]):
                        incounter += 1
                        remaining_edges.append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                [],  # suppliers
                                candidate_consumers,
                            )
                        )
            print(f"Counter: {counter}, InCounter: {incounter}")
            print(f"Remaining edges after prefiltering: {len(remaining_edges)}")
            # Pass 1
            for sig, group_edges in tqdm(
                prefiltered_groups.items(), desc="Processing dynamic groups (pass 1)"
            ):
                rep_supplier = group_edges[0][2]
                rep_consumer = group_edges[0][3]
                candidate_consumers = group_edges[0][-1]

                new_cf, matched_cf_obj = compute_average_cf(
                    candidate_suppliers=[],
                    candidate_consumers=candidate_consumers,
                    supplier_info=rep_supplier,
                    consumer_info=rep_consumer,
                    weight=self.weights,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    cf_index=self.cf_index,
                    logger=self.logger,
                )

                if new_cf:
                    for (
                        supplier_idx,
                        consumer_idx,
                        supplier_info,
                        consumer_info,
                        _,
                    ) in group_edges:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

            # Pass 2
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
                logger=self.logger,
            )

            grouped_edges = group_edges_by_signature(
                edge_list=[
                    (supplier_idx, consumer_idx, supplier_info, consumer_info)
                    for (supplier_idx, consumer_idx, supplier_info, consumer_info, _, _) in remaining_edges
                ],
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                geo=self.geo,
                weights=self.weights,
            )

            for (s_key, c_key, (_, candidate_consumers)), edge_group in tqdm(
                grouped_edges.items(), desc="Processing dynamic groups (pass 2)"
            ):
                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key, c_key, (), candidate_consumers
                )
                if new_cf:
                    for supplier_idx, consumer_idx in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

        self._update_unprocessed_edges()

    def map_contained_locations(self) -> None:
        """
        Resolve unmatched exchanges by assigning CFs from spatially containing regions.

        This method assigns a CF to an exchange based on a broader geographic area that
        contains the exchange's region. For example, if no CF exists for "Québec", but
        a CF exists for "Canada", that CF will be used.

        It is typically used when the method file contains national-level CFs but the
        inventory includes subnational or otherwise finer-grained locations.

        Notes
        -----
        - Uses a geographic containment hierarchy to resolve matches (e.g., geo aggregation trees).
        - Only uncharacterized exchanges are considered.
        - This is conceptually the inverse of `map_aggregate_locations()`.

        Preconditions
        -------------
        - `lci()` and `map_exchanges()` must be called
        - A geo containment structure must be defined or inferred

        Updates
        -------
        - Adds fallback CFs to `cfs_mapping`
        - Updates internal tracking of processed edges
        """

        self._initialize_weights()
        print("Handling contained locations...")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }
        candidate_supplier_keys = self._cached_supplier_keys

        consumer_locations = {
            self._get_consumer_info(idx).get("location")
            for _, idx in self.unprocessed_biosphere_edges
            + self.unprocessed_technosphere_edges
            if self._get_consumer_info(idx).get("location") not in ("RoW", "RoE")
        }

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            processed_flows = set(processed_flows)
            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = self._get_consumer_info(consumer_idx)
                cons_loc = consumer_info.get("location")
                if not cons_loc or cons_loc in ("RoW", "RoE"):
                    continue

                candidate_consumers = resolve_candidate_consumers(
                    geo=self.geo,
                    location=cons_loc,
                    weights=self.weights,
                    containing=False,
                )
                if not candidate_consumers:
                    continue

                sig = self._equality_supplier_signature(supplier_info)
                if sig in candidate_supplier_keys:
                    prefiltered_groups[sig].append(
                        (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            candidate_consumers,
                        )
                    )
                else:
                    if any(op in cf_operators for op in ["contains", "startswith"]):
                        remaining_edges.append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                [],  # suppliers
                                candidate_consumers,
                            )
                        )

            # Pass 1
            for sig, group_edges in tqdm(
                prefiltered_groups.items(), desc="Processing contained groups (pass 1)"
            ):
                rep_supplier = group_edges[0][2]
                rep_consumer = group_edges[0][3]
                candidate_consumers = group_edges[0][-1]

                new_cf, matched_cf_obj = compute_average_cf(
                    candidate_suppliers=[],
                    candidate_consumers=candidate_consumers,
                    supplier_info=rep_supplier,
                    consumer_info=rep_consumer,
                    weight=self.weights,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    cf_index=self.cf_index,
                    logger=self.logger,
                )

                if new_cf:
                    for (
                        supplier_idx,
                        consumer_idx,
                        supplier_info,
                        consumer_info,
                        _,
                    ) in group_edges:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

            # Pass 2
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
                logger=self.logger,
            )

            grouped_edges = group_edges_by_signature(
                edge_list=remaining_edges,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                geo=self.geo,
                weights=self.weights,
            )

            for (s_key, c_key, (_, candidate_consumers)), edge_group in tqdm(
                grouped_edges.items(), desc="Processing contained groups (pass 2)"
            ):
                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key, c_key, (), candidate_consumers
                )
                if new_cf:
                    for supplier_idx, consumer_idx in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

        self._update_unprocessed_edges()

    def map_remaining_locations_to_global(self) -> None:
        """
        Assign global fallback CFs to exchanges that remain unmatched after all regional mapping steps.

        This method ensures that all eligible exchanges are characterized by assigning a CF
        from the global region ("GLO") when no direct, aggregate, dynamic, or containing region match
        has been found.

        It is the last step in the regional mapping cascade.

        Notes
        -----
        - Uses a weighted global average if multiple CFs exist for the same exchange type.
        - If no global CF exists for a given exchange, it remains uncharacterized.
        - This step guarantees that the system-wide score is computable unless coverage is zero.

        Preconditions
        -------------
        - `lci()` and `map_exchanges()` must be called
        - Should follow other mapping steps: `map_aggregate_locations()`, `map_dynamic_locations()`, etc.

        Updates
        -------
        - Adds fallback CFs to `cfs_mapping`
        - Marks remaining exchanges as processed
        """

        self._initialize_weights()
        print("Handling remaining exchanges...")

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }
        candidate_supplier_keys = self._cached_supplier_keys

        # Resolve candidate locations for GLO once using utility
        global_locations = resolve_candidate_consumers(
            geo=self.geo,
            location="GLO",
            weights=self.weights,
            containing=True,
        )

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            processed_flows = set(processed_flows)
            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = self._get_consumer_info(consumer_idx)
                sig = self._equality_supplier_signature(supplier_info)

                if sig in candidate_supplier_keys:
                    prefiltered_groups[sig].append(
                        (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                            global_locations,
                            global_locations,
                        )
                    )
                else:
                    if any(op in cf_operators for op in ["contains", "startswith"]):
                        remaining_edges.append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                global_locations,
                                global_locations,
                            )
                        )

            # Pass 1
            for sig, group_edges in tqdm(
                prefiltered_groups.items(), desc="Processing global groups (pass 1)"
            ):
                rep_supplier = group_edges[0][2]
                rep_consumer = group_edges[0][3]

                new_cf, matched_cf_obj = compute_average_cf(
                    candidate_suppliers=global_locations,
                    candidate_consumers=global_locations,
                    supplier_info=rep_supplier,
                    consumer_info=rep_consumer,
                    weight=self.weights,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    cf_index=self.cf_index,
                    logger=self.logger,
                )

                if new_cf:
                    for (
                        supplier_idx,
                        consumer_idx,
                        supplier_info,
                        consumer_info,
                        _,
                        _,
                    ) in group_edges:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

            # Pass 2
            compute_cf_memoized = compute_cf_memoized_factory(
                cf_index=self.cf_index,
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                weights=self.weights,
                logger=self.logger,
            )

            grouped_edges = group_edges_by_signature(
                edge_list=[
                    (supplier_idx, consumer_idx, supplier_info, consumer_info)
                    for (supplier_idx, consumer_idx, supplier_info, consumer_info, _,_) in remaining_edges
                ],
                required_supplier_fields=self.required_supplier_fields,
                required_consumer_fields=self.required_consumer_fields,
                geo=self.geo,
                weights=self.weights,
            )

            for (
                s_key,
                c_key,
                (candidate_suppliers, candidate_consumers),
            ), edge_group in tqdm(
                grouped_edges.items(), desc="Processing global groups (pass 2)"
            ):
                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key, c_key, candidate_suppliers, candidate_consumers
                )
                if new_cf:
                    for supplier_idx, consumer_idx in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=(
                                matched_cf_obj.get("uncertainty")
                                if matched_cf_obj
                                else None
                            ),
                        )

        self._update_unprocessed_edges()

    def evaluate_cfs(self, scenario_idx: str | int = 0, scenario=None):
        """
        Evaluate the characterization factors (CFs) based on expressions, parameters, and uncertainty.

        This step computes the numeric CF values that will populate the characterization matrix.

        Depending on the method and configuration, it supports:
        - Symbolic CFs (e.g., "28 * (1 + 0.01 * (co2ppm - 410))")
        - Scenario-based parameter substitution
        - Uncertainty propagation via Monte Carlo simulation

        Parameters
        ----------
        scenario_idx : str or int, optional
            The scenario index (or year) for time/parameter-dependent evaluation. Defaults to 0.
        scenario : str, optional
            Name of the scenario to evaluate (overrides the default one set in `__init__`).

        Behavior
        --------
        - If `use_distributions=True` and `iterations > 1`, a 3D sparse matrix is created
          (i, j, k) where k indexes Monte Carlo iterations.
        - If symbolic expressions are present, they are resolved using the parameter set
          for the selected scenario and year.
        - If deterministic, builds a 2D matrix with direct values.

        Notes
        -----
        - Must be called before `lcia()` to populate the CF matrix.
        - Parameters are pulled from the method file or passed manually via `parameters`.


        Raises
        ------
        ValueError
            If the requested scenario is not found in the parameter dictionary.


        Updates
        -------
        - Sets `characterization_matrix`
        - Populates `scenario_cfs` with resolved CFs
        """

        if self.use_distributions and self.iterations > 1:
            coords_i, coords_j, coords_k = [], [], []
            data = []

            for cf in self.cfs_mapping:
                samples = sample_cf_distribution(
                    cf=cf,
                    n=self.iterations,
                    parameters=self.parameters,
                    random_state=self.random_state,
                    use_distributions=self.use_distributions,
                    SAFE_GLOBALS=self.SAFE_GLOBALS,
                )
                for i, j in cf["positions"]:
                    for k in range(self.iterations):
                        coords_i.append(i)
                        coords_j.append(j)
                        coords_k.append(k)
                        data.append(samples[k])

            matrix_type = (
                "biosphere" if len(self.biosphere_edges) > 0 else "technosphere"
            )
            n_rows, n_cols = (
                self.lca.inventory.shape
                if matrix_type == "biosphere"
                else self.lca.technosphere_matrix.shape
            )

            self.characterization_matrix = sparse.COO(
                coords=[coords_i, coords_j, coords_k],
                data=data,
                shape=(n_rows, n_cols, self.iterations),
            )

            self.scenario_cfs = [{"positions": [], "value": 0}]  # dummy

        else:
            # Fallback to 2D
            self.scenario_cfs = []
            scenario_name = None

            if scenario is not None:
                scenario_name = scenario
            elif self.scenario is not None:
                scenario_name = self.scenario

            if scenario_name is None:
                if isinstance(self.parameters, dict):
                    if len(self.parameters) > 0:
                        scenario_name = list(self.parameters.keys())[0]

            resolved_params = self._resolve_parameters_for_scenario(
                scenario_idx, scenario_name
            )

            for cf in self.cfs_mapping:
                if isinstance(cf["value"], str):

                    value = safe_eval_cached(
                        cf["value"],
                        parameters=resolved_params,
                        scenario_idx=scenario_idx,
                        SAFE_GLOBALS=self.SAFE_GLOBALS,
                    )
                else:
                    value = cf["value"]

                self.scenario_cfs.append(
                    {
                        "supplier": cf["supplier"],
                        "consumer": cf["consumer"],
                        "positions": cf["positions"],
                        "value": value,
                    }
                )

            matrix_type = (
                "biosphere" if len(self.biosphere_edges) > 0 else "technosphere"
            )
            self.characterization_matrix = initialize_lcia_matrix(
                self.lca, matrix_type=matrix_type
            )

            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    self.characterization_matrix[i, j] = cf["value"]

            self.characterization_matrix = self.characterization_matrix.tocsr()

    def lcia(self) -> None:
        """
        Perform the life cycle impact assessment (LCIA) using the evaluated characterization matrix.

        This method multiplies the inventory matrix with the CF matrix to produce a scalar score
        or a distribution of scores (for uncertainty propagation).


        Behavior
        --------
        - In deterministic mode: computes a single scalar LCIA score.
        - In uncertainty mode (3D matrix): computes a 1D array of LCIA scores across all iterations.


        Notes
        -----
        - Must be called after `evaluate_cfs()`.
        - Requires the inventory to be computed via `lci()`.
        - Technosphere or biosphere matrix is chosen based on exchange type.


        Updates
        -------
        - Sets `score` to the final impact value(s)
        - Stores `characterized_inventory` as a matrix or tensor

        If no exchanges are matched, the score defaults to 0.
        """

        # check that teh sum of processed biosphere and technosphere
        # edges is superior to zero, otherwise, we exit
        if (
            len(self.processed_biosphere_edges) + len(self.processed_technosphere_edges)
            == 0
        ):
            self.score = 0
            return

        is_biosphere = len(self.biosphere_edges) > 0

        if self.use_distributions and self.iterations > 1:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )

            # Convert 2D inventory to sparse.COO
            inventory_coo = sparse.COO.from_scipy_sparse(inventory)

            # Broadcast inventory shape for multiplication
            inv_expanded = inventory_coo[:, :, None]  # (i, j, 1)

            # Element-wise multiply
            characterized = self.characterization_matrix * inv_expanded

            # Sum across dimensions i and j to get 1 value per iteration
            self.characterized_inventory = characterized
            self.score = characterized.sum(axis=(0, 1))

        else:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )
            self.characterized_inventory = self.characterization_matrix.multiply(
                inventory
            )
            self.score = self.characterized_inventory.sum()

    def statistics(self):
        """
        Print a summary table of method metadata and coverage statistics.

        This includes:
        - Demand activity name
        - Method name and data file
        - Unit (if available)
        - Total CFs in the method file
        - Number of CFs used (i.e., matched to exchanges)
        - Number of unique CF values applied
        - Number of characterized vs. uncharacterized exchanges
        - Ignored locations or CFs that could not be applied

        This is a useful diagnostic tool to assess method coverage and
        identify missing or unmatched data.

        Output
        ------
        - Prints a PrettyTable to the console
        - Does not return a value

        Notes
        -----
        - Can be used after `lcia()` to assess method completeness
        - Will reflect both direct and fallback-based characterizations
        """

        # build PrettyTable
        table = PrettyTable()
        table.header = False
        rows = []
        try:
            rows.append(
                [
                    "Activity",
                    fill(
                        list(self.lca.demand.keys())[0]["name"],
                        width=45,
                    ),
                ]
            )
        except TypeError:
            rows.append(
                [
                    "Activity",
                    fill(
                        bw2data.get_activity(id=list(self.lca.demand.keys())[0])[
                            "name"
                        ],
                        width=45,
                    ),
                ]
            )
        rows.append(["Method name", fill(str(self.method), width=45)])
        if "unit" in self.method_metadata:
            rows.append(["Unit", fill(self.method_metadata["unit"], width=45)])
        rows.append(["Data file", fill(self.filepath.stem, width=45)])
        rows.append(["CFs in method", self.cfs_number])
        rows.append(
            [
                "CFs used",
                len([x["value"] for x in self.cfs_mapping if len(x["positions"]) > 0]),
            ]
        )
        unique_cfs = set(
            [
                x["value"]
                for x in self.cfs_mapping
                if len(x["positions"]) > 0 and x["value"] is not None
            ]
        )
        rows.append(
            [
                "Unique CFs used",
                len(unique_cfs),
            ]
        )

        if self.ignored_method_exchanges:
            rows.append(
                ["CFs without eligible exc.", len(self.ignored_method_exchanges)]
            )

        if self.ignored_locations:
            rows.append(["Product system locations ignored", self.ignored_locations])

        if len(self.processed_biosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(self.processed_biosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_biosphere_edges),
                ]
            )

        if len(self.processed_technosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(self.processed_technosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_technosphere_edges),
                ]
            )

        for row in rows:
            table.add_row(row)

        print(table)

    def generate_cf_table(self) -> pd.DataFrame:
        """
        Generate a detailed results table of characterized exchanges.

        Returns a pandas DataFrame with one row per characterized exchange,
        including the following fields:

        - Supplier and consumer activity name, reference product, and location
        - Flow amount
        - Characterization factor(s)
        - Characterized impact (CF × amount)

        Behavior
        --------
        - If uncertainty is enabled (`use_distributions=True`), the DataFrame contains:
          - Mean, std, percentiles, min/max for CFs and impact values
        - If deterministic: contains only point values for CF and impact

        Returns
        -------
        pd.DataFrame
            A table of all characterized exchanges with metadata and scores.

        Notes
        -----
        - Must be called after `evaluate_cfs()` and `lcia()`
        - Useful for debugging, reporting, or plotting contributions
        """

        if not self.scenario_cfs:
            print("You must run evaluate_cfs() first.")
            return pd.DataFrame()

        is_biosphere = True if self.technosphere_flow_matrix is None else False

        inventory = (
            self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
        )
        data = []

        if (
            self.use_distributions
            and hasattr(self, "characterization_matrix")
            and hasattr(self, "iterations")
        ):
            cm = self.characterization_matrix

            for i, j in zip(
                *cm.sum(axis=2).nonzero()
            ):  # Only loop over nonzero entries
                consumer = bw2data.get_activity(self.reversed_activity[j])
                supplier = (
                    bw2data.get_activity(self.reversed_biosphere[i])
                    if is_biosphere
                    else bw2data.get_activity(self.reversed_activity[i])
                )

                samples = np.array(cm[i, j, :].todense()).flatten().astype(float)
                amount = inventory[i, j]
                impact_samples = amount * samples

                # Percentiles
                cf_p = np.percentile(samples, [5, 25, 50, 75, 95])
                impact_p = np.percentile(impact_samples, [5, 25, 50, 75, 95])

                entry = {
                    "supplier name": supplier["name"],
                    "consumer name": consumer["name"],
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "amount": amount,
                    "CF (mean)": samples.mean(),
                    "CF (std)": samples.std(),
                    "CF (min)": samples.min(),
                    "CF (5th)": cf_p[0],
                    "CF (25th)": cf_p[1],
                    "CF (50th)": cf_p[2],
                    "CF (75th)": cf_p[3],
                    "CF (95th)": cf_p[4],
                    "CF (max)": samples.max(),
                    "impact (mean)": impact_samples.mean(),
                    "impact (std)": impact_samples.std(),
                    "impact (min)": impact_samples.min(),
                    "impact (5th)": impact_p[0],
                    "impact (25th)": impact_p[1],
                    "impact (50th)": impact_p[2],
                    "impact (75th)": impact_p[3],
                    "impact (95th)": impact_p[4],
                    "impact (max)": impact_samples.max(),
                }

                if is_biosphere:
                    entry["supplier categories"] = supplier.get("categories")
                else:
                    entry["supplier reference product"] = supplier.get(
                        "reference product"
                    )
                    entry["supplier location"] = supplier.get("location")

                data.append(entry)

        else:
            # Deterministic fallback
            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    consumer = bw2data.get_activity(self.reversed_activity[j])
                    supplier = (
                        bw2data.get_activity(self.reversed_biosphere[i])
                        if is_biosphere
                        else bw2data.get_activity(self.reversed_activity[i])
                    )

                    amount = inventory[i, j]
                    cf_value = cf["value"]
                    impact = amount * cf_value

                    entry = {
                        "supplier name": supplier["name"],
                        "consumer name": consumer["name"],
                        "consumer reference product": consumer.get("reference product"),
                        "consumer location": consumer.get("location"),
                        "amount": amount,
                        "CF": cf_value,
                        "impact": impact,
                    }

                    if is_biosphere:
                        entry["supplier categories"] = supplier.get("categories")
                    else:
                        entry["supplier reference product"] = supplier.get(
                            "reference product"
                        )
                        entry["supplier location"] = supplier.get("location")

                    data.append(entry)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Order columns
        preferred_columns = [
            "supplier name",
            "supplier categories",
            "supplier reference product",
            "supplier location",
            "consumer name",
            "consumer reference product",
            "consumer location",
            "amount",
        ]

        # Add CF or CF summary columns
        if self.use_distributions:
            preferred_columns += [
                "CF (mean)",
                "CF (std)",
                "CF (min)",
                "CF (5th)",
                "CF (25th)",
                "CF (50th)",
                "CF (75th)",
                "CF (95th)",
                "CF (max)",
                "impact (mean)",
                "impact (std)",
                "impact (min)",
                "impact (5th)",
                "impact (25th)",
                "impact (50th)",
                "impact (75th)",
                "impact (95th)",
                "impact (max)",
            ]
        else:
            preferred_columns += ["CF", "impact"]

        df = df[[col for col in preferred_columns if col in df.columns]]

        return df

    @property
    def geo(self):
        if getattr(self, "_geo", None) is None:
            self._geo = GeoResolver(self.weights)
        return self._geo
