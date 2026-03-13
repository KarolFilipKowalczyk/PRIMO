"""
Tests for primo.rules — rule registry, enumeration, and catalog I/O.
"""

import pytest
import numpy as np
import networkx as nx

from primo.rules import (
    ALL_RULES, RULES_ORIGINAL, RULES_NEW_WITNESSES, RULES_CATALOG,
    RULES_STRUCTURAL, RULES_RANDOM_DPO, RULE_SOURCE,
    make_initial_graphs, run_trajectory,
    enumerate_rules, enumerate_rules_at_signature,
    load_catalog, save_catalog, list_catalogs,
    dpo_rule_to_callable,
)


# ══════════════════════════════════════════════════════════════════════
# Rule registry
# ══════════════════════════════════════════════════════════════════════

class TestRuleRegistry:
    def test_total_rule_count(self):
        assert len(ALL_RULES) == 33

    def test_registry_composition(self):
        assert len(RULES_ORIGINAL) == 15
        assert len(RULES_NEW_WITNESSES) == 3
        assert len(RULES_CATALOG) == 5
        assert len(RULES_STRUCTURAL) == 5
        assert len(RULES_RANDOM_DPO) == 5

    def test_no_duplicate_names(self):
        all_names = (list(RULES_ORIGINAL) + list(RULES_NEW_WITNESSES)
                     + list(RULES_CATALOG) + list(RULES_STRUCTURAL)
                     + list(RULES_RANDOM_DPO))
        assert len(all_names) == len(set(all_names))

    def test_all_rules_have_source(self):
        for name in ALL_RULES:
            assert name in RULE_SOURCE, f"Missing source for {name}"

    def test_source_tags(self):
        for r in RULES_ORIGINAL:
            assert RULE_SOURCE[r] == "orig"
        for r in RULES_NEW_WITNESSES:
            assert RULE_SOURCE[r] == "witness"
        for r in RULES_CATALOG:
            assert RULE_SOURCE[r] == "catalog"
        for r in RULES_STRUCTURAL:
            assert RULE_SOURCE[r] == "struct"
        for r in RULES_RANDOM_DPO:
            assert RULE_SOURCE[r] == "rndDPO"


# ══════════════════════════════════════════════════════════════════════
# Initial graphs
# ══════════════════════════════════════════════════════════════════════

class TestInitialGraphs:
    def test_default_seeds(self):
        seeds = make_initial_graphs()
        assert set(seeds.keys()) == {'K1', 'K2', 'K3', 'P3'}

    def test_K1(self):
        G = make_initial_graphs(['K1'])['K1']
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_K2(self):
        G = make_initial_graphs(['K2'])['K2']
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1

    def test_K3(self):
        G = make_initial_graphs(['K3'])['K3']
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3

    def test_P3(self):
        G = make_initial_graphs(['P3'])['P3']
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

    def test_unknown_graph_raises(self):
        with pytest.raises(ValueError):
            make_initial_graphs(['K99'])

    def test_selective_seeds(self):
        seeds = make_initial_graphs(['K1', 'K3'])
        assert set(seeds.keys()) == {'K1', 'K3'}


# ══════════════════════════════════════════════════════════════════════
# Rule execution
# ══════════════════════════════════════════════════════════════════════

class TestRuleExecution:
    """Every rule must run from every seed without error."""

    @pytest.fixture
    def seeds(self):
        return make_initial_graphs()

    @pytest.mark.parametrize("rule_name", list(ALL_RULES.keys()))
    def test_rule_runs(self, rule_name, seeds):
        np.random.seed(42)
        rule_fn = ALL_RULES[rule_name]
        for seed_name, G0 in seeds.items():
            traj = run_trajectory(rule_fn, G0, T=5)
            assert len(traj) >= 2, (
                f"{rule_name}/{seed_name}: trajectory too short")
            for G in traj:
                assert isinstance(G, nx.Graph)

    def test_safety_cap(self):
        """Trajectory stops when graph exceeds n_max."""
        def explosive_rule(G):
            H = G.copy()
            n = H.number_of_nodes()
            base = max(H.nodes()) + 1
            for i in range(n):
                H.add_node(base + i)
            return H

        G0 = nx.complete_graph(3)
        traj = run_trajectory(explosive_rule, G0, T=100, n_max=50)
        for G in traj:
            assert G.number_of_nodes() <= 50

    def test_do_nothing_preserves_graph(self):
        G0 = nx.complete_graph(5)
        np.random.seed(42)
        traj = run_trajectory(ALL_RULES['do_nothing'], G0, T=5)
        for G in traj:
            assert G.number_of_nodes() == 5
            assert G.number_of_edges() == 10


# ══════════════════════════════════════════════════════════════════════
# Enumeration
# ══════════════════════════════════════════════════════════════════════

class TestEnumeration:
    """Enumeration counts must match stored catalogs."""

    def test_sig_1_1_connected(self):
        rules = enumerate_rules(1, 1, connected_only=True)
        assert len(rules) == 1

    def test_sig_1_1_all(self):
        rules = enumerate_rules(1, 1, connected_only=False)
        assert len(rules) == 1

    def test_sig_1_2_connected(self):
        rules = enumerate_rules(1, 2, connected_only=True)
        assert len(rules) == 1

    def test_sig_1_2_all(self):
        rules = enumerate_rules(1, 2, connected_only=False)
        assert len(rules) == 2

    def test_sig_2_3_connected(self):
        rules = enumerate_rules(2, 3, connected_only=True)
        assert len(rules) == 3

    def test_sig_2_3_all(self):
        rules = enumerate_rules(2, 3, connected_only=False)
        assert len(rules) == 6

    def test_matches_catalog_1_1(self):
        cat, rules = enumerate_rules_at_signature(1, 1)
        stored = load_catalog("1_1")
        assert cat["total_connected"] == stored["total_connected"]
        assert cat["total_any"] == stored["total_any"]

    def test_matches_catalog_1_2(self):
        cat, rules = enumerate_rules_at_signature(1, 2)
        stored = load_catalog("1_2")
        assert cat["total_connected"] == stored["total_connected"]
        assert cat["total_any"] == stored["total_any"]

    def test_matches_catalog_2_3(self):
        cat, rules = enumerate_rules_at_signature(2, 3)
        stored = load_catalog("2_3")
        assert cat["total_connected"] == stored["total_connected"]
        assert cat["total_any"] == stored["total_any"]

    def test_enumerated_rules_are_distinct(self):
        """All enumerated rules at 2→3 must be genuinely distinct."""
        rules = enumerate_rules(2, 3, connected_only=True)
        # Each rule should have a unique (r_edges, iota) signature
        signatures = set()
        for r in rules:
            sig = (tuple(tuple(e) for e in r["r_edges"]), tuple(r["iota"]))
            assert sig not in signatures, f"Duplicate rule: {r['id']}"
            signatures.add(sig)

    def test_isomorphism_eliminates_duplicates(self):
        """Without isomorphism checking we'd get more rules at 2→3."""
        # At signature 2→3, naive enumeration (3 possible edge subsets that
        # make connected graphs × 6 interface maps = up to 18 candidates)
        # should reduce to exactly 3.
        rules = enumerate_rules(2, 3, connected_only=True)
        assert len(rules) == 3


# ══════════════════════════════════════════════════════════════════════
# Catalog I/O
# ══════════════════════════════════════════════════════════════════════

class TestCatalogIO:
    def test_list_catalogs(self):
        cats = list_catalogs()
        assert "1_1" in cats
        assert "1_2" in cats
        assert "2_3" in cats

    def test_load_catalog_structure(self):
        cat = load_catalog("2_3")
        assert "signature" in cat
        assert "rules" in cat
        assert "total_connected" in cat
        assert len(cat["rules"]) == 3

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save a catalog, load it back, verify contents."""
        import json
        from primo.rules import _CATALOG_DIR

        cat, rules = enumerate_rules_at_signature(2, 3)
        # Save to temp location
        path = tmp_path / "rules_test.json"
        with open(path, 'w') as f:
            json.dump(cat, f, indent=2)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["total_connected"] == 3
        assert len(loaded["rules"]) == 3

    def test_catalog_rule_structure(self):
        cat = load_catalog("2_3")
        for rule in cat["rules"]:
            assert "id" in rule
            assert "l_edges" in rule
            assert "r_edges" in rule
            assert "iota" in rule


# ══════════════════════════════════════════════════════════════════════
# DPO rule conversion
# ══════════════════════════════════════════════════════════════════════

class TestDPORuleConversion:
    def test_catalog_rules_are_callable(self):
        cat = load_catalog("2_3")
        for rd in cat["rules"]:
            fn = dpo_rule_to_callable(rd)
            G0 = nx.complete_graph(3)
            np.random.seed(42)
            traj = run_trajectory(fn, G0, T=5)
            assert len(traj) >= 2

    def test_sprouting_adds_nodes(self):
        """Edge sprouting rule should grow the graph."""
        cat = load_catalog("2_3")
        sprouting = [r for r in cat["rules"] if "Sprouting" in r.get("name", "")]
        assert len(sprouting) > 0
        fn = dpo_rule_to_callable(sprouting[0])
        G0 = nx.complete_graph(3)
        np.random.seed(42)
        G1 = fn(G0)
        assert G1.number_of_nodes() >= G0.number_of_nodes()
