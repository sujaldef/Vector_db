"""Structural diagnostic for the handwritten HNSW implementation.

This script intentionally does not modify the notebook implementation. It loads
the current definitions from notebooks/05_hnsw_final.ipynb, builds only the
first 1,000 vectors, and reports graph invariants and connectivity problems.
It does not benchmark search, calculate recall, tune parameters, or build a
larger index.
"""

from __future__ import annotations

import ast
import heapq
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_FILE = PROJECT_DIR / "notebooks" / "05_hnsw_final.ipynb"
EMBEDDINGS_FILE = PROJECT_DIR / "data" / "embeddings" / "embeddings.npy"
IDS_FILE = PROJECT_DIR / "data" / "embeddings" / "ids.npy"
DATASET_SIZE = 1_000
M = 8
EF_CONSTRUCTION = 50
EF_SEARCH = 20
SEED = 42
MAX_EXAMPLES = 10
MAX_COMPONENTS = 10


def section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def examples(label: str, values: list, limit: int = MAX_EXAMPLES) -> None:
    if values:
        print(f"{label}: {len(values)}; examples: {values[:limit]}")
    else:
        print(f"{label}: 0")


def notebook_code_cells(path: Path) -> list[str]:
    """Read either the current VS Code notebook format or standard JSON."""
    text = path.read_text(encoding="utf-8-sig")

    if text.lstrip().startswith("<VSCode.Cell"):
        return re.findall(
            r'<VSCode\.Cell[^>]*language="python"[^>]*>\n?(.*?)</VSCode\.Cell>',
            text,
            flags=re.DOTALL,
        )

    notebook = json.loads(text)
    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            cells.append("".join(source) if isinstance(source, list) else source)
    return cells


def load_implementation(path: Path) -> dict:
    """Load only the actual implementation definitions from the notebook."""
    namespace = {
        "__name__": "debug_hnsw_notebook",
        "__file__": str(path),
        "heapq": heapq,
        "np": np,
        "time": time,
    }
    imported = False
    found_class = False
    found_builder = False

    for source in notebook_code_cells(path):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        has_cosine = any(
            isinstance(node, ast.FunctionDef) and node.name == "cosine_similarity"
            for node in tree.body
        )
        has_class = any(
            isinstance(node, ast.ClassDef) and node.name == "HNSWIndex"
            for node in tree.body
        )
        has_builder = any(
            isinstance(node, ast.FunctionDef) and node.name == "build_hnsw"
            for node in tree.body
        )

        if has_cosine:
            exec(compile(tree, str(path), "exec"), namespace)
            imported = True
        elif has_class:
            exec(compile(tree, str(path), "exec"), namespace)
            found_class = True
        elif has_builder:
            exec(compile(tree, str(path), "exec"), namespace)
            found_builder = True

    missing = []
    if not imported:
        missing.append("cosine_similarity")
    if not found_class:
        missing.append("HNSWIndex")
    if not found_builder:
        missing.append("build_hnsw")
    if missing:
        raise RuntimeError(
            f"Could not locate required definitions in {path}: {', '.join(missing)}"
        )

    return namespace


def active_nodes(index) -> set[int]:
    deleted = set(getattr(index, "deleted", set()))
    return set(range(len(index.vectors))) - deleted


def node_levels(index) -> dict[int, int]:
    levels = {}
    for layer, layer_map in enumerate(index.graph):
        for node in layer_map:
            levels[node] = max(levels.get(node, -1), layer)
    return levels


def valid_neighbor_ids(index, layer: int, node: int) -> list:
    total = len(index.vectors)
    return [
        neighbor
        for neighbor in index.graph[layer].get(node, [])
        if not isinstance(neighbor, int) or neighbor < 0 or neighbor >= total
    ]


def reachable(index, layer: int) -> set[int]:
    if not index.graph or layer >= len(index.graph):
        return set()
    entry = index.entry_point
    deleted = set(getattr(index, "deleted", set()))
    if entry is None or entry in deleted or entry not in index.graph[layer]:
        return set()

    visited = set()
    stack = [entry]
    while stack:
        node = stack.pop()
        if node in visited or node in deleted:
            continue
        visited.add(node)
        for neighbor in index.graph[layer].get(node, []):
            if neighbor not in visited and neighbor not in deleted:
                stack.append(neighbor)
    return visited


def components(index, layer: int) -> list[list[int]]:
    if not index.graph or layer >= len(index.graph):
        return []
    graph = index.graph[layer]
    remaining = set(graph)
    result = []
    while remaining:
        root = min(remaining)
        component = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            component.append(node)
            for neighbor in graph.get(node, []):
                if neighbor in remaining:
                    stack.append(neighbor)
            for other, neighbors in graph.items():
                if node in neighbors and other in remaining:
                    stack.append(other)
        result.append(sorted(component))
    return sorted(result, key=len, reverse=True)


def run_diagnostic(index, cosine_similarity) -> None:
    total_nodes = len(index.vectors)
    total_dimensions = getattr(index.vectors[0], "shape", ["unknown"])[0] if total_nodes else 0
    deleted = set(getattr(index, "deleted", set()))
    active = active_nodes(index)
    levels = node_levels(index)

    print("=" * 50)
    print("HNSW DIAGNOSTIC REPORT")
    print("=" * 50)

    section("DATASET")
    print(f"embedding_file: {EMBEDDINGS_FILE}")
    print(f"id_file: {IDS_FILE}")
    print(f"total_nodes: {total_nodes}")
    print(f"active_nodes: {len(active)}")
    print(f"embedding_dimension: {total_dimensions}")

    section("CONFIGURATION")
    print(f"M: {index.M}")
    print(f"ef_construction: {index.ef_construction}")
    print(f"ef_search: {index.ef_search}")
    print(f"entry_point: {index.entry_point}")
    print(f"max_level: {index.max_level}")

    section("GRAPH STRUCTURE")
    print(f"number_of_layers: {len(index.graph)}")
    layer_counts = []
    for layer, layer_map in enumerate(index.graph):
        layer_counts.append(len(layer_map))
        print(f"layer {layer}: nodes={len(layer_map)}")
    print(f"minimum_node_level: {min(levels.values()) if levels else 'unknown'}")
    print(f"maximum_node_level: {max(levels.values()) if levels else 'unknown'}")
    average_level = sum(levels.values()) / len(levels) if levels else 0.0
    print(f"average_node_level: {average_level:.3f}")

    section("NEIGHBOR STATISTICS")
    zero_neighbors = []
    over_m = []
    neighbor_counts_by_layer = {}
    for layer, layer_map in enumerate(index.graph):
        counts = []
        for node, neighbors in layer_map.items():
            if node in deleted:
                continue
            count = len(neighbors)
            counts.append(count)
            if count == 0:
                zero_neighbors.append((layer, node))
            if count > index.M:
                over_m.append((layer, node, count, index.M))
        neighbor_counts_by_layer[layer] = counts
        if counts:
            print(
                f"layer {layer}: min={min(counts)}, max={max(counts)}, "
                f"average={sum(counts) / len(counts):.3f}"
            )
        else:
            print(f"layer {layer}: no active nodes")
    examples("nodes_with_zero_neighbors", zero_neighbors)
    examples("nodes_exceeding_M", over_m)

    section("INVALID REFERENCES")
    invalid_refs = []
    for layer, layer_map in enumerate(index.graph):
        for node, neighbors in layer_map.items():
            for neighbor in neighbors:
                if not isinstance(neighbor, int) or neighbor < 0 or neighbor >= total_nodes:
                    invalid_refs.append((layer, node, neighbor))
    examples("invalid_neighbor_references", invalid_refs)

    section("SELF LOOPS")
    self_loops = []
    for layer, layer_map in enumerate(index.graph):
        for node, neighbors in layer_map.items():
            if node in neighbors:
                self_loops.append((layer, node))
    examples("self_loops", self_loops)

    section("DUPLICATES")
    duplicate_neighbors = []
    for layer, layer_map in enumerate(index.graph):
        for node, neighbors in layer_map.items():
            unique = set(neighbors)
            if len(unique) != len(neighbors):
                duplicate_neighbors.append((layer, node, neighbors))
    examples("duplicate_neighbor_lists", duplicate_neighbors)

    section("REVERSE EDGE CHECK")
    total_edges = 0
    reverse_mismatches = []
    for layer, layer_map in enumerate(index.graph):
        for node, neighbors in layer_map.items():
            for neighbor in neighbors:
                if not isinstance(neighbor, int) or neighbor < 0 or neighbor >= total_nodes:
                    continue
                total_edges += 1
                if node not in index.graph[layer].get(neighbor, []):
                    reverse_mismatches.append((layer, node, neighbor))
    mismatch_percent = (100 * len(reverse_mismatches) / total_edges) if total_edges else 0.0
    print(f"total_edges_checked: {total_edges}")
    print(f"reverse_edge_mismatches: {len(reverse_mismatches)}")
    print(f"reverse_edge_mismatch_percent: {mismatch_percent:.3f}%")
    examples("reverse_edge_mismatch_examples", reverse_mismatches)

    section("REACHABILITY")
    reachability_by_layer = {}
    for layer, layer_map in enumerate(index.graph):
        reached = reachable(index, layer)
        layer_nodes = set(layer_map) - deleted
        unreachable = sorted(layer_nodes - reached)
        percent = 100 * len(unreachable) / len(layer_nodes) if layer_nodes else 0.0
        reachability_by_layer[layer] = (reached, unreachable)
        print(
            f"layer {layer}: reachable={len(reached)}, unreachable={len(unreachable)}, "
            f"percent_unreachable={percent:.3f}%"
        )
        examples(f"layer_{layer}_unreachable_examples", unreachable, 20)

    section("ENTRY POINT")
    entry = index.entry_point
    entry_exists = entry is not None
    entry_valid = isinstance(entry, int) and 0 <= entry < total_nodes
    entry_level = levels.get(entry) if entry_valid else None
    print(f"exists: {entry_exists}")
    print(f"valid_node_index: {entry_valid}")
    print(f"level: {entry_level}")
    print(f"expected_highest_layer: {index.max_level}")
    print(f"present_in_highest_layer: {entry_valid and index.max_level < len(index.graph) and entry in index.graph[index.max_level]}")
    entry_problems = []
    if not entry_exists:
        entry_problems.append("entry point is missing")
    elif not entry_valid:
        entry_problems.append(f"entry point {entry} is outside node range")
    elif entry_level != index.max_level:
        entry_problems.append(f"entry point level {entry_level} != max_level {index.max_level}")
    examples("entry_point_problems", entry_problems)

    section("LAYER CONSISTENCY")
    layer_problems = []
    expected_max_level = len(index.graph) - 1
    if index.max_level != expected_max_level:
        layer_problems.append(("max_level", index.max_level, "expected", expected_max_level))
    for layer, layer_map in enumerate(index.graph):
        for node in layer_map:
            if node not in active:
                layer_problems.append(("inactive_or_missing_node", layer, node))
            elif levels.get(node) < layer:
                layer_problems.append(("node_level_below_layer", layer, node, levels.get(node)))
    present_nodes = set(levels)
    missing_from_graph = sorted(active - present_nodes)
    for node in missing_from_graph:
        layer_problems.append(("active_node_missing_from_graph", node))
    examples("layer_consistency_problems", layer_problems)

    section("COMPONENT ANALYSIS")
    layer0_components = components(index, 0)
    print(f"layer_0_component_count: {len(layer0_components)}")
    for rank, component in enumerate(layer0_components[:MAX_COMPONENTS], 1):
        print(f"component {rank}: size={len(component)}, examples={component[:MAX_EXAMPLES]}")
    if layer0_components:
        print(f"largest_component_size: {len(layer0_components[0])}")
        print(f"small_components_under_3: {sum(len(c) < 3 for c in layer0_components)}")
    else:
        print("largest_component_size: 0")

    section("NEIGHBOR QUALITY")
    quality_examples = []
    for layer, layer_map in enumerate(index.graph):
        for node, neighbors in list(layer_map.items())[:5]:
            if not neighbors or node in deleted:
                continue
            scores = []
            for neighbor in neighbors[:5]:
                if isinstance(neighbor, int) and 0 <= neighbor < total_nodes:
                    scores.append((neighbor, cosine_similarity(index.vectors[node], index.vectors[neighbor])))
            quality_examples.append((layer, node, scores))
    for item in quality_examples:
        print(f"layer={item[0]}, node={item[1]}, neighbor_scores={item[2]}")

    section("INSERTION COUNTS")
    id_values = [int(value) for value in index.ids]
    unique_ids = set(id_values)
    duplicate_ids = sorted({value for value in id_values if id_values.count(value) > 1})
    print(f"expected_node_count: {DATASET_SIZE}")
    print(f"actual_node_count: {total_nodes}")
    print(f"unique_id_count: {len(unique_ids)}")
    examples("duplicate_ids", duplicate_ids)

    section("VALIDATION RESULT")
    validate_method = getattr(index, "validate_graph", None)
    if callable(validate_method):
        try:
            print("validate_graph() complete result:")
            print(repr(validate_method()))
        except Exception as exc:
            print(f"validate_graph() raised {type(exc).__name__}: {exc}")
    else:
        print("validate_graph(): not exposed on HNSWIndex")

    diagnose_method = getattr(index, "diagnose_hnsw_graph", None)
    if callable(diagnose_method):
        try:
            print("diagnose_hnsw_graph() complete result:")
            print(repr(diagnose_method()))
        except Exception as exc:
            print(f"diagnose_hnsw_graph() raised {type(exc).__name__}: {exc}")
    else:
        print("diagnose_hnsw_graph(): not exposed on HNSWIndex")

    section("LIKELY PROBLEMS")
    ranked = []
    if reverse_mismatches:
        ranked.append(("Reverse-edge pruning", len(reverse_mismatches), "Reverse edges are missing after insertion, so graph navigation can become one-way and disconnected. Inspect _connect() and neighbor pruning."))
    if any(unreachable for _, unreachable in reachability_by_layer.values()):
        ranked.append(("Layer reachability", sum(len(u) for _, u in reachability_by_layer.values()), "Nodes cannot be reached from the entry point, so search may never visit relevant regions. Inspect insert() layer entry propagation and graph connections."))
    if len(layer0_components) > 1:
        ranked.append(("Disconnected layer-0 components", len(layer0_components), "Layer 0 is split into multiple components, which prevents a single entry point from reaching all nodes. Inspect insertion connectivity and pruning."))
    if invalid_refs:
        ranked.append(("Invalid neighbor references", len(invalid_refs), "Edges point outside the vector array, which can corrupt traversal. Inspect edge creation and node-index handling."))
    if duplicate_neighbors:
        ranked.append(("Duplicate neighbors", len(duplicate_neighbors), "Duplicate adjacency entries waste search budget and distort degree statistics. Inspect reverse insertion and pruning."))
    if over_m:
        ranked.append(("M capacity violations", len(over_m), "Some adjacency lists exceed M, so graph degree guarantees are not being enforced. Inspect neighbor pruning."))
    if layer_problems:
        ranked.append(("Layer metadata inconsistency", len(layer_problems), "Nodes and layer metadata disagree, which can make upper-layer traversal invalid. Inspect random_level(), max_level, and layer allocation."))
    if zero_neighbors:
        ranked.append(("Isolated nodes", len(zero_neighbors), "Nodes have no outgoing links and cannot contribute to traversal. Inspect insertion candidate selection and connection creation."))
    if duplicate_ids:
        ranked.append(("Duplicate document IDs", len(duplicate_ids), "Duplicate IDs make ID-based result interpretation ambiguous. Inspect input alignment and insertion ID handling."))

    if not ranked:
        print("No structural failure was detected by these checks; inspect the validator's own policy or search behavior next.")
    else:
        for rank, (problem, count, explanation) in enumerate(ranked, 1):
            print(f"{rank}. {problem} ({count})")
            print(f"   {explanation}")


def main() -> int:
    try:
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: NumPy is required: {exc}", file=sys.stderr)
        return 1

    for path in (NOTEBOOK_FILE, EMBEDDINGS_FILE, IDS_FILE):
        if not path.exists():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            return 1

    try:
        namespace = load_implementation(NOTEBOOK_FILE)
        embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")
        ids = np.load(IDS_FILE, mmap_mode="r")
        if len(embeddings) < DATASET_SIZE or len(ids) < DATASET_SIZE:
            raise RuntimeError("embedding or ID file contains fewer than 1,000 rows")
        if embeddings.ndim != 2 or embeddings.shape[1] != 384:
            raise RuntimeError(f"expected embeddings with dimension 384, got {embeddings.shape}")

        build_hnsw = namespace["build_hnsw"]
        start = time.perf_counter()
        index, build_time = build_hnsw(
            embeddings[:DATASET_SIZE],
            ids[:DATASET_SIZE],
            M=M,
            ef_construction=EF_CONSTRUCTION,
            ef_search=EF_SEARCH,
            seed=SEED,
        )
        print(f"build_time_seconds: {build_time:.3f}")
        print(f"diagnostic_wall_time_seconds: {time.perf_counter() - start:.3f}")
        run_diagnostic(index, namespace["cosine_similarity"])
    except Exception as exc:
        print(f"ERROR: diagnostic could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())