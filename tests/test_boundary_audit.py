"""Do not invent a single residual tensor when generated kernels consume a sum."""

from pathlib import Path

import pytest

from watching.boundary_audit import ExistingBoundaryObserver, inspect_boundary

SOURCES = Path(__file__).parents[1] / "results/compiler-mechanism-042r/compiler_sources"
NONE = SOURCES / "212913f5994a2c54bfc050b2f81257a860d0780386a7dbef888b12e18e7bc18b.py.txt"
REFERENCE = SOURCES / "17e9b1279afdf108c4a69f83610827c7ab6405cd77ffe6e7c3483f72d564fe52.py.txt"


@pytest.mark.parametrize("source", [NONE, REFERENCE])
def test_actual_archived_native_wrapper_stores_norm_not_preblock_residual(source):
    report = inspect_boundary(source.read_text(), 34)
    assert report["boundary_status"] == "separate_residual_summands_normalized_without_sum_store"
    assert report["all_stored_values_depend_on_rsqrt"]
    assert report["fp32_loaded_pointer_additions"] == [["in_ptr0", "in_ptr1"]]
    assert report["written_pointers"] == ["out_ptr1"]
    assert not report["single_native_tail_input_certified"]
    assert report["normalization_launch_line"] < report["query_projection_line"]


def test_unrecognized_graph_fails_instead_of_guessing_boundary():
    with pytest.raises(ValueError, match="exactly one"):
        inspect_boundary("def unrelated(): pass", 34)
    with pytest.raises(ValueError, match="uniquely identify"):
        inspect_boundary(NONE.read_text(), 37)


def test_tracer_never_claims_capture_if_no_native_wrapper_ran():
    with pytest.raises(RuntimeError, match="incomplete"):
        with ExistingBoundaryObserver(34):
            pass
