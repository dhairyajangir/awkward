"""Isolated Awkward/PyArrow differential diagnostic, not a CI replacement."""
import json
import os
import pathlib
import resource
import subprocess
import sys

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def child():
    import awkward as ak
    import numpy as np
    import pyarrow as pa

    source_root = pathlib.Path(os.environ["EXPECTED_SOURCE_ROOT"]).resolve()
    source_file = pathlib.Path(ak.__file__).resolve()
    assert source_file.is_relative_to(source_root), (str(source_file), str(source_root))
    assert pa.__version__ == os.environ["EXPECTED_ARROW"]
    assert np.__version__ == "1.21.3"
    mode, length_text, ext_text, dict_text = sys.argv[2:]
    length = int(length_text)
    extensionarray = ext_text == "true"
    categorical_as_dictionary = dict_text == "true"
    if mode == "nonempty":
        content = ak.Array(["hello"]).layout
        indices = np.array([-1, 0], dtype=np.int64)
    elif mode == "empty":
        content = ak.Array(["hello"])[:0].layout
        indices = np.full(length, -1, dtype=np.int64)
    else:
        raise ValueError(mode)
    array = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(indices), content, parameters={"__array__": "categorical"}
        )
    )
    print("before conversion", flush=True)
    out = ak.to_arrow(
        array,
        extensionarray=extensionarray,
        categorical_as_dictionary=categorical_as_dictionary,
    )
    print("before full validation", flush=True)
    out.validate(full=True)
    print("after full validation", flush=True)
    assert out.to_pylist() == ak.to_list(array)
    print("cell passed", flush=True)


def orchestrate():
    envs = json.loads(os.environ["DIAGNOSTIC_ENVS"])
    script = str(pathlib.Path(__file__).resolve())
    results = []
    for spec in envs:
        modes = [("nonempty", 2)]
        if spec["source"] == "merge":
            modes.extend([("empty", 0), ("empty", 3)])
        for mode, length in modes:
            for ext in (False, True):
                for dictionary in (False, True):
                    cmd = [
                        spec["python"], "-X", "faulthandler", script, "child", mode,
                        str(length), str(ext).lower(), str(dictionary).lower(),
                    ]
                    env = os.environ.copy()
                    env["EXPECTED_SOURCE_ROOT"] = spec["path"]
                    env["EXPECTED_ARROW"] = spec["arrow"]
                    case = {"source": spec["source"], "arrow": spec["arrow"], "mode": mode,
                            "length": length, "extensionarray": ext,
                            "categorical_as_dictionary": dictionary}
                    try:
                        result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=60)
                        case.update(returncode=result.returncode, stdout=result.stdout[-8000:], stderr=result.stderr[-8000:])
                    except subprocess.TimeoutExpired as exc:
                        case.update(returncode=None, timeout=True, stdout=str(exc.stdout)[-8000:], stderr=str(exc.stderr)[-8000:])
                    results.append(case)
                    print("CELL " + json.dumps(case, sort_keys=True), flush=True)
    failed = [r for r in results if r.get("returncode") != 0]
    print("DIAGNOSTIC_SUMMARY " + json.dumps({"cells": len(results), "failed": len(failed)}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    child() if len(sys.argv) > 1 and sys.argv[1] == "child" else orchestrate()
