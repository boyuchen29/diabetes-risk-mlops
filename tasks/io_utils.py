import pickle
import tempfile
from pathlib import Path


def dump_pickle(payload, output_path: str, *, dbutils=None) -> None:
    if _is_dbfs_path(output_path):
        if dbutils is None:
            raise ValueError("dbutils is required for DBFS paths")

        output_uri = _to_dbfs_uri(output_path)
        parent_uri = output_uri.rsplit("/", 1)[0]
        dbutils.fs.mkdirs(parent_uri)
        try:
            dbutils.fs.rm(output_uri, True)
        except Exception:
            pass

        with tempfile.TemporaryDirectory(dir=_workspace_temp_root()) as tmp_dir:
            local_path = Path(tmp_dir) / "payload.pkl"
            with open(local_path, "wb") as handle:
                pickle.dump(payload, handle)
            dbutils.fs.cp(f"file:{local_path}", output_uri)
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(input_path: str, *, dbutils=None):
    if _is_dbfs_path(input_path):
        if dbutils is None:
            raise ValueError("dbutils is required for DBFS paths")

        input_uri = _to_dbfs_uri(input_path)
        with tempfile.TemporaryDirectory(dir=_workspace_temp_root()) as tmp_dir:
            local_path = Path(tmp_dir) / "payload.pkl"
            dbutils.fs.cp(input_uri, f"file:{local_path}")
            with open(local_path, "rb") as handle:
                return pickle.load(handle)

    with open(input_path, "rb") as handle:
        return pickle.load(handle)


def _is_dbfs_path(path: str) -> bool:
    return path.startswith("/dbfs/") or path.startswith("dbfs:/")


def _to_dbfs_uri(path: str) -> str:
    if path.startswith("dbfs:/"):
        return path
    if path.startswith("/dbfs/"):
        return f"dbfs:/{path.removeprefix('/dbfs/')}"
    raise ValueError(f"Unsupported DBFS path: {path}")


def _workspace_temp_root() -> str:
    workspace_tmp = Path("/Workspace/tmp")
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    return str(workspace_tmp)
