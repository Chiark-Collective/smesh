from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
from .utils import get_logger

_log = get_logger()

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy
    _HAVE_VTK = True
except Exception:
    vtk = None  # type: ignore
    _HAVE_VTK = False

try:
    import trimesh  # type: ignore
    _HAVE_TRIMESH = True
except Exception:
    trimesh = None  # type: ignore
    _HAVE_TRIMESH = False


class MeshScene:
    """Holds a surface mesh and provides bounds / attribute probing.

    For best functionality, install VTK so attribute probing (RGB/normal) works.
    Fallbacks use trimesh for bounds only.
    """
    def __init__(
        self,
        mesh_path: str | Path | None = None,
        mesh_data: Optional["vtk.vtkPolyData"] = None,
        require_normals: bool = True,
        build_colors: bool = True
    ) -> None:
        self.mesh_path = Path(mesh_path) if mesh_path is not None else None
        self._vtk_poly: Optional["vtk.vtkPolyData"] = None
        self._trimesh: Optional["trimesh.Trimesh"] = None
        self._vertices: Optional[np.ndarray] = None
        self._faces: Optional[np.ndarray] = None
        self._vertex_normals: Optional[np.ndarray] = None
        self._vertex_colors: Optional[np.ndarray] = None

        if mesh_data is not None:
            if not _HAVE_VTK:
                raise RuntimeError("VTK is required to pass in vtkPolyData.")
            self._vtk_poly = mesh_data
            self._store_numpy_mesh_from_vtk(mesh_data)
        elif self.mesh_path is not None:
            self._load_from_path(self.mesh_path, require_normals=require_normals, build_colors=build_colors)
        else:
            raise ValueError("Provide either mesh_path or mesh_data.")

    # -- IO helpers --
    def _load_from_path(self, path: Path, require_normals: bool, build_colors: bool) -> None:
        suffix = path.suffix.lower()
        if _HAVE_VTK:
            reader = None
            if suffix in (".ply", ".vtp"):
                if suffix == ".ply":
                    reader = vtk.vtkPLYReader()
                    reader.SetFileName(str(path))
                else:
                    reader = vtk.vtkXMLPolyDataReader()
                    reader.SetFileName(str(path))
            elif suffix == ".obj":
                reader = vtk.vtkOBJReader()
                reader.SetFileName(str(path))
            else:
                _log.warning("Unknown mesh extension '%s'; trying VTK's generic reader.", suffix)
                reader = vtk.vtkGenericDataObjectReader()
                reader.SetFileName(str(path))
            reader.Update()
            poly = reader.GetOutput()
            # Ensure triangulated
            tri = vtk.vtkTriangleFilter()
            tri.SetInputData(poly)
            tri.PassLinesOff()
            tri.PassVertsOff()
            tri.Update()
            poly = tri.GetOutput()

            # Normals if needed
            if require_normals and poly.GetPointData().GetNormals() is None:
                n = vtk.vtkPolyDataNormals()
                n.SetInputData(poly)
                n.ComputePointNormalsOn()
                n.ComputeCellNormalsOn()
                n.SplittingOff()
                n.ConsistencyOn()
                n.AutoOrientNormalsOn()
                n.Update()
                poly = n.GetOutput()

            # If no RGB colors present and requested, try to create a default gray per-vertex color
            if build_colors and poly.GetPointData().GetScalars() is None:
                colors = vtk.vtkUnsignedCharArray()
                colors.SetNumberOfComponents(3)
                colors.SetName("RGB")
                npts = poly.GetNumberOfPoints()
                colors.SetNumberOfTuples(npts)
                for i in range(npts):
                    colors.SetTypedTuple(i, (200, 200, 200))
                poly.GetPointData().SetScalars(colors)

            self._vtk_poly = poly
            self._store_numpy_mesh_from_vtk(poly)
            return

        # Fallback options
        if _HAVE_TRIMESH:
            self._trimesh = trimesh.load_mesh(str(path), process=True)
            self._vertices = np.asarray(self._trimesh.vertices, dtype=np.float32)
            self._faces = np.asarray(self._trimesh.faces, dtype=np.int64)
            if self._trimesh.vertex_normals is not None:
                self._vertex_normals = np.asarray(self._trimesh.vertex_normals, dtype=np.float32)
            if self._trimesh.visual.kind == "vertex":
                self._vertex_colors = np.asarray(self._trimesh.visual.vertex_colors[:, :3], dtype=np.uint8)
            return

        if suffix == ".ply":
            self._load_ascii_ply(path)
            return

        raise RuntimeError("Install VTK/trimesh or provide ASCII PLY mesh.")

    # -- API --
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        if self._vtk_poly is not None:
            return tuple(self._vtk_poly.GetBounds())  # type: ignore
        if self._trimesh is not None:
            mn = self._trimesh.bounds[0]
            mx = self._trimesh.bounds[1]
            return (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
        if self._vertices is not None:
            mn = self._vertices.min(axis=0)
            mx = self._vertices.max(axis=0)
            return (float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2]))
        raise RuntimeError("Scene not loaded.")

    def vtk_polydata(self) -> "vtk.vtkPolyData":
        if self._vtk_poly is None:
            raise RuntimeError("VTK not available / mesh not loaded via VTK.")
        return self._vtk_poly

    def probe_attributes(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        """Probe per-point attributes like RGB and normal.

        Returns a dict possibly containing:
          - 'rgb': uint8 (N,3)
          - 'normal': float32 (N,3)

        Requires VTK. If VTK is unavailable, returns an empty dict.
        """
        if not _HAVE_VTK or self._vtk_poly is None:
            result: dict[str, np.ndarray] = {}
            if self._vertex_colors is not None:
                result["rgb"] = np.tile(self._vertex_colors.mean(axis=0, keepdims=True), (len(xyz), 1)).astype(np.uint8)
            if self._vertex_normals is not None:
                result["normal"] = np.tile(self._vertex_normals.mean(axis=0, keepdims=True), (len(xyz), 1)).astype(np.float32)
            return result

        # Build a point data set for probes
        import vtk  # local import for type checkers
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(xyz))
        for i, (x, y, z) in enumerate(xyz):
            pts.SetPoint(i, float(x), float(y), float(z))

        poly_pts = vtk.vtkPolyData()
        poly_pts.SetPoints(pts)

        probe = vtk.vtkProbeFilter()
        probe.SetInputData(poly_pts)
        probe.SetSourceData(self._vtk_poly)
        probe.CategoricalDataOff()
        probe.Update()

        out = probe.GetOutput()
        pdat = out.GetPointData()

        result: dict[str, np.ndarray] = {}

        # RGB may be in Scalars or a named array
        scalars = pdat.GetScalars()
        if scalars is not None and scalars.GetNumberOfComponents() >= 3:
            arr = vtk_to_numpy(scalars).reshape((-1, scalars.GetNumberOfComponents()))[:, :3]
            result["rgb"] = arr.astype(np.uint8, copy=False)

        normals = pdat.GetNormals()
        if normals is not None:
            arr = vtk_to_numpy(normals).reshape((-1, 3)).astype(np.float32, copy=False)
            result["normal"] = arr

        # Also check for named arrays
        arr_names = [pdat.GetArrayName(i) for i in range(pdat.GetNumberOfArrays())]
        for name in arr_names:
            if name and name.lower() in {"rgb", "rgba"} and name not in result:
                a = pdat.GetArray(name)
                arr = vtk_to_numpy(a).reshape((-1, a.GetNumberOfComponents()))[:, :3]
                result["rgb"] = arr.astype(np.uint8, copy=False)
            if name and name.lower() in {"normal", "normals"} and "normal" not in result:
                a = pdat.GetArray(name)
                arr = vtk_to_numpy(a).reshape((-1, 3)).astype(np.float32, copy=False)
                result["normal"] = arr

        return result

    # -- helpers for numpy-based pipelines --
    def has_numpy_mesh(self) -> bool:
        return self._vertices is not None and self._faces is not None

    def triangle_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._vertices is None or self._faces is None:
            raise RuntimeError("Mesh does not expose triangle arrays.")
        return self._vertices, self._faces

    def _store_numpy_mesh_from_vtk(self, poly: "vtk.vtkPolyData") -> None:
        import vtk  # local import
        from vtk.util.numpy_support import vtk_to_numpy

        pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32, copy=False)
        polys = vtk_to_numpy(poly.GetPolys().GetData())
        faces = polys.reshape(-1, 4)[:, 1:4].astype(np.int64, copy=False)
        self._vertices = pts
        self._faces = faces

        normals = poly.GetPointData().GetNormals()
        if normals is not None:
            self._vertex_normals = vtk_to_numpy(normals).astype(np.float32, copy=False)
        scalars = poly.GetPointData().GetScalars()
        if scalars is not None:
            arr = vtk_to_numpy(scalars)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                self._vertex_colors = arr[:, :3].astype(np.uint8, copy=False)

    def _load_ascii_ply(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            header: list[str] = []
            while True:
                line = f.readline()
                if not line:
                    raise RuntimeError("Unexpected EOF while reading PLY header.")
                line = line.strip()
                header.append(line)
                if line == "end_header":
                    break

            if header[0] != "ply":
                raise RuntimeError("Only ASCII PLY files are supported.")
            if "format ascii" not in header[1]:
                raise RuntimeError("Only ASCII PLY format is supported.")

            n_vertices = 0
            n_faces = 0
            vertex_props: list[str] = []
            face_props: list[str] = []
            current_element = None
            for line in header[2:]:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "element":
                    current_element = parts[1]
                    if current_element == "vertex":
                        n_vertices = int(parts[2])
                    elif current_element == "face":
                        n_faces = int(parts[2])
                elif parts[0] == "property" and current_element == "vertex":
                    vertex_props.append(parts[-1])
                elif parts[0] == "property" and current_element == "face":
                    face_props.append(parts[-1])

            vertices = []
            colors = []
            normals = []
            for _ in range(n_vertices):
                parts = f.readline().strip().split()
                if len(parts) < 3:
                    raise RuntimeError("Vertex line must contain at least xyz.")
                x, y, z = map(float, parts[:3])
                vertices.append((x, y, z))
                if len(parts) >= 6:
                    colors.append(tuple(int(float(v)) for v in parts[3:6]))
                if len(parts) >= 9:
                    normals.append(tuple(float(v) for v in parts[6:9]))

            faces = []
            for _ in range(n_faces):
                parts = f.readline().strip().split()
                if not parts:
                    continue
                count = int(parts[0])
                if count != 3:
                    raise RuntimeError("Only triangular faces are supported in ASCII PLY fallback.")
                face = tuple(int(v) for v in parts[1:4])
                faces.append(face)

        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.int64)
        if colors:
            col_arr = np.asarray(colors, dtype=np.float32)
            col_arr = np.clip(col_arr, 0, 255).astype(np.uint8)
            self._vertex_colors = col_arr
        else:
            self._vertex_colors = None
        self._vertex_normals = np.asarray(normals, dtype=np.float32) if normals else None
