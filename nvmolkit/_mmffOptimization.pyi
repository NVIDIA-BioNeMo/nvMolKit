from typing import Any, List

from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 200,
    properties: Any = None,
    hardwareOptions: Any = None,
    backend: str = "HYBRID",
    minimizerKind: str = "BFGS",
    fireOptions: Any = None,
) -> List[List[float]]: ...

def MMFFOptimizeMoleculesConfsDevice(
    molecules: List[Mol],
    maxIters: int = 200,
    properties: Any = None,
    hardwareOptions: Any = None,
    targetGpu: int = -1,
    backend: str = "HYBRID",
    minimizerKind: str = "BFGS",
    fireOptions: Any = None,
) -> Any: ...
