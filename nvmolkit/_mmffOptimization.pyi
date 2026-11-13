from typing import Any

from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: list[Mol],
    maxIters: int = 200,
    properties: Any = None,
    hardwareOptions: Any = None,
    backend: str = "HYBRID",
    minimizerKind: str = "BFGS",
    fireOptions: Any = None,
) -> list[list[float]]: ...
def MMFFOptimizeMoleculesConfsDevice(
    molecules: list[Mol],
    maxIters: int = 200,
    properties: Any = None,
    hardwareOptions: Any = None,
    targetGpu: int = -1,
    backend: str = "HYBRID",
    minimizerKind: str = "BFGS",
    fireOptions: Any = None,
) -> Any: ...
