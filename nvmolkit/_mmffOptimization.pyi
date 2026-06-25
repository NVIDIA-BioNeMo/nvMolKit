from typing import Any, List
from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int,
    properties: Any,
    hardwareOptions: Any,
) -> List[List[float]]: ...
