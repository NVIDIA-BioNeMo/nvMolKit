from typing import Any, List

from rdkit.Chem import Mol

from nvmolkit._embedMolecules import BatchHardwareOptions

def UFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int,
    vdwThresholds: List[float],
    ignoreInterfragInteractions: List[bool],
    hardwareOptions: BatchHardwareOptions,
    minimizerKind: str = "BFGS",
    fireOptions: object = ...,
) -> List[List[float]]: ...

def UFFOptimizeMoleculesConfsDevice(
    molecules: List[Mol],
    maxIters: int = 1000,
    vdwThresholds: List[float] = ...,
    ignoreInterfragInteractions: List[bool] = ...,
    hardwareOptions: BatchHardwareOptions = ...,
    targetGpu: int = -1,
    minimizerKind: str = "BFGS",
    fireOptions: object = ...,
) -> Any: ...
