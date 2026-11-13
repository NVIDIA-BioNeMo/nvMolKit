from typing import Any

from rdkit.Chem import Mol

from nvmolkit._embedMolecules import BatchHardwareOptions

def UFFOptimizeMoleculesConfs(
    molecules: list[Mol],
    maxIters: int,
    vdwThresholds: list[float],
    ignoreInterfragInteractions: list[bool],
    hardwareOptions: BatchHardwareOptions,
    minimizerKind: str = "BFGS",
    fireOptions: object = ...,
) -> list[list[float]]: ...
def UFFOptimizeMoleculesConfsDevice(
    molecules: list[Mol],
    maxIters: int = 1000,
    vdwThresholds: list[float] = ...,
    ignoreInterfragInteractions: list[bool] = ...,
    hardwareOptions: BatchHardwareOptions = ...,
    targetGpu: int = -1,
    minimizerKind: str = "BFGS",
    fireOptions: object = ...,
) -> Any: ...
