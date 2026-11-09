from rdkit.Chem import Mol
from rdkit.Chem.rdDistGeom import EmbedParameters

class BatchHardwareOptions: ...

def EmbedMolecules(
    molecules: list[Mol],
    params: EmbedParameters,
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: BatchHardwareOptions | None = ...,
) -> None: ...
