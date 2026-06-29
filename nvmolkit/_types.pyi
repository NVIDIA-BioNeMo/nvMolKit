class FireOptions:
    dtInit: float
    dtMinFactor: float
    dtMaxFactor: float
    dMax: float
    timeStepIncrement: float
    timeStepDecrement: float
    nMinForIncrease: int
    alphaInit: float
    alphaDecrement: float
    useMass: bool
    gradTol: float
    takeHalfStepBack: bool
    abcCorrection: bool
    stuckDetectionEnabled: bool
    stuckEnergyRelTol: float
    stuckStreakLength: int
    stuckEvalEveryNPolls: int

    def __init__(self) -> None: ...
