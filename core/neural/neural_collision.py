"""Alias module for collision operator models (PFNO, PRFNO, ConservativeFNO, ConservativeRFNO).
"""
from core.neural.collision_op import (
    PFNO,
    PRFNO,
    ConservativeFNO,
    ConservativeRFNO,
)


__all__: list[str] = [
    "PFNO",
    "PRFNO",
    "ConservativeFNO",
    "ConservativeRFNO",
]


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()
