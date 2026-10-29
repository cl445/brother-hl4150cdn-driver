"""Printer models: paper geometry and the XL2HB/PJL values each model's filter writes.

`MODELS` maps the CLI/CUPS name of every supported model to its
`PrinterModel`; `DEFAULT_MODEL` is used where none is given.
"""

from brother_xl2hb.models.base import PrinterModel
from brother_xl2hb.models.hl4150cdn import HL4150CDN

MODELS: dict[str, PrinterModel] = {"hl4150cdn": HL4150CDN}
DEFAULT_MODEL = HL4150CDN
