"""Job Sheets module.

Prepares PD080 "Packaging Sign Out & In Record" sheets from SAP production
plan screenshots / pasted tables. Kept as a self-contained package so it can be
extended later (Excel/PDF upload, trace-code DB, GIRO setups, learned aliases)
without touching the core Pallet Tracker.
"""

from .router import router  # noqa: F401
