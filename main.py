from __future__ import annotations

from sqlalchemy import text

from fastapi import Request

import csv
from pathlib import Path
import re
from datetime import datetime

from fastapi import UploadFile, File

from fastapi import FastAPI, Request, Form, Cookie
from starlette.responses import Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime,
    ForeignKey, UniqueConstraint, select, func, distinct
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from openpyxl import Workbook
from fastapi.responses import StreamingResponse
import io

OPERATORS = {
    "Valik": "1111",
    "Valentin": "2222",
    "Paul": "3333",
    "Mick": "4444",
    "Michal": "5555",
}


import os

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

ROWS = ["K", "L", "M", "N", "O", "P", "Q", "J"]
LEVELS = ["A", "B", "C"]

ALLOWED_SLOTS: dict[str, set[int]] = {
    "J": set(range(4, 23)),                      # Tray 1 only                       
    "K": set(range(4, 30)),                      # 04-29
    "L": set(range(4, 30)) - {13},               # 04-29 but no 13
    "M": set(range(4, 30)),                      # 04-29
    "N": set(range(4, 30)),                      # 04-29
    "O": set(range(18, 30)),                     # 18-29
    "P": set(range(16, 28)),                     # 16-27
    "Q": set(range(1, 9))                       # 01-08  
}

templates = Jinja2Templates(directory="templates")
app = FastAPI(title="Pallet Tracker")  # New function

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True)
    row = Column(String, nullable=False)
    slot = Column(Integer, nullable=False)
    level = Column(String, nullable=False)

    pallet = relationship("Pallet", back_populates="location", uselist=False)

    __table_args__ = (UniqueConstraint("row", "slot", "level", name="uq_location"),)

    @property
    def code(self) -> str:
        return f"{self.row}{self.slot:02d}{self.level}"

from sqlalchemy import Boolean

class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    has_positions = Column(Boolean, default=False)

    

class Pallet(Base):
    __tablename__ = "pallets"
    id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey("locations.id"), unique=True)

    category_name = Column(String, nullable=False)
    punnets = Column(Integer, nullable=False, default=0)
    description = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="pallet")

    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=True)
    zone = relationship("Zone")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)

    pallet_name = Column(String, nullable=False)
    action = Column(String, nullable=False)  # ADD / REMOVE / MOVE

    from_location = Column(String)  # e.g. L03C
    to_location = Column(String)    # e.g. K01A

    actor = Column(String, nullable=False, default="unknown")

def log_event(
    db,
    actor: str,
    pallet_name: str,
    action: str,
    from_loc: str | None = None,
    to_loc: str | None = None,
):
    event = Event(
        actor=actor,
        pallet_name=pallet_name,
        action=action,
        from_location=from_loc,
        to_location=to_loc,
        ts=datetime.utcnow(),
    )
    db.add(event)

from datetime import timedelta


def get_active_operators(db, minutes: int = 30):
    since = datetime.utcnow() - timedelta(minutes=minutes)

    q = (
        select(Event.actor)
        .where(Event.ts >= since)
        .distinct()
    )

    return db.execute(q).scalars().all()


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "", "names": list(OPERATORS.keys())},
    )


@app.post("/login", response_class=HTMLResponse)
def login_submit(request: Request, name: str = Form(...), pin: str = Form(...)):
    name = (name or "").strip()
    pin = (pin or "").strip()

    if name not in OPERATORS or OPERATORS[name] != pin:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Wrong name or PIN", "names": list(OPERATORS.keys())},
        )

    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("operator", name, max_age=7 * 24 * 3600, samesite="lax")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("operator")
    return resp




def init_db():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        count = db.execute(select(func.count(Location.id))).scalar_one()
        if count > 0:
            return

        for r in ROWS:
            slots = sorted(ALLOWED_SLOTS[r])
            for lvl in LEVELS:
                for s in slots:
                    db.add(Location(row=r, slot=s, level=lvl))
        db.commit()


def parse_location(code: str):
    if not code:
        return None

    code = code.strip().upper()

    m = re.fullmatch(r"([A-Z])\s*(\d{1,2})\s*([A-C])", code)
    if not m:
        return None

    row = m.group(1)
    slot = int(m.group(2))
    level = m.group(3)

    # 🔥 проверяем через конфиг
    if row not in ROWS:
        return None

    allowed = ALLOWED_SLOTS.get(row)
    if not allowed or slot not in allowed:
        return None

    return row, slot, level


def find_location(db, code: str):
    parsed = parse_location(code)
    if not parsed:
        return None
    row, slot, level = parsed
    return db.execute(
        select(Location).where(Location.row == row, Location.slot == slot, Location.level == level)
    ).scalar_one_or_none()


def get_suggestions(db):
    raw = db.execute(
        select(distinct(Pallet.category_name)).order_by(Pallet.category_name)
    ).scalars().all()

    out = []
    for name in raw:
        n = (name or "").strip()
        if not n:
            continue

        # убираем короткие обрубки типа "Ea", "Bb"
        if len(n) < 3:
            continue

        # убираем то, что похоже на координаты: L1, L12, L12C
        if re.fullmatch(r"[J-Q]\d{1,2}([A-C])?", n.upper()):
            continue

        out.append(n)

    return out


@app.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    since: str = "",
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    with SessionLocal() as db:
        q = select(Event).order_by(Event.ts.desc()).limit(500)
        if since:
            try:
                dt = datetime.fromisoformat(since)  # YYYY-MM-DD
                q = q.where(Event.ts >= dt)
            except Exception:
                pass

        events = db.execute(q).scalars().all()

    return templates.TemplateResponse(
        "history.html",
        {"request": request, "events": events, "since": since},
    )

@app.get("/export/history")
def export_history(operator: str | None = Cookie(default=None)):

    if not operator:
        return RedirectResponse("/login", status_code=303)

    with SessionLocal() as db:
        events = db.execute(
            select(Event).order_by(Event.ts.desc())
        ).scalars().all()

    wb = Workbook()
    ws = wb.active
    ws.title = "History"

    # Заголовки
    ws.append([
        "Pallet",
        "Action",
        "From",
        "To",
        "Operator",
        "Date/Time (UTC)",
    ])

    # Данные
    for e in events:
        ws.append([
            e.pallet_name,
            e.action,
            e.from_location or "",
            e.to_location or "",
            e.actor,
            e.ts.strftime("%Y-%m-%d %H:%M"),
        ])

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=history.xlsx"
        },
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    q: str = "",
    toast: str = "",
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    q = (q or "").strip()

    with SessionLocal() as db:
        suggestions = get_suggestions(db)
        active_ops = get_active_operators(db)

        # empty search => show nothing
        if not q:
            return templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "pallets": [],
                    "q": "",
                    "total": None,
                    "toast": toast,
                    "suggestions": suggestions,
                    "has_query": False,
                    "active_ops": active_ops,
                },
            )

        items_q = select(Pallet).join(Zone, isouter=True).join(Location, isouter=True)
        total_q = select(func.coalesce(func.sum(Pallet.punnets), 0)).select_from(Pallet).join(Location)

        parsed = parse_location(q)
        if parsed:
            r, s, l = parsed
            items_q = items_q.where(Location.row == r, Location.slot == s, Location.level == l)
            total_q = total_q.where(Location.row == r, Location.slot == s, Location.level == l)
        else:
            items_q = items_q.where(func.lower(Pallet.category_name) == q.lower())
            total_q = total_q.where(func.lower(Pallet.category_name) == q.lower())


        pallets = db.execute(items_q).scalars().all()
        total = db.execute(total_q).scalar_one()

        active_ops = get_active_operators(db)

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "pallets": pallets,
                "active_ops": active_ops,
                "q": q,
                "total": total,
                "toast": toast,
                "suggestions": suggestions,
                "has_query": True,
            },
        )


@app.get("/add", response_class=HTMLResponse)
def add_form(
    request: Request,
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    with SessionLocal() as db:
        suggestions = get_suggestions(db)
        zones = db.query(Zone).all()   # ← ДОБАВИЛИ

        return templates.TemplateResponse(
            "add.html",
            {
                "request": request,
                "values": {
                    "category_name": "",
                    "location_code": "",
                    "punnets": "",
                    "description": "",
                    "zone_id": zones[0].id if zones else None,
                },
                "error": "",
                "occupied": "",
                "invalid": "",
                "suggestions": suggestions,
                "zones": zones,   # ← ГЛАВНОЕ
            },
        )

@app.post("/add", response_class=HTMLResponse)
def add_pallet(
    request: Request,
    category_name: str = Form(...),
    zone_id: int = Form(...),
    location_code: str = Form(""),
    punnets: int = Form(...),
    description: str = Form(""),
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    values = {
        "category_name": (category_name or "").strip(),
        "location_code": (location_code or "").strip().upper(),
        "punnets": punnets,
        "description": (description or "").strip(),
        "zone_id": zone_id,
    }

    with SessionLocal() as db:
        suggestions = get_suggestions(db)
        zones = db.query(Zone).all()

        zone = db.get(Zone, zone_id)
        if not zone:
            return templates.TemplateResponse(
                "add.html",
                {
                    "request": request,
                    "values": values,
                    "error": "Invalid zone selected.",
                    "occupied": "",
                    "invalid": "",
                    "suggestions": suggestions,
                    "zones": zones,
                },
            )

        # 🧊 Если зона с позициями
        if zone.has_positions:

            loc = find_location(db, values["location_code"])
            if not loc:
                return templates.TemplateResponse(
                    "add.html",
                    {
                        "request": request,
                        "values": values,
                        "error": "",
                        "occupied": "",
                        "invalid": "Invalid location (use L03C).",
                        "suggestions": suggestions,
                        "zones": zones,
                    },
                )

            exists = db.execute(
                select(Pallet).where(Pallet.location_id == loc.id)
            ).scalar_one_or_none()

            if exists:
                return templates.TemplateResponse(
                    "add.html",
                    {
                        "request": request,
                        "values": values,
                        "error": "",
                        "occupied": f"Location {loc.code} is occupied.",
                        "invalid": "",
                        "suggestions": suggestions,
                        "zones": zones,
                    },
                )

            location_id = loc.id
            to_loc_value = loc.code

        # 🟢 Если зона без позиций
        else:
            location_id = None
            to_loc_value = zone.name

        pallet = Pallet(
            location_id=location_id,
            zone_id=zone.id,
            category_name=values["category_name"],
            punnets=int(punnets),
            description=values["description"] or None,
            updated_at=datetime.utcnow(),
        )

        db.add(pallet)

        log_event(
            db,
            actor=operator,
            pallet_name=pallet.category_name,
            action="ADD",
            to_loc=to_loc_value,
        )

        db.commit()

    return RedirectResponse("/?toast=Added", status_code=303)



@app.post("/remove/{pid}")
def remove(
    pid: int,
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    with SessionLocal() as db:
        pallet = db.get(Pallet, pid)
        if pallet:
            name = pallet.category_name
            loc_code = pallet.location.code if pallet.location else None

            log_event(
                db,
                actor=operator,
                pallet_name=name,
                action="REMOVE",
                from_loc=loc_code,
            )

            db.delete(pallet)
            db.commit()

    return RedirectResponse("/?toast=Removed", status_code=303)

@app.get("/move/{pid}", response_class=HTMLResponse)
def move_form(
    request: Request,
    pid: int,
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    with SessionLocal() as db:
        pallet = db.get(Pallet, pid)
        if not pallet:
            return RedirectResponse("/", status_code=303)

        zones = db.query(Zone).all()

        return templates.TemplateResponse(
            "move.html",
            {
                "request": request,
                "pallet": pallet,
                "zones": zones,
                "values": {"to_location": "", "zone_id": pallet.zone_id},
                "occupied": "",
                "invalid": "",
                "error": "",
            },
        )

@app.post("/move/{pid}", response_class=HTMLResponse)
def move_submit(
    request: Request,
    pid: int,
    zone_id: int = Form(...),
    to_location: str = Form(""),
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    to_location = (to_location or "").strip().upper()

    with SessionLocal() as db:
        pallet = db.get(Pallet, pid)
        if not pallet:
            return RedirectResponse("/", status_code=303)

        zones = db.query(Zone).all()
        zone = db.get(Zone, zone_id)

        if not zone:
            return RedirectResponse("/", status_code=303)

        from_zone = pallet.zone
        from_loc = pallet.location.code if pallet.location else None

        # Если зона с позициями
        if zone.has_positions:

            target = find_location(db, to_location)
            if not target:
                return templates.TemplateResponse(
                    "move.html",
                    {
                        "request": request,
                        "pallet": pallet,
                        "zones": zones,
                        "values": {"to_location": to_location, "zone_id": zone_id},
                        "occupied": "",
                        "invalid": "Invalid location format. Use like L03C.",
                        "error": "",
                    },
                )

            occupied = db.execute(
                select(Pallet).where(Pallet.location_id == target.id)
            ).scalar_one_or_none()

            if occupied:
                return templates.TemplateResponse(
                    "move.html",
                    {
                        "request": request,
                        "pallet": pallet,
                        "zones": zones,
                        "values": {"to_location": to_location, "zone_id": zone_id},
                        "occupied": f"Location {target.code} is already occupied.",
                        "invalid": "",
                        "error": "",
                    },
                )

            pallet.location_id = target.id
            pallet.zone_id = zone.id
            to_loc_value = target.code

        # Если зона без позиций
        else:
            pallet.location_id = None
            pallet.zone_id = zone.id
            to_loc_value = zone.name

        log_event(
            db,
            actor=operator,
            pallet_name=pallet.category_name,
            action="MOVE",
            from_loc=from_loc or (from_zone.name if from_zone else None),
            to_loc=to_loc_value,
        )

        db.commit()

    return RedirectResponse("/?toast=Moved", status_code=303)


@app.post("/import-csv")
async def import_csv(
    request: Request,
    file: UploadFile = File(...),
    operator: str | None = Cookie(default=None),
):
    if not operator:
        return RedirectResponse("/login", status_code=303)

    if not file.filename.endswith(".csv"):
        return templates.TemplateResponse(
            "import_result.html",
            {
                "request": request,
                "imported": 0,
                "errors": ["Only CSV files allowed"]
            }
        )

    imported = 0
    errors = []

    contents = await file.read()
    decoded = contents.decode("utf-8").splitlines()
    reader = csv.DictReader(decoded)

    with SessionLocal() as db:
        for row in reader:
            category = (row.get("category_name") or "").strip()
            location_code = (row.get("location_code") or "").strip().upper()
            punnets = int(row.get("punnets") or 0)
            description = (row.get("description") or "").strip() or None

            loc = find_location(db, location_code)
            if not loc:
                errors.append(f"Invalid location: {location_code}")
                continue

            exists = db.execute(
                select(Pallet).where(Pallet.location_id == loc.id)
            ).scalar_one_or_none()

            if exists:
                errors.append(f"Occupied: {location_code}")
                continue

            pallet = Pallet(
                location_id=loc.id,
                category_name=category,
                punnets=punnets,
                description=description,
                updated_at=datetime.utcnow(),
            )

            db.add(pallet)
            imported += 1

        db.commit()

    return templates.TemplateResponse(
        "import_result.html",
        {
            "request": request,
            "imported": imported,
            "errors": errors
        }
    )
from sqlalchemy import delete


# ================== End Points ================== #

@app.get("/reset-db")
def reset_db(operator: str | None = Cookie(default=None)):
    if not operator:
        return {"error": "Not logged in"}

    with SessionLocal() as db:
        db.execute(delete(Pallet))
        db.commit()

    return {"status": "Pallets cleared"}


from sqlalchemy import delete

from sqlalchemy import delete

@app.get("/rebuild-locations")
def rebuild_locations():
    with SessionLocal() as db:
        # сначала удалить pallets (иначе FK ошибка)
        db.execute(delete(Pallet))
        db.commit()

        # потом удалить locations
        db.execute(delete(Location))
        db.commit()

        # создать заново
        for r in ROWS:
            slots = sorted(ALLOWED_SLOTS[r])
            for lvl in LEVELS:
                for s in slots:
                    db.add(Location(row=r, slot=s, level=lvl))

        db.commit()

    return {"status": "Locations rebuilt clean"}
    
@app.get("/init-db")
def force_init_db():
    Base.metadata.create_all(bind=engine)
    return {"status": "tables created"}

@app.get("/debug-j")
def debug_j():
    with SessionLocal() as db:
        rows = db.execute(
            select(Location).where(Location.row == "J")
        ).scalars().all()

        return {"count": len(rows)}

@app.get("/create-tables")
def create_tables():
    Base.metadata.create_all(bind=engine)
    return {"status": "tables created"}

@app.get("/debug-locations")
def debug_locations():
    with SessionLocal() as db:
        rows = db.execute(select(Location.row).distinct()).scalars().all()
        count = db.execute(select(func.count(Location.id))).scalar_one()
    return {"rows": rows, "count": count}

@app.get("/rebuild-locations")
def rebuild_locations():
    with SessionLocal() as db:
        db.execute(delete(Location))
        db.commit()

        for r in ROWS:
            slots = sorted(ALLOWED_SLOTS[r])
            for lvl in LEVELS:
                for s in slots:
                    db.add(Location(row=r, slot=s, level=lvl))

        db.commit()

    return {"status": "Locations rebuilt"}

from sqlalchemy import delete

from sqlalchemy import delete

@app.get("/force-rebuild-locations")
def force_rebuild_locations(operator: str | None = Cookie(default=None)):
    if not operator:
        return {"error": "Not logged in"}

    with SessionLocal() as db:
        # 1. Удаляем паллеты
        db.execute(delete(Pallet))

        # 2. Удаляем локации
        db.execute(delete(Location))

        db.commit()

        # 3. Пересоздаём локации
        for r in ROWS:
            slots = sorted(ALLOWED_SLOTS[r])
            for lvl in LEVELS:
                for s in slots:
                    db.add(Location(row=r, slot=s, level=lvl))

        db.commit()

    return {"status": "Locations rebuilt clean"}


@app.get("/check-j-count")
def check_j_count():
    with SessionLocal() as db:
        count = db.execute(
            select(func.count()).where(Location.row == "J")
        ).scalar()
    return {"J_count": count}

@app.get("/debug-find/{code}")
def debug_find(code: str):
    with SessionLocal() as db:
        loc = find_location(db, code)
        return {"found": bool(loc)}

@app.get("/debug-db")
def debug_db():
    return {"db_url": DB_URL}


from sqlalchemy import delete

@app.get("/nuke")
def nuke():
    with SessionLocal() as db:
        db.execute(delete(Pallet))
        db.execute(delete(Location))
        db.commit()
    return {"status": "nuked"}
#============================================#

@app.get("/create-zones-table")
def create_zones_table():
    Base.metadata.create_all(bind=engine)
    return {"status": "zones table created"}

@app.get("/init-zones")
def init_zones():
    with SessionLocal() as db:
        existing = db.query(Zone).filter_by(name="Chiller 10").first()
        if not existing:
            zone = Zone(name="Chiller 10", has_positions=True)
            db.add(zone)
            db.commit()
    return {"status": "Chiller 10 ready"}

@app.get("/assign-old-pallets")
def assign_old_pallets():
    with SessionLocal() as db:
        zone = db.query(Zone).filter_by(name="Chiller 10").first()
        if not zone:
            return {"error": "Zone not found"}

        pallets = db.query(Pallet).filter(Pallet.zone_id == None).all()

        for p in pallets:
            p.zone_id = zone.id

        db.commit()

    return {"status": "Old pallets assigned"}

from sqlalchemy import text

@app.get("/add-zone-column")
def add_zone_column():
    with engine.connect() as conn:
        conn.execute(
            text("ALTER TABLE pallets ADD COLUMN IF NOT EXISTS zone_id INTEGER")
        )
        conn.commit()
    return {"status": "zone_id column ensured"}

@app.get("/debug-zones")
def debug_zones():
    with SessionLocal() as db:
        zones = db.query(Zone).all()
        pallets = db.query(Pallet).limit(5).all()
        return {
            "zones": [z.name for z in zones],
            "sample_pallets": [
                {"id": p.id, "zone_id": p.zone_id}
                for p in pallets
            ]
        }

#===============================================#

























