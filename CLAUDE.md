# Insider Tracker

## Arkitektur
FI:s insynsregister → `providers/` (httpx) → `pipeline/` (pandas-rensning, laddning) → `db/` (SQLAlchemy, SQLite i dev) → `ai/` (Claude-summeringar) → `api/` (FastAPI) + `ui/` (Streamlit).

## Datakälla: FI:s insynsregister
- Export: `GET https://marknadssok.fi.se/Publiceringsklient/sv-SE/Search/Search` med `SearchFunctionType=Insyn`, `Publiceringsdatum.From`/`.To` (YYYY-MM-DD), övriga filter tomma, och `button=export`.
- Svar: CSV, **UTF-16LE**, semikolon, decimalkomma, datum med klockslag (`2026-11-02 00:00:00`), 22 kolumner.
- **Max 1000 rader per sökning** — hämta i små datumintervall.
- Exempelfil (riktig data, aldrig i git): `data/Insyn2026-11-03.csv`.

## Analysfönster runt rapporter
- Handelsstopp för personer i ledande ställning: 30 kalenderdagar före delårs- eller bokslutsrapport (MAR art. 19.11). Stoppet upphör när rapporten publiceras.
- Dag 0 = rapportdagen. Fönster: dag −44 till −31 (14 dagar före stoppet) och dag 0 till +14 (efter rapporten).
- Placera affärer i fönster efter transaktionsdatum, inte publiceringsdatum. Dag 0 är osäker: registret visar inte om affären gjordes före eller efter publiceringen.
- Rapportdatum och aktiekurser finns inte i FI:s register. Källa är inte vald.
- Datan visar när affärer skett, inte varför. Påstå aldrig motiv.

## Konventioner
- Python 3.12+, src-layout, paket `insider_tracker`.
- Kod, identifierare och commit-meddelanden på engelska.
- Bygg bara det som behövs nu. Inga spekulativa features.

## Regler
- Inga hemligheter i kod — allt via miljövariabler (`.env`, se `.env.example`).
- Logga aldrig personnamn — logga ID:n.
- AI:n lägger aldrig till siffror i databasen; den tolkar bara befintlig data. AI-text märks som genererad. Inga köp/sälj-råd.
- Varje rad i `trades` ska ha `source_url` till FI:s original.
- Tester anropar aldrig riktiga externa API:er (FI, Claude) — använd fixturer och mockar. Test-data innehåller bara fejkade namn.

## Kommandon
- Skapa miljön: `conda env create -f environment.yml`
- Aktivera miljön: `conda activate ipro`
- Tester: `pytest`
- Lint: `ruff check src/ tests/`
