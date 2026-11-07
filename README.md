# Insider Tracker

När en vd eller styrelseledamot köper eller säljer aktier i sitt eget bolag måste det anmälas till Finansinspektionen, och allt hamnar i ett öppet register. Datan finns alltså där, men den är svår att följa och få överblick över.

Insider Tracker hämtar insynsaffärerna från FI, kopplar dem till bolagens rapportdagar och visar hur ledningar handlar runt rapporterna. Vad gjorde styrelsen två veckor innan det lagstadgade handelsstoppet inför rapporten, och under två veckor efter rapporten? Claude kan sammanfatta ett bolags affärer, och en liten agent kan svara på öppna frågor om datan.

## Vad den gör

- Hämtar insynsaffärer från FI:s register, två veckor åt gången och med nya försök vid tillfälliga fel.
- Rensar datan: tal, datum, korrigerade rader och personuppgifter som inte behövs.
- Kopplar affärerna till bolagens rapportdagar via ISIN och LEI.
- Placerar affärerna i två fönster: dag −44 till −31 före rapporten och dag 0 till +14 efter.
- Filtrerar fram marknadsköp och marknadsförsäljningar av aktier, slår ihop affärer från samma dag och tillämpar beloppsgränser per segment.
- Sparar allt i en databas utan dubbletter, och senare körningar hämtar bara det nya.
- Skriver AI-sammanfattningar per bolag med Claude.
- Har en agent som själv väljer bland verktyg som bara läser data.
- Visar allt i en dashboard med filter, diagram och tabell.
- Har ett API som ger affärer, bolag, fönsterstatistik och sammanfattningar som JSON.

## Byggt med

Python, pandas, SQLAlchemy och SQLite, FastAPI, Streamlit och Altair, Claude API, pytest, ruff och GitHub Actions.

## Struktur

```
src/insider_tracker/
├── providers/   hämtar data från FI och rapportkalendrar
├── pipeline/    rensning, koppling, filter, inläsning och analys
├── db/          tabeller och frågor
├── ai/          sammanfattning och agent
├── api/         API med FastAPI
└── ui/          dashboard
```

## Köra lokalt

```bash
conda env create -f environment.yml
conda activate ipro
cp .env.example .env              # lägg in ANTHROPIC_API_KEY
pytest
```

Kedjan förutsätter att rapportdatum finns i `data/report_dates.csv`. Nya rapportdatum kan läggas till från en egen lista, till exempel hämtad från bolagens IR-sidor, med `python -m insider_tracker.pipeline.loader --reports <fil>`.

```bash
python -m insider_tracker.pipeline.loader       # hämtar affärer och sparar i databasen
python -m insider_tracker.pipeline.analysis     # räknar affärer per fönster och befattning
streamlit run src/insider_tracker/ui/app.py     # öppnar dashboarden
uvicorn insider_tracker.api.main:app --reload   # startar API:t, dokumentation på /docs
```

API:t läser bara och visar inga personnamn:

| Anrop | Ger |
|---|---|
| `GET /companies` | Bolag med affärer i fönstren |
| `GET /trades` | Affärer, filtrerbara på bolag, fönster, typ och befattning, högst 500 per anrop |
| `GET /windows` | Antal affärer per fönster, befattning och typ |
| `GET /companies/{bolag}/summary` | Bolagets senaste AI-sammanfattning |

AI-delarna gör betalda anrop till Claude:

```bash
python -m insider_tracker.ai.summarizer --company "Bolagsnamn" --dry-run   # visar vad som skickas, gratis
python -m insider_tracker.ai.summarizer --company "Bolagsnamn"
python -m insider_tracker.ai.agent "Vilka bolag sticker ut med köp efter rapporterna?"
```

## AI-delarna

- **Sammanfattning:** ett enda anrop per bolag. Koden väljer datan och Claude tolkar den.
- **Agent:** en egen agentloop där Claude väljer bland tre verktyg som bara läser data, med tak för antal steg.
- **Gemensamt:** personnamn skickas aldrig. Svaren märks som AI-genererade, och reglerna förbjuder råd, påhittade siffror och påståenden om varför någon handlade.
- **Testning:** testerna använder en fejkad modell och anropar aldrig API:t.

AI-texten ska alltid granskas mot källdatan. Språkmodeller kan formulera fel även när siffrorna stämmer.

## Om datan

Affärerna kommer från FI:s offentliga register, och varje rad har en länk till originalet. Registret innehåller namn på personer, så projektet sparar bara det som behövs, loggar inga namn och visar inga namn i dashboarden. Riktig data och databasen ligger aldrig i git.

Datan visar när affärer skett, inte varför. Inget här är investeringsråd.

## Begränsningar

- Rapportdatum finns ungefär ett år bakåt, och alla bolag har inte rapportdatum.
- Registret visar inte personens innehav, så det går inte att se hur stor del som såldes.
- Belopp räknas bara i SEK. Affärer i andra valutor saknar belopp.
- Dag 0 är osäker, eftersom en affär samma dag kan ha gjorts före rapporten.
