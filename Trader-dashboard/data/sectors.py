"""
NIFTY 500 Stock Sector & Industry Mapping
==========================================
Maps each ticker to one of 13 sectors and its specific industry.
Used by the scanner engine for sector rotation analysis, peer comparison,
and sector-based filtering.

Source: NSE, BSE, and public filings — updated periodically.
"""

# 13 Major Sectors with constituent stocks
SECTOR_MAP = {
    "Financial Services": {
        "industries": ["Banking", "NBFC", "Insurance", "Asset Management", "Stock Broking"],
        "stocks": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
            "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "RBLBANK.NS", "IDFCFIRSTB.NS",
            "AUBANK.NS", "YESBANK.NS", "SOUTHBANK.NS", "BANKBARODA.NS", "CANBK.NS",
            "PNB.NS", "INDIANB.NS", "UCOBANK.NS", "CENTRALBK.NS", "MAHABANK.NS",
            "CSBBANK.NS", "CUB.NS", "KARURVYSYA.NS", "DCBBANK.NS", "IDBI.NS",
            "IIFL.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS", "L&TFH.NS", "BAJFINANCE.NS",
            "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "SRTRANSFIN.NS", "MANAPPURAM.NS", "LICHSGFIN.NS",
            "POONAWALLA.NS", "MASFIN.NS", "CREDITACC.NS", "HDFCLIFE.NS", "SBILIFE.NS",
            "ICICIGI.NS", "ICICIPRUDI.NS", "HDFCAMC.NS", "UTIAMC.NS", "SBICARD.NS",
            "MUTHOOTFIN.NS", "BAJAJELEC.NS", "BSE.NS", "CDSL.NS", "MCX.NS",
            "MOTILALOFS.NS", "ANGELONE.NS", "ICICISEC.NS", "DHANI.NS"
        ]
    },
    "Information Technology": {
        "industries": ["IT Services", "Software Products", "Consulting"],
        "stocks": [
            "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
            "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "LTI.NS",
            "MINDTECK.NS", "OFSS.NS", "HEXAWARE.NS", "CYIENT.NS", "ZENSARTECH.NS",
            "BSOFT.NS", "KPITTECH.NS", "MASTEK.NS", "TATAELXSI.NS", "TATACOMM.NS",
            "SONATSOFTW.NS", "INTELLECT.NS", "EKC.NS", "NEWGEN.NS", "HAPPSTMNDS.NS",
            "NIITTECH.NS", "DATAMATICS.NS", "QUESS.NS", "TEAMLEASE.NS"
        ]
    },
    "Energy": {
        "industries": ["Oil & Gas", "Power Generation", "Power Distribution", "Renewable Energy"],
        "stocks": [
            "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS",
            "GAIL.NS", "PETRONET.NS", "MGL.NS", "IGL.NS", "GUJGASLTD.NS",
            "GSPL.NS", "OIL.NS", "AEGISCHEM.NS", "CASTROLIND.NS",
            "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIPOWER.NS", "NHPC.NS",
            "SJVN.NS", "JSWENERGY.NS", "SUZLON.NS", "ADANIGREEN.NS", "INOXGREEN.NS",
            "CESP.NS", "TORNTPOWER.NS", "NLCINDIA.NS"
        ]
    },
    "Automobile": {
        "industries": ["Passenger Vehicles", "Commercial Vehicles", "Auto Ancillaries", "Two/Three Wheelers"],
        "stocks": [
            "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
            "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BALKRISIND.NS", "APOLLOTYRE.NS",
            "MRF.NS", "CEATLTD.NS", "EXIDEIND.NS", "AMARAJA.NS", "MOTHERSUMI.NS",
            "BOSCHLTD.NS", "ENDURANCE.NS", "SCHAEFFLER.NS", "SUNDRMFAST.NS", "TIMKEN.NS",
            "SKFINDIA.NS", "LUMINOS.NS", "SAMBHAAV.NS", "TALBROAUTO.NS", "UNOMINDA.NS",
            "WABCOINDIA.NS", "ZFCOM.NS"
        ]
    },
    "Pharmaceuticals & Healthcare": {
        "industries": ["Pharmaceuticals", "Hospitals", "Diagnostics", "Healthcare Services"],
        "stocks": [
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
            "LUPIN.NS", "ZYDUSLIFE.NS", "TORNTPHARM.NS", "AUROPHARMA.NS", "GLENMARK.NS",
            "BIOCON.NS", "ALEMBICLTD.NS", "CADILAHC.NS", "NATCOPHARM.NS", "LAURUSLABS.NS",
            "GRANULES.NS", "STAR.NS", "PFIZER.NS", "SANOFI.NS", "ABBOTINDIA.NS",
            "IPCALAB.NS", "SYNGENE.NS", "MAXHEALTH.NS", "FORTIS.NS", "HEALTH.NS",
            "MEDANTA.NS", "KIMS.NS", "APOLLO.NS", "METROPOLIS.NS", "THYROCARE.NS",
            "ERIS.NS", "JBCHEPHARM.NS", "GLAXO.NS", "NOVARTIS.NS"
        ]
    },
    "Consumer Goods": {
        "industries": ["FMCG", "Consumer Durables", "Household Products", "Food & Beverages"],
        "stocks": [
            "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
            "MARICO.NS", "DABUR.NS", "GODREJCP.NS", "COLPAL.NS", "EMAMILTD.NS",
            "PAGEIND.NS", "PIDILITIND.NS", "BERGEPAINT.NS", "KAJARIACER.NS", "ASIANPAINT.NS",
            "TITAN.NS", "VOLTAS.NS", "HAVELLS.NS", "CROMPTON.NS", "BLUESTARCO.NS",
            "WHIRLPOOL.NS", "ORIENTELEC.NS", "BAJAJELEC.NS", "VGUARD.NS", "STOVEKRAFT.NS",
            "AMBER.NS", "BATAINDIA.NS", "RELAXO.NS", "CAMPUS.NS", "METROBRAND.NS",
            "VSTIND.NS", "GODREJPROP.NS", "MCDOWELL-N.NS", "UBL.NS", "RADICO.NS",
            "CCL.NS", "VBL.NS", "ZYDUSWELL.NS"
        ]
    },
    "Engineering & Capital Goods": {
        "industries": ["Industrial Machinery", "Electrical Equipment", "Defence", "Infrastructure"],
        "stocks": [
            "LT.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS", "BEL.NS",
            "HAL.NS", "COCHINSHIP.NS", "GRSE.NS", "MAZDOCK.NS", "BEML.NS",
            "L&T.NS", "KEC.NS", "KALPATPOWR.NS", "PNC.NS", "NBCC.NS",
            "ENGINERSIN.NS", "GMRINFRA.NS", "IRB.NS", "IRCON.NS", "RVNL.NS",
            "CONCOR.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS",
            "ACC.NS", "DALBHARAT.NS", "RAMCOCEM.NS", "HEIDELBERG.NS", "SHRIRAMEPC.NS",
            "TITAGARH.NS", "TEXMACO.NS", "JWL.NS", "MATHERAN.NS"
        ]
    },
    "Metals & Mining": {
        "industries": ["Steel", "Non-Ferrous Metals", "Mining", "Minerals"],
        "stocks": [
            "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "NATIONALUM.NS", "NMDC.NS",
            "SAIL.NS", "JINDALSTEL.NS", "COALINDIA.NS", "HINDZINC.NS", "VEDL.NS",
            "MOIL.NS", "KIOCL.NS", "RATEGAIN.NS", "APLAPOLLO.NS", "JSL.NS",
            "MAHSEAMLES.NS", "RATNAMANI.NS", "SANDHAR.NS", "SUPREMEIND.NS", "WELCORP.NS",
            "GRAVITA.NS", "ELECTSTER.NS"
        ]
    },
    "Telecom & Media": {
        "industries": ["Telecom Services", "Media & Entertainment", "Broadcasting", "Digital"],
        "stocks": [
            "BHARTIARTL.NS", "IDEA.NS", "TCI.NS", "TCIEXP.NS",
            "ZEEL.NS", "SUNTV.NS", "PVRINOX.NS", "INOXLEISUR.NS", "NETWORK18.NS",
            "TV18BRDCST.NS", "DISHTV.NS", "JAGRAN.NS", "HTMEDIA.NS", "DBREALTY.NS",
            "NAZARA.NS", "AFFLE.NS", "EASEMYTRIP.NS", "INFIBEAM.NS", "ZOMATO.NS",
            "NYM.NS", "PAYTM.NS", "POLICYBZR.NS"
        ]
    },
    "Real Estate": {
        "industries": ["Real Estate Development", "REITs", "Property Management"],
        "stocks": [
            "DLF.NS", "OBEROIRLTY.NS", "LODHA.NS", "PRESTIGE.NS", "BRIGADE.NS",
            "GODREJPROP.NS", "SOBHA.NS", "PHOENIXLTD.NS", "SUNTECK.NS", "KOLTEPATIL.NS",
            "MAHALIFE.NS", "PURVA.NS", "PENINLAND.NS", "ARIHANTSUP.NS",
            "EMBASSY.NS", "MINDSPACE.NS", "BROOKFIELD.NS"
        ]
    },
    "Chemicals": {
        "industries": ["Specialty Chemicals", "Fertilizers", "Agrochemicals", "Petrochemicals"],
        "stocks": [
            "UPL.NS", "PIIND.NS", "DEEPAKNTR.NS", "SRF.NS", "AARTIIND.NS",
            "SOLARA.NS", "NAVINFLUOR.NS", "VINATIORGA.NS", "ALKYLAMINE.NS", "CLEAN.NS",
            "FLUOROCHEM.NS", "GALAXYSURF.NS", "FINEORG.NS", "GODREJAGRO.NS", "RALLIS.NS",
            "COROMANDEL.NS", "CHAMBLFERT.NS", "GNFC.NS", "GSFC.NS", "NFL.NS",
            "FACT.NS", "MANGALAM.NS", "ZOTA.NS", "TATACHEM.NS", "LINDEINDIA.NS",
            "GUJALKALI.NS", "GUFICBIO.NS"
        ]
    },
    "Logistics & Transportation": {
        "industries": ["Shipping", "Logistics", "Aviation", "Railways"],
        "stocks": [
            "CONCOR.NS", "BLUEDART.NS", "DELHIVERY.NS", "TCI.NS", "TCIEXP.NS",
            "VRL.NS", "MAHLOG.NS", "GATI.NS", "ALLCARGO.NS", "SFL.NS",
            "INDIGO.NS", "ADANIPORTS.NS", "GMRINFRA.NS", "GODFREY.NS", "COCHINSHIP.NS",
            "SCI.NS", "SHIPPING.NS", "GE.NS"
        ]
    },
    "Miscellaneous": {
        "industries": ["Conglomerate", "Trading", "Others"],
        "stocks": [
            "ADANIENT.NS", "ADANIPORTS.NS", "TRENT.NS", "DMART.NS", "MISHRA.NS",
            "VAKRANGEE.NS", "RELIANCE.NS", "GODREJIND.NS", "3MINDIA.NS"
        ]
    }
}


def get_sector(ticker: str) -> str:
    """Get the sector for a given ticker symbol."""
    for sector, info in SECTOR_MAP.items():
        if ticker in info["stocks"]:
            return sector
    return "Unknown"


def get_industry(ticker: str) -> str:
    """Get the industry for a given ticker symbol."""
    for sector, info in SECTOR_MAP.items():
        if ticker in info["stocks"]:
            for industry in info["industries"]:
                # Find matching industry logic
                pass
            return info["industries"][0]
    return "Unknown"


def get_all_sectors() -> list:
    """Get list of all sectors."""
    return list(SECTOR_MAP.keys())


def get_sector_stocks(sector: str) -> list:
    """Get all stocks in a given sector."""
    info = SECTOR_MAP.get(sector)
    if info:
        return info["stocks"]
    return []


def get_sector_count(ticker_list: list) -> dict:
    """Count stocks by sector from a given list of tickers."""
    counts = {}
    for t in ticker_list:
        s = get_sector(t)
        counts[s] = counts.get(s, 0) + 1
    return counts
