"""Auditable equivalence rules. Precision belongs to the observation, not the grouping key."""
import re
import unicodedata
from decimal import Decimal

ALIASES = {alias: rod for rod in ("xn","xp","yn","yp") for alias in
           (rod, rod+" rod", "rod "+rod, rod[0]+"方向"+("非较多孔杆" if rod[1]=="n" else "较多孔杆"))}
PREDICATES = {"diameter":"has_diameter","length":"has_length","density":"has_density",
              "build_direction":"has_build_direction","test_of_specimen":"has_test_record_inverse"}
UNITS = {"m":("m",1),"mm":("m","0.001"),"cm":("m","0.01"),"kg/m3":("kg/m3",1),
         "g/cm3":("kg/m3",1000),"k":("K",1),"%":("1","0.01"),"m/s":("m/s",1)}
DEFAULT = {"has_diameter":"m","has_length":"m","has_density":"kg/m3"}
def entity(text):
    text=unicodedata.normalize("NFKC",text).strip()
    return ALIASES.get(text.lower(),ALIASES.get(text.lower().replace(" ",""),text))

def normalize(s,p,o,source_unit=None):
    s,p,o=entity(s),PREDICATES.get(p,p),entity(o)
    if p=="has_test_record_inverse": s,p,o=o,"has_test",s
    if p=="records_field": s=re.sub(r"^([xy][np])[1-5]$",r"\1",s)
    if p=="has_build_direction": o=o.lower()
    if p=="has_condition" and re.fullmatch(r"[+-]?[\d.]+\s*m/s",o):
        p="has_wind_speed"
    if p=="has_condition":
        temp=re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*°C boundary",o)
        if temp:
            value=Decimal(temp[1])
            kelvin=format(value+Decimal("273.15"),"f")
            return (s,"has_boundary_temperature",kelvin+" K"),dict(value=kelvin,si_unit="K",significant_digits=len(value.as_tuple().digits))
    m=re.fullmatch(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(mm|cm|m|kg/m3|g/cm3|K|%|m/s)?",o)
    if m and (m[2] or source_unit):
        unit=(m[2] or source_unit).lower()
        if unit in UNITS:
            si,factor=UNITS[unit]; number=Decimal(m[1])
            value=format((number*Decimal(factor)).normalize(),"f")
            return (s,p,value+" "+si),dict(value=value,si_unit=si,significant_digits=len(number.as_tuple().digits))
    return (s,p,o),None
