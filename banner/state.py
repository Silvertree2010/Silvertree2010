"""Current conditions: time, sun, moon, weather, season."""
import json, math, os, urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import astro

# Location comes from repo secrets so it never appears in the code
LAT = float(os.environ.get("BANNER_LAT") or "46.80")
LON = float(os.environ.get("BANNER_LON") or "8.23")
TZ = ZoneInfo("Europe/Zurich")
API = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&current=temperature_2m,weather_code,cloud_cover,precipitation,wind_speed_10m,"
       "wind_direction_10m&timezone=Europe%2FZurich")


@dataclass
class Weather:
    code: int = 0            # WMO code
    cloud: float = 0.1       # 0..1
    temp: float = 10.0       # °C
    wind: float = 5.0        # km/h
    wind_dir: float = 270.0  # from, degrees
    source: str = "default"

    @property
    def fog(self):
        return self.code in (45, 48)

    @property
    def thunder(self):
        return self.code >= 95

    @property
    def precip(self):
        """-> (kind, intensity 0..1) with kind in {None, 'rain', 'snow'}"""
        c = self.code
        if c in (71, 73, 75, 77, 85, 86):
            return "snow", {71: .35, 73: .65, 75: 1, 77: .3, 85: .5, 86: .9}[c]
        if c in (51, 53, 55, 56, 57):
            kind = "snow" if self.temp < 0.5 else "rain"
            return kind, {51: .2, 53: .3, 55: .4, 56: .3, 57: .45}[c]
        if c in (61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
            kind = "snow" if self.temp < 0.5 else "rain"
            return kind, {61: .4, 63: .65, 65: 1, 66: .5, 67: .8, 80: .45, 81: .7, 82: 1,
                          95: .9, 96: 1, 99: 1}[c]
        return None, 0.0

    @property
    def label(self):
        kind, k = self.precip
        if self.thunder: return "thunderstorm"
        if self.fog: return "fog"
        if kind == "snow": return "snow"
        if kind == "rain": return "rain"
        if self.cloud > .85: return "overcast"
        if self.cloud > .45: return "cloudy"
        if self.cloud > .15: return "partly cloudy"
        return "clear"


def fetch_weather(timeout=10) -> Weather:
    try:
        with urllib.request.urlopen(API.format(lat=LAT, lon=LON), timeout=timeout) as r:
            cur = json.load(r)["current"]
        return Weather(code=int(cur["weather_code"]), cloud=cur["cloud_cover"] / 100,
                       temp=cur["temperature_2m"], wind=cur["wind_speed_10m"],
                       wind_dir=cur["wind_direction_10m"], source="open-meteo")
    except Exception as e:  # no network → clear default weather, banner still renders
        print(f"weather unavailable ({e}), using default")
        return Weather()


SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}

# snow line as fraction of mountain height (0 = all white, 1 = snow-free), mid-month
SNOWLINE = [.12, .14, .28, .48, .68, .84, .92, .93, .86, .7, .45, .2]


@dataclass
class Scene:
    now: datetime
    weather: Weather
    sun_alt: float = 0
    sun_az: float = 0
    moon_alt: float = 0
    moon_az: float = 0
    moon_frac: float = 0
    moon_waxing: bool = True
    season: str = "autumn"
    snowline: float = .7
    year_phase: float = 0       # 0..1 through the year
    seed: int = 0
    extra: dict = field(default_factory=dict)


def build_scene(now: datetime, weather: Weather) -> Scene:
    now = now.astimezone(TZ)
    s_alt, s_az = astro.sun(now, LAT, LON)
    m_alt, m_az, frac, wax = astro.moon(now, LAT, LON)
    doy = now.timetuple().tm_yday
    # interpolate the snow line smoothly between mid-months
    m = (now.month - 1) + (now.day - 15) / 30
    i0 = math.floor(m) % 12
    t = m - math.floor(m)
    snow = SNOWLINE[i0] * (1 - t) + SNOWLINE[(i0 + 1) % 12] * t
    snow -= max(0, -weather.temp) * .02        # frost pushes the snow line down
    if weather.precip[0] == "snow":
        snow -= .12
    return Scene(now=now, weather=weather, sun_alt=s_alt, sun_az=s_az,
                 moon_alt=m_alt, moon_az=m_az, moon_frac=frac, moon_waxing=wax,
                 season=SEASONS[now.month], snowline=max(0.02, min(.97, snow)),
                 year_phase=doy / 365.25,
                 seed=int(now.strftime("%Y%m%d%H")))
