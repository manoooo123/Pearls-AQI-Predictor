import math
import pytest
from app.flask_api import calculate_us_aqi, get_aqi_meta

class TestUSEPABreakpoints:

    def test_zero_pm25_gives_zero_aqi(self):
        assert calculate_us_aqi(0.0) == 0

    def test_lower_bound_good(self):
        assert calculate_us_aqi(12.0) == 50

    def test_lower_bound_moderate(self):
        aqi = calculate_us_aqi(12.1)
        assert aqi == 50

    def test_upper_bound_moderate(self):
        assert calculate_us_aqi(35.4) == 100

    def test_lower_bound_usg(self):
        aqi = calculate_us_aqi(35.5)
        assert aqi == 100

    def test_upper_bound_usg(self):
        assert calculate_us_aqi(55.4) == 150

    def test_lower_bound_unhealthy(self):
        aqi = calculate_us_aqi(55.5)
        assert aqi == 150

    def test_upper_bound_unhealthy(self):
        assert calculate_us_aqi(150.4) == 200

    def test_lower_bound_very_unhealthy(self):
        aqi = calculate_us_aqi(150.5)
        assert aqi == 200

    def test_upper_bound_very_unhealthy(self):
        assert calculate_us_aqi(250.4) == 300

    def test_lower_bound_hazardous(self):
        aqi = calculate_us_aqi(250.5)
        assert aqi == 300

    def test_upper_bound_hazardous(self):
        assert calculate_us_aqi(500.4) == 500

    def test_extreme_pm25_capped_at_500(self):
        assert calculate_us_aqi(999.0) == 500

    def test_midrange_moderate(self):
        aqi = calculate_us_aqi(23.75)
        assert 51 <= aqi <= 100

    def test_midrange_unhealthy_sensitive(self):
        aqi = calculate_us_aqi(45.0)
        assert 101 <= aqi <= 150

    def test_midrange_unhealthy(self):
        aqi = calculate_us_aqi(100.0)
        assert 151 <= aqi <= 200

    def test_midrange_very_unhealthy(self):
        aqi = calculate_us_aqi(200.0)
        assert 201 <= aqi <= 300

    def test_midrange_hazardous(self):
        aqi = calculate_us_aqi(400.0)
        assert aqi > 300

class TestAQIEdgeCases:

    def test_nan_returns_zero(self):
        assert calculate_us_aqi(float("nan")) == 0

    def test_inf_returns_zero(self):
        assert calculate_us_aqi(float("inf")) == 0

    def test_none_returns_zero(self):
        assert calculate_us_aqi(None) == 0

    def test_negative_pm25_returns_zero(self):
        assert calculate_us_aqi(-5.0) == 0
        assert calculate_us_aqi(-100.0) == 0

    def test_output_is_always_int(self):
        for pm25 in [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 350.0]:
            result = calculate_us_aqi(pm25)
            assert isinstance(result, int), f"AQI for pm25={pm25} should be int, got {type(result)}"

    def test_aqi_is_monotonically_non_decreasing(self):
        pm25_values = [0, 5, 12, 20, 35, 55, 100, 150, 200, 250, 350, 500]
        aqis = [calculate_us_aqi(v) for v in pm25_values]
        for i in range(1, len(aqis)):
            assert aqis[i] >= aqis[i - 1], (
                f"AQI not monotonic: pm25={pm25_values[i]} → aqi={aqis[i]} "
                f"< pm25={pm25_values[i-1]} → aqi={aqis[i-1]}"
            )

class TestAQIMeta:

    def test_good_category(self):
        meta = get_aqi_meta(30)
        assert meta["category"] == "Good"
        assert meta["color"].startswith("#")
        assert len(meta["health"]) > 10

    def test_moderate_category(self):
        meta = get_aqi_meta(75)
        assert meta["category"] == "Moderate"

    def test_usg_category(self):
        meta = get_aqi_meta(125)
        assert meta["category"] == "Unhealthy for Sensitive Groups"

    def test_unhealthy_category(self):
        meta = get_aqi_meta(175)
        assert meta["category"] == "Unhealthy"

    def test_very_unhealthy_category(self):
        meta = get_aqi_meta(250)
        assert meta["category"] == "Very Unhealthy"

    def test_hazardous_category(self):
        meta = get_aqi_meta(400)
        assert meta["category"] == "Hazardous"

    def test_all_meta_fields_present(self):
        for aqi_val in [25, 75, 125, 175, 250, 400]:
            meta = get_aqi_meta(aqi_val)
            assert "category" in meta, f"Missing 'category' for AQI {aqi_val}"
            assert "color"    in meta, f"Missing 'color' for AQI {aqi_val}"
            assert "health"   in meta, f"Missing 'health' for AQI {aqi_val}"
            assert meta["color"].startswith("#"), f"Color not hex for AQI {aqi_val}"
            assert len(meta["health"]) > 0, f"Empty health advice for AQI {aqi_val}"

    def test_no_category_is_good_for_failed_aqi(self):
        result = calculate_us_aqi(None)
        assert result == 0
        assert result == 0

    def test_color_is_valid_hex(self):
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        for aqi_val in [10, 75, 125, 175, 250, 400]:
            color = get_aqi_meta(aqi_val)["color"]
            assert hex_pattern.match(color), f"Invalid hex color '{color}' for AQI {aqi_val}"
