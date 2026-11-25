import xml.etree.ElementTree as ET

import numpy as np
import pytest

from run import validate_payload_geometry


def test_usd_float_mass_roundtrip_and_native_exact_mass():
    config = dict(size_m=[.15, .15, .15], mass_kg=.1)
    for mass in (.1, float(np.float32(.1))):
        geom = ET.Element('geom', size='.075 .075 .075', mass=str(mass))
        validate_payload_geometry(geom, config)


@pytest.mark.parametrize('mass', ['0.100001', '0.25', '0', '-0.1', 'nan', 'inf'])
def test_mass_mismatch_is_still_rejected(mass):
    geom = ET.Element('geom', size='.075 .075 .075', mass=mass)
    with pytest.raises(ValueError, match='Payload'):
        validate_payload_geometry(geom, dict(size_m=[.15]*3, mass_kg=.1))


def test_dimension_mismatch_is_still_rejected():
    geom = ET.Element('geom', size='.075 .075 .08', mass='.1')
    with pytest.raises(ValueError, match='Payload'):
        validate_payload_geometry(geom, dict(size_m=[.15]*3, mass_kg=.1))
